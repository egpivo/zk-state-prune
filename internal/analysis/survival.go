package analysis

import (
	"encoding/json"
	"fmt"
	"math"
	"sort"

	"github.com/kshedden/statmodel/duration"

	"github.com/egpivo/zk-state-prune/internal/model"
)

// KMResult is a fitted Kaplan–Meier curve with the diagnostics a pruning
// policy actually needs. Time[i], Surv[i], SE[i], NumRisk[i] are aligned.
//
// For pruning we typically ask "what is S(t) at some horizon t?" — use
// SurvAt for that; the curve is a step function so we pick the largest
// sampled time that is <= t.
type KMResult struct {
	Label       string // stratum label, e.g. "erc20" or "balance"
	N           int
	NumEvents   int
	NumCensored int
	Time        []float64
	Surv        []float64
	SE          []float64
	NumRisk     []float64
	// MedianSurv is the smallest sampled time t with Surv(t) <= 0.5, or
	// NaN if the curve never crosses 0.5 (heavy censoring).
	MedianSurv float64
}

// MarshalJSON renders a KMResult safely for `encoding/json`, replacing
// the NaN-valued MedianSurv (which Go's encoder refuses to emit) with
// a JSON `null`. Every other field is a plain numeric or slice and
// marshals with default semantics.
func (k *KMResult) MarshalJSON() ([]byte, error) {
	type alias struct {
		Label       string    `json:"label"`
		N           int       `json:"n"`
		NumEvents   int       `json:"num_events"`
		NumCensored int       `json:"num_censored"`
		Time        []float64 `json:"time"`
		Surv        []float64 `json:"surv"`
		SE          []float64 `json:"se"`
		NumRisk     []float64 `json:"num_risk"`
		MedianSurv  *float64  `json:"median_surv"`
	}
	out := alias{
		Label:       k.Label,
		N:           k.N,
		NumEvents:   k.NumEvents,
		NumCensored: k.NumCensored,
		Time:        k.Time,
		Surv:        k.Surv,
		SE:          k.SE,
		NumRisk:     k.NumRisk,
	}
	if !math.IsNaN(k.MedianSurv) && !math.IsInf(k.MedianSurv, 0) {
		v := k.MedianSurv
		out.MedianSurv = &v
	}
	return json.Marshal(out)
}

// SurvAt returns S(t) via right-continuous step interpolation. For t before
// the first event, returns 1. For t after the last sample, returns the last
// observed survival probability (the curve may not reach 0 under censoring).
func (k *KMResult) SurvAt(t float64) float64 {
	if len(k.Time) == 0 {
		return 1
	}
	if t < k.Time[0] {
		return 1
	}
	// Binary search: largest i with Time[i] <= t.
	i := sort.Search(len(k.Time), func(i int) bool { return k.Time[i] > t }) - 1
	if i < 0 {
		return 1
	}
	return k.Surv[i]
}

// SurvivalFitter abstracts the underlying survival library so we can swap
// implementations without touching callers. The full interface is the
// Phase-1 final shape:
//
//   - FitKaplanMeier and FitCoxPH return raw fits.
//   - CheckPH runs the Schoenfeld-residual PH-assumption test on a Cox fit
//     and returns per-covariate p-values plus a global p-value.
//   - Calibrate fits an isotonic recalibration on a held-out interval set
//     and returns a CalibratedModel that overrides PredictAccessProb to
//     emit recalibrated probabilities.
//
// Segments 9–11 implement CheckPH / Calibrate; today they return
// ErrNotImplemented but the interface is already final so callers can be
// written and tested against the eventual shape.
type SurvivalFitter interface {
	FitKaplanMeier(intervals []model.InterAccessInterval) (*KMResult, error)
	FitCoxPH(intervals []model.InterAccessInterval, covariates []string) (*CoxResult, error)
	CheckPH(result *CoxResult) (*PHTestResult, error)
	Calibrate(result *CoxResult, holdout []model.InterAccessInterval) (*CalibratedModel, error)
}

// StatmodelFitter is the kshedden/statmodel/duration-backed implementation
// of SurvivalFitter.
type StatmodelFitter struct{}

// NewStatmodelFitter returns a ready-to-use StatmodelFitter. It is stateless
// and safe for concurrent use.
func NewStatmodelFitter() StatmodelFitter { return StatmodelFitter{} }

// FitKaplanMeier fits a single (non-stratified) KM curve over all intervals.
func (StatmodelFitter) FitKaplanMeier(intervals []model.InterAccessInterval) (*KMResult, error) {
	if len(intervals) == 0 {
		return nil, fmt.Errorf("FitKaplanMeier: empty intervals")
	}
	ds := ToSurvivalDataset(intervals)
	// NOTE: we deliberately pass nil (no EntryVar). kshedden/statmodel's
	// SurvfuncRight aborts the process via os.Exit(1) when any row has
	// entry >= time, which in our gap-time formulation fires for every
	// Duration == 0 interval (legitimate co-accesses in the same block).
	// Gap-time KM does not need delayed entry anyway — left truncation is
	// absorbed into the interval construction in BuildIntervals. When we
	// add calendar-time Cox in Phase 2 we'll reintroduce EntryVar there.
	sf, err := duration.NewSurvfuncRight(ds, ColDuration, ColStatus, nil)
	if err != nil {
		return nil, fmt.Errorf("NewSurvfuncRight: %w", err)
	}
	sf.Fit()

	res := &KMResult{
		N:       len(intervals),
		Time:    append([]float64(nil), sf.Time()...),
		Surv:    append([]float64(nil), sf.SurvProb()...),
		SE:      append([]float64(nil), sf.SurvProbSE()...),
		NumRisk: append([]float64(nil), sf.NumRisk()...),
	}
	for _, it := range intervals {
		if it.IsObserved {
			res.NumEvents++
		} else {
			res.NumCensored++
		}
	}
	res.MedianSurv = medianSurvivalTime(res.Time, res.Surv)
	return res, nil
}

// medianSurvivalTime is the smallest t with S(t) <= 0.5, or NaN.
func medianSurvivalTime(t, s []float64) float64 {
	for i, sv := range s {
		if sv <= 0.5 {
			return t[i]
		}
	}
	return math.NaN()
}

// FitKaplanMeierStratified partitions intervals by key and fits one curve
// per stratum. The returned map is keyed on the stratum label (stable,
// deterministic iteration-order inside this function).
func FitKaplanMeierStratified(
	fitter SurvivalFitter,
	intervals []model.InterAccessInterval,
	key func(model.InterAccessInterval) string,
) (map[string]*KMResult, error) {
	if fitter == nil {
		return nil, fmt.Errorf("FitKaplanMeierStratified: nil fitter")
	}
	groups := make(map[string][]model.InterAccessInterval)
	for _, it := range intervals {
		k := key(it)
		groups[k] = append(groups[k], it)
	}
	labels := make([]string, 0, len(groups))
	for k := range groups {
		labels = append(labels, k)
	}
	sort.Strings(labels)

	out := make(map[string]*KMResult, len(groups))
	for _, label := range labels {
		res, err := fitter.FitKaplanMeier(groups[label])
		if err != nil {
			return nil, fmt.Errorf("stratum %q: %w", label, err)
		}
		res.Label = label
		out[label] = res
	}
	return out, nil
}

// StratumByContractType is a convenience key function for stratification.
func StratumByContractType(it model.InterAccessInterval) string { return it.ContractType.String() }

// StratumBySlotType is a convenience key function for stratification.
func StratumBySlotType(it model.InterAccessInterval) string { return it.SlotType.String() }
