package analysis

import (
	"fmt"
	"math"

	"github.com/kshedden/statmodel/duration"
	"github.com/kshedden/statmodel/statmodel"

	"github.com/egpivo/zk-state-prune/internal/model"
)

// CoxResult is a fitted Cox proportional-hazards model with the bookkeeping
// that subsequent calibration / diagnostics passes need.
//
// The fit is gap-time: TimeVar=Duration, StatusVar=Status, EntryVar=Entry.
// Predictors are continuous covariates pulled out of InterAccessInterval and
// standardized (mean-zero, unit-variance) before the fit, because the raw
// covariates span 0..1e6 and the optimizer becomes ill-conditioned without
// rescaling. The Scales slice records each predictor's pre-fit mean and
// std so PredictAccessProb can apply the same transformation to fresh rows.
//
// BaselineTime / BaselineCumHaz are read straight from
// duration.PHReg.BaselineCumHaz on stratum 0 (no stratification yet — that
// is a Phase-1 follow-up if the PH check confirms severe per-category
// violations).
type CoxResult struct {
	Predictors []string
	Coef       []float64
	StdErr     []float64
	PValue     []float64
	ZScore     []float64
	LogLike    float64

	NumObs    int
	NumEvents int

	Scales []CovarScale

	BaselineTime   []float64
	BaselineCumHaz []float64

	// intervals is the training set, retained so post-fit diagnostics
	// (CheckPH, calibration) don't have to be re-loaded from the DB.
	// Phase-1 sizes (≤1e5 intervals × ~100B each) make the memory cost
	// trivial compared with the convenience of a self-contained result.
	intervals []model.InterAccessInterval
}

// CovarScale records the centering / scaling applied to one predictor at
// fit time so prediction can re-apply the same transform.
type CovarScale struct {
	Name string
	Mean float64
	Std  float64
}

// DefaultCoxPredictors is the predictor set used by Phase-1 callers that
// don't want to think about which columns to feed Cox. They are the
// continuous covariates that the interval builder snapshots at
// IntervalStart, which are the things an online pruning policy could
// realistically read off the live state.
//
// ContractAge is intentionally absent: in Phase-1 BuildIntervals does not
// yet fetch a slot's parent contract deploy block, so it falls back to
// using slot.CreatedAt for both ages, making ContractAge perfectly
// collinear with SlotAge. Feeding both to Cox produces a singular
// Hessian and a non-convergent fit. We add ContractAge back the moment
// IterateSlotEvents surfaces deploy_block — see the TODO in
// internal/analysis/intervals.go's contractAgeAt closure.
var DefaultCoxPredictors = []string{ColAccessCount, ColSlotAge}

// FitCoxPH fits a Cox proportional-hazards model on intervals using
// kshedden/statmodel/duration. predictors is the ordered list of column
// names from the dstream adapter; pass DefaultCoxPredictors for the
// Phase-1 standard set.
//
// Landmines we navigated:
//   - PHReg panics (not error returns) on entry > time and on negative
//     time. Our intervals always have Entry=0 and Duration>=1 (the
//     dedupe + entry-skip work in BuildIntervals enforces this), so both
//     guards are satisfied. We still defensively reject any negative
//     duration here so the panic stays in test territory.
//   - Predictors at vastly different scales (AccessCount ~1, ContractAge
//     ~1e5, SlotAge ~1e5) make the optimizer wander or hit Inf. We
//     standardize before the fit and store the inverse transform.
//   - The library's "Status must be 0/1" check is satisfied by the
//     dstream adapter's encoding of IsObserved.
func (StatmodelFitter) FitCoxPH(intervals []model.InterAccessInterval, covariates []string) (*CoxResult, error) {
	if len(intervals) == 0 {
		return nil, fmt.Errorf("FitCoxPH: empty intervals")
	}
	if len(covariates) == 0 {
		covariates = DefaultCoxPredictors
	}
	for _, it := range intervals {
		if it.Duration > math.MaxInt64 { // unreachable in practice; keeps the panic in tests only
			return nil, fmt.Errorf("FitCoxPH: duration overflow at slot %s", it.SlotID)
		}
	}

	// Materialize a fresh column set so we can standardize in place
	// without mutating the caller's view of the dataset.
	cols, names := buildCoxColumns(intervals)
	scales := standardizeColumns(cols, names, covariates)

	ds := statmodel.NewDataset(cols, names)
	cfg := duration.DefaultPHRegConfig()
	cfg.EntryVar = ColEntry

	ph, err := duration.NewPHReg(ds, ColDuration, ColStatus, covariates, cfg)
	if err != nil {
		return nil, fmt.Errorf("NewPHReg: %w", err)
	}
	res, err := ph.Fit()
	if err != nil {
		return nil, fmt.Errorf("Fit: %w", err)
	}

	out := &CoxResult{
		Predictors: append([]string(nil), covariates...),
		Coef:       append([]float64(nil), res.Params()...),
		StdErr:     append([]float64(nil), res.StdErr()...),
		PValue:     append([]float64(nil), res.PValues()...),
		ZScore:     append([]float64(nil), res.ZScores()...),
		LogLike:    res.LogLike(),
		NumObs:     len(intervals),
		Scales:     scales,
	}
	for _, it := range intervals {
		if it.IsObserved {
			out.NumEvents++
		}
	}
	t, h := ph.BaselineCumHaz(0, out.Coef)
	out.BaselineTime = append([]float64(nil), t...)
	out.BaselineCumHaz = append([]float64(nil), h...)
	out.intervals = append([]model.InterAccessInterval(nil), intervals...)
	return out, nil
}

// rawCovariate pulls a single covariate value out of an interval by
// predictor-column name. Returns 0 for unknown names so callers don't
// have to special-case missing optional predictors.
func rawCovariate(it model.InterAccessInterval, name string) float64 {
	switch name {
	case ColAccessCount:
		return float64(it.AccessCount)
	case ColContractAge:
		return float64(it.ContractAge)
	case ColSlotAge:
		return float64(it.SlotAge)
	case ColContractType:
		return float64(it.ContractType)
	case ColSlotType:
		return float64(it.SlotType)
	}
	return 0
}

// PredictAccessProb returns the probability that a slot with the given
// raw covariate vector is accessed within the next tau blocks, assuming
// it has just been touched (idle = 0).
//
//	S(t | x) = exp(-H_0(t) * exp(x'β))
//	P(access by t) = 1 - S(t | x)
//
// covariates must be keyed by predictor name and contain at least the
// keys in r.Predictors. Missing keys are treated as zero (post-scaling
// equivalent to "the population mean") so callers can omit ones they
// don't have on hand.
func (r *CoxResult) PredictAccessProb(covariates map[string]float64, tau float64) float64 {
	if r == nil || len(r.BaselineTime) == 0 {
		return 0
	}
	lp := 0.0
	for i, name := range r.Predictors {
		raw := covariates[name]
		var z float64
		if i < len(r.Scales) && r.Scales[i].Std > 0 {
			z = (raw - r.Scales[i].Mean) / r.Scales[i].Std
		}
		lp += r.Coef[i] * z
	}
	h := baselineCumHazAt(r.BaselineTime, r.BaselineCumHaz, tau)
	surv := math.Exp(-h * math.Exp(lp))
	if surv < 0 {
		surv = 0
	}
	if surv > 1 {
		surv = 1
	}
	return 1 - surv
}

// PredictAccessProbForInterval is a convenience wrapper that pulls
// covariates straight off an InterAccessInterval. It iterates whatever
// predictor set the model was fit with — hard-coding a fixed map would
// silently zero out any predictor outside the default set (e.g. a model
// fit with ContractType / SlotType would be evaluated on incomplete
// covariates and return wrong probabilities).
func (r *CoxResult) PredictAccessProbForInterval(it model.InterAccessInterval, tau float64) float64 {
	if r == nil {
		return 0
	}
	cov := make(map[string]float64, len(r.Predictors))
	for _, name := range r.Predictors {
		cov[name] = rawCovariate(it, name)
	}
	return r.PredictAccessProb(cov, tau)
}

// baselineCumHazAt is a right-continuous step lookup on a (time, H0) grid.
// For tau before the first time we return 0; for tau after the last we
// return the final cumulative hazard.
func baselineCumHazAt(time, hz []float64, tau float64) float64 {
	if len(time) == 0 {
		return 0
	}
	if tau < time[0] {
		return 0
	}
	if tau >= time[len(time)-1] {
		return hz[len(hz)-1]
	}
	// Binary search for the largest index i with time[i] <= tau.
	lo, hi := 0, len(time)-1
	for lo < hi {
		mid := (lo + hi + 1) / 2
		if time[mid] <= tau {
			lo = mid
		} else {
			hi = mid - 1
		}
	}
	return hz[lo]
}

// buildCoxColumns is the same shape as ToSurvivalDataset but returns the
// raw [][]float64 + names so we can standardize in place before handing
// the result to statmodel.NewDataset. Keeping it separate from
// ToSurvivalDataset means the KM path stays untouched.
func buildCoxColumns(intervals []model.InterAccessInterval) ([][]float64, []string) {
	n := len(intervals)
	dur := make([]float64, n)
	status := make([]float64, n)
	entry := make([]float64, n)
	accessCount := make([]float64, n)
	contractAge := make([]float64, n)
	slotAge := make([]float64, n)
	contractType := make([]float64, n)
	slotType := make([]float64, n)
	for i, it := range intervals {
		dur[i] = float64(it.Duration)
		if it.IsObserved {
			status[i] = 1
		}
		accessCount[i] = float64(it.AccessCount)
		contractAge[i] = float64(it.ContractAge)
		slotAge[i] = float64(it.SlotAge)
		contractType[i] = float64(it.ContractType)
		slotType[i] = float64(it.SlotType)
	}
	cols := [][]float64{dur, status, entry, accessCount, contractAge, slotAge, contractType, slotType}
	names := []string{ColDuration, ColStatus, ColEntry, ColAccessCount, ColContractAge, ColSlotAge, ColContractType, ColSlotType}
	return cols, names
}

// standardizeColumns rescales each predictor column to mean zero, unit
// variance, mutating the slice in place. It returns the inverse transform
// so prediction can re-scale fresh rows. Predictors with zero variance
// are left untouched and recorded with Std=1 so PredictAccessProb's
// (raw-mean)/std step is a no-op.
func standardizeColumns(cols [][]float64, names []string, predictors []string) []CovarScale {
	scales := make([]CovarScale, 0, len(predictors))
	for _, p := range predictors {
		idx := -1
		for i, n := range names {
			if n == p {
				idx = i
				break
			}
		}
		if idx < 0 {
			scales = append(scales, CovarScale{Name: p, Mean: 0, Std: 1})
			continue
		}
		col := cols[idx]
		mean := 0.0
		for _, v := range col {
			mean += v
		}
		if len(col) > 0 {
			mean /= float64(len(col))
		}
		variance := 0.0
		for _, v := range col {
			d := v - mean
			variance += d * d
		}
		if len(col) > 1 {
			variance /= float64(len(col) - 1)
		}
		std := math.Sqrt(variance)
		if std == 0 {
			scales = append(scales, CovarScale{Name: p, Mean: mean, Std: 1})
			continue
		}
		for i := range col {
			col[i] = (col[i] - mean) / std
		}
		scales = append(scales, CovarScale{Name: p, Mean: mean, Std: std})
	}
	return scales
}
