package analysis

import (
	"encoding/json"
	"fmt"
	"log/slog"
	"math"
	"sort"

	"github.com/kshedden/statmodel/duration"
	"github.com/kshedden/statmodel/statmodel"

	"github.com/egpivo/zk-state-prune/internal/domain"
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

	// --- Stratified fit (Phase 3) ----------------------------------
	//
	// StratumColumn is the dstream column name the fit stratified on
	// (e.g. ColContractType). Empty for unstratified fits. When set,
	// Survival routes through per-stratum baseline hazards keyed on
	// the interval's value in that column.
	StratumColumn string
	// StratumLabels[i] is the training-time value of StratumColumn
	// for stratum i. Prediction-time lookup matches the row's column
	// value against this list to pick the right baseline; rows whose
	// value never appeared in training fall back to the first stratum.
	StratumLabels []float64
	// StratumBaselineTimes[i] / StratumBaselineCumHaz[i] are the
	// per-stratum baseline hazard grids. Same length as StratumLabels.
	StratumBaselineTimes  [][]float64
	StratumBaselineCumHaz [][]float64

	// intervals is the training set, retained so post-fit diagnostics
	// (CheckPH, calibration) don't have to be re-loaded from the DB.
	// Phase-1 sizes (≤1e5 intervals × ~100B each) make the memory cost
	// trivial compared with the convenience of a self-contained result.
	intervals []domain.InterAccessInterval
}

// CovarScale records the centering / scaling applied to one predictor at
// fit time so prediction can re-apply the same transform.
type CovarScale struct {
	Name string
	Mean float64
	Std  float64
}

// MarshalJSON renders a CoxResult safely for encoding/json: NaN and
// Inf float fields (which the encoder refuses to serialize) are
// rewritten as null. Partial fits from gonum's linesearch often leave
// StdErr / PValue / ZScore padded with NaN, and log-likelihood can
// be -Inf on degenerate inputs.
func (r *CoxResult) MarshalJSON() ([]byte, error) {
	nullable := func(xs []float64) []*float64 {
		out := make([]*float64, len(xs))
		for i, v := range xs {
			if math.IsNaN(v) || math.IsInf(v, 0) {
				continue
			}
			vv := v
			out[i] = &vv
		}
		return out
	}
	scalar := func(v float64) *float64 {
		if math.IsNaN(v) || math.IsInf(v, 0) {
			return nil
		}
		vv := v
		return &vv
	}
	type alias struct {
		Predictors            []string     `json:"Predictors"`
		Coef                  []*float64   `json:"Coef"`
		StdErr                []*float64   `json:"StdErr"`
		PValue                []*float64   `json:"PValue"`
		ZScore                []*float64   `json:"ZScore"`
		LogLike               *float64     `json:"LogLike"`
		NumObs                int          `json:"NumObs"`
		NumEvents             int          `json:"NumEvents"`
		Scales                []CovarScale `json:"Scales"`
		BaselineTime          []float64    `json:"BaselineTime"`
		BaselineCumHaz        []float64    `json:"BaselineCumHaz"`
		StratumColumn         string       `json:"StratumColumn,omitempty"`
		StratumLabels         []float64    `json:"StratumLabels,omitempty"`
		StratumBaselineTimes  [][]float64  `json:"StratumBaselineTimes,omitempty"`
		StratumBaselineCumHaz [][]float64  `json:"StratumBaselineCumHaz,omitempty"`
	}
	return json.Marshal(alias{
		Predictors:            r.Predictors,
		Coef:                  nullable(r.Coef),
		StdErr:                nullable(r.StdErr),
		PValue:                nullable(r.PValue),
		ZScore:                nullable(r.ZScore),
		LogLike:               scalar(r.LogLike),
		NumObs:                r.NumObs,
		NumEvents:             r.NumEvents,
		Scales:                r.Scales,
		BaselineTime:          r.BaselineTime,
		BaselineCumHaz:        r.BaselineCumHaz,
		StratumColumn:         r.StratumColumn,
		StratumLabels:         r.StratumLabels,
		StratumBaselineTimes:  r.StratumBaselineTimes,
		StratumBaselineCumHaz: r.StratumBaselineCumHaz,
	})
}

// DefaultCoxPredictors is the predictor set used by callers that don't
// want to think about which columns to feed Cox. They are the continuous
// covariates that the interval builder snapshots at IntervalStart, which
// are the things an online tiering policy could realistically read off
// the live state.
//
// ContractAge re-joins the set in Phase 2 now that storage surfaces
// contracts.deploy_block via SlotWithMeta.DeployBlock and the mock
// extractor sets slot.CreatedAt to the slot's first event block. Those
// two changes together make ContractAge ≥ SlotAge with strict inequality
// in the typical case, removing the Phase-1 collinearity that gave the
// optimizer a singular Hessian.
var DefaultCoxPredictors = []string{ColAccessCount, ColContractAge, ColSlotAge}

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
func (f StatmodelFitter) FitCoxPH(intervals []domain.InterAccessInterval, covariates []string) (*CoxResult, error) {
	return f.FitCoxPHStratified(intervals, covariates, "")
}

// FitCoxPHStratified is FitCoxPH with an additional stratification
// variable. When strataColumn is non-empty, statmodel fits a shared
// coefficient vector but a separate baseline hazard per unique value
// of that column. The plan recommends stratifying on contract type
// because Phase-2 CheckPH confirmed the aging covariates do not
// satisfy PH in aggregate: different application archetypes age in
// different ways, and giving each one its own clock absorbs that
// non-proportionality into the baseline.
//
// strataColumn must be a dstream column name from the adapter
// (typically ColContractType). It must NOT appear in covariates —
// stratification substitutes for including it as a predictor. If
// strataColumn is empty, the fit behaves exactly like the unstratified
// FitCoxPH and the returned CoxResult's Stratum* fields stay empty.
func (StatmodelFitter) FitCoxPHStratified(
	intervals []domain.InterAccessInterval,
	covariates []string,
	strataColumn string,
) (*CoxResult, error) {
	if err := validateCoxInputs(intervals, covariates, strataColumn); err != nil {
		return nil, err
	}
	if len(covariates) == 0 {
		covariates = DefaultCoxPredictors
	}

	// Materialize a fresh column set so we can standardize in place
	// without mutating the caller's view of the dataset.
	cols, names := buildCoxColumns(intervals)
	scales := standardizeColumns(cols, names, covariates)

	ph, err := newPHRegForCov(cols, names, covariates, strataColumn)
	if err != nil {
		return nil, err
	}
	res, err := fitAndValidate(ph)
	if err != nil {
		return nil, err
	}

	out := assembleCoxResult(res, covariates, scales, strataColumn, len(intervals))
	for _, it := range intervals {
		if it.IsObserved {
			out.NumEvents++
		}
	}
	fillBaselineHazards(out, ph, intervals, strataColumn)
	out.intervals = append([]domain.InterAccessInterval(nil), intervals...)
	return out, nil
}

// validateCoxInputs rejects shapes that PHReg can't handle (empty
// intervals, predictor/stratum collision, duration overflow). Empty
// covariates is fine — the caller substitutes DefaultCoxPredictors
// after the validation barrier.
func validateCoxInputs(intervals []domain.InterAccessInterval, covariates []string, strataColumn string) error {
	if len(intervals) == 0 {
		return fmt.Errorf("FitCoxPH: empty intervals")
	}
	for _, c := range covariates {
		if c == strataColumn {
			return fmt.Errorf("FitCoxPH: stratification column %q must not appear in predictors", strataColumn)
		}
	}
	for _, it := range intervals {
		if it.Duration > math.MaxInt64 { // unreachable in practice; keeps the panic in tests only
			return fmt.Errorf("FitCoxPH: duration overflow at slot %s", it.SlotID)
		}
	}
	return nil
}

// newPHRegForCov assembles the statmodel duration.PHReg handle. The
// L2 ridge is tiny (1e-6) — enough to keep gonum's optimizer stable
// when ContractAge/SlotAge correlate, small enough to leave
// coefficients essentially unchanged on well-conditioned data.
func newPHRegForCov(cols [][]float64, names, covariates []string, strataColumn string) (*duration.PHReg, error) {
	ds := statmodel.NewDataset(cols, names)
	cfg := duration.DefaultPHRegConfig()
	cfg.EntryVar = ColEntry
	if strataColumn != "" {
		cfg.StrataVar = strataColumn
	}
	cfg.L2Penalty = make(map[string]float64, len(covariates))
	for _, p := range covariates {
		cfg.L2Penalty[p] = 1e-6
	}
	ph, err := duration.NewPHReg(ds, ColDuration, ColStatus, covariates, cfg)
	if err != nil {
		return nil, fmt.Errorf("NewPHReg: %w", err)
	}
	return ph, nil
}

// fitAndValidate runs the PHReg fit and tolerates partial results
// (gonum's "linesearch: failed to converge" usually leaves usable
// coefficients). Errors only when Fit returns nil (truly failed) or
// when a coefficient is non-finite — downstream consumers can't
// handle NaN / Inf in the hazard exponent.
func fitAndValidate(ph *duration.PHReg) (*duration.PHResults, error) {
	res, fitErr := ph.Fit()
	if res == nil {
		return nil, fmt.Errorf("Fit: %w", fitErr)
	}
	if fitErr != nil {
		slog.Warn("cox optimizer returned non-fatal error, using partial fit", "err", fitErr)
	}
	for i, c := range res.Params() {
		if math.IsNaN(c) || math.IsInf(c, 0) {
			return nil, fmt.Errorf("Fit: coef[%d] not finite (%v); err=%v", i, c, fitErr)
		}
	}
	return res, nil
}

// assembleCoxResult packs PHReg output into the CoxResult shape
// downstream consumers expect, padding StdErr / PValue / ZScore /
// Coef with NaN when a partial fit left them short. NumEvents and
// the baseline hazards are filled in by the caller — they depend on
// the training intervals, not on res.
func assembleCoxResult(
	res *duration.PHResults,
	covariates []string,
	scales []CovarScale,
	strataColumn string,
	nObs int,
) *CoxResult {
	out := &CoxResult{
		Predictors:    append([]string(nil), covariates...),
		Coef:          append([]float64(nil), res.Params()...),
		StdErr:        padOrCopy(res.StdErr(), len(covariates)),
		PValue:        padOrCopy(res.PValues(), len(covariates)),
		ZScore:        padOrCopy(res.ZScores(), len(covariates)),
		LogLike:       res.LogLike(),
		NumObs:        nObs,
		Scales:        scales,
		StratumColumn: strataColumn,
	}
	if len(out.Coef) < len(covariates) {
		padded := make([]float64, len(covariates))
		for i := range padded {
			if i < len(out.Coef) {
				padded[i] = out.Coef[i]
			} else {
				padded[i] = math.NaN()
			}
		}
		out.Coef = padded
	}
	return out
}

// fillBaselineHazards pulls the baseline cumulative hazard grid out
// of PHReg — a single (time, cumhaz) pair for unstratified fits, or
// per-stratum grids aligned with StratumLabels when strataColumn is
// set.
func fillBaselineHazards(out *CoxResult, ph *duration.PHReg, intervals []domain.InterAccessInterval, strataColumn string) {
	if strataColumn == "" {
		t, h := ph.BaselineCumHaz(0, out.Coef)
		out.BaselineTime = append([]float64(nil), t...)
		out.BaselineCumHaz = append([]float64(nil), h...)
		return
	}
	// Discover the unique stratum values in the training data and
	// pull a per-stratum baseline cumulative hazard for each.
	// statmodel assigns stratum indices in ascending value order
	// of the stratum column, so StratumLabels ends up sorted.
	strataValues := collectUniqueStrata(intervals, strataColumn)
	out.StratumLabels = strataValues
	out.StratumBaselineTimes = make([][]float64, len(strataValues))
	out.StratumBaselineCumHaz = make([][]float64, len(strataValues))
	for i := range strataValues {
		t, h := ph.BaselineCumHaz(i, out.Coef)
		out.StratumBaselineTimes[i] = append([]float64(nil), t...)
		out.StratumBaselineCumHaz[i] = append([]float64(nil), h...)
	}
}

// collectUniqueStrata returns the sorted set of stratum-column values
// present in intervals. Matches statmodel's ascending-sort ordering so
// the i-th value corresponds to statmodel's stratum index i in
// BaselineCumHaz(i, params).
func collectUniqueStrata(intervals []domain.InterAccessInterval, col string) []float64 {
	seen := make(map[float64]struct{})
	for _, it := range intervals {
		seen[rawCovariate(it, col)] = struct{}{}
	}
	out := make([]float64, 0, len(seen))
	for v := range seen {
		out = append(out, v)
	}
	sort.Float64s(out)
	return out
}

// padOrCopy returns a copy of xs padded out to length n with NaN.
// Used to normalize statmodel partial-fit outputs where optional
// fields (StdErr, PValues, ZScores) may come back empty.
func padOrCopy(xs []float64, n int) []float64 {
	out := make([]float64, n)
	for i := range out {
		if i < len(xs) {
			out[i] = xs[i]
		} else {
			out[i] = math.NaN()
		}
	}
	return out
}

// rawCovariate pulls a single covariate value out of an interval by
// predictor-column name. Returns 0 for unknown names so callers don't
// have to special-case missing optional predictors.
func rawCovariate(it domain.InterAccessInterval, name string) float64 {
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

// linearPredictor computes the standardized x'β for an interval or a
// caller-provided covariate map, using the scaling parameters stored at
// fit time. Centralized so PredictAccessProb and SurvivalAt stay
// numerically consistent.
func (r *CoxResult) linearPredictor(covariates map[string]float64) float64 {
	if r == nil {
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
	return lp
}

// Survival returns S(t | x) for the fitted Cox model, clipped to [0, 1].
//
//	S(t | x) = exp(-H_0(t) * exp(x'β))
//
// On a stratified fit H_0 is the per-stratum baseline hazard, selected
// by the interval's value in r.StratumColumn (which must appear in
// covariates, or defaults to the first-observed stratum if absent).
//
// Use Survival rather than PredictAccessProb when you need the survival
// curve at arbitrary t — e.g. the tiering policy's conditional-survival
// search. PredictAccessProb at time τ is equivalent to 1 − Survival(τ).
func (r *CoxResult) Survival(covariates map[string]float64, t float64) float64 {
	if r == nil {
		return 1
	}
	lp := r.linearPredictor(covariates)
	var stratumVal float64
	if r.StratumColumn != "" {
		stratumVal = covariates[r.StratumColumn]
	}
	times, hazards := r.baselineFor(stratumVal)
	if len(times) == 0 {
		return 1
	}
	h := baselineCumHazAt(times, hazards, t)
	s := math.Exp(-h * math.Exp(lp))
	if s < 0 {
		return 0
	}
	if s > 1 {
		return 1
	}
	return s
}

// baselineFor resolves the baseline hazard grid to use for a row whose
// stratum column takes value stratumVal. Unstratified fits return the
// single BaselineTime / BaselineCumHaz; stratified fits match
// stratumVal against StratumLabels. Unknown values fall through to the
// first stratum because the fit saw no data from that category —
// extrapolating via a real stratum is preferable to returning 0 hazard.
func (r *CoxResult) baselineFor(stratumVal float64) (times, hazards []float64) {
	if r.StratumColumn == "" || len(r.StratumLabels) == 0 {
		return r.BaselineTime, r.BaselineCumHaz
	}
	for i, l := range r.StratumLabels {
		if l == stratumVal {
			return r.StratumBaselineTimes[i], r.StratumBaselineCumHaz[i]
		}
	}
	return r.StratumBaselineTimes[0], r.StratumBaselineCumHaz[0]
}

// SurvivalForInterval is the InterAccessInterval analogue of Survival.
// Uses the full predictor set fit on the model, plus the stratum
// column (if any) so stratified fits can pick the right baseline.
func (r *CoxResult) SurvivalForInterval(it domain.InterAccessInterval, t float64) float64 {
	if r == nil {
		return 1
	}
	cov := make(map[string]float64, len(r.Predictors)+1)
	for _, name := range r.Predictors {
		cov[name] = rawCovariate(it, name)
	}
	if r.StratumColumn != "" {
		cov[r.StratumColumn] = rawCovariate(it, r.StratumColumn)
	}
	return r.Survival(cov, t)
}

// PredictAccessProb returns the probability that a slot with the given
// raw covariate vector is accessed within the next tau blocks, assuming
// it has just been touched (idle = 0). Thin wrapper around Survival.
func (r *CoxResult) PredictAccessProb(covariates map[string]float64, tau float64) float64 {
	return 1 - r.Survival(covariates, tau)
}

// PredictAccessProbForInterval is a convenience wrapper that pulls
// covariates straight off an InterAccessInterval. It iterates whatever
// predictor set the model was fit with — hard-coding a fixed map would
// silently zero out any predictor outside the default set (e.g. a model
// fit with ContractType / SlotType would be evaluated on incomplete
// covariates and return wrong probabilities).
func (r *CoxResult) PredictAccessProbForInterval(it domain.InterAccessInterval, tau float64) float64 {
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
func buildCoxColumns(intervals []domain.InterAccessInterval) ([][]float64, []string) {
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
