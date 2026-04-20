package analysis

import (
	"fmt"
	"math"
	"sort"

	"gonum.org/v1/gonum/stat/distuv"

	"github.com/egpivo/zk-state-prune/internal/model"
)

// PHTestResult holds the outcome of a Schoenfeld-residual test of the
// proportional-hazards assumption. Filled in by Segment 9.
//
// PerCovariatePValue is keyed by predictor name. GlobalPValue is the
// overall test that no predictor violates PH. Schoenfeld is implemented
// as: residual r_ij = x_ij - x̄(t_i), correlation with event time used
// as the test statistic.
type PHTestResult struct {
	Predictors         []string
	PerCovariatePValue map[string]float64
	GlobalPValue       float64
}

// CalibratedModel wraps a CoxResult with a recalibration map fitted on a
// holdout split. Filled in by Segment 11. The recalibration is an
// isotonic regression from raw predicted probability to empirical
// access rate, so PredictAccessProb post-calibration preserves ranking
// but matches the empirical CDF.
type CalibratedModel struct {
	Base *CoxResult
	// Tau is the horizon at which the recalibration was fitted. It
	// becomes the canonical horizon for PredictAccessProb on this model:
	// raw Cox predictions at any other horizon would not match the grid.
	Tau float64
	// Recalibration grid: PredX[i] is a sorted predicted probability,
	// CalibratedY[i] is the isotonic-fitted empirical rate at that
	// quantile. PredictAccessProb does a step lookup on this grid.
	PredX       []float64
	CalibratedY []float64

	// --- Split-conformal intervals -----------------------------------
	//
	// Two ε values, one per predictive quantity, each fit on the same
	// conformal half of the holdout but with a different label/
	// prediction pair:
	//
	//   Epsilon covers the *calibrated point* prediction at horizon Tau
	//   evaluated from idle = 0 — the isotonic-recalibrated
	//   probability of access within the next Tau blocks given the
	//   slot has just been touched. Valid for CalibratedModel.
	//   PredictUpperAccessProbForInterval (which the robust policy
	//   uses at the idle = 0 search sample).
	//
	//   ConditionalEpsilon covers the *raw Cox conditional* prediction
	//   1 − S(u+τ)/S(u) for a SINGLE idle duration u drawn from the
	//   per-row u distribution the cal pass used. It is a marginal
	//   (finite-sample) upper-bound quantile on one fresh (interval,
	//   u, label_at_u) triple drawn from the same distribution as
	//   the cal half. It is NOT a simultaneous-coverage bound across
	//   all idle samples the statistical policy evaluates during its
	//   T*-search: the policy probes ~21 u values per interval and
	//   the marginal coverage claim does not extend to the worst of
	//   those probes. Treat the robust policy's "valid at every
	//   idle" as engineering-level defensive, not a theorem.
	//
	// Both guarantees are valid only for (i) the target quantity they
	// were fit against, (ii) data drawn from a distribution
	// exchangeable with the cal half, and (iii) at the same CoverageLevel.
	//
	// ConditionalEpsilon == 0 means conditional conformal was not
	// fitted (too few unambiguous cal-half samples), in which case the
	// robust policy falls back to the raw point estimate at idle > 0.
	//
	// CoverageLevel is the nominal 1 − α (e.g. 0.9 for α = 0.1). Zero
	// means conformal prediction was not fitted at all (intervals are
	// width-zero point estimates).
	Epsilon            float64
	ConditionalEpsilon float64
	CoverageLevel      float64
}

// PredictAccessProb returns the recalibrated probability of access by
// horizon c.Tau for the given covariates.
//
// The horizon is fixed at fit time (CalibrateAt's tau argument) — the
// isotonic grid only knows how to map raw predictions at exactly that
// horizon back to empirical rates. Calling at any other tau would mean
// looking up the wrong calibration curve, so the API does not let the
// caller pass one. Use Base.PredictAccessProb directly for off-horizon
// raw queries.
func (c *CalibratedModel) PredictAccessProb(covariates map[string]float64) float64 {
	if c == nil || c.Base == nil {
		return 0
	}
	raw := c.Base.PredictAccessProb(covariates, c.Tau)
	if len(c.PredX) == 0 {
		return raw
	}
	return isotonicLookup(c.PredX, c.CalibratedY, raw)
}

// PredictAccessProbForInterval is the InterAccessInterval analogue of
// PredictAccessProb. It uses the model's full predictor set so a model
// fit with extra covariates (ContractType, SlotType, …) does not get
// silently evaluated on a truncated map.
func (c *CalibratedModel) PredictAccessProbForInterval(it model.InterAccessInterval) float64 {
	if c == nil || c.Base == nil {
		return 0
	}
	cov := make(map[string]float64, len(c.Base.Predictors))
	for _, name := range c.Base.Predictors {
		cov[name] = rawCovariate(it, name)
	}
	return c.PredictAccessProb(cov)
}

// PredictInterval returns the split-conformal prediction interval
// [lower, upper] around the calibrated access probability, with
// marginal coverage ≥ c.CoverageLevel. The interval collapses to the
// point estimate when Epsilon == 0 (conformal prediction not fitted).
// Both endpoints are clipped to [0, 1].
func (c *CalibratedModel) PredictInterval(covariates map[string]float64) (lower, upper float64) {
	p := c.PredictAccessProb(covariates)
	lower = p - c.Epsilon
	upper = p + c.Epsilon
	if lower < 0 {
		lower = 0
	}
	if upper > 1 {
		upper = 1
	}
	return lower, upper
}

// PredictIntervalForInterval is the InterAccessInterval analogue of
// PredictInterval.
func (c *CalibratedModel) PredictIntervalForInterval(it model.InterAccessInterval) (lower, upper float64) {
	if c == nil || c.Base == nil {
		return 0, 0
	}
	cov := make(map[string]float64, len(c.Base.Predictors))
	for _, name := range c.Base.Predictors {
		cov[name] = rawCovariate(it, name)
	}
	return c.PredictInterval(cov)
}

// PredictUpperAccessProbForInterval returns just the upper endpoint of
// the conformal interval, which is the quantity the robust decision
// rule plugs into the surrogate for d=cold. Uses Epsilon — the ε
// fit for the calibrated point prediction at τ from idle=0.
func (c *CalibratedModel) PredictUpperAccessProbForInterval(it model.InterAccessInterval) float64 {
	_, upper := c.PredictIntervalForInterval(it)
	return upper
}

// PredictUpperConditionalAccessProb returns a pessimism-adjusted upper
// bound on the conditional access probability at idle duration u:
//
//	p_upper(u) = min(1, (1 − S(u+τ)/S(u)) + ConditionalEpsilon)
//
// ConditionalEpsilon was fit as a split-conformal quantile on one u
// per holdout row, so the coverage claim is MARGINAL (a fresh single
// probe drawn from the same (row, u) distribution lies in
// [raw − ε, raw + ε] at the nominal CoverageLevel). It is NOT a
// simultaneous-coverage bound: the statistical policy probes many u
// values per interval during its T*-search, and the worst over those
// probes can exceed the marginal quantile. The robust policy still
// profits from the extra margin in practice — ε shifts decisions
// toward keeping slots hot when the model is noisier — but "robust"
// here means "cheapest valid pessimism layer", not "uniform over the
// search grid". When ConditionalEpsilon is zero (no conformal fit)
// the function returns the raw point estimate.
func (c *CalibratedModel) PredictUpperConditionalAccessProb(it model.InterAccessInterval, idle float64) float64 {
	if c == nil || c.Base == nil {
		return 0
	}
	sU := c.Base.SurvivalForInterval(it, idle)
	if sU <= 0 {
		return 1
	}
	sUTau := c.Base.SurvivalForInterval(it, idle+c.Tau)
	p := 1 - sUTau/sU
	if p < 0 {
		p = 0
	}
	p += c.ConditionalEpsilon
	if p > 1 {
		p = 1
	}
	return p
}

// isotonicLookup is a placeholder right-continuous step lookup used until
// Segment 11 builds the real PAV-fitted grid. Returning the input value
// when no grid is present makes CalibratedModel a no-op wrapper for the
// duration of segments 8–10.
func isotonicLookup(x, y []float64, q float64) float64 {
	if len(x) == 0 {
		return q
	}
	if q <= x[0] {
		return y[0]
	}
	if q >= x[len(x)-1] {
		return y[len(y)-1]
	}
	lo, hi := 0, len(x)-1
	for lo < hi {
		mid := (lo + hi + 1) / 2
		if x[mid] <= q {
			lo = mid
		} else {
			hi = mid - 1
		}
	}
	return y[lo]
}

// CalibrationCurve summarizes the agreement between Cox-predicted
// access probabilities and the empirical access rate on a held-out
// interval set, evaluated at a single horizon tau. It is the standard
// reliability-diagram input: each bin pairs the mean predicted
// probability with the observed rate among rows whose prediction landed
// in that bin.
//
// Bin observed rates are computed via **bin-wise Kaplan–Meier**: for
// each quantile bin we fit a KM curve on the bin's intervals and read
// 1 − S_KM(τ) as the empirical access-by-tau rate. This keeps all
// rows in the bin population — including the "censored before tau"
// rows that Phase 1 used to drop — at the cost of one KM fit per bin.
// KM handles the censoring correctly: a row censored at time t < τ
// still contributes to the risk set for the event times ≤ t and then
// leaves, which is exactly the treatment a naive drop couldn't express.
//
// BrierScore still uses row-level binary labels and therefore excludes
// censored-before-τ rows from its denominator (they don't have an
// unambiguous {0, 1} label). NumKept counts all holdout rows used for
// bin construction; BrierN is the subset with unambiguous labels.
type CalibrationCurve struct {
	Tau float64

	Bins []CalibrationBin

	// BrierScore is the mean squared error between predicted probability
	// and the binary label, averaged over BrierN rows that had
	// unambiguous labels. Lower is better.
	BrierScore float64

	// NumKept is the number of holdout rows that ended up in a bin.
	// Under the bin-wise KM policy every non-empty holdout row is kept,
	// so NumKept == len(holdout).
	NumKept int
	// BrierN is the subset of NumKept used as the Brier denominator
	// (rows where the {0, 1} label is unambiguous). Kept separate so
	// the reader can see how much censoring pressure the Brier was
	// computed under.
	BrierN int
	// NumDropped is retained for source compatibility. It is always
	// zero under bin-wise KM and will be removed in a later phase.
	NumDropped int
}

// CalibrationBin is one row of a reliability diagram.
type CalibrationBin struct {
	PredictedMean float64
	ObservedRate  float64
	N             int
}

// PredictForIntervalFunc is the predict-by-interval abstraction both
// CoxResult and CalibratedModel satisfy. Pulling it out lets the
// calibration machinery serve both model flavours without a switch.
type PredictForIntervalFunc func(it model.InterAccessInterval) float64

// CalibrationCurveFromCox evaluates a fitted Cox model on the holdout
// intervals at horizon tau and produces a reliability diagram with
// `nBins` equal-population quantile bins. Thin wrapper around
// CalibrationCurveFromPredict that delegates prediction to the raw
// Cox baseline hazard.
func CalibrationCurveFromCox(
	res *CoxResult,
	holdout []model.InterAccessInterval,
	tau float64,
	nBins int,
) (*CalibrationCurve, error) {
	if res == nil {
		return nil, fmt.Errorf("CalibrationCurveFromCox: nil model")
	}
	predict := func(it model.InterAccessInterval) float64 {
		return res.PredictAccessProbForInterval(it, tau)
	}
	return CalibrationCurveFromPredict(predict, holdout, tau, nBins)
}

// CalibrationCurveFromPredict is the reusable reliability-diagram
// builder. It runs the Phase-2 bin-wise KM observed-rate policy so
// censored-before-τ rows stay in the bins instead of being dropped.
func CalibrationCurveFromPredict(
	predict PredictForIntervalFunc,
	holdout []model.InterAccessInterval,
	tau float64,
	nBins int,
) (*CalibrationCurve, error) {
	if predict == nil {
		return nil, fmt.Errorf("CalibrationCurveFromPredict: nil predict")
	}
	if tau <= 0 {
		return nil, fmt.Errorf("CalibrationCurveFromPredict: tau must be > 0")
	}
	if len(holdout) == 0 {
		return nil, fmt.Errorf("CalibrationCurveFromPredict: empty holdout")
	}
	if nBins <= 0 {
		nBins = 10
	}

	type row struct {
		p  float64
		it model.InterAccessInterval
	}
	rows := make([]row, 0, len(holdout))
	for _, it := range holdout {
		rows = append(rows, row{
			p:  predict(it),
			it: it,
		})
	}
	if len(rows) < nBins {
		return nil, fmt.Errorf("CalibrationCurveFromPredict: only %d holdout rows for %d bins", len(rows), nBins)
	}
	sort.Slice(rows, func(i, j int) bool { return rows[i].p < rows[j].p })

	fitter := NewStatmodelFitter()
	bins := make([]CalibrationBin, 0, nBins)
	binSize := len(rows) / nBins
	brier := 0.0
	brierN := 0
	for b := 0; b < nBins; b++ {
		lo := b * binSize
		hi := lo + binSize
		if b == nBins-1 {
			hi = len(rows)
		}
		var sumP float64
		ivs := make([]model.InterAccessInterval, 0, hi-lo)
		for j := lo; j < hi; j++ {
			r := rows[j]
			sumP += r.p
			ivs = append(ivs, r.it)
			// Row-level Brier contribution for rows with unambiguous
			// labels. Censored-before-τ rows are still in the bin (KM
			// consumes them) but contribute no Brier term.
			switch {
			case r.it.IsObserved && r.it.Duration <= uint64(tau):
				d := r.p - 1
				brier += d * d
				brierN++
			case r.it.Duration > uint64(tau):
				d := r.p - 0
				brier += d * d
				brierN++
			}
		}
		n := hi - lo
		predictedMean := sumP / float64(n)

		// Bin-wise KM. If the fit fails (extremely small bin, pathological
		// data) we fall back to 0 so the reliability diagram still prints
		// rather than blowing up the whole report.
		observedRate := 0.0
		if km, err := fitter.FitKaplanMeier(ivs); err == nil && km != nil {
			observedRate = 1 - km.SurvAt(tau)
			if observedRate < 0 {
				observedRate = 0
			}
		}
		bins = append(bins, CalibrationBin{
			PredictedMean: predictedMean,
			ObservedRate:  observedRate,
			N:             n,
		})
	}

	brierScore := 0.0
	if brierN > 0 {
		brierScore = brier / float64(brierN)
	}
	return &CalibrationCurve{
		Tau:        tau,
		Bins:       bins,
		BrierScore: brierScore,
		NumKept:    len(rows),
		BrierN:     brierN,
		NumDropped: 0,
	}, nil
}

// CheckPH runs a Schoenfeld-residuals proportional-hazards test on a
// fitted Cox model. The result reports a per-covariate p-value plus a
// Fisher-combined global p-value. Small p means the residuals correlate
// with event time, i.e. the covariate's effect is itself time-dependent
// and the PH assumption is violated.
//
// Algorithm (gap-time, right-censored, no strata):
//
//  1. Standardize each covariate the same way the fit did, using
//     CoxResult.Scales. Compute the per-row linear predictor lp_i.
//  2. Sort intervals by Duration ascending and walk *backward* so the
//     running sums {sum_w, sum_wx_k} for the risk set R(t) are built
//     incrementally in O(N). Tied event times are added together
//     before any of their residuals are read so a tied event sees
//     itself in its own risk set.
//  3. At every observed event the Schoenfeld residual for covariate k
//     is r_ik = z_ik − x̄_k(t_i), where the bar denotes a Cox-weighted
//     mean over the risk set. We don't use scaled residuals — for a
//     Phase-1 misspecification screen the raw form against time-rank
//     is enough and avoids a second matrix inversion.
//  4. Per covariate: Pearson correlation between residuals and the
//     event-time *rank* (rank, not raw time, makes the test invariant
//     to monotone time transforms — equivalent to the "transform=km"
//     option in R's cox.zph). The t-statistic of that correlation
//     gives the per-covariate p-value.
//  5. Global: Fisher's method combines the per-covariate p-values into
//     a chi-square with df = 2K.
//
// Phase-1 limitations to be aware of:
//   - We assume no stratification (single baseline). When FitCoxPH
//     grows a StrataVar, this routine has to walk per-stratum risk
//     sets in lockstep.
//   - Tie handling matches the Breslow approximation (one risk-set
//     summary per tie group). Efron tie correction is not applied;
//     for typical Phase-1 data with mostly distinct gap times the
//     difference is negligible.
func (StatmodelFitter) CheckPH(res *CoxResult) (*PHTestResult, error) {
	if res == nil {
		return nil, fmt.Errorf("CheckPH: nil CoxResult")
	}
	if len(res.Predictors) == 0 {
		return nil, fmt.Errorf("CheckPH: CoxResult has no predictors")
	}
	if len(res.intervals) == 0 {
		return nil, fmt.Errorf("CheckPH: CoxResult has no retained intervals")
	}
	K := len(res.Predictors)
	n := len(res.intervals)

	// Step 1: standardize and linear predictor.
	z := make([][]float64, n)
	lp := make([]float64, n)
	for i, it := range res.intervals {
		zi := make([]float64, K)
		for k, name := range res.Predictors {
			raw := rawCovariate(it, name)
			sc := CovarScale{Std: 1}
			if k < len(res.Scales) {
				sc = res.Scales[k]
			}
			var zik float64
			if sc.Std > 0 {
				zik = (raw - sc.Mean) / sc.Std
			}
			zi[k] = zik
			lp[i] += res.Coef[k] * zik
		}
		z[i] = zi
	}

	// Step 2: sort indices ascending by Duration. Tie order doesn't
	// matter so SliceStable is the cheap choice.
	order := make([]int, n)
	for i := range order {
		order[i] = i
	}
	sort.SliceStable(order, func(a, b int) bool {
		return res.intervals[order[a]].Duration < res.intervals[order[b]].Duration
	})

	// Step 3: walk backward grouping by tied time, computing residuals.
	// residTime[e] is the time of the e-th observed event in the order
	// it was processed (high-to-low time). resid[e] is the K-vector of
	// raw Schoenfeld residuals for that event. The orientation is
	// arbitrary — the correlation with rank does not care.
	type ev struct {
		time float64
		r    []float64
	}
	events := make([]ev, 0, n)

	sumW := 0.0
	sumWX := make([]float64, K)
	i := n - 1
	for i >= 0 {
		// Find the tie group [j+1, i] sharing this time.
		j := i
		curT := res.intervals[order[i]].Duration
		for j >= 0 && res.intervals[order[j]].Duration == curT {
			j--
		}
		// Add every member of the tie group to the running sums
		// before reading any of them — a tied event must see its
		// peers in its own risk set.
		for m := j + 1; m <= i; m++ {
			ii := order[m]
			w := math.Exp(lp[ii])
			sumW += w
			for k := 0; k < K; k++ {
				sumWX[k] += w * z[ii][k]
			}
		}
		if sumW > 0 {
			xbar := make([]float64, K)
			for k := 0; k < K; k++ {
				xbar[k] = sumWX[k] / sumW
			}
			for m := j + 1; m <= i; m++ {
				ii := order[m]
				if !res.intervals[ii].IsObserved {
					continue
				}
				r := make([]float64, K)
				for k := 0; k < K; k++ {
					r[k] = z[ii][k] - xbar[k]
				}
				events = append(events, ev{time: float64(curT), r: r})
			}
		}
		i = j
	}

	if len(events) < 5 {
		return nil, fmt.Errorf("CheckPH: too few observed events for PH test (%d)", len(events))
	}

	// Step 4: per-covariate Pearson correlation against time rank.
	times := make([]float64, len(events))
	for e := range events {
		times[e] = events[e].time
	}
	ranks := averageRanks(times)

	out := &PHTestResult{
		Predictors:         append([]string(nil), res.Predictors...),
		PerCovariatePValue: make(map[string]float64, K),
	}
	chi2 := 0.0
	for k := 0; k < K; k++ {
		col := make([]float64, len(events))
		for e := range events {
			col[e] = events[e].r[k]
		}
		rho := pearsonCorrelation(col, ranks)
		p := correlationPValue(rho, len(events))
		out.PerCovariatePValue[res.Predictors[k]] = p
		// Fisher's method needs strictly positive p; clamp away from
		// zero to keep -2 ln(p) finite.
		if p < 1e-300 {
			p = 1e-300
		}
		chi2 += -2 * math.Log(p)
	}
	// Step 5: global p from Fisher's combined statistic, df = 2K.
	chi := distuv.ChiSquared{K: float64(2 * K)}
	out.GlobalPValue = chi.Survival(chi2)
	return out, nil
}

// pearsonCorrelation is a small zero-mean / unit-norm Pearson coefficient
// implementation. Returns 0 when either side has zero variance to keep
// the downstream t-test well-defined instead of producing NaN.
func pearsonCorrelation(x, y []float64) float64 {
	if len(x) != len(y) || len(x) < 2 {
		return 0
	}
	var sx, sy float64
	for i := range x {
		sx += x[i]
		sy += y[i]
	}
	mx := sx / float64(len(x))
	my := sy / float64(len(y))
	var num, dx, dy float64
	for i := range x {
		ax := x[i] - mx
		ay := y[i] - my
		num += ax * ay
		dx += ax * ax
		dy += ay * ay
	}
	if dx == 0 || dy == 0 {
		return 0
	}
	return num / math.Sqrt(dx*dy)
}

// correlationPValue is a two-sided p-value for H0: ρ = 0 using the
// standard t = ρ √((n−2) / (1−ρ²)) transform.
func correlationPValue(rho float64, n int) float64 {
	if n <= 2 {
		return 1
	}
	if math.Abs(rho) >= 1 {
		return 0
	}
	t := rho * math.Sqrt(float64(n-2)/(1-rho*rho))
	st := distuv.StudentsT{Mu: 0, Sigma: 1, Nu: float64(n - 2)}
	return 2 * st.Survival(math.Abs(t))
}

// averageRanks returns the rank of each input value with ties resolved
// to the average rank of the tied positions. Output ranks are 1..n.
// Used by CheckPH so the time axis is invariant to monotone transforms.
func averageRanks(xs []float64) []float64 {
	n := len(xs)
	idx := make([]int, n)
	for i := range idx {
		idx[i] = i
	}
	sort.SliceStable(idx, func(a, b int) bool { return xs[idx[a]] < xs[idx[b]] })
	ranks := make([]float64, n)
	i := 0
	for i < n {
		j := i
		for j < n && xs[idx[j]] == xs[idx[i]] {
			j++
		}
		// Average rank for the tied block [i, j) is (i+1 + j)/2.
		avg := float64(i+1+j) / 2
		for k := i; k < j; k++ {
			ranks[idx[k]] = avg
		}
		i = j
	}
	return ranks
}

// CalibrateAt fits an isotonic recalibration map from raw Cox-predicted
// access probability to empirical access rate on the holdout set, at
// horizon tau. Returns a CalibratedModel that wraps the original
// CoxResult and recovers a calibrated PredictAccessProb via a
// PAV-fitted step lookup, plus a split-conformal ε around that
// prediction.
//
// Data discipline:
//
//   - The non-ambiguous holdout rows are split 50/50 by a deterministic
//     FNV hash of SlotID into "iso" and "cal" sets. iso fits the PAV
//     isotonic map; cal is then scored against that fitted map and its
//     residuals become the conformal non-conformity scores. This is
//     the standard split-conformal recipe — no row helps fit its own
//     ε, so the finite-sample coverage claim actually holds.
//
//   - Censoring policy is the same as CalibrationCurveFromCox: rows
//     with Duration > tau are label 0 (no access yet at tau), observed
//     rows with Duration <= tau are label 1, and censored rows with
//     Duration <= tau are dropped because their label is unknowable
//     without a survival adjustment.
//
// CalibrateAt is the lower-level entry point that takes tau explicitly.
// The interface method Calibrate(res, holdout) defaults tau to the
// median Duration of the training set (read off res.intervals).
func (StatmodelFitter) CalibrateAt(
	res *CoxResult,
	holdout []model.InterAccessInterval,
	tau float64,
) (*CalibratedModel, error) {
	if res == nil {
		return nil, fmt.Errorf("CalibrateAt: nil CoxResult")
	}
	if tau <= 0 {
		return nil, fmt.Errorf("CalibrateAt: tau must be > 0")
	}
	if len(holdout) == 0 {
		return nil, fmt.Errorf("CalibrateAt: empty holdout")
	}

	all := make([]calPoint, 0, len(holdout))
	for _, it := range holdout {
		var label float64
		switch {
		case it.IsObserved && it.Duration <= uint64(tau):
			label = 1
		case it.Duration > uint64(tau):
			label = 0
		default:
			continue
		}
		all = append(all, calPoint{
			p:  res.PredictAccessProbForInterval(it, tau),
			y:  label,
			it: it,
		})
	}
	if len(all) < 10 {
		return nil, fmt.Errorf("CalibrateAt: only %d non-ambiguous holdout rows (need >= 10 for split conformal)", len(all))
	}

	// Deterministic 50/50 partition on SlotID → even hash bucket
	// fits PAV, odd bucket measures residuals. Done before the sort
	// below so each half is independent of the predicted-probability
	// order.
	var isoPts, calPts []calPoint
	for _, p := range all {
		if slotIDBucket(p.it.SlotID) == 0 {
			isoPts = append(isoPts, p)
		} else {
			calPts = append(calPts, p)
		}
	}
	if len(isoPts) < 5 || len(calPts) < 5 {
		return nil, fmt.Errorf("CalibrateAt: split-conformal partition too small (iso=%d, cal=%d)", len(isoPts), len(calPts))
	}

	// Fit the PAV isotonic map on the iso half.
	sort.Slice(isoPts, func(i, j int) bool { return isoPts[i].p < isoPts[j].p })
	xIso := make([]float64, len(isoPts))
	yIso := make([]float64, len(isoPts))
	for i, p := range isoPts {
		xIso[i] = p.p
		yIso[i] = p.y
	}
	yHatIso := poolAdjacentViolators(yIso)

	// Compute conformal residuals on the cal half by looking up each
	// row's prediction on the PAV grid fitted above.
	const defaultCoverage = 0.9
	resid := make([]float64, len(calPts))
	for i, p := range calPts {
		fitted := isotonicLookup(xIso, yHatIso, p.p)
		d := p.y - fitted
		if d < 0 {
			d = -d
		}
		resid[i] = d
	}
	sort.Float64s(resid)
	alpha := 1 - defaultCoverage
	k := int(math.Ceil((1-alpha)*float64(len(resid)+1))) - 1
	if k < 0 {
		k = 0
	}
	if k >= len(resid) {
		k = len(resid) - 1
	}
	epsilon := resid[k]

	// Conditional split-conformal (Segment 19). Fit a second ε on the
	// raw Cox conditional prediction 1 − S(u+τ)/S(u) at an arbitrary
	// idle u > 0, so the robust policy can apply a valid margin at
	// every T*-search sample, not just idle = 0. Uses the same cal
	// half as above but scores a different (prediction, label) pair
	// per row — each row contributes exactly one residual, sampled at
	// a deterministic u drawn from the slot id.
	condEpsilon := fitConditionalEpsilon(res, calPts, tau, defaultCoverage)

	return &CalibratedModel{
		Base:               res,
		Tau:                tau,
		PredX:              xIso,
		CalibratedY:        yHatIso,
		Epsilon:            epsilon,
		ConditionalEpsilon: condEpsilon,
		CoverageLevel:      defaultCoverage,
	}, nil
}

// fitConditionalEpsilon computes the split-conformal quantile for the
// conditional access probability at arbitrary idle u, using one
// deterministically-sampled u per cal row. Returns 0 if fewer than 10
// rows end up with unambiguous labels — a signal to the consumer that
// the conditional margin could not be fit (common on small holdouts
// or datasets where every row falls into the "don't know beyond
// Duration" bucket).
//
// Per-row label rules given u < Duration (rows with u ≥ Duration are
// dropped — the conditional "slot still idle at u" fails):
//
//   - u + τ ≤ Duration: label = 0 for every row. Observed rows have
//     their event at t = Duration > u+τ (outside horizon); censored
//     rows have no event at all in [0, Duration].
//   - u + τ > Duration AND IsObserved: label = 1 (event at Duration
//     falls inside (u, u+τ]).
//   - u + τ > Duration AND !IsObserved: drop. The horizon extends past
//     our observation end — we don't know what happens in (Duration,
//     u+τ], so neither 0 nor 1 is defensible.
func fitConditionalEpsilon(base *CoxResult, calPts []calPoint, tau, coverage float64) float64 {
	resid := make([]float64, 0, len(calPts))
	for _, p := range calPts {
		it := p.it
		u := conformalIdleSample(it.SlotID, it.Duration)
		if u >= it.Duration {
			continue
		}
		reach := u + uint64(tau+0.5)
		var label float64
		switch {
		case reach <= it.Duration:
			label = 0
		case it.IsObserved:
			label = 1
		default:
			continue
		}
		sU := base.SurvivalForInterval(it, float64(u))
		if sU <= 0 {
			continue
		}
		sUTau := base.SurvivalForInterval(it, float64(u)+tau)
		pred := 1 - sUTau/sU
		if pred < 0 {
			pred = 0
		}
		if pred > 1 {
			pred = 1
		}
		d := label - pred
		if d < 0 {
			d = -d
		}
		resid = append(resid, d)
	}
	if len(resid) < 10 {
		return 0
	}
	sort.Float64s(resid)
	alpha := 1 - coverage
	k := int(math.Ceil((1-alpha)*float64(len(resid)+1))) - 1
	if k < 0 {
		k = 0
	}
	if k >= len(resid) {
		k = len(resid) - 1
	}
	return resid[k]
}

// conformalIdleSample deterministically derives an idle duration in
// [0, duration) from a slot id. Uses the same FNV-1a base as
// slotIDBucket but with an extra "u" salt so the conditional sample
// is independent of the iso/cal partition choice.
func conformalIdleSample(slotID string, duration uint64) uint64 {
	if duration == 0 {
		return 0
	}
	var h uint32 = 2166136261
	for i := 0; i < len(slotID); i++ {
		h ^= uint32(slotID[i])
		h *= 16777619
	}
	h ^= 'u'
	h *= 16777619
	return uint64(h) % duration
}

// calPoint is the per-row record the calibration pipeline carries
// through CalibrateAt: the raw Cox-predicted point probability at τ,
// the binary label at τ, and the source interval so downstream passes
// (iso-fit, conformal-residual, conditional-conformal) can re-derive
// whatever quantity they need.
type calPoint struct {
	p, y float64
	it   model.InterAccessInterval
}

// slotIDBucket deterministically hashes a slot id to {0, 1}. Used by
// CalibrateAt to build disjoint iso-fit / conformal-residual halves
// from a single holdout set without introducing a stateful RNG.
func slotIDBucket(id string) uint32 {
	// FNV-1a on the id bytes, low bit for the split.
	var h uint32 = 2166136261
	for i := 0; i < len(id); i++ {
		h ^= uint32(id[i])
		h *= 16777619
	}
	return h & 1
}

// Calibrate is the SurvivalFitter interface method. It selects a
// default horizon (median Duration of the training intervals retained
// inside res) and delegates to CalibrateAt. Callers that want to pin
// tau should invoke CalibrateAt directly.
func (f StatmodelFitter) Calibrate(res *CoxResult, holdout []model.InterAccessInterval) (*CalibratedModel, error) {
	if res == nil {
		return nil, fmt.Errorf("Calibrate: nil CoxResult")
	}
	if len(res.intervals) == 0 {
		return nil, fmt.Errorf("Calibrate: training intervals not retained on CoxResult")
	}
	durs := make([]float64, 0, len(res.intervals))
	for _, it := range res.intervals {
		durs = append(durs, float64(it.Duration))
	}
	sort.Float64s(durs)
	tau := durs[len(durs)/2]
	if tau <= 0 {
		tau = 1
	}
	return f.CalibrateAt(res, holdout, tau)
}
