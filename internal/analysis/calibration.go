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
// Phase-1 censoring policy: each holdout interval is labelled as
//
//   - 1 ("happened") if IsObserved && Duration <= tau
//   - 0 ("did not happen by tau") if Duration > tau, regardless of
//     observed/censored — at time tau we still hadn't seen an access
//   - dropped if !IsObserved && Duration <= tau ("censored before tau"):
//     we don't know whether an access would have happened by tau, so
//     using either label biases the curve.
//
// The drop policy is the cleanest small-N approach. A future revision
// can replace it with bin-wise Kaplan–Meier observed rates that
// incorporate the dropped rows correctly.
type CalibrationCurve struct {
	Tau float64

	Bins []CalibrationBin

	// BrierScore is the mean squared error between predicted probability
	// and the binary label, averaged over kept rows. Lower is better; a
	// well-calibrated and well-discriminating model has a small Brier.
	BrierScore float64

	// NumKept and NumDropped account for every holdout row. Reporting
	// both lets the reader sanity-check the censoring drop rate.
	NumKept    int
	NumDropped int
}

// CalibrationBin is one row of a reliability diagram.
type CalibrationBin struct {
	PredictedMean float64
	ObservedRate  float64
	N             int
}

// CalibrationCurveFromCox evaluates a fitted Cox model on the holdout
// intervals at horizon tau and produces a reliability diagram with
// `nBins` equal-population quantile bins.
func CalibrationCurveFromCox(
	model *CoxResult,
	holdout []model.InterAccessInterval,
	tau float64,
	nBins int,
) (*CalibrationCurve, error) {
	if model == nil {
		return nil, fmt.Errorf("CalibrationCurveFromCox: nil model")
	}
	if tau <= 0 {
		return nil, fmt.Errorf("CalibrationCurveFromCox: tau must be > 0")
	}
	if len(holdout) == 0 {
		return nil, fmt.Errorf("CalibrationCurveFromCox: empty holdout")
	}
	if nBins <= 0 {
		nBins = 10
	}

	type point struct{ p, y float64 }
	pts := make([]point, 0, len(holdout))
	dropped := 0
	for _, it := range holdout {
		var label float64
		switch {
		case it.IsObserved && it.Duration <= uint64(tau):
			label = 1
		case it.Duration > uint64(tau):
			label = 0
		default:
			// censored && Duration <= tau → ambiguous, skip.
			dropped++
			continue
		}
		p := model.PredictAccessProbForInterval(it, tau)
		pts = append(pts, point{p: p, y: label})
	}
	if len(pts) < nBins {
		return nil, fmt.Errorf("CalibrationCurveFromCox: only %d non-ambiguous rows for %d bins", len(pts), nBins)
	}

	sort.Slice(pts, func(i, j int) bool { return pts[i].p < pts[j].p })

	bins := make([]CalibrationBin, 0, nBins)
	binSize := len(pts) / nBins
	brier := 0.0
	for b := 0; b < nBins; b++ {
		lo := b * binSize
		hi := lo + binSize
		if b == nBins-1 {
			hi = len(pts)
		}
		var sumP, sumY float64
		for j := lo; j < hi; j++ {
			sumP += pts[j].p
			sumY += pts[j].y
			d := pts[j].p - pts[j].y
			brier += d * d
		}
		n := hi - lo
		bins = append(bins, CalibrationBin{
			PredictedMean: sumP / float64(n),
			ObservedRate:  sumY / float64(n),
			N:             n,
		})
	}

	return &CalibrationCurve{
		Tau:        tau,
		Bins:       bins,
		BrierScore: brier / float64(len(pts)),
		NumKept:    len(pts),
		NumDropped: dropped,
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
// CoxResult and recovers a calibrated PredictAccessProb via PAV-fitted
// step lookup.
//
// Censoring handling matches CalibrationCurveFromCox: rows that are
// censored before tau are dropped because their label is unknowable
// without a survival adjustment; rows with Duration > tau are 0
// regardless of their observed-flag.
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

	type point struct{ p, y float64 }
	pts := make([]point, 0, len(holdout))
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
		pts = append(pts, point{
			p: res.PredictAccessProbForInterval(it, tau),
			y: label,
		})
	}
	if len(pts) < 5 {
		return nil, fmt.Errorf("CalibrateAt: only %d non-ambiguous holdout rows (need >= 5)", len(pts))
	}

	sort.Slice(pts, func(i, j int) bool { return pts[i].p < pts[j].p })

	x := make([]float64, len(pts))
	y := make([]float64, len(pts))
	for i, p := range pts {
		x[i] = p.p
		y[i] = p.y
	}
	yHat := poolAdjacentViolators(y)

	return &CalibratedModel{
		Base:        res,
		Tau:         tau,
		PredX:       x,
		CalibratedY: yHat,
	}, nil
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
