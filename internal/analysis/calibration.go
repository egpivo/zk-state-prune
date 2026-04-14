package analysis

import "github.com/egpivo/zk-state-prune/internal/model"

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
	// Recalibration grid: PredX[i] is a sorted predicted probability,
	// CalibratedY[i] is the isotonic-fitted empirical rate at that
	// quantile. PredictAccessProb does a step lookup on this grid.
	PredX       []float64
	CalibratedY []float64
}

// PredictAccessProb returns the recalibrated probability of access by
// horizon tau for the given covariates. Until Segment 11 lands this
// just delegates to the base model; the wrapper exists so calling code
// can be written against the final API today.
func (c *CalibratedModel) PredictAccessProb(covariates map[string]float64, tau float64) float64 {
	if c == nil || c.Base == nil {
		return 0
	}
	raw := c.Base.PredictAccessProb(covariates, tau)
	if len(c.PredX) == 0 {
		return raw
	}
	return isotonicLookup(c.PredX, c.CalibratedY, raw)
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

// CheckPH runs the Schoenfeld-residuals PH-assumption test. Stub for
// Segment 9; returns ErrNotImplemented today so callers can compile.
func (StatmodelFitter) CheckPH(_ *CoxResult) (*PHTestResult, error) {
	return nil, ErrNotImplemented
}

// Calibrate fits an isotonic recalibration on the holdout interval set.
// Stub for Segment 11; returns ErrNotImplemented today.
func (StatmodelFitter) Calibrate(_ *CoxResult, _ []model.InterAccessInterval) (*CalibratedModel, error) {
	return nil, ErrNotImplemented
}
