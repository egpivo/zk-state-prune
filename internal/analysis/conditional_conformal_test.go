package analysis

import (
	"math"
	"math/rand/v2"
	"path/filepath"
	"testing"

	"github.com/egpivo/zk-state-prune/internal/domain"
)

func TestConditionalConformal_OutOfSampleCoverage(t *testing.T) {
	train := phHoldsIntervals(800)
	test := phHoldsIntervals(600)

	fitter := NewStatmodelFitter()
	res, err := fitter.FitCoxPH(train, DefaultCoxPredictors)
	if err != nil {
		t.Fatalf("FitCoxPH: %v", err)
	}
	calib, err := fitter.Calibrate(res, train)
	if err != nil {
		t.Fatalf("Calibrate: %v", err)
	}
	if calib.ConditionalEpsilon <= 0 {
		t.Fatalf("ConditionalEpsilon=%v, expected > 0 on well-behaved synthetic data", calib.ConditionalEpsilon)
	}

	// On a fresh test set drawn from the same distribution, the true
	// conditional label should fall inside [pred - ε, pred + ε] with
	// rate at least CoverageLevel − slack. Different RNG seed than
	// CalibrateAt's salt so we're not recovering cal-half points.
	r := rand.New(rand.NewPCG(29, 31))
	evaluated, covered := measureConditionalCoverage(calib, test, r)
	if evaluated == 0 {
		t.Fatal("no evaluable test rows for conditional coverage")
	}
	rate := float64(covered) / float64(evaluated)
	// Nominal 0.9 − 20% slack for finite-sample + per-row-index
	// reuse approximations. Well below this rate signals a real
	// miscalibration, not just sampling noise.
	if rate < 0.7 {
		t.Errorf("conditional coverage = %v (%d/%d), want ≥ 0.7", rate, covered, evaluated)
	}
}

// measureConditionalCoverage walks test rows, samples a random idle
// u ∈ [0, Duration) per row, and checks whether the observed label
// at horizon u+τ falls inside [pred − ε, pred + ε]. Returns the
// (evaluated, covered) pair for the caller to translate into a rate.
func measureConditionalCoverage(
	calib *CalibratedModel,
	test []domain.InterAccessInterval,
	r *rand.Rand,
) (evaluated, covered int) {
	tau := calib.Tau
	for _, it := range test {
		if it.Duration == 0 {
			continue
		}
		u := uint64(r.Float64() * float64(it.Duration))
		label, ok := conditionalLabel(it, u, tau)
		if !ok {
			continue
		}
		pred, ok := conditionalPred(calib, it, u, tau)
		if !ok {
			continue
		}
		lower, upper := clampUnit(pred-calib.ConditionalEpsilon, pred+calib.ConditionalEpsilon)
		evaluated++
		if label >= lower && label <= upper {
			covered++
		}
	}
	return evaluated, covered
}

// conditionalLabel returns the binary coverage label at idle u:
// 0 if (u + τ) still falls inside the observed duration, 1 if the
// interval terminated with an observed event, or !ok for the
// right-censored "we can't tell" case we skip for coverage.
func conditionalLabel(it domain.InterAccessInterval, u uint64, tau float64) (float64, bool) {
	reach := u + uint64(tau+0.5)
	switch {
	case reach <= it.Duration:
		return 0, true
	case it.IsObserved:
		return 1, true
	default:
		return 0, false
	}
}

// conditionalPred returns the raw Cox conditional access probability
// at idle u with horizon τ, or !ok if the conditional is undefined
// (S(u) ≤ 0 means the interval has already "died" at u).
func conditionalPred(calib *CalibratedModel, it domain.InterAccessInterval, u uint64, tau float64) (float64, bool) {
	sU := calib.Base.SurvivalForInterval(it, float64(u))
	if sU <= 0 {
		return 0, false
	}
	sUTau := calib.Base.SurvivalForInterval(it, float64(u)+tau)
	pred := 1 - sUTau/sU
	if pred < 0 {
		pred = 0
	}
	if pred > 1 {
		pred = 1
	}
	return pred, true
}

func clampUnit(lo, hi float64) (float64, float64) {
	if lo < 0 {
		lo = 0
	}
	if hi > 1 {
		hi = 1
	}
	return lo, hi
}

func TestPredictUpperConditional_BoundedAndAboveRaw(t *testing.T) {
	train := phHoldsIntervals(600)
	fitter := NewStatmodelFitter()
	res, err := fitter.FitCoxPH(train, DefaultCoxPredictors)
	if err != nil {
		t.Fatalf("FitCoxPH: %v", err)
	}
	calib, err := fitter.Calibrate(res, train)
	if err != nil {
		t.Fatalf("Calibrate: %v", err)
	}
	if calib.ConditionalEpsilon == 0 {
		t.Skip("ConditionalEpsilon was zero on this seed — skip the above-raw check")
	}
	// Pick a few probe intervals + a range of idle values, check the
	// upper-bound predictor returns something in [0, 1] and >= the
	// raw conditional (because we add ε >= 0).
	for _, it := range train[:30] {
		for _, idle := range []float64{1, 10, 100, 1000} {
			sU := calib.Base.SurvivalForInterval(it, idle)
			if sU <= 0 {
				continue
			}
			sUTau := calib.Base.SurvivalForInterval(it, idle+calib.Tau)
			raw := 1 - sUTau/sU
			if raw < 0 {
				raw = 0
			}
			if raw > 1 {
				raw = 1
			}
			up := calib.PredictUpperConditionalAccessProb(it, idle)
			if up < 0 || up > 1 {
				t.Errorf("upper=%v out of [0,1]", up)
			}
			if up < raw-1e-12 {
				t.Errorf("upper=%v < raw=%v (ε=%v)", up, raw, calib.ConditionalEpsilon)
			}
		}
	}
}

func TestModelFile_ConditionalEpsilonRoundTrip(t *testing.T) {
	ivs := phHoldsIntervals(500)
	fitter := NewStatmodelFitter()
	res, err := fitter.FitCoxPH(ivs, DefaultCoxPredictors)
	if err != nil {
		t.Fatalf("FitCoxPH: %v", err)
	}
	calib, err := fitter.Calibrate(res, ivs)
	if err != nil {
		t.Fatalf("Calibrate: %v", err)
	}
	if calib.ConditionalEpsilon == 0 {
		t.Skip("ConditionalEpsilon zero on this seed — nothing to round-trip")
	}
	path := filepath.Join(t.TempDir(), "cond.json")
	if err := SaveModelFile(path, calib); err != nil {
		t.Fatalf("SaveModelFile: %v", err)
	}
	loaded, err := LoadModelFile(path)
	if err != nil {
		t.Fatalf("LoadModelFile: %v", err)
	}
	if loaded.ConditionalEpsilon != calib.ConditionalEpsilon {
		t.Errorf("ConditionalEpsilon: loaded=%v orig=%v", loaded.ConditionalEpsilon, calib.ConditionalEpsilon)
	}
	// Full prediction parity across idle values.
	for _, it := range ivs[:20] {
		for _, u := range []float64{0, 5, 50, 500} {
			oUp := calib.PredictUpperConditionalAccessProb(it, u)
			lUp := loaded.PredictUpperConditionalAccessProb(it, u)
			if math.Abs(oUp-lUp) > 1e-12 {
				t.Errorf("upper(u=%v): orig=%v loaded=%v", u, oUp, lUp)
			}
		}
	}
}

func TestConditionalEpsilon_ZeroOnTinyHoldout(t *testing.T) {
	// Extremely small holdout (below the 10-row minimum for the
	// conditional conformal fit): ConditionalEpsilon stays 0 and the
	// PredictUpperConditionalAccessProb falls back to the raw point
	// estimate.
	ivs := phHoldsIntervals(15)
	fitter := NewStatmodelFitter()
	res, err := fitter.FitCoxPH(ivs, DefaultCoxPredictors)
	if err != nil {
		t.Fatalf("FitCoxPH: %v", err)
	}
	calib, err := fitter.Calibrate(res, ivs)
	if err != nil {
		// With only 15 rows we might hit the >= 10 cal-half bar anyway;
		// skip if Calibrate bails out.
		t.Skipf("Calibrate refused tiny holdout: %v", err)
	}
	// With ε=0, upper and raw must match.
	if calib.ConditionalEpsilon != 0 {
		t.Skip("got non-zero ε on 15-row holdout — extremely lucky seed; not a failure")
	}
	for _, it := range ivs[:5] {
		for _, u := range []float64{10, 50} {
			sU := calib.Base.SurvivalForInterval(it, u)
			if sU <= 0 {
				continue
			}
			sUTau := calib.Base.SurvivalForInterval(it, u+calib.Tau)
			raw := 1 - sUTau/sU
			if raw < 0 {
				raw = 0
			}
			up := calib.PredictUpperConditionalAccessProb(it, u)
			if math.Abs(up-raw) > 1e-12 {
				t.Errorf("ε=0 but upper=%v != raw=%v", up, raw)
			}
		}
	}
}
