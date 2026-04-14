package analysis

import (
	"context"
	"math"
	"math/rand/v2"
	"testing"

	"github.com/egpivo/zk-state-prune/internal/extractor"
	"github.com/egpivo/zk-state-prune/internal/model"
)

// syntheticCoxIntervals fabricates intervals where Duration is monotone in
// AccessCount: higher AccessCount → shorter gap. Cox should learn a
// positive coefficient on AccessCount (higher hazard → shorter survival).
// All intervals are observed (no censoring) so the test is unambiguous.
func syntheticCoxIntervals(n int) []model.InterAccessInterval {
	r := rand.New(rand.NewPCG(7, 11))
	out := make([]model.InterAccessInterval, 0, n)
	for i := 0; i < n; i++ {
		ac := uint64(r.IntN(20))
		// Inverse relationship + jitter, clamped to >=1.
		base := 100.0 / float64(ac+1)
		dur := uint64(math.Max(1, base+r.NormFloat64()*2))
		out = append(out, model.InterAccessInterval{
			SlotID:       "s",
			Duration:     dur,
			IsObserved:   true,
			AccessCount:  ac,
			// Distinct values so the test does not stumble into the
			// same collinearity pitfall the Phase-1 mock has.
			ContractAge:  uint64(i * 10),
			SlotAge:      uint64(i*7 + r.IntN(5)),
			ContractType: model.ContractERC20,
			SlotType:     model.SlotTypeBalance,
		})
	}
	return out
}

func TestFitCoxPH_LearnsExpectedSign(t *testing.T) {
	ivs := syntheticCoxIntervals(500)
	res, err := NewStatmodelFitter().FitCoxPH(ivs, DefaultCoxPredictors)
	if err != nil {
		t.Fatalf("FitCoxPH: %v", err)
	}
	if got, want := len(res.Coef), len(DefaultCoxPredictors); got != want {
		t.Fatalf("len(Coef)=%d want %d", got, want)
	}
	if res.NumObs != len(ivs) || res.NumEvents != len(ivs) {
		t.Errorf("NumObs=%d NumEvents=%d, want %d/%d", res.NumObs, res.NumEvents, len(ivs), len(ivs))
	}
	// AccessCount is the first predictor; its coef should be positive
	// (higher prior accesses → higher hazard → shorter gap).
	idx := -1
	for i, p := range res.Predictors {
		if p == ColAccessCount {
			idx = i
		}
	}
	if idx < 0 {
		t.Fatalf("AccessCount predictor missing: %v", res.Predictors)
	}
	if res.Coef[idx] <= 0 {
		t.Errorf("AccessCount coef = %v, want > 0", res.Coef[idx])
	}
	// Sanity: log-likelihood and SE all finite.
	if math.IsNaN(res.LogLike) || math.IsInf(res.LogLike, 0) {
		t.Errorf("LogLike not finite: %v", res.LogLike)
	}
	for i, se := range res.StdErr {
		if math.IsNaN(se) || math.IsInf(se, 0) || se <= 0 {
			t.Errorf("StdErr[%d]=%v, want positive finite", i, se)
		}
	}
	// BaselineCumHaz should be non-empty and monotone non-decreasing.
	if len(res.BaselineCumHaz) == 0 {
		t.Fatalf("BaselineCumHaz empty")
	}
	for i := 1; i < len(res.BaselineCumHaz); i++ {
		if res.BaselineCumHaz[i] < res.BaselineCumHaz[i-1]-1e-9 {
			t.Errorf("BaselineCumHaz not monotone at %d: %v", i, res.BaselineCumHaz)
			break
		}
	}
}

func TestFitCoxPH_PredictAccessProbBounds(t *testing.T) {
	ivs := syntheticCoxIntervals(300)
	res, err := NewStatmodelFitter().FitCoxPH(ivs, DefaultCoxPredictors)
	if err != nil {
		t.Fatalf("FitCoxPH: %v", err)
	}
	// Probability of access at tau=0 must be 0; at tau=∞ must be ≤ 1.
	covar := map[string]float64{
		ColAccessCount: 5,
		ColContractAge: 100,
		ColSlotAge:     100,
	}
	if p := res.PredictAccessProb(covar, 0); p != 0 {
		t.Errorf("PredictAccessProb(tau=0)=%v want 0", p)
	}
	if p := res.PredictAccessProb(covar, 1e12); p < 0 || p > 1 {
		t.Errorf("PredictAccessProb(tau=∞)=%v out of [0,1]", p)
	}
	// Monotone in tau.
	prev := 0.0
	for _, tau := range []float64{1, 5, 20, 80, 200} {
		p := res.PredictAccessProb(covar, tau)
		if p < prev-1e-9 {
			t.Errorf("PredictAccessProb decreased: tau=%v p=%v prev=%v", tau, p, prev)
		}
		prev = p
	}
	// Higher AccessCount should give higher access probability at fixed tau.
	low := res.PredictAccessProb(map[string]float64{ColAccessCount: 0, ColContractAge: 100, ColSlotAge: 100}, 50)
	high := res.PredictAccessProb(map[string]float64{ColAccessCount: 15, ColContractAge: 100, ColSlotAge: 100}, 50)
	if !(high > low) {
		t.Errorf("high-access prob (%v) not greater than low-access (%v)", high, low)
	}
}

func TestFitCoxPH_EmptyError(t *testing.T) {
	if _, err := NewStatmodelFitter().FitCoxPH(nil, nil); err == nil {
		t.Fatal("expected error on empty intervals")
	}
}

func TestFitCoxPH_OnMockEndToEnd(t *testing.T) {
	ctx := context.Background()
	db := openDB(t)

	cfg := extractor.DefaultMockConfig()
	cfg.NumContracts = 30
	cfg.SlotsPerContractXmin = 5
	cfg.SlotsPerContractMax = 30
	cfg.TotalBlocks = 20_000
	cfg.Window = model.ObservationWindow{Start: 4_000, End: 20_000}
	cfg.AccessRateXmin = 1e-4
	cfg.MaxEventsPerSlot = 200
	cfg.PeriodBlocks = 2_000

	if err := extractor.NewMockExtractor(cfg).Extract(ctx, db); err != nil {
		t.Fatalf("Extract: %v", err)
	}
	built, err := BuildIntervals(ctx, db, cfg.Window)
	if err != nil {
		t.Fatalf("BuildIntervals: %v", err)
	}
	if len(built.Intervals) == 0 {
		t.Fatal("no intervals built")
	}

	res, err := NewStatmodelFitter().FitCoxPH(built.Intervals, DefaultCoxPredictors)
	if err != nil {
		t.Fatalf("FitCoxPH on mock: %v", err)
	}
	if res.NumObs != len(built.Intervals) {
		t.Errorf("NumObs=%d want %d", res.NumObs, len(built.Intervals))
	}
	for _, c := range res.Coef {
		if math.IsNaN(c) || math.IsInf(c, 0) {
			t.Errorf("non-finite coef %v in %v", c, res.Coef)
		}
	}
}

