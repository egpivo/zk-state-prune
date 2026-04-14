package analysis

import (
	"context"
	"math"
	"math/rand/v2"
	"testing"

	"github.com/egpivo/zk-state-prune/internal/extractor"
	"github.com/egpivo/zk-state-prune/internal/model"
)

// phHoldsIntervals fabricates a Cox-friendly dataset: the hazard depends
// on a single covariate with a *constant* multiplicative effect. Higher
// AccessCount → shorter Duration, but the multiplier doesn't drift with
// time. The Schoenfeld test should *fail to reject* PH (large p-values).
func phHoldsIntervals(n int) []model.InterAccessInterval {
	r := rand.New(rand.NewPCG(2, 3))
	out := make([]model.InterAccessInterval, 0, n)
	for i := 0; i < n; i++ {
		ac := uint64(r.IntN(20))
		// Exponential gap with rate proportional to (1 + ac), constant
		// over time. -log(U) gives an exponential.
		u := r.Float64()
		if u < 1e-9 {
			u = 1e-9
		}
		dur := uint64(math.Max(1, -math.Log(u)*100/float64(ac+1)))
		out = append(out, model.InterAccessInterval{
			SlotID:      "s",
			Duration:    dur,
			IsObserved:  true,
			AccessCount: ac,
			SlotAge:     uint64(i*7 + r.IntN(5)),
		})
	}
	return out
}

// phViolatedIntervals fabricates a dataset where the AccessCount effect
// flips sign across time: for "early" rows higher AccessCount → shorter
// Duration, for "late" rows higher AccessCount → longer Duration. That
// is precisely what Cox cannot fit with a single proportional coefficient,
// and Schoenfeld residuals should correlate strongly with time.
func phViolatedIntervals(n int) []model.InterAccessInterval {
	r := rand.New(rand.NewPCG(5, 13))
	out := make([]model.InterAccessInterval, 0, n)
	for i := 0; i < n; i++ {
		ac := uint64(r.IntN(20))
		// Sign of effect flips depending on row index, which after
		// sorting by Duration becomes a flip across the time axis.
		var dur uint64
		if i%2 == 0 {
			dur = uint64(math.Max(1, 20+float64(ac)*5+r.NormFloat64()*3))
		} else {
			dur = uint64(math.Max(1, 200-float64(ac)*5+r.NormFloat64()*3))
		}
		out = append(out, model.InterAccessInterval{
			SlotID:      "s",
			Duration:    dur,
			IsObserved:  true,
			AccessCount: ac,
			SlotAge:     uint64(i*7 + r.IntN(5)),
		})
	}
	return out
}

func TestCheckPH_NotRejectedWhenPHHolds(t *testing.T) {
	ivs := phHoldsIntervals(400)
	res, err := NewStatmodelFitter().FitCoxPH(ivs, DefaultCoxPredictors)
	if err != nil {
		t.Fatalf("FitCoxPH: %v", err)
	}
	out, err := NewStatmodelFitter().CheckPH(res)
	if err != nil {
		t.Fatalf("CheckPH: %v", err)
	}
	if got := len(out.PerCovariatePValue); got != len(DefaultCoxPredictors) {
		t.Fatalf("len(PerCovariatePValue)=%d want %d", got, len(DefaultCoxPredictors))
	}
	for name, p := range out.PerCovariatePValue {
		if p < 0 || p > 1 {
			t.Errorf("p-value for %s = %v out of [0,1]", name, p)
		}
	}
	// On well-behaved PH-respecting data we expect at least one
	// covariate's p-value to be comfortably above 0.05. We don't pin
	// the global because Fisher's combined test can still be small if
	// any single covariate noisily violates.
	largest := 0.0
	for _, p := range out.PerCovariatePValue {
		if p > largest {
			largest = p
		}
	}
	if largest < 0.05 {
		t.Errorf("all covariate p-values are tiny on PH-holding data: %+v", out.PerCovariatePValue)
	}
}

func TestCheckPH_RejectsWhenEffectIsTimeDependent(t *testing.T) {
	ivs := phViolatedIntervals(600)
	res, err := NewStatmodelFitter().FitCoxPH(ivs, DefaultCoxPredictors)
	if err != nil {
		t.Fatalf("FitCoxPH: %v", err)
	}
	out, err := NewStatmodelFitter().CheckPH(res)
	if err != nil {
		t.Fatalf("CheckPH: %v", err)
	}
	// The AccessCount effect was constructed to flip across time, so
	// its Schoenfeld residuals should correlate strongly with time.
	pAccess, ok := out.PerCovariatePValue[ColAccessCount]
	if !ok {
		t.Fatalf("AccessCount p-value missing: %+v", out.PerCovariatePValue)
	}
	if pAccess >= 0.05 {
		t.Errorf("AccessCount PH test p=%v, want < 0.05 on time-dependent data", pAccess)
	}
	if out.GlobalPValue >= 0.05 {
		t.Errorf("Global PH p=%v, want < 0.05 on time-dependent data", out.GlobalPValue)
	}
}

func TestCheckPH_NilAndEmpty(t *testing.T) {
	f := NewStatmodelFitter()
	if _, err := f.CheckPH(nil); err == nil {
		t.Error("expected error on nil")
	}
	if _, err := f.CheckPH(&CoxResult{}); err == nil {
		t.Error("expected error on empty CoxResult")
	}
}

func TestCheckPH_OnMockEndToEnd(t *testing.T) {
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
	res, err := NewStatmodelFitter().FitCoxPH(built.Intervals, DefaultCoxPredictors)
	if err != nil {
		t.Fatalf("FitCoxPH: %v", err)
	}
	out, err := NewStatmodelFitter().CheckPH(res)
	if err != nil {
		t.Fatalf("CheckPH: %v", err)
	}
	for name, p := range out.PerCovariatePValue {
		if math.IsNaN(p) || p < 0 || p > 1 {
			t.Errorf("%s: p=%v out of [0,1]", name, p)
		}
	}
	if math.IsNaN(out.GlobalPValue) || out.GlobalPValue < 0 || out.GlobalPValue > 1 {
		t.Errorf("GlobalPValue=%v out of [0,1]", out.GlobalPValue)
	}
}

func TestAverageRanks_HandlesTies(t *testing.T) {
	// Input: [10, 20, 20, 30] → sorted ranks: 1, 2.5, 2.5, 4
	got := averageRanks([]float64{10, 20, 20, 30})
	want := []float64{1, 2.5, 2.5, 4}
	for i := range got {
		if math.Abs(got[i]-want[i]) > 1e-9 {
			t.Errorf("rank[%d]=%v want %v", i, got[i], want[i])
		}
	}
}

func TestPearsonCorrelation_KnownAnswer(t *testing.T) {
	// Perfect linear relationship → ρ = 1.
	x := []float64{1, 2, 3, 4, 5}
	y := []float64{2, 4, 6, 8, 10}
	if got := pearsonCorrelation(x, y); math.Abs(got-1) > 1e-9 {
		t.Errorf("pearson = %v, want 1", got)
	}
	// Anti-correlated → ρ = -1.
	y2 := []float64{10, 8, 6, 4, 2}
	if got := pearsonCorrelation(x, y2); math.Abs(got+1) > 1e-9 {
		t.Errorf("pearson = %v, want -1", got)
	}
	// Zero variance on one side → 0 (no NaN).
	if got := pearsonCorrelation(x, []float64{3, 3, 3, 3, 3}); got != 0 {
		t.Errorf("pearson on flat = %v, want 0", got)
	}
}
