package analysis

import (
	"context"
	"math"
	"testing"

	"github.com/egpivo/zk-state-prune/internal/extractor"
	"github.com/egpivo/zk-state-prune/internal/model"
)

// makeIntervals is a tiny helper: each pair (duration, observed) becomes an
// interval with fresh identifiers and default covariates. Useful for unit
// tests that want to drive the KM fitter with hand-crafted data.
func makeIntervals(pairs ...struct {
	Dur      uint64
	Observed bool
	Cat      model.ContractCategory
	Slot     model.SlotType
}) []model.InterAccessInterval {
	out := make([]model.InterAccessInterval, 0, len(pairs))
	for i, p := range pairs {
		out = append(out, model.InterAccessInterval{
			SlotID:        "s",
			IntervalStart: 0,
			IntervalEnd:   p.Dur,
			Duration:      p.Dur,
			IsObserved:    p.Observed,
			EntryTime:     0,
			ContractType:  p.Cat,
			SlotType:      p.Slot,
			AccessCount:   uint64(i),
		})
	}
	return out
}

func TestKaplanMeier_AllObserved(t *testing.T) {
	// Three events at 10, 20, 30. KM should drop from 1 → 2/3 → 1/3 → 0.
	ivs := makeIntervals(
		struct {
			Dur      uint64
			Observed bool
			Cat      model.ContractCategory
			Slot     model.SlotType
		}{10, true, model.ContractERC20, model.SlotTypeBalance},
		struct {
			Dur      uint64
			Observed bool
			Cat      model.ContractCategory
			Slot     model.SlotType
		}{20, true, model.ContractERC20, model.SlotTypeBalance},
		struct {
			Dur      uint64
			Observed bool
			Cat      model.ContractCategory
			Slot     model.SlotType
		}{30, true, model.ContractERC20, model.SlotTypeBalance},
	)

	res, err := NewStatmodelFitter().FitKaplanMeier(ivs)
	if err != nil {
		t.Fatalf("FitKaplanMeier: %v", err)
	}
	if res.NumEvents != 3 || res.NumCensored != 0 {
		t.Fatalf("events=%d censored=%d want 3,0", res.NumEvents, res.NumCensored)
	}
	// Survival curve should be monotone decreasing and end near 0.
	if len(res.Surv) == 0 {
		t.Fatalf("empty Surv")
	}
	for i := 1; i < len(res.Surv); i++ {
		if res.Surv[i] > res.Surv[i-1]+1e-9 {
			t.Fatalf("Surv not monotone: %v", res.Surv)
		}
	}
	if last := res.Surv[len(res.Surv)-1]; last > 1e-9 {
		t.Errorf("final Surv = %v, want ~0 for all-observed", last)
	}
	// Median should exist and sit between the first and last event.
	if math.IsNaN(res.MedianSurv) || res.MedianSurv < 10 || res.MedianSurv > 30 {
		t.Errorf("MedianSurv = %v, want in [10,30]", res.MedianSurv)
	}
}

func TestKaplanMeier_AllCensored(t *testing.T) {
	ivs := makeIntervals(
		struct {
			Dur      uint64
			Observed bool
			Cat      model.ContractCategory
			Slot     model.SlotType
		}{100, false, model.ContractERC20, model.SlotTypeBalance},
		struct {
			Dur      uint64
			Observed bool
			Cat      model.ContractCategory
			Slot     model.SlotType
		}{200, false, model.ContractERC20, model.SlotTypeBalance},
	)

	res, err := NewStatmodelFitter().FitKaplanMeier(ivs)
	if err != nil {
		t.Fatalf("FitKaplanMeier: %v", err)
	}
	if res.NumEvents != 0 || res.NumCensored != 2 {
		t.Fatalf("events=%d censored=%d want 0,2", res.NumEvents, res.NumCensored)
	}
	for i, sv := range res.Surv {
		if math.Abs(sv-1.0) > 1e-9 {
			t.Errorf("Surv[%d] = %v, want 1 (no events)", i, sv)
		}
	}
	if !math.IsNaN(res.MedianSurv) {
		t.Errorf("MedianSurv = %v, want NaN for fully-censored curve", res.MedianSurv)
	}
}

func TestKaplanMeier_MixedCensoring(t *testing.T) {
	ivs := makeIntervals(
		struct {
			Dur      uint64
			Observed bool
			Cat      model.ContractCategory
			Slot     model.SlotType
		}{10, true, model.ContractERC20, model.SlotTypeBalance},
		struct {
			Dur      uint64
			Observed bool
			Cat      model.ContractCategory
			Slot     model.SlotType
		}{20, false, model.ContractERC20, model.SlotTypeBalance},
		struct {
			Dur      uint64
			Observed bool
			Cat      model.ContractCategory
			Slot     model.SlotType
		}{30, true, model.ContractERC20, model.SlotTypeBalance},
		struct {
			Dur      uint64
			Observed bool
			Cat      model.ContractCategory
			Slot     model.SlotType
		}{40, false, model.ContractERC20, model.SlotTypeBalance},
	)
	res, err := NewStatmodelFitter().FitKaplanMeier(ivs)
	if err != nil {
		t.Fatalf("FitKaplanMeier: %v", err)
	}
	if res.NumEvents != 2 || res.NumCensored != 2 {
		t.Fatalf("events=%d censored=%d want 2,2", res.NumEvents, res.NumCensored)
	}
	// Monotone non-increasing.
	for i := 1; i < len(res.Surv); i++ {
		if res.Surv[i] > res.Surv[i-1]+1e-9 {
			t.Fatalf("Surv not monotone: %v", res.Surv)
		}
	}
	// SurvAt bracketing: before first event → 1, after last → last value.
	if got := res.SurvAt(5); got != 1 {
		t.Errorf("SurvAt(5)=%v, want 1", got)
	}
	if got := res.SurvAt(1e9); got != res.Surv[len(res.Surv)-1] {
		t.Errorf("SurvAt(large) = %v, want %v", got, res.Surv[len(res.Surv)-1])
	}
}

func TestKaplanMeier_EmptyIntervalsError(t *testing.T) {
	if _, err := NewStatmodelFitter().FitKaplanMeier(nil); err == nil {
		t.Fatal("expected error on empty intervals")
	}
}

func TestFitKaplanMeierStratified_OnMock(t *testing.T) {
	ctx := context.Background()
	db := openDB(t)

	cfg := extractor.DefaultMockConfig()
	cfg.NumContracts = 40
	cfg.SlotsPerContractXmin = 5
	cfg.SlotsPerContractMax = 30
	cfg.TotalBlocks = 10_000
	cfg.Window = model.ObservationWindow{Start: 2_000, End: 10_000}
	cfg.AccessRateXmin = 1e-4
	cfg.MaxEventsPerSlot = 100
	cfg.PeriodBlocks = 1_000

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

	curves, err := FitKaplanMeierStratified(NewStatmodelFitter(), built.Intervals, StratumByContractType)
	if err != nil {
		t.Fatalf("FitKaplanMeierStratified: %v", err)
	}
	if len(curves) == 0 {
		t.Fatal("no strata")
	}
	// Each stratum must carry its own label, consistent counts, and a
	// monotone curve.
	for label, c := range curves {
		if c.Label != label {
			t.Errorf("label mismatch: map %q vs struct %q", label, c.Label)
		}
		if c.N != c.NumEvents+c.NumCensored {
			t.Errorf("%s: N=%d != events+censored=%d", label, c.N, c.NumEvents+c.NumCensored)
		}
		for i := 1; i < len(c.Surv); i++ {
			if c.Surv[i] > c.Surv[i-1]+1e-9 {
				t.Errorf("%s: Surv not monotone at idx %d: %v", label, i, c.Surv)
				break
			}
		}
		// SurvAt should be bounded in [0,1] and non-increasing under
		// larger probes. A minority of intervals have Duration==0
		// (legitimate intra-block co-accesses), so S(0) can legitimately
		// sit below 1.
		if s := c.SurvAt(0); s < 0.5 || s > 1.0 {
			t.Errorf("%s: SurvAt(0)=%v out of [0.5,1]", label, s)
		}
		if s0, s1 := c.SurvAt(0), c.SurvAt(1e9); s1 > s0+1e-9 {
			t.Errorf("%s: SurvAt(∞)=%v > SurvAt(0)=%v", label, s1, s0)
		}
	}
}
