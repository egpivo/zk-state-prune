package pruning

import (
	"math"
	"testing"

	"github.com/egpivo/zk-state-prune/internal/model"
)

// constPredict returns a fixed probability regardless of the interval
// and idle duration. Lets the unit tests pin the policy output
// deterministically when the T*-search isn't the thing under test.
func constPredict(p float64) CondAccessProbFunc {
	return func(_ model.InterAccessInterval, _ float64) float64 { return p }
}

func TestStatisticalPolicy_PStarComputation(t *testing.T) {
	costs := CostParams{RAMUnitCost: 2, MissPenalty: 100}
	p, err := NewStatisticalPolicy("s", constPredict(0), 10, costs)
	if err != nil {
		t.Fatalf("NewStatisticalPolicy: %v", err)
	}
	want := 0.2 // (2 * 10) / 100
	if math.Abs(p.PStar()-want) > 1e-12 {
		t.Errorf("PStar=%v want %v", p.PStar(), want)
	}
	if p.Tau() != 10 {
		t.Errorf("Tau=%v want 10", p.Tau())
	}
}

func TestStatisticalPolicy_PStarSaturatesAt1(t *testing.T) {
	p, _ := NewStatisticalPolicy("s", constPredict(0.5), 1000, CostParams{RAMUnitCost: 1, MissPenalty: 10})
	if p.PStar() != 1 {
		t.Errorf("PStar=%v want 1 (clamped)", p.PStar())
	}
}

func TestStatisticalPolicy_DemotesWhenProbBelowPStarFromIdleZero(t *testing.T) {
	// Constant p = 0.2 everywhere < p* = 0.5 → policy demotes at u=0
	// and the slot is fully cold for its entire interval.
	p, _ := NewStatisticalPolicy("s", constPredict(0.2), 10, CostParams{RAMUnitCost: 5, MissPenalty: 100})
	got := p.HotBlocks(model.InterAccessInterval{Duration: 100})
	if got != 0 {
		t.Errorf("HotBlocks=%d want 0 (demoted at u=0)", got)
	}
}

func TestStatisticalPolicy_KeepsHotWhenProbAbovePStarEverywhere(t *testing.T) {
	// Constant p = 0.9 > p* = 0.5 at every idle duration → never demotes.
	p, _ := NewStatisticalPolicy("s", constPredict(0.9), 10, CostParams{RAMUnitCost: 5, MissPenalty: 100})
	got := p.HotBlocks(model.InterAccessInterval{Duration: 100})
	if got != 100 {
		t.Errorf("HotBlocks=%d want 100 (kept hot)", got)
	}
}

func TestStatisticalPolicy_MidIntervalTransition(t *testing.T) {
	// The whole point of the T*-search: p(u) starts above p* and then
	// drops. The policy should find the crossover partway through the
	// interval. With N=20 samples over Duration=100, the sampling grid
	// has 5-block spacing; asking the predict function to flip at
	// u=50 lets us check the crossover lands within that step.
	predict := func(_ model.InterAccessInterval, idle float64) float64 {
		if idle < 50 {
			return 0.9
		}
		return 0.05
	}
	p, _ := NewStatisticalPolicy("s", predict, 10, CostParams{RAMUnitCost: 5, MissPenalty: 100}) // p* = 0.5
	got := p.HotBlocks(model.InterAccessInterval{Duration: 100})
	// Search grid is 0, 5, 10, …, 100 at 5-block spacing. The first u
	// with p < p* is 50. Allow ±5 slack for the grid.
	if got < 45 || got > 55 {
		t.Errorf("HotBlocks=%d, want ~50 (mid-interval crossover)", got)
	}
}

func TestStatisticalPolicy_PerSlotBranching(t *testing.T) {
	// Predict: high for slot "hot", low for slot "cold". Verify the
	// policy reads the interval and branches accordingly. Idle is
	// ignored so both slots produce the same decision at every u.
	predict := func(it model.InterAccessInterval, _ float64) float64 {
		if it.SlotID == "hot" {
			return 0.9
		}
		return 0.01
	}
	p, _ := NewStatisticalPolicy("s", predict, 10, CostParams{RAMUnitCost: 5, MissPenalty: 100}) // p* = 0.5
	hot := p.HotBlocks(model.InterAccessInterval{SlotID: "hot", Duration: 100})
	cold := p.HotBlocks(model.InterAccessInterval{SlotID: "cold", Duration: 100})
	if hot != 100 || cold != 0 {
		t.Errorf("hot=%d cold=%d, want 100,0", hot, cold)
	}
}

func TestStatisticalPolicy_ConstructorValidation(t *testing.T) {
	cases := []struct {
		name  string
		p     CondAccessProbFunc
		tau   float64
		costs CostParams
	}{
		{"nil predict", nil, 1, CostParams{1, 10}},
		{"tau=0", constPredict(0.5), 0, CostParams{1, 10}},
		{"miss=0", constPredict(0.5), 1, CostParams{1, 0}},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			if _, err := NewStatisticalPolicy("s", c.p, c.tau, c.costs); err == nil {
				t.Errorf("expected error for %s", c.name)
			}
		})
	}
}

func TestStatisticalPolicy_RunsThroughSimulator(t *testing.T) {
	// End-to-end: mid-interval demotion for a single slot. The
	// predict function returns hot at idle=0 and cold at idle>=10.
	// With Duration=20 the simulator should record HotBlocks ≈ 10,
	// cold tail ≈ 10, and trigger one reactivation (the access at the
	// end of the interval landed on the cold slot). The second
	// interval (censored trailing tail) demotes immediately because
	// its idle is counted from 0 and the same predict kicks in at 10.
	predict := func(_ model.InterAccessInterval, idle float64) float64 {
		if idle < 10 {
			return 0.9
		}
		return 0.05
	}
	p, _ := NewStatisticalPolicy("s", predict, 1, CostParams{RAMUnitCost: 1, MissPenalty: 10}) // p* = 0.1

	ivs := []model.InterAccessInterval{
		{SlotID: "x", IntervalStart: 0, Duration: 20, IsObserved: true},
		{SlotID: "x", IntervalStart: 20, Duration: 100, IsObserved: false},
	}
	res, err := Run(p, ivs, CostParams{RAMUnitCost: 1, MissPenalty: 10})
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	// HotBlocks on first interval: grid 0,1,2,…,20 at step 1 — first
	// u≥10 lands at exactly 10, so Hot=10, Cold=10 → one miss.
	// HotBlocks on trailing censored interval (Duration=100): grid is
	// 0,5,10,…,100 at step 5, first u≥10 lands at 10, so Hot=10,
	// Cold=90, no miss (censored). Total Hot = 20 across both.
	if res.Reactivations != 1 {
		t.Errorf("Reactivations=%d want 1", res.Reactivations)
	}
	if res.SlotBlocksHot < 18 || res.SlotBlocksHot > 22 {
		t.Errorf("SlotBlocksHot=%d want ≈ 20 (10 per interval)", res.SlotBlocksHot)
	}
	// Trailing interval has Duration=100; its demote happens at
	// idle≈10, so Hot≈10 + Cold≈90 → slot ends in cold tier.
	if res.FinalPrunedSlots != 1 {
		t.Errorf("FinalPrunedSlots=%d want 1", res.FinalPrunedSlots)
	}
}
