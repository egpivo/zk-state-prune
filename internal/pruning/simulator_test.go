package pruning

import (
	"context"
	"math"
	"path/filepath"
	"testing"

	"github.com/egpivo/zk-state-prune/internal/analysis"
	"github.com/egpivo/zk-state-prune/internal/extractor"
	"github.com/egpivo/zk-state-prune/internal/model"
	"github.com/egpivo/zk-state-prune/internal/storage"
)

// mkInterval is a small helper to build a minimal InterAccessInterval for
// the hand-crafted unit tests. Covariates default to zero.
func mkInterval(slot string, dur uint64, observed bool) model.InterAccessInterval {
	return model.InterAccessInterval{
		SlotID:     slot,
		Duration:   dur,
		IsObserved: observed,
	}
}

func TestRun_NoPruneIsFreeAndLoseless(t *testing.T) {
	ivs := []model.InterAccessInterval{
		mkInterval("s1", 10, true),
		mkInterval("s1", 50, true),
		mkInterval("s1", 200, false),
	}
	res, err := Run(NoPrune{}, ivs, DefaultCostParams())
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if res.SlotBlocksPruned != 0 || res.StorageSavedFrac != 0 {
		t.Errorf("no-prune should save nothing: %+v", res)
	}
	if res.Reactivations != 0 || res.FalsePruneRate != 0 {
		t.Errorf("no-prune should never reactivate: %+v", res)
	}
	if res.FinalPrunedSlots != 0 {
		t.Errorf("no-prune should leave zero pruned slots, got %d", res.FinalPrunedSlots)
	}
}

func TestRun_FixedIdleThresholdHit(t *testing.T) {
	// Policy prunes at idle >= 20.
	policy := FixedIdle{Label: "test", IdleBlocks: 20}
	ivs := []model.InterAccessInterval{
		mkInterval("s1", 10, true),  // below threshold → no savings
		mkInterval("s1", 30, true),  // crosses threshold → reactivation + 10 saved
		mkInterval("s1", 50, false), // trailing censored, 30 saved, final=pruned
	}
	res, err := Run(policy, ivs, DefaultCostParams())
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if res.Reactivations != 1 {
		t.Errorf("Reactivations=%d want 1", res.Reactivations)
	}
	if res.SlotBlocksPruned != 10+30 {
		t.Errorf("SlotBlocksPruned=%d want 40", res.SlotBlocksPruned)
	}
	if res.TotalExposure != 90 {
		t.Errorf("TotalExposure=%d want 90", res.TotalExposure)
	}
	wantSaved := float64(40) / float64(90)
	if res.StorageSavedFrac < wantSaved-1e-9 || res.StorageSavedFrac > wantSaved+1e-9 {
		t.Errorf("StorageSavedFrac=%v want %v", res.StorageSavedFrac, wantSaved)
	}
	if res.ObservedIntervals != 2 {
		t.Errorf("ObservedIntervals=%d want 2", res.ObservedIntervals)
	}
	wantRate := 0.5
	if res.FalsePruneRate < wantRate-1e-9 || res.FalsePruneRate > wantRate+1e-9 {
		t.Errorf("FalsePruneRate=%v want %v", res.FalsePruneRate, wantRate)
	}
	if res.FinalPrunedSlots != 1 {
		t.Errorf("FinalPrunedSlots=%d want 1", res.FinalPrunedSlots)
	}
}

func TestRun_ReactivationResetsFinalState(t *testing.T) {
	// Slot gets pruned, reactivated, then left short-idle censored → not
	// pruned at window end.
	policy := FixedIdle{Label: "t", IdleBlocks: 10}
	ivs := []model.InterAccessInterval{
		mkInterval("s1", 50, true), // pruned then reactivated
		mkInterval("s1", 5, false), // trailing short censored, not pruned
	}
	res, _ := Run(policy, ivs, DefaultCostParams())
	if res.Reactivations != 1 {
		t.Errorf("Reactivations=%d want 1", res.Reactivations)
	}
	if res.FinalPrunedSlots != 0 {
		t.Errorf("FinalPrunedSlots=%d want 0 (reactivation cleared it)", res.FinalPrunedSlots)
	}
}

func TestRun_ThresholdLargerThanAllIntervals(t *testing.T) {
	policy := FixedIdle{Label: "huge", IdleBlocks: 1_000_000}
	ivs := []model.InterAccessInterval{
		mkInterval("s1", 100, true),
		mkInterval("s1", 200, false),
	}
	res, _ := Run(policy, ivs, DefaultCostParams())
	if res.SlotBlocksPruned != 0 || res.Reactivations != 0 || res.FinalPrunedSlots != 0 {
		t.Errorf("oversized threshold should be a no-op: %+v", res)
	}
}

func TestRun_CostAccounting(t *testing.T) {
	// Hand-crafted intervals with a threshold of 20 → one observed
	// reactivation (Duration=30, 10 cold slot-blocks) + one censored
	// cold tail (Duration=50, 30 cold slot-blocks). Exposure 10+30+50=90.
	// Hot = 90 − (10+30) = 50 slot-blocks.
	policy := FixedIdle{Label: "t", IdleBlocks: 20}
	ivs := []model.InterAccessInterval{
		mkInterval("s1", 10, true),  // hot throughout, no reactivation
		mkInterval("s1", 30, true),  // hot 20 + cold 10 → reactivation
		mkInterval("s1", 50, false), // hot 20 + cold 30 → final=cold
	}
	costs := CostParams{RAMUnitCost: 2, MissPenalty: 100}
	res, err := Run(policy, ivs, costs)
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if res.TotalExposure != 90 {
		t.Errorf("TotalExposure=%d want 90", res.TotalExposure)
	}
	if res.SlotBlocksPruned != 40 {
		t.Errorf("SlotBlocksPruned=%d want 40", res.SlotBlocksPruned)
	}
	if res.SlotBlocksHot != 50 {
		t.Errorf("SlotBlocksHot=%d want 50", res.SlotBlocksHot)
	}
	wantRAM := 50.0 / 90.0
	if math.Abs(res.RAMRatio-wantRAM) > 1e-9 {
		t.Errorf("RAMRatio=%v want %v", res.RAMRatio, wantRAM)
	}
	// HotHitCoverage = 1 - 1/2 = 0.5 (two observed intervals, one miss).
	if math.Abs(res.HotHitCoverage-0.5) > 1e-9 {
		t.Errorf("HotHitCoverage=%v want 0.5", res.HotHitCoverage)
	}
	if math.Abs(res.RAMCost-100) > 1e-9 { // 50 * 2
		t.Errorf("RAMCost=%v want 100", res.RAMCost)
	}
	if math.Abs(res.MissPenaltyAgg-100) > 1e-9 { // 1 * 100
		t.Errorf("MissPenaltyAgg=%v want 100", res.MissPenaltyAgg)
	}
	if math.Abs(res.TotalCost-200) > 1e-9 {
		t.Errorf("TotalCost=%v want 200", res.TotalCost)
	}
	// Legacy aliases must round-trip.
	if math.Abs(res.StorageSavedFrac-(1-wantRAM)) > 1e-9 {
		t.Errorf("StorageSavedFrac=%v want %v", res.StorageSavedFrac, 1-wantRAM)
	}
	if math.Abs(res.FalsePruneRate-0.5) > 1e-9 {
		t.Errorf("FalsePruneRate=%v want 0.5", res.FalsePruneRate)
	}
}

func TestRun_NoPruneAllHotBaseline(t *testing.T) {
	// No-prune keeps every slot-block hot → RAMRatio=1, misses=0,
	// TotalCost = RAMUnitCost * exposure. Sanity check the upper bound.
	ivs := []model.InterAccessInterval{
		mkInterval("s1", 10, true),
		mkInterval("s1", 50, false),
	}
	costs := CostParams{RAMUnitCost: 1, MissPenalty: 100}
	res, _ := Run(NoPrune{}, ivs, costs)
	if res.RAMRatio != 1 {
		t.Errorf("RAMRatio=%v want 1", res.RAMRatio)
	}
	if res.HotHitCoverage != 1 {
		t.Errorf("HotHitCoverage=%v want 1", res.HotHitCoverage)
	}
	if res.MissPenaltyAgg != 0 {
		t.Errorf("MissPenaltyAgg=%v want 0", res.MissPenaltyAgg)
	}
	if res.TotalCost != 60 {
		t.Errorf("TotalCost=%v want 60", res.TotalCost)
	}
}

func TestRun_NilPolicyError(t *testing.T) {
	if _, err := Run(nil, nil, DefaultCostParams()); err == nil {
		t.Fatal("expected error on nil policy")
	}
}

func TestPolicyByName(t *testing.T) {
	cases := []struct {
		name    string
		wantErr bool
		// wantLabel, if non-empty, asserts the resolved policy's Name().
		// Lets us verify that the underscore / dash / whitespace / case
		// normalization all route to the same concrete policy.
		wantLabel string
	}{
		{name: "no-prune", wantLabel: "no-prune"},
		{name: "fixed-30d", wantLabel: "fixed-30d"},
		{name: "fixed-90d", wantLabel: "fixed-90d"},
		// YAML idiom with underscores must resolve identically to the
		// canonical dashed form — this is the config/CLI bridge.
		{name: "fixed_30d", wantLabel: "fixed-30d"},
		{name: "fixed_90d", wantLabel: "fixed-90d"},
		{name: "  Fixed-30d  ", wantLabel: "fixed-30d"},
		// "statistical" without deps still errors because the registry
		// needs a pre-built StatisticalPolicy; the dedicated branch
		// below verifies the positive path.
		{name: "statistical", wantErr: true},
		{name: "bogus", wantErr: true},
	}
	for _, c := range cases {
		p, err := PolicyByName(c.name, PolicyDeps{})
		if (err != nil) != c.wantErr {
			t.Errorf("PolicyByName(%q) err=%v want err=%v", c.name, err, c.wantErr)
			continue
		}
		if err == nil && c.wantLabel != "" && p.Name() != c.wantLabel {
			t.Errorf("PolicyByName(%q).Name() = %q, want %q", c.name, p.Name(), c.wantLabel)
		}
	}

	// Positive path: with a pre-built StatisticalPolicy in deps,
	// PolicyByName("statistical") resolves to it.
	stat, err := NewStatisticalPolicy("statistical",
		func(_ model.InterAccessInterval, _ float64) float64 { return 0.5 },
		10, CostParams{RAMUnitCost: 1, MissPenalty: 10})
	if err != nil {
		t.Fatalf("NewStatisticalPolicy: %v", err)
	}
	got, err := PolicyByName("statistical", PolicyDeps{Statistical: stat})
	if err != nil {
		t.Fatalf("PolicyByName(statistical, deps): %v", err)
	}
	if got.Name() != "statistical" {
		t.Errorf("resolved name=%q want statistical", got.Name())
	}
}

func TestRunAll_EndToEndOnMock_MonotonicOrdering(t *testing.T) {
	// Full pipeline: mock → intervals → simulate. We assert that for the
	// baseline trio {no-prune, fixed-90d, fixed-30d} both StorageSavedFrac
	// and FalsePruneRate are non-decreasing as the policy gets stricter.
	// This is the weakest correctness bar the comparison report can offer
	// against the pure no-prune ceiling, but it catches several classes of
	// regression at once (double-counting, sign errors, threshold logic).
	ctx := context.Background()
	db, err := storage.Open(ctx, filepath.Join(t.TempDir(), "x.db"))
	if err != nil {
		t.Fatalf("open: %v", err)
	}
	t.Cleanup(func() { _ = db.Close() })

	cfg := extractor.DefaultMockConfig()
	cfg.NumContracts = 40
	cfg.SlotsPerContractXmin = 5
	cfg.SlotsPerContractMax = 30
	cfg.TotalBlocks = 50_000
	cfg.Window = model.ObservationWindow{Start: 10_000, End: 50_000}
	cfg.AccessRateXmin = 1e-4
	cfg.MaxEventsPerSlot = 300
	cfg.PeriodBlocks = 5_000
	if err := extractor.NewMockExtractor(cfg).Extract(ctx, db); err != nil {
		t.Fatalf("Extract: %v", err)
	}
	built, err := analysis.BuildIntervals(ctx, db, cfg.Window)
	if err != nil {
		t.Fatalf("BuildIntervals: %v", err)
	}
	if len(built.Intervals) == 0 {
		t.Fatal("no intervals built")
	}

	// Tune the fixed windows to the test's 50k-block horizon so they are
	// actually reachable inside the trace: the production presets (216k
	// and 648k) are much larger than this window and would trivially
	// evaluate to zero savings.
	lenient := FixedIdle{Label: "lenient", IdleBlocks: 20_000}
	strict := FixedIdle{Label: "strict", IdleBlocks: 5_000}

	results, err := RunAll([]Policy{NoPrune{}, lenient, strict}, built.Intervals, DefaultCostParams())
	if err != nil {
		t.Fatalf("RunAll: %v", err)
	}
	if len(results) != 3 {
		t.Fatalf("len(results)=%d want 3", len(results))
	}

	// no-prune baseline
	if r := results[0]; r.SlotBlocksPruned != 0 || r.Reactivations != 0 {
		t.Errorf("no-prune not inert: %+v", r)
	}
	// Savings must be non-decreasing as threshold shrinks.
	if results[1].StorageSavedFrac > results[2].StorageSavedFrac {
		t.Errorf("stricter policy saved less: lenient=%v strict=%v",
			results[1].StorageSavedFrac, results[2].StorageSavedFrac)
	}
	// False-prune rate must also be non-decreasing — stricter policy
	// catches more real accesses on pruned slots.
	if results[1].FalsePruneRate > results[2].FalsePruneRate {
		t.Errorf("stricter policy had lower false-prune rate: lenient=%v strict=%v",
			results[1].FalsePruneRate, results[2].FalsePruneRate)
	}
	// Sanity: strict policy saved *something* over the 50k window.
	if results[2].StorageSavedFrac <= 0 {
		t.Errorf("strict policy saved nothing: %+v", results[2])
	}
	// Sanity: every result agrees on the slot universe.
	for _, r := range results {
		if r.TotalSlots != results[0].TotalSlots {
			t.Errorf("TotalSlots drift: %d vs %d", r.TotalSlots, results[0].TotalSlots)
		}
	}
}
