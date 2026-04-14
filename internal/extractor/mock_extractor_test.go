package extractor

import (
	"context"
	"path/filepath"
	"testing"

	"github.com/egpivo/zk-state-prune/internal/model"
	"github.com/egpivo/zk-state-prune/internal/storage"
)

func smallConfig() MockConfig {
	c := DefaultMockConfig()
	c.NumContracts = 20
	c.SlotsPerContractXmin = 5
	c.SlotsPerContractMax = 50
	c.TotalBlocks = 10_000
	c.Window = model.ObservationWindow{Start: 2_000, End: 10_000}
	c.AccessRateXmin = 1e-4
	c.MaxEventsPerSlot = 200
	c.PeriodBlocks = 1_000
	return c
}

func openDB(t *testing.T) *storage.DB {
	t.Helper()
	db, err := storage.Open(context.Background(), filepath.Join(t.TempDir(), "x.db"))
	if err != nil {
		t.Fatalf("open: %v", err)
	}
	t.Cleanup(func() { _ = db.Close() })
	return db
}

func TestMockExtractor_PopulatesDB(t *testing.T) {
	ctx := context.Background()
	db := openDB(t)

	if err := NewMockExtractor(smallConfig()).Extract(ctx, db); err != nil {
		t.Fatalf("Extract: %v", err)
	}
	slots, err := db.CountSlots(ctx)
	if err != nil || slots == 0 {
		t.Fatalf("CountSlots = %d, err=%v; want > 0", slots, err)
	}
	events, err := db.CountAccessEvents(ctx)
	if err != nil || events == 0 {
		t.Fatalf("CountAccessEvents = %d, err=%v; want > 0", events, err)
	}

	// Sanity: every event must point at an existing slot.
	var orphan int64
	if err := db.SQL().QueryRowContext(ctx,
		`SELECT COUNT(*) FROM access_events e
		   LEFT JOIN state_slots s ON s.slot_id = e.slot_id
		  WHERE s.slot_id IS NULL`).Scan(&orphan); err != nil {
		t.Fatalf("orphan check: %v", err)
	}
	if orphan != 0 {
		t.Fatalf("found %d orphan access events", orphan)
	}
}

func TestMockExtractor_IdempotentOnSameDB(t *testing.T) {
	ctx := context.Background()
	db := openDB(t)
	ex := NewMockExtractor(smallConfig())

	if err := ex.Extract(ctx, db); err != nil {
		t.Fatalf("Extract #1: %v", err)
	}
	s1, _ := db.CountSlots(ctx)
	e1, _ := db.CountAccessEvents(ctx)

	if err := ex.Extract(ctx, db); err != nil {
		t.Fatalf("Extract #2: %v", err)
	}
	s2, _ := db.CountSlots(ctx)
	e2, _ := db.CountAccessEvents(ctx)

	if s1 != s2 || e1 != e2 {
		t.Fatalf("not idempotent: slots %d->%d, events %d->%d", s1, s2, e1, e2)
	}
}

func TestMockExtractor_DeterministicForSeed(t *testing.T) {
	ctx := context.Background()
	cfg := smallConfig()

	run := func() (int64, int64) {
		db := openDB(t)
		if err := NewMockExtractor(cfg).Extract(ctx, db); err != nil {
			t.Fatalf("Extract: %v", err)
		}
		s, _ := db.CountSlots(ctx)
		e, _ := db.CountAccessEvents(ctx)
		return s, e
	}
	s1, e1 := run()
	s2, e2 := run()
	if s1 != s2 || e1 != e2 {
		t.Fatalf("non-deterministic: (%d,%d) vs (%d,%d)", s1, e1, s2, e2)
	}
}

func TestMockExtractor_SeedChangesOutput(t *testing.T) {
	ctx := context.Background()
	cfg := smallConfig()

	dbA := openDB(t)
	if err := NewMockExtractor(cfg).Extract(ctx, dbA); err != nil {
		t.Fatalf("Extract A: %v", err)
	}
	a, _ := dbA.CountAccessEvents(ctx)

	cfg.Seed = cfg.Seed + 1
	dbB := openDB(t)
	if err := NewMockExtractor(cfg).Extract(ctx, dbB); err != nil {
		t.Fatalf("Extract B: %v", err)
	}
	b, _ := dbB.CountAccessEvents(ctx)
	if a == b {
		t.Logf("warning: seeds produced equal event counts (%d); not impossible but unexpected", a)
	}
}

func TestMockExtractor_ContractCategoryDistribution(t *testing.T) {
	ctx := context.Background()
	cfg := smallConfig()
	cfg.NumContracts = 500
	db := openDB(t)
	if err := NewMockExtractor(cfg).Extract(ctx, db); err != nil {
		t.Fatalf("Extract: %v", err)
	}

	rows, err := db.SQL().QueryContext(ctx,
		`SELECT contract_type, COUNT(*) FROM contracts GROUP BY contract_type`)
	if err != nil {
		t.Fatalf("query: %v", err)
	}
	defer rows.Close()
	counts := map[string]int{}
	for rows.Next() {
		var name string
		var n int
		if err := rows.Scan(&name, &n); err != nil {
			t.Fatal(err)
		}
		counts[name] = n
	}
	// ERC20 has the largest weight (0.40), so it should dominate.
	if counts["erc20"] == 0 {
		t.Fatalf("no erc20 contracts produced: %v", counts)
	}
	maxCat, maxN := "", 0
	for k, v := range counts {
		if v > maxN {
			maxCat, maxN = k, v
		}
	}
	if maxCat != "erc20" {
		t.Logf("warning: dominant category was %q (%d), expected erc20; sample size may be small", maxCat, maxN)
	}
}

func TestSamplePareto_TailHeavy(t *testing.T) {
	// Pareto(xmin=1, alpha=2) has theoretical mean 2; sanity-check the
	// empirical mean stays in a reasonable interval.
	r := newTestRand()
	const n = 20000
	sum := 0.0
	for i := 0; i < n; i++ {
		sum += samplePareto(r, 1, 2)
	}
	mean := sum / n
	if mean < 1.5 || mean > 3.0 {
		t.Fatalf("Pareto(1,2) empirical mean = %.3f, want in [1.5, 3.0]", mean)
	}
}

func TestSamplePoisson_MeanMatchesLambda(t *testing.T) {
	r := newTestRand()
	for _, lambda := range []float64{0.5, 5, 50} {
		const n = 5000
		sum := 0
		for i := 0; i < n; i++ {
			sum += samplePoisson(r, lambda)
		}
		mean := float64(sum) / n
		if mean < 0.7*lambda || mean > 1.3*lambda {
			t.Fatalf("Poisson(%v) mean = %.3f, want within 30%%", lambda, mean)
		}
	}
}

func TestSlotTypeMix_ERC20DominatedByBalance(t *testing.T) {
	r := newTestRand()
	counts := map[model.SlotType]int{}
	for i := 0; i < 5000; i++ {
		counts[sampleSlotType(r, model.ContractERC20)]++
	}
	if counts[model.SlotTypeBalance] < counts[model.SlotTypeMapping] {
		t.Fatalf("expected balance to dominate ERC20 mix, got %v", counts)
	}
}
