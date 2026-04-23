package analysis

import (
	"context"
	"math"
	"testing"

	"github.com/egpivo/zk-state-prune/internal/domain"
	"github.com/egpivo/zk-state-prune/internal/extractor"
)

func TestRunSpatial_DetectsMockClustering(t *testing.T) {
	ctx := context.Background()
	db := openDB(t)

	// Mock extractor generates intra-contract co-accesses with
	// ρ = 0.7 by default. RunSpatial should report a meaningfully
	// positive MeanJaccard. We don't pin an exact value because the
	// co-access machinery interacts with the per-slot Poisson rate
	// sampler in non-trivial ways, but > 0.05 rules out noise.
	cfg := extractor.DefaultMockConfig()
	cfg.NumContracts = 30
	cfg.SlotsPerContractXmin = 5
	cfg.SlotsPerContractMax = 30
	cfg.TotalBlocks = 20_000
	cfg.Window = domain.ObservationWindow{Start: 4_000, End: 20_000}
	cfg.AccessRateXmin = 5e-4
	cfg.MaxEventsPerSlot = 300
	cfg.PeriodBlocks = 2_000
	cfg.IntraContractCorrelation = 0.7

	if err := extractor.NewMockExtractor(cfg).Extract(ctx, db); err != nil {
		t.Fatalf("Extract: %v", err)
	}

	rep, err := RunSpatial(ctx, db, cfg.Window)
	if err != nil {
		t.Fatalf("RunSpatial: %v", err)
	}
	if rep.NumContracts == 0 {
		t.Fatalf("no contracts measured")
	}
	if rep.MeanJaccard <= 0.05 {
		t.Errorf("MeanJaccard=%v, want > 0.05 on ρ=0.7 mock", rep.MeanJaccard)
	}
	if math.IsNaN(rep.MeanJaccard) || rep.MeanJaccard > 1 {
		t.Errorf("MeanJaccard=%v out of [0,1]", rep.MeanJaccard)
	}
	// Per-contract map must be consistent with NumContracts.
	if len(rep.PerContract) != rep.NumContracts {
		t.Errorf("PerContract size %d != NumContracts %d", len(rep.PerContract), rep.NumContracts)
	}
}

func TestRunSpatial_ZeroClusteringWhenRhoZero(t *testing.T) {
	ctx := context.Background()
	db := openDB(t)

	cfg := extractor.DefaultMockConfig()
	cfg.NumContracts = 30
	cfg.SlotsPerContractXmin = 5
	cfg.SlotsPerContractMax = 30
	cfg.TotalBlocks = 20_000
	cfg.Window = domain.ObservationWindow{Start: 4_000, End: 20_000}
	cfg.AccessRateXmin = 5e-4
	cfg.MaxEventsPerSlot = 300
	cfg.PeriodBlocks = 2_000
	cfg.IntraContractCorrelation = 0.0 // disable co-access injection

	if err := extractor.NewMockExtractor(cfg).Extract(ctx, db); err != nil {
		t.Fatalf("Extract: %v", err)
	}
	rep, err := RunSpatial(ctx, db, cfg.Window)
	if err != nil {
		t.Fatalf("RunSpatial: %v", err)
	}
	// Without co-access injection, shared blocks across slots arise
	// only by random chance. Jaccard should be near zero but we allow
	// some slack for the Poisson noise floor.
	if rep.MeanJaccard > 0.05 {
		t.Errorf("MeanJaccard=%v, want ~0 without co-access", rep.MeanJaccard)
	}
}

func TestJaccardUint64(t *testing.T) {
	cases := []struct {
		a, b []uint64
		want float64
	}{
		{[]uint64{1, 2, 3}, []uint64{2, 3, 4}, 2.0 / 4.0}, // |∩|=2, |∪|=4
		{[]uint64{1, 2, 3}, []uint64{1, 2, 3}, 1.0},
		{[]uint64{1, 2, 3}, []uint64{4, 5, 6}, 0.0},
		{nil, nil, 0.0},
		{[]uint64{1}, nil, 0.0},
	}
	for _, c := range cases {
		got := jaccardUint64(c.a, c.b)
		if math.Abs(got-c.want) > 1e-12 {
			t.Errorf("jaccard(%v, %v) = %v, want %v", c.a, c.b, got, c.want)
		}
	}
}
