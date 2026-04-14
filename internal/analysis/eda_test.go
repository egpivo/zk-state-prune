package analysis

import (
	"context"
	"testing"

	"github.com/egpivo/zk-state-prune/internal/extractor"
	"github.com/egpivo/zk-state-prune/internal/model"
)

func smallMock() extractor.MockConfig {
	c := extractor.DefaultMockConfig()
	c.NumContracts = 30
	c.SlotsPerContractXmin = 5
	c.SlotsPerContractMax = 50
	c.TotalBlocks = 10_000
	c.Window = model.ObservationWindow{Start: 2_000, End: 10_000}
	c.AccessRateXmin = 1e-4
	c.MaxEventsPerSlot = 200
	c.PeriodBlocks = 1_000
	return c
}

func TestRunEDA_SmokeOnMock(t *testing.T) {
	ctx := context.Background()
	db := openDB(t)
	if err := extractor.NewMockExtractor(smallMock()).Extract(ctx, db); err != nil {
		t.Fatalf("Extract: %v", err)
	}
	window := smallMock().Window

	rep, err := RunEDA(ctx, db, window)
	if err != nil {
		t.Fatalf("RunEDA: %v", err)
	}
	if rep.TotalIntervals == 0 {
		t.Fatalf("no intervals produced")
	}
	if rep.SlotCount == 0 {
		t.Fatalf("no slots contributed")
	}
	// Every non-empty trace ends in a right-censored interval, so the
	// floor on the censoring rate is 1/avg_events_per_slot. With our
	// power-law rate most slots produce just a couple of events, so a
	// realistic rate sits anywhere between ~0.1 and ~1.0. We only
	// assert it's non-zero and sane.
	if rep.RightCensoredRate <= 0 || rep.RightCensoredRate > 1 {
		t.Errorf("RightCensoredRate = %v out of range", rep.RightCensoredRate)
	}
	if rep.Frequency.Count != rep.SlotCount {
		t.Errorf("frequency count %d != slot count %d", rep.Frequency.Count, rep.SlotCount)
	}
	if len(rep.ByContractType) == 0 {
		t.Errorf("ByContractType empty")
	}
}

func TestRunEDA_LeftTruncationDetected(t *testing.T) {
	ctx := context.Background()
	db := openDB(t)
	cfg := smallMock()
	cfg.PreWindowSlotFraction = 1.0 // force every contract pre-window
	if err := extractor.NewMockExtractor(cfg).Extract(ctx, db); err != nil {
		t.Fatalf("Extract: %v", err)
	}
	rep, err := RunEDA(ctx, db, cfg.Window)
	if err != nil {
		t.Fatalf("RunEDA: %v", err)
	}
	if rep.LeftTruncatedCount == 0 {
		t.Fatalf("expected left-truncated slots, got zero")
	}
	// LeftTruncatedRate is per slot. Every slot pre-exists the window →
	// rate should be 1.0.
	if rep.LeftTruncatedRate < 0.99 {
		t.Errorf("LeftTruncatedRate = %v, want ~1.0", rep.LeftTruncatedRate)
	}
}

func TestHillEstimator_OnParetoSample(t *testing.T) {
	// Synthetic Pareto(xmin=1, α=2). Inverse CDF: x = u^(-1/2).
	n := 5000
	xs := make([]float64, n)
	for i := 0; i < n; i++ {
		u := float64(i+1) / float64(n+1) // deterministic quasi-uniform
		xs[i] = 1.0 / sqrt(1-u)
	}
	sortFloats(xs)
	alpha := hillEstimator(xs, 0.5)
	if alpha < 1.5 || alpha > 2.5 {
		t.Errorf("hill α = %v, want ~2", alpha)
	}
}

func sortFloats(xs []float64) {
	// tiny insertion sort — avoids importing sort just for a test helper
	for i := 1; i < len(xs); i++ {
		for j := i; j > 0 && xs[j-1] > xs[j]; j-- {
			xs[j-1], xs[j] = xs[j], xs[j-1]
		}
	}
}

func sqrt(x float64) float64 {
	// Newton's method — keeps the test file dependency-free.
	if x <= 0 {
		return 0
	}
	z := x
	for i := 0; i < 20; i++ {
		z = 0.5 * (z + x/z)
	}
	return z
}
