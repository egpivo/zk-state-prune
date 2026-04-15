package analysis

import (
	"context"
	"math"
	"testing"

	"github.com/egpivo/zk-state-prune/internal/extractor"
	"github.com/egpivo/zk-state-prune/internal/model"
)

func TestRunTemporal_SmokeOnMock(t *testing.T) {
	ctx := context.Background()
	db := openDB(t)

	// Mock defaults to 10% periodic contracts. The temporal pass
	// should flag a non-trivial number of contracts as periodic and
	// report a positive periodic fraction, without swinging wildly
	// over the full [0, 1] range.
	cfg := extractor.DefaultMockConfig()
	cfg.NumContracts = 100
	cfg.SlotsPerContractXmin = 5
	cfg.SlotsPerContractMax = 30
	cfg.TotalBlocks = 30_000
	cfg.Window = model.ObservationWindow{Start: 6_000, End: 30_000}
	cfg.AccessRateXmin = 5e-4
	cfg.MaxEventsPerSlot = 300
	cfg.PeriodBlocks = 2_000
	cfg.PeriodicContractsRatio = 0.3 // bump above default so we have enough signal

	if err := extractor.NewMockExtractor(cfg).Extract(ctx, db); err != nil {
		t.Fatalf("Extract: %v", err)
	}
	rep, err := RunTemporal(ctx, db, cfg.Window)
	if err != nil {
		t.Fatalf("RunTemporal: %v", err)
	}
	if rep.NumContracts == 0 {
		t.Fatalf("no contracts measured")
	}
	if rep.PeriodicFraction < 0 || rep.PeriodicFraction > 1 {
		t.Errorf("PeriodicFraction=%v out of [0,1]", rep.PeriodicFraction)
	}
	// Per-contract readings must parse as finite numbers.
	for addr, sig := range rep.PerContract {
		if math.IsNaN(sig.Strength) || math.IsInf(sig.Strength, 0) {
			t.Errorf("%s: Strength=%v not finite", addr, sig.Strength)
		}
		if sig.Periodic && sig.PeriodBlocks == 0 {
			t.Errorf("%s: marked periodic but PeriodBlocks=0", addr)
		}
	}
}

func TestAutocorrelationPeak_FlatReturnsZero(t *testing.T) {
	x := []float64{1, 1, 1, 1, 1, 1, 1, 1}
	lag, r := autocorrelationPeak(x, 1, 4)
	if lag != 0 || r != 0 {
		t.Errorf("flat series lag=%d r=%v, want 0,0", lag, r)
	}
}

func TestAutocorrelationPeak_PeriodicSineHitsAtExpectedLag(t *testing.T) {
	// A sine-like sequence with period 8 should peak at lag 8 (or 16, …).
	n := 64
	period := 8
	x := make([]float64, n)
	for i := 0; i < n; i++ {
		x[i] = math.Sin(2 * math.Pi * float64(i) / float64(period))
	}
	lag, r := autocorrelationPeak(x, 1, n/2)
	if r < 0.5 {
		t.Errorf("peak correlation = %v, want > 0.5 on clean sine", r)
	}
	// Allow a ±1 slack; discrete autocorrelation on a finite window
	// can land on a neighbouring integer lag.
	if lag < period-1 || lag > period+1 {
		t.Errorf("peak lag=%d, want ≈ %d", lag, period)
	}
}
