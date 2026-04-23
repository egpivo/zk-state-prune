package analysis

import (
	"context"
	"fmt"
	"sort"

	"github.com/egpivo/zk-state-prune/internal/domain"
	"github.com/egpivo/zk-state-prune/internal/storage"
)

// TemporalReport describes per-contract access-pattern periodicity.
// The mock extractor injects seasonality into a configurable fraction
// of contracts via a sinusoidal envelope (PeriodicContractsRatio); this
// report fires the same pattern detection on a real-data pipeline would
// need. Phase 2 ships a basic autocorrelation-peak detector — good
// enough to flag the synthetic periodic contracts as a sanity check.
type TemporalReport struct {
	Window domain.ObservationWindow

	// Number of contracts that had enough in-window events
	// (>= temporalMinEvents) for a periodicity call.
	NumContracts int

	// PerContract is keyed by contract address. PeriodBlocks is the
	// estimated dominant period in blocks, or 0 if the autocorrelation
	// peak was below the significance threshold. Strength is the value
	// of the autocorrelation at the peak in [-1, 1].
	PerContract map[string]ContractTemporalSignal

	// Summary counts.
	PeriodicContracts int
	PeriodicFraction  float64
}

// ContractTemporalSignal is the per-contract temporal readout.
type ContractTemporalSignal struct {
	PeriodBlocks uint64
	Strength     float64
	Periodic     bool
}

// temporalNumBins is the histogram resolution used to convert raw
// event blocks into a uniformly-sampled time series before running
// autocorrelation. 100 bins gives reasonable resolution for Phase-2
// mock horizons without overwhelming the per-contract math.
const temporalNumBins = 100

// temporalMinEvents is the minimum in-window event count for a contract
// to be eligible for periodicity testing. Too few events and the
// autocorrelation peak is just noise.
const temporalMinEvents = 20

// temporalSignificance is the minimum autocorrelation-at-peak value for
// a contract to be labelled periodic. Tuned so the mock's seasonal
// contracts flag reliably while non-periodic ones stay below.
const temporalSignificance = 0.3

// RunTemporal computes per-contract periodicity via autocorrelation on
// a binned event-count time series. For each contract with enough
// in-window events, the routine:
//
//  1. Bins the event blocks into temporalNumBins uniform-width bins
//     spanning the observation window.
//  2. Computes the sample autocorrelation r(k) for lags
//     k = 1 … temporalNumBins/2.
//  3. Reports the lag with the largest correlation. If that value is
//     above temporalSignificance the contract is flagged periodic and
//     its dominant period (in blocks) is lag × bin_width.
func RunTemporal(ctx context.Context, db *storage.DB, window domain.ObservationWindow) (*TemporalReport, error) {
	if window.End <= window.Start {
		return nil, fmt.Errorf("RunTemporal: invalid window [%d, %d)", window.Start, window.End)
	}

	// Collect per-contract event blocks (in-window, deduped).
	byContract := make(map[string][]uint64)
	err := db.IterateSlotEvents(ctx, func(sm storage.SlotWithMeta, events []domain.AccessEvent) error {
		addr := sm.Slot.ContractAddr
		for _, e := range events {
			if !window.Contains(e.BlockNumber) {
				continue
			}
			byContract[addr] = append(byContract[addr], e.BlockNumber)
		}
		return nil
	})
	if err != nil {
		return nil, fmt.Errorf("iterate slots: %w", err)
	}

	winSpan := window.End - window.Start
	binWidth := float64(winSpan) / float64(temporalNumBins)
	if binWidth <= 0 {
		return nil, fmt.Errorf("RunTemporal: degenerate bin width")
	}

	out := &TemporalReport{
		Window:      window,
		PerContract: make(map[string]ContractTemporalSignal, len(byContract)),
	}

	for addr, blocks := range byContract {
		if len(blocks) < temporalMinEvents {
			continue
		}
		sort.Slice(blocks, func(i, j int) bool { return blocks[i] < blocks[j] })

		series := make([]float64, temporalNumBins)
		for _, b := range blocks {
			idx := int(float64(b-window.Start) / binWidth)
			if idx < 0 {
				idx = 0
			}
			if idx >= temporalNumBins {
				idx = temporalNumBins - 1
			}
			series[idx]++
		}

		peakLag, peakR := autocorrelationPeak(series, 1, temporalNumBins/2)
		periodic := peakR >= temporalSignificance
		periodBlocks := uint64(0)
		if periodic {
			periodBlocks = uint64(float64(peakLag) * binWidth)
		}
		out.PerContract[addr] = ContractTemporalSignal{
			PeriodBlocks: periodBlocks,
			Strength:     peakR,
			Periodic:     periodic,
		}
		out.NumContracts++
		if periodic {
			out.PeriodicContracts++
		}
	}
	if out.NumContracts > 0 {
		out.PeriodicFraction = float64(out.PeriodicContracts) / float64(out.NumContracts)
	}
	return out, nil
}

// autocorrelationPeak returns the lag k ∈ [minLag, maxLag] maximizing
// the sample autocorrelation of the series x, along with the
// correlation value at that lag. Returns (0, 0) if the series has zero
// variance or if all lags produce non-finite values.
//
// Sample autocorrelation at lag k:
//
//	r(k) = Σ (x[t] − x̄)(x[t+k] − x̄) / Σ (x[t] − x̄)²
//
// computed with the full-length denominator convention so r(0) = 1.
func autocorrelationPeak(x []float64, minLag, maxLag int) (int, float64) {
	n := len(x)
	if n < 4 || minLag <= 0 || maxLag < minLag {
		return 0, 0
	}
	if maxLag > n-1 {
		maxLag = n - 1
	}
	var mean float64
	for _, v := range x {
		mean += v
	}
	mean /= float64(n)
	var variance float64
	for _, v := range x {
		d := v - mean
		variance += d * d
	}
	if variance == 0 {
		return 0, 0
	}

	bestLag := 0
	bestR := 0.0
	for k := minLag; k <= maxLag; k++ {
		var num float64
		for t := 0; t+k < n; t++ {
			num += (x[t] - mean) * (x[t+k] - mean)
		}
		r := num / variance
		if r > bestR {
			bestR = r
			bestLag = k
		}
	}
	return bestLag, bestR
}
