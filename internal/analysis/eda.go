package analysis

import (
	"context"
	"fmt"
	"math"
	"sort"

	"gonum.org/v1/gonum/stat"

	"github.com/egpivo/zk-state-prune/internal/model"
	"github.com/egpivo/zk-state-prune/internal/storage"
)

// EDAReport is the structured output of Phase-1 exploratory data analysis.
// It intentionally mirrors the things the motivation story depends on:
//   - the access-frequency distribution should be heavy-tailed (power-law
//     like), which is the precondition for pruning being worthwhile
//   - the inter-access-time distribution should have a long right tail,
//     which is what the survival model is trying to predict
//   - per-contract-type breakdowns let us see whether the heavy tail is
//     uniform or comes from specific application archetypes
//   - the censoring diagnostic lets us confirm the window is well-chosen
//     before we trust any survival fit
type EDAReport struct {
	Window model.ObservationWindow

	Frequency        DistributionSummary
	InterAccessTime  DistributionSummary
	ByContractType   map[model.ContractCategory]ContractTypeSummary
	BySlotType       map[model.SlotType]SlotTypeSummary

	// Censoring diagnostics. Rate is a fraction in [0,1].
	TotalIntervals     int
	RightCensoredCount int // interval-level
	RightCensoredRate  float64
	LeftTruncatedCount int // slot-level (distinct slots with CreatedAt < Window.Start)
	LeftTruncatedRate  float64
	SlotsWithNoEvents  int
	SlotsSkipped       int

	// SlotCount is the number of slots that contributed at least one
	// interval (i.e. had any risk exposure inside the window).
	SlotCount int
}

// DistributionSummary is a compact first-moment / tail summary. We keep it
// small on purpose: this is Phase 1 EDA, not a histogram dump.
type DistributionSummary struct {
	Count      int
	Mean       float64
	StdDev     float64
	Min        float64
	P50        float64
	P90        float64
	P99        float64
	Max        float64
	// PowerLawAlphaMLE is the Hill-estimator exponent α for the upper tail,
	// fitted above the median. Power-law tails give α ~ 1-3; values much
	// larger than that indicate an exponential-like tail. Zero if Count<10.
	PowerLawAlphaMLE float64
}

// ContractTypeSummary aggregates per-category metrics that inform
// stratification choices for the survival model.
type ContractTypeSummary struct {
	Slots              int
	Intervals          int
	AccessFrequency    DistributionSummary
	InterAccessTime    DistributionSummary
	RightCensoredRate  float64
}

// SlotTypeSummary is the same idea but cut by slot kind (balance, mapping…).
type SlotTypeSummary struct {
	Slots              int
	Intervals          int
	InterAccessTime    DistributionSummary
	RightCensoredRate  float64
}

// RunEDA builds intervals for the given window and produces an EDAReport.
func RunEDA(ctx context.Context, db *storage.DB, window model.ObservationWindow) (*EDAReport, error) {
	built, err := BuildIntervals(ctx, db, window)
	if err != nil {
		return nil, fmt.Errorf("build intervals: %w", err)
	}
	return summarize(built, window), nil
}

func summarize(built IntervalBuildResult, window model.ObservationWindow) *EDAReport {
	intervals := built.Intervals
	n := len(intervals)

	// Per-slot access count inside the window. We cannot simply count
	// observed intervals here: when BuildIntervals skips a degenerate
	// entry-equal-first interval the first access is consumed silently
	// and the observed-interval count is (total_accesses - 1) rather
	// than total. Instead we use the invariant that every interval's
	// AccessCount field holds the cumulative access count up to its
	// start, so `AccessCount + [observed?1:0]` equals the access count
	// immediately after this interval. Taking the per-slot maximum of
	// that quantity recovers the true in-window access total, which is
	// what the frequency distribution should summarize.
	slotFreq := make(map[string]int)
	// Observed-only durations, used for IAT summary. Censored durations
	// would bias the mean/percentiles downward, so they're excluded from
	// DistributionSummary; censoring is reported separately.
	iat := make([]float64, 0, n)

	byCat := make(map[model.ContractCategory]*ctAgg)
	bySlot := make(map[model.SlotType]*stAgg)

	for _, it := range intervals {
		total := int(it.AccessCount)
		if it.IsObserved {
			total++
		}
		if v, ok := slotFreq[it.SlotID]; !ok || v < total {
			slotFreq[it.SlotID] = total
		}
		if it.IsObserved {
			iat = append(iat, float64(it.Duration))
		}
		cat := getOrInit(byCat, it.ContractType)
		cat.intervals++
		if it.IsObserved {
			cat.iat = append(cat.iat, float64(it.Duration))
		} else {
			cat.censored++
		}
		st := getOrInitSlot(bySlot, it.SlotType)
		st.intervals++
		if it.IsObserved {
			st.iat = append(st.iat, float64(it.Duration))
		} else {
			st.censored++
		}
	}

	freqVals := make([]float64, 0, len(slotFreq))
	for _, v := range slotFreq {
		freqVals = append(freqVals, float64(v))
	}

	// Populate per-category slot counts by re-scanning intervals and
	// attributing each unique slot to its category. A map-per-slot is
	// fine at Phase-1 sizes.
	seen := make(map[string]bool)
	for _, it := range intervals {
		if seen[it.SlotID] {
			continue
		}
		seen[it.SlotID] = true
		byCat[it.ContractType].slots++
		bySlot[it.SlotType].slots++
		byCat[it.ContractType].freq = append(byCat[it.ContractType].freq, float64(slotFreq[it.SlotID]))
	}

	report := &EDAReport{
		Window:             window,
		Frequency:          summarizeDistribution(freqVals),
		InterAccessTime:    summarizeDistribution(iat),
		ByContractType:     make(map[model.ContractCategory]ContractTypeSummary, len(byCat)),
		BySlotType:         make(map[model.SlotType]SlotTypeSummary, len(bySlot)),
		TotalIntervals:     n,
		RightCensoredCount: built.RightCensored,
		LeftTruncatedCount: built.LeftTruncatedSlots,
		SlotsWithNoEvents:  built.SlotsWithNoEvents,
		SlotsSkipped:       built.SlotsSkipped,
		SlotCount:          len(slotFreq),
	}
	if n > 0 {
		report.RightCensoredRate = float64(built.RightCensored) / float64(n)
	}
	if report.SlotCount > 0 {
		report.LeftTruncatedRate = float64(built.LeftTruncatedSlots) / float64(report.SlotCount)
	}
	for cat, agg := range byCat {
		cr := 0.0
		if agg.intervals > 0 {
			cr = float64(agg.censored) / float64(agg.intervals)
		}
		report.ByContractType[cat] = ContractTypeSummary{
			Slots:             agg.slots,
			Intervals:         agg.intervals,
			AccessFrequency:   summarizeDistribution(agg.freq),
			InterAccessTime:   summarizeDistribution(agg.iat),
			RightCensoredRate: cr,
		}
	}
	for st, agg := range bySlot {
		cr := 0.0
		if agg.intervals > 0 {
			cr = float64(agg.censored) / float64(agg.intervals)
		}
		report.BySlotType[st] = SlotTypeSummary{
			Slots:             agg.slots,
			Intervals:         agg.intervals,
			InterAccessTime:   summarizeDistribution(agg.iat),
			RightCensoredRate: cr,
		}
	}
	return report
}

type ctAgg struct {
	slots     int
	intervals int
	censored  int
	iat       []float64
	freq      []float64
}

type stAgg struct {
	slots     int
	intervals int
	censored  int
	iat       []float64
}

func getOrInit(m map[model.ContractCategory]*ctAgg, k model.ContractCategory) *ctAgg {
	if a, ok := m[k]; ok {
		return a
	}
	a := &ctAgg{}
	m[k] = a
	return a
}

func getOrInitSlot(m map[model.SlotType]*stAgg, k model.SlotType) *stAgg {
	if a, ok := m[k]; ok {
		return a
	}
	a := &stAgg{}
	m[k] = a
	return a
}

// summarizeDistribution produces a DistributionSummary from a sample. The
// sample is sorted in place; callers should pass a slice they own.
func summarizeDistribution(xs []float64) DistributionSummary {
	s := DistributionSummary{Count: len(xs)}
	if len(xs) == 0 {
		return s
	}
	sort.Float64s(xs)
	s.Min = xs[0]
	s.Max = xs[len(xs)-1]
	// stat.Quantile requires sorted input and a CumulantKind.
	s.P50 = stat.Quantile(0.5, stat.Empirical, xs, nil)
	s.P90 = stat.Quantile(0.9, stat.Empirical, xs, nil)
	s.P99 = stat.Quantile(0.99, stat.Empirical, xs, nil)
	s.Mean, s.StdDev = stat.MeanStdDev(xs, nil)
	if len(xs) >= 10 {
		s.PowerLawAlphaMLE = hillEstimator(xs, 0.5)
	}
	return s
}

// hillEstimator returns the Hill MLE of the tail index α for the upper
// (1-p) fraction of a sorted sample. Zero / NaN-safe.
//
//	α̂ = 1 / ( (1/k) * Σ ln(x_i / x_min) )
//
// where x_min is the p-quantile and the sum runs over the k values above it.
func hillEstimator(sorted []float64, p float64) float64 {
	n := len(sorted)
	k := n - int(math.Floor(p*float64(n)))
	if k < 5 {
		return 0
	}
	xmin := sorted[n-k]
	if xmin <= 0 {
		return 0
	}
	sum := 0.0
	count := 0
	for i := n - k; i < n; i++ {
		if sorted[i] <= 0 {
			continue
		}
		sum += math.Log(sorted[i] / xmin)
		count++
	}
	if sum <= 0 || count == 0 {
		return 0
	}
	return float64(count) / sum
}
