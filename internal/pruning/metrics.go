package pruning

// CostParams bundles the per-slot cost knobs the simulator uses to score
// tiering policies. Values come from configs/default.yaml (Phase 2 wires
// the YAML loader); callers that construct a sim programmatically should
// prefer DefaultCostParams so defaults stay in one place.
//
// Units are intentionally abstract. RAMUnitCost is the cost of keeping one
// slot hot for one block; MissPenalty is the cost of a single cold-tier
// fetch. They share an arbitrary cost axis — only their ratio matters for
// ranking policies, so the defaults are normalized to RAMUnitCost=1.
type CostParams struct {
	// RAMUnitCost is the cost of keeping one slot in the hot tier for one
	// block. Multiplied by the hot-tier slot-block integral to get the
	// RAM part of TotalCost.
	RAMUnitCost float64

	// MissPenalty is the cost of serving one access from the cold tier
	// (reactivation proof + fetch latency charge). Multiplied by the
	// reactivation count to get the miss part of TotalCost.
	MissPenalty float64
}

// DefaultCostParams returns a (RAMUnitCost, MissPenalty) pair tuned so
// the statistical policy's threshold p* = (RAMUnitCost · τ) / MissPenalty
// lands inside (0, 1) for the horizons that show up on Phase-1 mock
// data (τ ∼ 10⁴ blocks). Concretely RAMUnitCost = 1.0 normalized,
// MissPenalty = 1e5 → p* ≈ 0.1 at τ = 10 000, which is the right
// neighbourhood for an "is the slot likely to be touched soon?" cutoff.
//
// The ratio is what drives policy ranking; absolute units are
// arbitrary. Real-data callers should override via the CLI
// --ram-unit-cost / --miss-penalty flags or by loading from
// configs/default.yaml once Phase 2 wires the YAML loader.
func DefaultCostParams() CostParams {
	return CostParams{RAMUnitCost: 1.0, MissPenalty: 1e5}
}

// SimResult is the output of a single tiering simulation run. Every rate is
// in [0,1]; counts are raw integers so callers can compute additional
// derived metrics without losing precision.
//
// The four Phase-2 headline metrics map onto the cost-aware surrogate
//
//	d_i* = argmin_d  c(d) + ℓ(d) · p_i(τ)
//
// the way the statistical policy uses them:
//
//   - RAMRatio: share of slot-block exposure that stayed in the hot tier,
//     the direct analogue of the sequencer RAM / witness footprint we are
//     trying to control. Complement of the legacy "storage saved" fraction.
//
//   - HotHitCoverage: fraction of observed accesses served straight out of
//     the hot tier without paying a fetch penalty. Complement of the legacy
//     "false prune rate".
//
//   - MissPenalty: total miss penalty under the configured CostParams,
//     computed as Reactivations × CostParams.MissPenalty.
//
//   - TotalCost: the aggregate surrogate score for the whole trace,
//     RAMCost + MissPenalty, in the same arbitrary cost units as
//     CostParams. This is the single number to compare policies on.
//
// Legacy fields (StorageSavedFrac, FalsePruneRate) are kept populated so
// Phase-1 call sites don't break; they are equivalent to
// 1 − RAMRatio and 1 − HotHitCoverage respectively. Phase 3 removes them.
type SimResult struct {
	Policy string

	TotalSlots        int
	ObservedIntervals int
	CensoredIntervals int

	// TotalExposure is the sum of all interval durations, i.e. the
	// slot-block risk-set integral across the window. It is the
	// denominator for RAMRatio.
	TotalExposure uint64
	// SlotBlocksPruned is the sum of (interval.Duration − threshold) over
	// intervals whose duration exceeds the policy threshold. It is the
	// cold-tier slot-block count, numerator for the legacy saved fraction.
	SlotBlocksPruned uint64
	// SlotBlocksHot is TotalExposure − SlotBlocksPruned, the numerator for
	// RAMRatio. Kept as an explicit field for readability.
	SlotBlocksHot uint64

	// Phase-2 headline metrics.
	RAMRatio       float64 // in [0,1]
	HotHitCoverage float64 // in [0,1]
	RAMCost        float64 // SlotBlocksHot × CostParams.RAMUnitCost
	MissPenaltyAgg float64 // Reactivations × CostParams.MissPenalty
	TotalCost      float64 // RAMCost + MissPenaltyAgg

	Reactivations int

	FinalPrunedSlots int

	// --- Legacy Phase-1 aliases ---
	// StorageSavedFrac == 1 − RAMRatio; the cold-tier slot-block fraction.
	StorageSavedFrac float64
	// FalsePruneRate == 1 − HotHitCoverage; the miss rate.
	FalsePruneRate float64
}
