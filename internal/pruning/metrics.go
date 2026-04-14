package pruning

// SimResult is the output of a single tiering simulation run. Every rate is
// in [0,1]; counts are raw integers so callers can compute additional
// derived metrics without losing precision.
//
// The four headline metrics map onto the cost-aware surrogate
//
//	d_i* = argmin_d  c(d) + ℓ(d) · p_i(τ)
//
// the way Phase 2's statistical policy will use them:
//
//   - StorageSavedFrac == cold-tier ratio. Fraction of slot-block exposure
//     the policy demotes out of the hot tier. The complementary number
//     (1 − StorageSavedFrac) is the **RAM ratio**: the share of slot-block
//     exposure that stays in the hot working set, i.e. the sequencer RAM
//     and witness-generation footprint we are trying to control.
//
//   - FalsePruneRate == miss rate. Fraction of observed accesses
//     (gap-closing events) that found their slot in the cold tier. The
//     complement (1 − FalsePruneRate) is the **hot-hit coverage**: the
//     fraction of real accesses served straight out of the hot tier
//     without paying a fetch penalty.
//
//   - Reactivations == miss count, scaled up to a **miss penalty** by
//     multiplying by the per-fetch cost ℓ from the config.
//
//   - FinalPrunedSlots is the cold-tier population at window end. Pairs
//     naturally with the RAM ratio when answering "how big is the hot
//     working set right now?".
//
// Field names are kept on the legacy "pruning" terminology for Phase 1 to
// avoid a churn-y rename across simulator, CLI printers, and tests.
// Phase 2 will introduce explicit RAMRatio / HotHitCoverage / MissPenalty
// / TotalCost fields once the cost parameters from configs/default.yaml
// are wired through the simulator.
type SimResult struct {
	Policy string

	TotalSlots        int
	ObservedIntervals int
	CensoredIntervals int

	// TotalExposure is the sum of all interval durations, i.e. the
	// slot-block risk-set integral across the window. It is the
	// denominator for the RAM-ratio computation
	// (RAMRatio = 1 − SlotBlocksPruned / TotalExposure).
	TotalExposure uint64
	// SlotBlocksPruned is the sum of (interval.Duration − threshold) over
	// intervals whose duration exceeds the policy threshold. Numerator
	// for StorageSavedFrac (== cold-tier ratio).
	SlotBlocksPruned uint64
	// StorageSavedFrac is the cold-tier ratio in [0,1]. RAM ratio is
	// 1 − StorageSavedFrac.
	StorageSavedFrac float64

	// Reactivations is the miss count: observed accesses that landed on
	// a cold-tier slot.
	Reactivations int
	// FalsePruneRate is the miss rate. Hot-hit coverage is
	// 1 − FalsePruneRate.
	FalsePruneRate float64

	// FinalPrunedSlots is the cold-tier slot count at window end.
	FinalPrunedSlots int
}
