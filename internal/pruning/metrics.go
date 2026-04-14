package pruning

// SimResult is the output of a single pruning simulation run. Every rate is
// in [0,1]; counts are raw integers so callers can compute additional
// derived metrics without losing precision.
//
// The four headline metrics for Phase-1 comparison are:
//
//   - StorageSavedFrac: fraction of slot-block exposure the policy removes.
//     This is the closest proxy for "prover memory saved": a slot that is
//     pruned for half the observation window contributes 0.5 to the saved
//     side of the ledger, and only the slot-blocks remaining after pruning
//     have to live inside the prover's Merkle tree.
//
//   - FalsePruneRate: fraction of observed accesses (gap-closing events)
//     that found their slot in the pruned state. This is the cost side of
//     the ledger — every false prune is a reactivation proof in production.
//
//   - Reactivations: absolute count, useful for cost estimates (multiply
//     by the reactivation proof cost from the config).
//
//   - FinalPrunedSlots: slots sitting in the pruned state at window end.
//     Pairs naturally with StorageSavedFrac for the "how much smaller is
//     the live state at t=now?" question.
type SimResult struct {
	Policy string

	TotalSlots        int
	ObservedIntervals int
	CensoredIntervals int

	// TotalExposure is the sum of all interval durations, i.e. the
	// slot-block risk set integral across the window. It is the
	// denominator for StorageSavedFrac.
	TotalExposure uint64
	// SlotBlocksPruned is the sum of (interval.Duration - threshold) over
	// intervals whose duration exceeds the policy threshold. Numerator
	// for StorageSavedFrac.
	SlotBlocksPruned uint64
	StorageSavedFrac float64

	Reactivations  int
	FalsePruneRate float64

	FinalPrunedSlots int
}
