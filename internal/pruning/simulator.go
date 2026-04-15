package pruning

import (
	"fmt"

	"github.com/egpivo/zk-state-prune/internal/model"
)

// Run simulates a tiering policy over a precomputed slice of survival
// intervals and scores it under a CostParams budget. The simulator is a
// pure function over intervals: no random state, no wall clock, and no
// dependency on raw access events, which makes it trivially deterministic
// and cheap to run across many policies back-to-back.
//
// Semantics per interval (gap time from previous access to next event or
// window end):
//
//   - The policy reports HotBlocks(it) ∈ [0, it.Duration] — how long the
//     slot stayed in the hot tier during this interval.
//   - Cold tail = it.Duration − HotBlocks(it). It contributes slot-blocks
//     to SlotBlocksPruned (the cold ledger).
//   - Observed intervals ending with a real access count as a miss iff
//     the cold tail is non-zero (the access landed on a demoted slot).
//   - Censored trailing intervals never contribute misses but their cold
//     tail decides whether the slot ends the window in the cold tier.
//
// TotalCost = SlotBlocksHot × RAMUnitCost + Reactivations × MissPenalty,
// which is the surrogate score the statistical policy minimizes
// slot-by-slot.
func Run(policy Policy, intervals []model.InterAccessInterval, costs CostParams) (*SimResult, error) {
	if policy == nil {
		return nil, fmt.Errorf("pruning.Run: nil policy")
	}

	res := &SimResult{Policy: policy.Name()}
	slotFinalPruned := make(map[string]bool)

	for _, it := range intervals {
		if _, seen := slotFinalPruned[it.SlotID]; !seen {
			slotFinalPruned[it.SlotID] = false
		}
		res.TotalExposure += it.Duration

		hot := policy.HotBlocks(it)
		if hot > it.Duration {
			hot = it.Duration
		}
		res.SlotBlocksPruned += it.Duration - hot
		// Slot ended this interval in the cold tier iff hot < Duration.
		// For observed intervals that means the access at IntervalEnd
		// found the slot demoted (a miss); for censored intervals it
		// means the trailing tail decides the slot's final state.
		endedCold := hot < it.Duration

		if it.IsObserved {
			res.ObservedIntervals++
			if endedCold {
				res.Reactivations++
			}
			slotFinalPruned[it.SlotID] = false
		} else {
			res.CensoredIntervals++
			slotFinalPruned[it.SlotID] = endedCold
		}
	}

	res.TotalSlots = len(slotFinalPruned)
	for _, p := range slotFinalPruned {
		if p {
			res.FinalPrunedSlots++
		}
	}

	res.SlotBlocksHot = res.TotalExposure - res.SlotBlocksPruned
	if res.TotalExposure > 0 {
		res.RAMRatio = float64(res.SlotBlocksHot) / float64(res.TotalExposure)
		res.StorageSavedFrac = 1 - res.RAMRatio
	}
	if res.ObservedIntervals > 0 {
		res.FalsePruneRate = float64(res.Reactivations) / float64(res.ObservedIntervals)
		res.HotHitCoverage = 1 - res.FalsePruneRate
	} else {
		// No observed events means no cold-tier hits are even possible,
		// which is the best-case coverage rather than the worst — a
		// policy that never gets tested trivially "serves" every access.
		res.HotHitCoverage = 1
	}

	res.RAMCost = float64(res.SlotBlocksHot) * costs.RAMUnitCost
	res.MissPenaltyAgg = float64(res.Reactivations) * costs.MissPenalty
	res.TotalCost = res.RAMCost + res.MissPenaltyAgg
	return res, nil
}

// RunAll runs every policy against the same intervals and returns the
// results in the order the policies were passed. Convenience wrapper for
// the comparison report.
func RunAll(policies []Policy, intervals []model.InterAccessInterval, costs CostParams) ([]*SimResult, error) {
	out := make([]*SimResult, 0, len(policies))
	for _, p := range policies {
		r, err := Run(p, intervals, costs)
		if err != nil {
			return nil, fmt.Errorf("policy %q: %w", p.Name(), err)
		}
		out = append(out, r)
	}
	return out, nil
}
