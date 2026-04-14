package pruning

import (
	"fmt"

	"github.com/egpivo/zk-state-prune/internal/model"
)

// Run simulates a pruning policy over a precomputed slice of survival
// intervals and returns a SimResult. The simulator is a pure function over
// intervals: no random state, no wall clock, and no dependency on raw
// access events, which makes it trivially deterministic and cheap to run
// across many policies back-to-back.
//
// Semantics per interval (gap time from previous access to next event or
// window end):
//
//   - If Duration < threshold: slot was never pruned during this interval,
//     no savings, no reactivation.
//   - If Duration >= threshold: the policy would have pruned the slot at
//     (IntervalStart + threshold). Savings = Duration - threshold.
//   - Observed intervals ending with a real access constitute a reactivation
//     iff Duration >= threshold (the access landed on a pruned slot).
//   - Censored trailing intervals never contribute reactivations — no access
//     actually happened — but they do contribute savings and they determine
//     whether the slot ends the window in the pruned state.
//
// Because every slot's trailing interval is censored by construction in
// BuildIntervals, "final pruned state" reduces to "is the trailing
// interval longer than the policy threshold?" which we read straight off
// the stream.
func Run(policy Policy, intervals []model.InterAccessInterval) (*SimResult, error) {
	if policy == nil {
		return nil, fmt.Errorf("pruning.Run: nil policy")
	}

	res := &SimResult{Policy: policy.Name()}
	threshold := policy.PruneThreshold()
	slotFinalPruned := make(map[string]bool)

	for _, it := range intervals {
		if _, seen := slotFinalPruned[it.SlotID]; !seen {
			slotFinalPruned[it.SlotID] = false
		}
		res.TotalExposure += it.Duration

		pruned := policy.IsPruned(it.Duration)
		if pruned && threshold > 0 {
			res.SlotBlocksPruned += it.Duration - threshold
		}

		if it.IsObserved {
			res.ObservedIntervals++
			if pruned {
				res.Reactivations++
			}
			// A real access clears the pruned state going forward; the
			// next interval starts fresh. Reset the per-slot flag so the
			// trailing censored interval is what determines final state.
			slotFinalPruned[it.SlotID] = false
		} else {
			res.CensoredIntervals++
			// The trailing censored interval dictates the final state.
			slotFinalPruned[it.SlotID] = pruned
		}
	}

	res.TotalSlots = len(slotFinalPruned)
	for _, p := range slotFinalPruned {
		if p {
			res.FinalPrunedSlots++
		}
	}
	if res.TotalExposure > 0 {
		res.StorageSavedFrac = float64(res.SlotBlocksPruned) / float64(res.TotalExposure)
	}
	if res.ObservedIntervals > 0 {
		res.FalsePruneRate = float64(res.Reactivations) / float64(res.ObservedIntervals)
	}
	return res, nil
}

// RunAll runs every policy against the same intervals and returns the
// results in the order the policies were passed. Convenience wrapper for
// the comparison report.
func RunAll(policies []Policy, intervals []model.InterAccessInterval) ([]*SimResult, error) {
	out := make([]*SimResult, 0, len(policies))
	for _, p := range policies {
		r, err := Run(p, intervals)
		if err != nil {
			return nil, fmt.Errorf("policy %q: %w", p.Name(), err)
		}
		out = append(out, r)
	}
	return out, nil
}
