package analysis

import (
	"fmt"
	"math/rand/v2"
	"sort"

	"github.com/egpivo/zk-state-prune/internal/domain"
)

// TrainHoldoutSplitBySlot partitions intervals into a (train, holdout)
// pair, with `holdoutFrac` of the *slots* assigned to the holdout side.
// All intervals belonging to a single slot stay together — splitting at
// the interval level would let the model peek at consecutive accesses on
// the same slot during training and then evaluate on its own neighbours,
// inflating apparent calibration.
//
// Determinism: the slot order is sorted before shuffling so the same
// `seed` always yields the same partition regardless of the input
// interval order or map iteration randomness inside upstream callers.
//
// Returns an error rather than panicking on out-of-range fractions so
// the calling CLI flag can surface it cleanly.
func TrainHoldoutSplitBySlot(
	intervals []domain.InterAccessInterval,
	holdoutFrac float64,
	seed uint64,
) (train, holdout []domain.InterAccessInterval, err error) {
	if holdoutFrac <= 0 || holdoutFrac >= 1 {
		return nil, nil, fmt.Errorf("TrainHoldoutSplitBySlot: holdoutFrac %v not in (0,1)", holdoutFrac)
	}
	if len(intervals) == 0 {
		return nil, nil, fmt.Errorf("TrainHoldoutSplitBySlot: empty intervals")
	}

	slotSet := make(map[string]struct{})
	for _, it := range intervals {
		slotSet[it.SlotID] = struct{}{}
	}
	slots := make([]string, 0, len(slotSet))
	for s := range slotSet {
		slots = append(slots, s)
	}
	sort.Strings(slots)

	r := rand.New(rand.NewPCG(seed, seed^0x9E3779B97F4A7C15))
	r.Shuffle(len(slots), func(i, j int) { slots[i], slots[j] = slots[j], slots[i] })

	nHoldout := int(float64(len(slots)) * holdoutFrac)
	// Guarantee both sides are non-empty when there is at least one
	// slot to spare. Pruning analyses with an empty side are useless.
	if nHoldout == 0 && len(slots) >= 2 {
		nHoldout = 1
	}
	if nHoldout >= len(slots) && len(slots) >= 2 {
		nHoldout = len(slots) - 1
	}
	holdoutSet := make(map[string]struct{}, nHoldout)
	for i := 0; i < nHoldout; i++ {
		holdoutSet[slots[i]] = struct{}{}
	}

	for _, it := range intervals {
		if _, ok := holdoutSet[it.SlotID]; ok {
			holdout = append(holdout, it)
		} else {
			train = append(train, it)
		}
	}
	return train, holdout, nil
}
