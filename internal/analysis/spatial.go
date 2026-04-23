package analysis

import (
	"context"
	"fmt"
	"math/rand/v2"
	"sort"

	"github.com/egpivo/zk-state-prune/internal/domain"
	"github.com/egpivo/zk-state-prune/internal/storage"
)

// SpatialReport describes intra-contract slot clustering: the degree to
// which pairs of slots belonging to the same contract share access
// blocks. High Jaccard across slot pairs means a contract's storage
// tends to move together (e.g. a DEX pool where every swap touches
// reserves + price + fee slots in the same tx), which the tiering
// policy can exploit by promoting/demoting the whole contract together.
//
// The metric is a grounded sanity check against the mock extractor's
// IntraContractCorrelation ρ parameter: on a synthetic dataset with
// ρ=0.7, MeanJaccard should be meaningfully > 0.
type SpatialReport struct {
	Window domain.ObservationWindow

	// Number of contracts whose slot set was large enough (>= 2 slots
	// with events) to measure clustering on.
	NumContracts int

	// Global means over contracts that contributed to the measurement.
	MeanJaccard   float64
	MedianJaccard float64

	// PerContract is the per-contract Jaccard score, keyed by address.
	PerContract map[string]float64
}

// spatialMaxPairs caps the per-contract pair sample to keep the
// quadratic cost bounded on contracts with hundreds of slots. 50 pairs
// is plenty to estimate a contract-wide mean within a few percent.
const spatialMaxPairs = 50

// RunSpatial computes per-contract intra-contract access clustering via
// the mean pairwise Jaccard coefficient on slot access-block sets,
// restricted to the observation window.
//
// For each contract the routine collects
//
//	S_i = { block_number : slot i was touched at block_number ∧ window.Contains(block_number) }
//
// for every slot i in the contract. It then draws up to spatialMaxPairs
// slot pairs (i, j) and computes Jaccard(S_i, S_j) = |S_i ∩ S_j| / |S_i ∪ S_j|.
// The contract's score is the mean over sampled pairs. A contract with
// fewer than 2 eligible slots is skipped.
func RunSpatial(ctx context.Context, db *storage.DB, window domain.ObservationWindow) (*SpatialReport, error) {
	if window.End <= window.Start {
		return nil, fmt.Errorf("RunSpatial: invalid window [%d, %d)", window.Start, window.End)
	}
	byContract, err := collectContractSlotBlocks(ctx, db, window)
	if err != nil {
		return nil, err
	}

	r := rand.New(rand.NewPCG(17, 23))
	perContract := make(map[string]float64, len(byContract))
	scores := make([]float64, 0, len(byContract))
	for addr, slots := range byContract {
		score, ok := meanJaccardForContract(slots, r)
		if !ok {
			continue
		}
		perContract[addr] = score
		scores = append(scores, score)
	}

	out := &SpatialReport{
		Window:       window,
		NumContracts: len(scores),
		PerContract:  perContract,
	}
	if len(scores) > 0 {
		sort.Float64s(scores)
		var sum float64
		for _, s := range scores {
			sum += s
		}
		out.MeanJaccard = sum / float64(len(scores))
		out.MedianJaccard = scores[len(scores)/2]
	}
	return out, nil
}

// collectContractSlotBlocks streams the storage iterator and builds a
// contract → slot → sorted-access-blocks map for every slot that has
// at least one access inside the window. Same-block duplicates are
// dropped to match BuildIntervals so the Jaccard set semantics agree
// with survival input.
func collectContractSlotBlocks(
	ctx context.Context,
	db *storage.DB,
	window domain.ObservationWindow,
) (map[string]map[string][]uint64, error) {
	byContract := make(map[string]map[string][]uint64)
	err := db.IterateSlotEvents(ctx, func(sm storage.SlotWithMeta, events []domain.AccessEvent) error {
		blocks := make([]uint64, 0, len(events))
		var prev uint64
		havePrev := false
		for _, e := range events {
			if !window.Contains(e.BlockNumber) {
				continue
			}
			if havePrev && e.BlockNumber == prev {
				continue
			}
			blocks = append(blocks, e.BlockNumber)
			prev = e.BlockNumber
			havePrev = true
		}
		if len(blocks) == 0 {
			return nil
		}
		addr := sm.Slot.ContractAddr
		if byContract[addr] == nil {
			byContract[addr] = make(map[string][]uint64)
		}
		byContract[addr][sm.Slot.SlotID] = blocks
		return nil
	})
	if err != nil {
		return nil, fmt.Errorf("iterate slots: %w", err)
	}
	return byContract, nil
}

// meanJaccardForContract computes the mean pairwise Jaccard score over
// this contract's slot access sets. Enumerates all pairs when the
// count is small; otherwise samples up to spatialMaxPairs uniformly.
// Returns (0, false) when the contract has fewer than 2 slots or the
// sample produced no valid pairs.
func meanJaccardForContract(slots map[string][]uint64, r *rand.Rand) (float64, bool) {
	if len(slots) < 2 {
		return 0, false
	}
	keys := make([]string, 0, len(slots))
	for k := range slots {
		keys = append(keys, k)
	}
	sort.Strings(keys)

	nSlots := len(keys)
	totalPairs := nSlots * (nSlots - 1) / 2
	var sum float64
	var count int
	if totalPairs <= spatialMaxPairs {
		for i := 0; i < nSlots; i++ {
			for j := i + 1; j < nSlots; j++ {
				sum += jaccardUint64(slots[keys[i]], slots[keys[j]])
				count++
			}
		}
	} else {
		for n := 0; n < spatialMaxPairs; n++ {
			i := r.IntN(nSlots)
			j := r.IntN(nSlots)
			if i == j {
				j = (j + 1) % nSlots
			}
			sum += jaccardUint64(slots[keys[i]], slots[keys[j]])
			count++
		}
	}
	if count == 0 {
		return 0, false
	}
	return sum / float64(count), true
}

// jaccardUint64 is |A ∩ B| / |A ∪ B| for two sorted (or to-be-sorted)
// uint64 sequences. Returns 0 for two empty sets.
func jaccardUint64(a, b []uint64) float64 {
	if len(a) == 0 && len(b) == 0 {
		return 0
	}
	setA := make(map[uint64]struct{}, len(a))
	for _, v := range a {
		setA[v] = struct{}{}
	}
	inter := 0
	for _, v := range b {
		if _, ok := setA[v]; ok {
			inter++
		}
	}
	union := len(setA) + len(b) - inter
	if union == 0 {
		return 0
	}
	return float64(inter) / float64(union)
}
