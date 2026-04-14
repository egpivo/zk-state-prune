package analysis

import (
	"github.com/kshedden/statmodel/statmodel"

	"github.com/egpivo/zk-state-prune/internal/model"
)

// Column names used throughout the analysis package. They are the contract
// between ToSurvivalDataset and any SurvivalFitter that consumes the
// resulting statmodel.Dataset.
const (
	ColDuration     = "Duration"     // float: gap-time length of the interval
	ColStatus       = "Status"       // 1 = observed event, 0 = right censored
	ColEntry        = "Entry"        // delayed-entry time; 0 in gap-time KM
	ColAccessCount  = "AccessCount"  // covariate: prior in-window accesses
	ColContractAge  = "ContractAge"  // covariate: blocks since contract deploy
	ColSlotAge      = "SlotAge"      // covariate: blocks since slot creation
	ColContractType = "ContractType" // covariate: int-coded category enum
	ColSlotType     = "SlotType"     // covariate: int-coded slot-type enum
)

// ToSurvivalDataset converts a slice of InterAccessIntervals into the
// columnar format that github.com/kshedden/statmodel/duration consumes.
//
// The plan's naming ("dstream_adapter") predates the current statmodel API,
// which accepts a plain statmodel.Dataset built from [][]float64 + names;
// we no longer need the full dstream pipeline, so the adapter is just a
// column-packer. The file name is kept to match the plan's layout.
//
// Time-scale conventions:
//
//   - TimeVar = Duration (gap time between consecutive accesses)
//   - StatusVar = Status (1 observed, 0 censored at window end)
//   - EntryVar = Entry — always 0 in the gap-time formulation, because every
//     non-first-interval enters the risk set exactly at its own start and
//     first intervals of left-truncated slots use Window.Start as a
//     surrogate "previous access". The column is still populated so that a
//     later calendar-time Cox fit can reuse the same adapter with a
//     different time scale.
func ToSurvivalDataset(intervals []model.InterAccessInterval) statmodel.Dataset {
	n := len(intervals)
	dur := make([]float64, n)
	status := make([]float64, n)
	entry := make([]float64, n)
	accessCount := make([]float64, n)
	contractAge := make([]float64, n)
	slotAge := make([]float64, n)
	contractType := make([]float64, n)
	slotType := make([]float64, n)

	for i, it := range intervals {
		dur[i] = float64(it.Duration)
		if it.IsObserved {
			status[i] = 1
		}
		// entry[i] intentionally left at 0 — see package doc above.
		accessCount[i] = float64(it.AccessCount)
		contractAge[i] = float64(it.ContractAge)
		slotAge[i] = float64(it.SlotAge)
		contractType[i] = float64(it.ContractType)
		slotType[i] = float64(it.SlotType)
	}

	cols := [][]float64{dur, status, entry, accessCount, contractAge, slotAge, contractType, slotType}
	names := []string{ColDuration, ColStatus, ColEntry, ColAccessCount, ColContractAge, ColSlotAge, ColContractType, ColSlotType}
	return statmodel.NewDataset(cols, names)
}
