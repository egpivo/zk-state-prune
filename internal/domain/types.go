// Package domain contains core domain types shared across zksp.
package domain

import "fmt"

// SlotType classifies a storage slot by its semantic role in a contract's
// storage layout. It influences both lifecycle (e.g. balances are touched
// frequently, fixed config slots rarely) and the survival-model stratum.
type SlotType int

const (
	SlotTypeUnknown SlotType = iota
	SlotTypeBalance
	SlotTypeMapping
	SlotTypeArray
	SlotTypeFixed
)

func (s SlotType) String() string {
	switch s {
	case SlotTypeBalance:
		return "balance"
	case SlotTypeMapping:
		return "mapping"
	case SlotTypeArray:
		return "array"
	case SlotTypeFixed:
		return "fixed"
	case SlotTypeUnknown:
		return "unknown"
	}
	return fmt.Sprintf("slot_type(%d)", int(s))
}

// ParseSlotType is the inverse of SlotType.String for round-tripping through
// SQLite text columns.
func ParseSlotType(s string) SlotType {
	switch s {
	case "balance":
		return SlotTypeBalance
	case "mapping":
		return SlotTypeMapping
	case "array":
		return SlotTypeArray
	case "fixed":
		return SlotTypeFixed
	default:
		return SlotTypeUnknown
	}
}

// AccessType distinguishes reads from writes. Both touch the slot from a
// liveness standpoint, but writes also reset the "last modified" cursor.
type AccessType int

const (
	AccessRead AccessType = iota
	AccessWrite
)

func (a AccessType) String() string {
	switch a {
	case AccessRead:
		return "read"
	case AccessWrite:
		return "write"
	}
	return fmt.Sprintf("access_type(%d)", int(a))
}

func ParseAccessType(s string) AccessType {
	if s == "write" {
		return AccessWrite
	}
	return AccessRead
}

// ContractCategory groups contracts by application archetype. Different
// categories exhibit very different access lifecycles, so the survival model
// stratifies on this.
type ContractCategory int

const (
	ContractOther ContractCategory = iota
	ContractERC20
	ContractDEX
	ContractNFT
	ContractBridge
	ContractGovernance
)

func (c ContractCategory) String() string {
	switch c {
	case ContractERC20:
		return "erc20"
	case ContractDEX:
		return "dex"
	case ContractNFT:
		return "nft"
	case ContractBridge:
		return "bridge"
	case ContractGovernance:
		return "governance"
	case ContractOther:
		return "other"
	}
	return fmt.Sprintf("contract_category(%d)", int(c))
}

func ParseContractCategory(s string) ContractCategory {
	switch s {
	case "erc20":
		return ContractERC20
	case "dex":
		return ContractDEX
	case "nft":
		return ContractNFT
	case "bridge":
		return ContractBridge
	case "governance":
		return ContractGovernance
	default:
		return ContractOther
	}
}

// StateSlot is one storage slot tracked by zksp. SlotID is an opaque
// deterministic key that serves as the primary key. The exact format
// is chosen per-extractor (e.g. the mock uses a synthetic string; the
// Transfer-log surrogate uses "contract:holder"; the statediff source
// uses "contract:<32-byte-slotKey>") and callers downstream treat it
// strictly as an opaque token — it is never parsed back into contract
// / index pairs. Two extractions by the same extractor must produce
// the same SlotID for the same underlying slot.
type StateSlot struct {
	SlotID       string
	ContractAddr string
	SlotIndex    uint64
	SlotType     SlotType
	CreatedAt    uint64
	LastAccess   uint64
	AccessCount  uint64
	IsActive     bool
}

// AccessEvent records a single touch of a slot at a given block.
type AccessEvent struct {
	SlotID      string
	BlockNumber uint64
	AccessType  AccessType
	TxHash      string
}

// ObservationWindow is the [Start, End) block range that analysis looks at.
// Events outside the window are invisible to EDA and survival fits, so the
// window choice directly controls left-truncation and right-censoring rates.
type ObservationWindow struct {
	Start uint64
	End   uint64
}

// Contains reports whether block b falls inside the window.
func (w ObservationWindow) Contains(b uint64) bool { return b >= w.Start && b < w.End }

// Span returns the window length in blocks.
func (w ObservationWindow) Span() uint64 { return w.End - w.Start }

// InterAccessInterval is one [start, stop] survival observation built from a
// slot's access trace, restricted to an observation window. It is the unit of
// input for both EDA inter-access analysis and Kaplan–Meier / Cox fits.
//
// Censoring & truncation:
//   - IsObserved=true means the interval ended on a real access (event); false
//     means it was right-censored at window.End.
//   - IsLeftTrunc=true marks the first interval of a slot whose creation
//     block precedes window.Start. The slot enters the risk set at EntryTime
//     (= window.Start), not at its true creation time, which avoids the
//     survivorship bias from over-representing long-lived slots.
//   - For non-first intervals EntryTime equals IntervalStart and the flag is
//     false: the slot is already in the risk set from the previous access.
type InterAccessInterval struct {
	SlotID        string
	IntervalStart uint64
	IntervalEnd   uint64
	Duration      uint64
	IsObserved    bool
	IsLeftTrunc   bool
	EntryTime     uint64

	// Cox covariates captured at IntervalStart so the model sees the state
	// of the world the way an online policy would.
	ContractType ContractCategory
	SlotType     SlotType
	AccessCount  uint64
	ContractAge  uint64
	SlotAge      uint64
}

// ContractMeta is per-contract metadata, populated by the extractor and
// refreshed by analysis passes.
type ContractMeta struct {
	Address      string
	ContractType ContractCategory
	DeployBlock  uint64
	TotalSlots   uint64
	ActiveSlots  uint64
}
