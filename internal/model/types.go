// Package model contains core domain types shared across zksp.
package model

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

// StateSlot is one storage slot tracked by zksp. SlotID is the canonical
// keccak256 of (contract, index) and serves as the primary key.
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

// ContractMeta is per-contract metadata, populated by the extractor and
// refreshed by analysis passes.
type ContractMeta struct {
	Address      string
	ContractType ContractCategory
	DeployBlock  uint64
	TotalSlots   uint64
	ActiveSlots  uint64
}
