package storage

import (
	"context"
	"path/filepath"
	"testing"

	"github.com/egpivo/zk-state-prune/internal/model"
)

func TestOpenAndRoundTrip(t *testing.T) {
	ctx := context.Background()
	path := filepath.Join(t.TempDir(), "test.db")

	db, err := Open(ctx, path)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	t.Cleanup(func() { _ = db.Close() })

	contract := model.ContractMeta{
		Address:      "0xabc",
		ContractType: model.ContractERC20,
		DeployBlock:  100,
		TotalSlots:   2,
		ActiveSlots:  2,
	}
	if err := db.UpsertContract(ctx, contract); err != nil {
		t.Fatalf("UpsertContract: %v", err)
	}

	slots := []model.StateSlot{
		{SlotID: "s1", ContractAddr: "0xabc", SlotIndex: 0, SlotType: model.SlotTypeBalance, CreatedAt: 100, LastAccess: 150, AccessCount: 3, IsActive: true},
		{SlotID: "s2", ContractAddr: "0xabc", SlotIndex: 1, SlotType: model.SlotTypeMapping, CreatedAt: 110, LastAccess: 200, AccessCount: 1, IsActive: true},
	}
	for _, s := range slots {
		if err := db.UpsertSlot(ctx, s); err != nil {
			t.Fatalf("UpsertSlot: %v", err)
		}
	}

	events := []model.AccessEvent{
		{SlotID: "s1", BlockNumber: 100, AccessType: model.AccessWrite, TxHash: "0x1"},
		{SlotID: "s1", BlockNumber: 130, AccessType: model.AccessRead, TxHash: "0x2"},
		{SlotID: "s1", BlockNumber: 150, AccessType: model.AccessRead, TxHash: "0x3"},
		{SlotID: "s2", BlockNumber: 200, AccessType: model.AccessWrite, TxHash: "0x4"},
	}
	if err := db.InsertAccessEvents(ctx, events); err != nil {
		t.Fatalf("InsertAccessEvents: %v", err)
	}

	gotSlots, err := db.CountSlots(ctx)
	if err != nil || gotSlots != 2 {
		t.Fatalf("CountSlots = %d, err=%v; want 2", gotSlots, err)
	}
	gotEvents, err := db.CountAccessEvents(ctx)
	if err != nil || gotEvents != 4 {
		t.Fatalf("CountAccessEvents = %d, err=%v; want 4", gotEvents, err)
	}

	// Idempotent re-upsert with new lifecycle state.
	updated := slots[0]
	updated.LastAccess = 999
	updated.AccessCount = 4
	if err := db.UpsertSlot(ctx, updated); err != nil {
		t.Fatalf("re-upsert: %v", err)
	}
	if got, _ := db.CountSlots(ctx); got != 2 {
		t.Fatalf("CountSlots after upsert = %d; want 2", got)
	}
}

func TestParseEnumsRoundTrip(t *testing.T) {
	cases := []model.SlotType{
		model.SlotTypeBalance, model.SlotTypeMapping, model.SlotTypeArray,
		model.SlotTypeFixed, model.SlotTypeUnknown,
	}
	for _, c := range cases {
		if got := model.ParseSlotType(c.String()); got != c {
			t.Errorf("SlotType round trip %v -> %q -> %v", c, c.String(), got)
		}
	}
	cats := []model.ContractCategory{
		model.ContractERC20, model.ContractDEX, model.ContractNFT,
		model.ContractBridge, model.ContractGovernance, model.ContractOther,
	}
	for _, c := range cats {
		if got := model.ParseContractCategory(c.String()); got != c {
			t.Errorf("ContractCategory round trip %v -> %q -> %v", c, c.String(), got)
		}
	}
}
