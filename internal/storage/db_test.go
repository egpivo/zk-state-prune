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
	if _, err := db.InsertAccessEvents(ctx, events); err != nil {
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

func TestMigrate_V1ToV2DedupsAccessEvents(t *testing.T) {
	// Simulate an on-disk v1 DB that was populated by an older build
	// which used plain INSERT (no dedup). The new migrate() must
	// (a) spot the v1 stamp, (b) run the dedup DELETE, (c) apply
	// model.Schema's unique index without error, and (d) stamp the
	// new version.
	ctx := context.Background()
	path := filepath.Join(t.TempDir(), "legacy.db")
	db, err := Open(ctx, path)
	if err != nil {
		t.Fatalf("Open #1: %v", err)
	}
	// Seed a slot + contract so the events have FK parents.
	if err := db.UpsertContract(ctx, model.ContractMeta{
		Address: "0xlegacy", ContractType: model.ContractERC20,
	}); err != nil {
		t.Fatalf("UpsertContract: %v", err)
	}
	if err := db.UpsertSlot(ctx, model.StateSlot{
		SlotID: "s1", ContractAddr: "0xlegacy", SlotType: model.SlotTypeBalance,
		CreatedAt: 1, LastAccess: 1, IsActive: true,
	}); err != nil {
		t.Fatalf("UpsertSlot: %v", err)
	}

	// Simulate the "legacy" state: rewind the version stamp to "1"
	// and drop the unique index + re-insert a duplicate by hand
	// (bypassing our new INSERT OR IGNORE). This mimics what a v1
	// install would have left behind.
	raw := db.SQL()
	if _, err := raw.ExecContext(ctx,
		`UPDATE schema_meta SET value='1' WHERE key='version'`); err != nil {
		t.Fatalf("rewind version: %v", err)
	}
	if _, err := raw.ExecContext(ctx, `DROP INDEX IF EXISTS idx_events_unique`); err != nil {
		t.Fatalf("drop unique index: %v", err)
	}
	for i := 0; i < 3; i++ {
		if _, err := raw.ExecContext(ctx,
			`INSERT INTO access_events(slot_id, block_number, access_type, tx_hash)
			 VALUES('s1', 100, 'write', '0xdup')`); err != nil {
			t.Fatalf("legacy insert #%d: %v", i, err)
		}
	}
	// Sanity: all 3 dup rows are there.
	if got, _ := db.CountAccessEvents(ctx); got != 3 {
		t.Fatalf("legacy state: CountAccessEvents=%d want 3", got)
	}
	if err := db.Close(); err != nil {
		t.Fatalf("close: %v", err)
	}

	// Re-open → migrate runs. Must succeed (no UNIQUE INDEX failure)
	// and leave exactly 1 of the 3 duplicates.
	db2, err := Open(ctx, path)
	if err != nil {
		t.Fatalf("Open #2 (migration): %v", err)
	}
	t.Cleanup(func() { _ = db2.Close() })

	got, err := db2.CountAccessEvents(ctx)
	if err != nil || got != 1 {
		t.Fatalf("post-migration CountAccessEvents=%d err=%v; want 1", got, err)
	}
	var v string
	if err := db2.SQL().QueryRowContext(ctx,
		`SELECT value FROM schema_meta WHERE key='version'`).Scan(&v); err != nil {
		t.Fatalf("read version: %v", err)
	}
	if v != "2" {
		t.Errorf("schema_meta version=%q, want %q", v, "2")
	}
}

func TestInsertAccessEvents_IsIdempotent(t *testing.T) {
	// Guarantees the unique (slot_id, block_number, access_type,
	// tx_hash) index + INSERT OR IGNORE silently drops exact
	// duplicate events on re-insert. This is the RPC extractor's
	// crash-between-flush-and-hwm resume path: block is re-fetched,
	// same events re-emitted, re-insertion must not double-count.
	ctx := context.Background()
	path := filepath.Join(t.TempDir(), "idem.db")
	db, err := Open(ctx, path)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	t.Cleanup(func() { _ = db.Close() })

	if err := db.UpsertContract(ctx, model.ContractMeta{
		Address:      "0xabc",
		ContractType: model.ContractERC20,
	}); err != nil {
		t.Fatalf("UpsertContract: %v", err)
	}
	if err := db.UpsertSlot(ctx, model.StateSlot{
		SlotID:       "s1",
		ContractAddr: "0xabc",
		SlotType:     model.SlotTypeBalance,
		CreatedAt:    1,
		LastAccess:   1,
		IsActive:     true,
	}); err != nil {
		t.Fatalf("UpsertSlot: %v", err)
	}

	events := []model.AccessEvent{
		{SlotID: "s1", BlockNumber: 100, AccessType: model.AccessWrite, TxHash: "0xaa"},
		{SlotID: "s1", BlockNumber: 101, AccessType: model.AccessRead, TxHash: "0xbb"},
	}
	inserted, err := db.InsertAccessEvents(ctx, events)
	if err != nil {
		t.Fatalf("first insert: %v", err)
	}
	if inserted != 2 {
		t.Errorf("first insert returned %d, want 2", inserted)
	}
	if got, _ := db.CountAccessEvents(ctx); got != 2 {
		t.Fatalf("CountAccessEvents after first insert = %d, want 2", got)
	}

	// Re-insert the same batch — simulated resume after crash.
	inserted, err = db.InsertAccessEvents(ctx, events)
	if err != nil {
		t.Fatalf("second insert: %v", err)
	}
	if inserted != 0 {
		t.Errorf("second insert of exact duplicates returned %d, want 0", inserted)
	}
	if got, _ := db.CountAccessEvents(ctx); got != 2 {
		t.Errorf("CountAccessEvents after duplicate insert = %d, want 2 (INSERT OR IGNORE should drop dups)", got)
	}

	// Distinct tuples still insert fine.
	fresh := []model.AccessEvent{
		{SlotID: "s1", BlockNumber: 102, AccessType: model.AccessWrite, TxHash: "0xcc"},
	}
	inserted, err = db.InsertAccessEvents(ctx, fresh)
	if err != nil {
		t.Fatalf("third insert: %v", err)
	}
	if inserted != 1 {
		t.Errorf("third insert returned %d, want 1", inserted)
	}
	if got, _ := db.CountAccessEvents(ctx); got != 3 {
		t.Errorf("CountAccessEvents after fresh insert = %d, want 3", got)
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
