package analysis

import (
	"context"
	"path/filepath"
	"testing"

	"github.com/egpivo/zk-state-prune/internal/model"
	"github.com/egpivo/zk-state-prune/internal/storage"
)

func openDB(t *testing.T) *storage.DB {
	t.Helper()
	db, err := storage.Open(context.Background(), filepath.Join(t.TempDir(), "x.db"))
	if err != nil {
		t.Fatalf("open: %v", err)
	}
	t.Cleanup(func() { _ = db.Close() })
	return db
}

// seed installs a single contract + slot + ordered events into db.
func seed(t *testing.T, db *storage.DB, slotID string, createdAt uint64, events ...uint64) {
	t.Helper()
	ctx := context.Background()
	contract := "0xaa"
	if err := db.UpsertContract(ctx, model.ContractMeta{
		Address:      contract,
		ContractType: model.ContractERC20,
		DeployBlock:  createdAt,
	}); err != nil {
		t.Fatalf("UpsertContract: %v", err)
	}
	last := createdAt
	if len(events) > 0 {
		last = events[len(events)-1]
	}
	if err := db.UpsertSlot(ctx, model.StateSlot{
		SlotID:       slotID,
		ContractAddr: contract,
		SlotType:     model.SlotTypeBalance,
		CreatedAt:    createdAt,
		LastAccess:   last,
		AccessCount:  uint64(len(events)),
		IsActive:     true,
	}); err != nil {
		t.Fatalf("UpsertSlot: %v", err)
	}
	if len(events) == 0 {
		return
	}
	evs := make([]model.AccessEvent, 0, len(events))
	for i, b := range events {
		at := model.AccessRead
		if i == 0 {
			at = model.AccessWrite
		}
		evs = append(evs, model.AccessEvent{
			SlotID:      slotID,
			BlockNumber: b,
			AccessType:  at,
			TxHash:      "0x00",
		})
	}
	if err := db.InsertAccessEvents(ctx, evs); err != nil {
		t.Fatalf("InsertAccessEvents: %v", err)
	}
}

func TestBuildIntervals_AllObserved(t *testing.T) {
	ctx := context.Background()
	db := openDB(t)
	seed(t, db, "s1", 100, 200, 300, 400) // created inside window, all events before window.End

	res, err := BuildIntervals(ctx, db, model.ObservationWindow{Start: 100, End: 500})
	if err != nil {
		t.Fatalf("BuildIntervals: %v", err)
	}
	// Expected intervals:
	//   [100,200] obs, [200,300] obs, [300,400] obs, [400,500] censored
	if got := len(res.Intervals); got != 4 {
		t.Fatalf("len=%d want 4: %+v", got, res.Intervals)
	}
	observed := 0
	for _, it := range res.Intervals[:3] {
		if !it.IsObserved {
			t.Errorf("expected observed interval, got censored: %+v", it)
		}
		if it.IsLeftTrunc {
			t.Errorf("slot created inside window should not be left-truncated: %+v", it)
		}
		observed++
	}
	if res.Intervals[3].IsObserved {
		t.Errorf("trailing interval should be censored")
	}
	if res.RightCensored != 1 {
		t.Errorf("RightCensored=%d want 1", res.RightCensored)
	}
	if res.LeftTruncatedIntervals != 0 || res.LeftTruncatedSlots != 0 {
		t.Errorf("LT intervals=%d slots=%d want 0,0",
			res.LeftTruncatedIntervals, res.LeftTruncatedSlots)
	}
}

func TestBuildIntervals_FullyCensoredNoEvents(t *testing.T) {
	ctx := context.Background()
	db := openDB(t)
	seed(t, db, "s1", 100) // created inside window, zero events

	res, err := BuildIntervals(ctx, db, model.ObservationWindow{Start: 100, End: 500})
	if err != nil {
		t.Fatalf("BuildIntervals: %v", err)
	}
	if len(res.Intervals) != 1 {
		t.Fatalf("len=%d want 1", len(res.Intervals))
	}
	it := res.Intervals[0]
	if it.IsObserved {
		t.Error("want censored")
	}
	if it.Duration != 400 {
		t.Errorf("Duration=%d want 400", it.Duration)
	}
	if res.SlotsWithNoEvents != 1 {
		t.Errorf("SlotsWithNoEvents=%d want 1", res.SlotsWithNoEvents)
	}
}

func TestBuildIntervals_LeftTruncated(t *testing.T) {
	ctx := context.Background()
	db := openDB(t)
	// Slot created at block 50, window starts at 100. Earliest visible
	// event is at 150, so the first interval must span [100, 150] and be
	// flagged as left-truncated.
	seed(t, db, "s1", 50, 80 /*before window, must be dropped*/, 150, 300)

	res, err := BuildIntervals(ctx, db, model.ObservationWindow{Start: 100, End: 500})
	if err != nil {
		t.Fatalf("BuildIntervals: %v", err)
	}
	// Intervals: [100,150] obs LT, [150,300] obs, [300,500] censored
	if got := len(res.Intervals); got != 3 {
		t.Fatalf("len=%d want 3: %+v", got, res.Intervals)
	}
	first := res.Intervals[0]
	if !first.IsLeftTrunc || first.IntervalStart != 100 || first.IntervalEnd != 150 {
		t.Errorf("first interval wrong: %+v", first)
	}
	if res.LeftTruncatedIntervals != 1 || res.LeftTruncatedSlots != 1 {
		t.Errorf("LT intervals=%d slots=%d want 1,1",
			res.LeftTruncatedIntervals, res.LeftTruncatedSlots)
	}
	if res.RightCensored != 1 {
		t.Errorf("RightCensored=%d want 1", res.RightCensored)
	}
}

func TestBuildIntervals_SlotBornAfterWindow(t *testing.T) {
	ctx := context.Background()
	db := openDB(t)
	seed(t, db, "s1", 600, 700) // born after window.End — skipped

	res, err := BuildIntervals(ctx, db, model.ObservationWindow{Start: 100, End: 500})
	if err != nil {
		t.Fatalf("BuildIntervals: %v", err)
	}
	if len(res.Intervals) != 0 {
		t.Errorf("expected zero intervals, got %d", len(res.Intervals))
	}
	if res.SlotsSkipped != 1 {
		t.Errorf("SlotsSkipped=%d want 1", res.SlotsSkipped)
	}
}

func TestBuildIntervals_LTCountedEvenWhenFirstEventOnWindowStart(t *testing.T) {
	// Regression: left-truncated slot whose first visible access lands
	// exactly on Window.Start. The entry fast-path consumes the opener
	// so no emitted interval carries IsLeftTrunc=true, but the slot must
	// still be counted as left-truncated at the slot-level diagnostic.
	ctx := context.Background()
	db := openDB(t)
	seed(t, db, "s1", 50 /*pre-window*/, 100 /*==Window.Start*/, 300)

	res, err := BuildIntervals(ctx, db, model.ObservationWindow{Start: 100, End: 500})
	if err != nil {
		t.Fatalf("BuildIntervals: %v", err)
	}
	// No interval should have IsLeftTrunc=true under this fast-path.
	for _, it := range res.Intervals {
		if it.IsLeftTrunc {
			t.Errorf("did not expect any LT interval, got %+v", it)
		}
	}
	if res.LeftTruncatedIntervals != 0 {
		t.Errorf("LeftTruncatedIntervals=%d want 0", res.LeftTruncatedIntervals)
	}
	if res.LeftTruncatedSlots != 1 {
		t.Errorf("LeftTruncatedSlots=%d want 1", res.LeftTruncatedSlots)
	}
}

func TestBuildIntervals_DedupesSameBlockEvents(t *testing.T) {
	ctx := context.Background()
	db := openDB(t)
	// Three "events" at the same block 200 collapse to one logical touch.
	seed(t, db, "s1", 100, 200, 200, 200, 400)

	res, err := BuildIntervals(ctx, db, model.ObservationWindow{Start: 100, End: 500})
	if err != nil {
		t.Fatalf("BuildIntervals: %v", err)
	}
	// After dedupe: effective events [200, 400]. Intervals:
	//   [100,200] obs, [200,400] obs, [400,500] censored → 3 total.
	if got := len(res.Intervals); got != 3 {
		t.Fatalf("len=%d want 3: %+v", got, res.Intervals)
	}
	for i, it := range res.Intervals {
		if it.Duration == 0 {
			t.Errorf("interval[%d] has Duration=0: %+v", i, it)
		}
	}
}

func TestBuildIntervals_FirstEventAtEntryIsConsumed(t *testing.T) {
	ctx := context.Background()
	db := openDB(t)
	// CreatedAt == first event block; the degenerate [100,100] interval
	// must be elided, leaving only the gap [100,200] and trailing censor.
	seed(t, db, "s1", 100, 100, 200)

	res, err := BuildIntervals(ctx, db, model.ObservationWindow{Start: 100, End: 500})
	if err != nil {
		t.Fatalf("BuildIntervals: %v", err)
	}
	if got := len(res.Intervals); got != 2 {
		t.Fatalf("len=%d want 2: %+v", got, res.Intervals)
	}
	for _, it := range res.Intervals {
		if it.Duration == 0 {
			t.Errorf("unexpected zero-duration interval: %+v", it)
		}
	}
	// AccessCount invariant: the trailing censored interval's
	// AccessCount must equal the true number of in-window accesses (2).
	last := res.Intervals[len(res.Intervals)-1]
	if last.IsObserved || last.AccessCount != 2 {
		t.Errorf("trailing interval = %+v, want censored ac=2", last)
	}
}

func TestBuildIntervals_NoZeroDurationOnMockBurst(t *testing.T) {
	// Seed a slot with many co-accesses clumped in a few blocks, which is
	// what the mock extractor's intra-contract correlation pass produces.
	// Every emitted interval must have Duration > 0.
	ctx := context.Background()
	db := openDB(t)
	seed(t, db, "s1", 100, 120, 120, 120, 121, 130, 130, 200, 200, 300)

	res, err := BuildIntervals(ctx, db, model.ObservationWindow{Start: 100, End: 500})
	if err != nil {
		t.Fatalf("BuildIntervals: %v", err)
	}
	if len(res.Intervals) == 0 {
		t.Fatalf("no intervals")
	}
	for i, it := range res.Intervals {
		if it.Duration == 0 {
			t.Errorf("interval[%d] Duration=0: %+v", i, it)
		}
	}
}

func TestBuildIntervals_CovariatesAtIntervalStart(t *testing.T) {
	ctx := context.Background()
	db := openDB(t)
	seed(t, db, "s1", 100, 150, 200, 300)

	res, err := BuildIntervals(ctx, db, model.ObservationWindow{Start: 100, End: 500})
	if err != nil {
		t.Fatalf("BuildIntervals: %v", err)
	}
	// AccessCount should be 0,1,2,3 across the four emitted intervals.
	want := []uint64{0, 1, 2, 3}
	for i, it := range res.Intervals {
		if it.AccessCount != want[i] {
			t.Errorf("interval[%d] AccessCount=%d want %d", i, it.AccessCount, want[i])
		}
	}
}
