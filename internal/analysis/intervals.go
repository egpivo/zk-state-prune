// Package analysis contains the statistical passes that operate on the
// zksp DB: interval construction, EDA, survival models, spatial/temporal
// analysis. Every pass takes a *storage.DB (read-only from its point of
// view) and an observation window, and returns plain structs — no hidden
// DB writes, so passes compose cleanly inside the CLI.
package analysis

import (
	"context"
	"fmt"

	"github.com/egpivo/zk-state-prune/internal/model"
	"github.com/egpivo/zk-state-prune/internal/storage"
)

// IntervalBuildResult bundles the interval set with the diagnostics the EDA
// report needs. All fields are raw counts; consumers format them as rates.
type IntervalBuildResult struct {
	Intervals     []model.InterAccessInterval
	RightCensored int
	// LeftTruncatedIntervals counts intervals that carry IsLeftTrunc=true,
	// i.e. intervals that use Window.Start as a surrogate prior access.
	// This is the right input for Cox models that filter on the flag.
	LeftTruncatedIntervals int
	// LeftTruncatedSlots counts distinct slots whose creation block
	// precedes Window.Start, regardless of whether any emitted interval
	// ended up carrying IsLeftTrunc=true. A left-truncated slot whose
	// first visible event sits exactly on Window.Start has its opener
	// consumed by the entry fast-path and contributes zero LT intervals,
	// but it is still a left-truncated slot for diagnostic purposes.
	// Use this field for the plan's "left_truncated_count / total_slots"
	// metric.
	LeftTruncatedSlots int
	// SlotsWithNoEvents counts slots that existed during the window but were
	// never touched in it. Each contributes exactly one fully-censored
	// interval [entry, window.End].
	SlotsWithNoEvents int
	// SlotsSkipped counts slots whose creation block is at or after
	// window.End — they're not in the risk set at all, so they produce
	// zero intervals.
	SlotsSkipped int
}

// BuildIntervals walks every slot in db and emits a survival observation
// (or several, for slots with many accesses) restricted to window. The
// censoring / truncation rules follow the plan:
//
//   - entry  = max(slot.CreatedAt, window.Start)
//   - first interval uses entry as its start; IsLeftTrunc is true iff the
//     slot pre-existed the window
//   - each event inside the window closes the current interval (observed)
//     and opens a new one starting at the event block
//   - the trailing interval from the last in-window event to window.End is
//     right-censored (IsObserved=false)
//   - slots with no in-window events yield one fully-censored interval
//     [entry, window.End]
//   - slots born at or after window.End are skipped (no risk exposure)
//
// Covariates (AccessCount, ContractAge, SlotAge) are snapshotted at the
// interval's start so downstream Cox fits get time-consistent features.
func BuildIntervals(
	ctx context.Context,
	db *storage.DB,
	window model.ObservationWindow,
) (IntervalBuildResult, error) {
	if window.End <= window.Start {
		return IntervalBuildResult{}, fmt.Errorf("invalid window: [%d, %d)", window.Start, window.End)
	}

	res := IntervalBuildResult{Intervals: make([]model.InterAccessInterval, 0, 1024)}

	err := db.IterateSlotEvents(ctx, func(sm storage.SlotWithMeta, events []model.AccessEvent) error {
		slot := sm.Slot
		if slot.CreatedAt >= window.End {
			res.SlotsSkipped++
			return nil
		}

		entry := slot.CreatedAt
		leftTrunc := false
		if slot.CreatedAt < window.Start {
			entry = window.Start
			leftTrunc = true
		}

		// Collect in-window events and dedupe same-block touches. Multiple
		// accesses to one slot inside the same block (intra-tx read+write,
		// co-access buddies injected by the mock, etc.) count as ONE
		// logical event for survival purposes: treating them as distinct
		// would emit Duration=0 intervals, which (a) skew IAT summaries
		// toward zero and (b) are fragile inputs for KM/Cox libraries that
		// assume strictly positive durations or need explicit tie handling.
		// Storage already hands us events sorted by (slot_id, block_number),
		// so "previous block equals current block" is sufficient for dedup.
		inWindow := make([]uint64, 0, len(events))
		var prevBlock uint64
		havePrev := false
		for _, e := range events {
			if !window.Contains(e.BlockNumber) {
				continue
			}
			if e.BlockNumber < entry {
				continue
			}
			if havePrev && e.BlockNumber == prevBlock {
				continue
			}
			inWindow = append(inWindow, e.BlockNumber)
			prevBlock = e.BlockNumber
			havePrev = true
		}

		slotAgeAt := func(t uint64) uint64 {
			if t < slot.CreatedAt {
				return 0
			}
			return t - slot.CreatedAt
		}
		// Contract deploy block is not fetched by IterateSlotEvents today —
		// use CreatedAt as a stand-in. In Phase 1 the mock sets them equal,
		// and the Cox pass can swap this out once we surface deploy_block.
		contractAgeAt := slotAgeAt

		emit := func(start, end uint64, observed, lt bool, accessCount uint64) {
			res.Intervals = append(res.Intervals, model.InterAccessInterval{
				SlotID:        slot.SlotID,
				IntervalStart: start,
				IntervalEnd:   end,
				Duration:      end - start,
				IsObserved:    observed,
				IsLeftTrunc:   lt,
				EntryTime:     start,
				ContractType:  sm.Category,
				SlotType:      slot.SlotType,
				AccessCount:   accessCount,
				ContractAge:   contractAgeAt(start),
				SlotAge:       slotAgeAt(start),
			})
			if !observed {
				res.RightCensored++
			}
			if lt {
				res.LeftTruncatedIntervals++
			}
		}

		if leftTrunc {
			// Count the slot as left-truncated up front, regardless of
			// whether the emit loop below ends up flagging any individual
			// interval. The entry fast-path may consume the first event
			// and leave no interval with IsLeftTrunc=true, but the slot
			// still pre-existed the window and should appear in the
			// diagnostic "slots whose history we can't see the start of".
			res.LeftTruncatedSlots++
		}

		if len(inWindow) == 0 {
			res.SlotsWithNoEvents++
			emit(entry, window.End, false, leftTrunc, 0)
			return nil
		}

		// If the first in-window event coincides with the slot's entry
		// block (e.g. CreatedAt == first access, or a left-truncated slot
		// whose first visible touch is Window.Start) the "entry → first"
		// interval has zero length and carries no information. Consume
		// the event as the chain's starting reference and emit gap-time
		// intervals from there. Left-truncation attribution flows to the
		// first *emitted* interval, so if we skip the degenerate opener
		// no interval for this slot inherits the LT flag — which matches
		// reality: once we've seen a real access at window.Start, the
		// remaining gaps are honest gap times, not truncated.
		prev := entry
		accessCount := uint64(0)
		startIdx := 0
		if inWindow[0] == entry {
			prev = inWindow[0]
			accessCount = 1
			startIdx = 1
		}
		for i := startIdx; i < len(inWindow); i++ {
			blk := inWindow[i]
			lt := false
			if i == startIdx && startIdx == 0 {
				lt = leftTrunc
			}
			emit(prev, blk, true, lt, accessCount)
			prev = blk
			accessCount++
		}
		// Trailing censored interval from the final in-window event to
		// window.End. Never left-truncated (slot is already in the risk
		// set at this point). accessCount now equals the total number of
		// unique in-window access blocks for this slot, so downstream EDA
		// can recover "accesses per slot" without retaining raw events.
		emit(prev, window.End, false, false, accessCount)
		return nil
	})
	if err != nil {
		return IntervalBuildResult{}, fmt.Errorf("iterate slots: %w", err)
	}
	return res, nil
}
