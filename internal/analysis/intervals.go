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
// report needs. RightCensored / LeftTruncated are counts, not rates, so the
// consumer can format them as either.
type IntervalBuildResult struct {
	Intervals     []model.InterAccessInterval
	RightCensored int
	LeftTruncated int
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

		// The extractor inserts events in any order across slots but we
		// asked storage to sort by (slot_id, block_number), so the per-slot
		// stream is already ordered. Still, be defensive about duplicates
		// at the boundary.
		inWindow := make([]uint64, 0, len(events))
		for _, e := range events {
			if !window.Contains(e.BlockNumber) {
				continue
			}
			if e.BlockNumber < entry {
				// Event precedes the slot's effective entry (can happen
				// if CreatedAt was wrong); clamp.
				continue
			}
			inWindow = append(inWindow, e.BlockNumber)
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
				res.LeftTruncated++
			}
		}

		if len(inWindow) == 0 {
			res.SlotsWithNoEvents++
			emit(entry, window.End, false, leftTrunc, 0)
			return nil
		}

		// First interval: entry → first event. Left-truncated iff the slot
		// pre-existed the window.
		prev := entry
		accessCount := uint64(0)
		for i, blk := range inWindow {
			lt := false
			if i == 0 {
				lt = leftTrunc
			}
			emit(prev, blk, true, lt, accessCount)
			prev = blk
			accessCount++
		}
		// Trailing censored interval from the final in-window event to
		// window.End. Never left-truncated (slot is already in the risk
		// set at this point).
		emit(prev, window.End, false, false, accessCount)
		return nil
	})
	if err != nil {
		return IntervalBuildResult{}, fmt.Errorf("iterate slots: %w", err)
	}
	return res, nil
}
