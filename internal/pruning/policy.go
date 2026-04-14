// Package pruning hosts the pruning policy engine, the baseline simulator,
// and the evaluation metrics that let us compare policies on a trace.
//
// Phase 1 only ships two families of policies — NoPrune and fixed-idle-window
// rules — which together form the baseline the statistical policy in Phase 2
// will be measured against. The Policy interface is deliberately narrow so
// the simulator can stay a pure function over pre-built survival intervals
// without growing a stateful slot-tracking loop.
package pruning

import (
	"fmt"
	"strings"
)

// Policy is a pruning decision rule expressed on gap-time. Given the number
// of blocks a slot has been idle (i.e. time since its last access), the
// policy reports whether the slot is currently in the pruned state. A
// stateless duration predicate is enough because the simulator operates on
// inter-access intervals and can recompute idle durations directly.
type Policy interface {
	// Name is used for logging, reports, and CLI selection.
	Name() string

	// IsPruned reports whether a slot idle for `idle` blocks is pruned.
	// Must be monotone: once true for some idle, stays true for larger.
	IsPruned(idle uint64) bool

	// PruneThreshold is the smallest idle duration at which the slot
	// transitions to the pruned state. Returning 0 means "never prunes"
	// and short-circuits savings computation in the simulator.
	PruneThreshold() uint64
}

// NoPrune is the no-op baseline: slots live forever. Useful as the upper
// bound on storage cost and the lower bound on false-prune rate (zero).
type NoPrune struct{}

func (NoPrune) Name() string            { return "no-prune" }
func (NoPrune) IsPruned(uint64) bool    { return false }
func (NoPrune) PruneThreshold() uint64  { return 0 }

// FixedIdle prunes any slot that has been idle for at least IdleBlocks
// blocks. This is the strawman that statistical policies must beat.
type FixedIdle struct {
	Label      string
	IdleBlocks uint64
}

func (f FixedIdle) Name() string           { return f.Label }
func (f FixedIdle) IsPruned(idle uint64) bool { return idle >= f.IdleBlocks }
func (f FixedIdle) PruneThreshold() uint64  { return f.IdleBlocks }

// Phase-1 presets, matching the idle-window values in configs/default.yaml
// (30 and 90 days at 12-second blocks).
var (
	Fixed30d = FixedIdle{Label: "fixed-30d", IdleBlocks: 216_000}
	Fixed90d = FixedIdle{Label: "fixed-90d", IdleBlocks: 648_000}
)

// PolicyByName resolves a CLI flag or config key into a concrete Policy.
// It is the single entry point for `zksp simulate --policy` so that adding
// a new policy touches one place.
//
// Input is normalized before lookup: the name is lowercased and underscores
// are rewritten to dashes, so CLI idiom ("fixed-30d") and YAML idiom
// ("fixed_30d") both resolve to the same policy. Surrounding whitespace is
// trimmed.
func PolicyByName(name string) (Policy, error) {
	key := strings.ReplaceAll(strings.ToLower(strings.TrimSpace(name)), "_", "-")
	switch key {
	case "no-prune":
		return NoPrune{}, nil
	case "fixed-30d":
		return Fixed30d, nil
	case "fixed-90d":
		return Fixed90d, nil
	case "statistical":
		return nil, fmt.Errorf("policy %q: not yet implemented (Phase 2)", name)
	default:
		return nil, fmt.Errorf("unknown policy %q", name)
	}
}
