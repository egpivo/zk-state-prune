// Package sim hosts the tiering policy engine, the baseline runner,
// and the evaluation metrics used to compare policies on a trace.
//
// Tiering here means the broader hot/cold placement decision:
// each slot is classified into a tier and the runner scores how well a
// policy trades RAM (the cost of keeping a slot hot) against the miss
// penalty (the cost of fetching a cold slot when an access actually lands
// on it). The cost-aware decision rule a future statistical policy will
// minimize is the per-slot surrogate
//
//	d_i* = argmin_d  c(d) + ℓ(d) · p_i(τ)
//
// where c(d) is the per-tier holding cost, ℓ(d) the miss penalty for that
// tier, and p_i(τ) the probability the slot is accessed within the
// horizon τ.
//
// Phase 1 only ships two families of policies — NoPrune and fixed-idle-window
// rules — which together form the baseline the statistical policy in Phase 2
// will be measured against. The Policy interface is deliberately narrow so
// the simulator can stay a pure function over pre-built survival intervals
// without growing a stateful slot-tracking loop.
package sim

import (
	"fmt"
	"strings"

	"github.com/egpivo/zk-state-prune/internal/domain"
)

// Policy is a tiering decision rule. Given an inter-access interval, the
// policy returns the number of blocks the slot was held in the hot tier
// during that interval. The remainder of the interval — Duration minus
// HotBlocks — is the cold-tier residency, which the simulator turns into
// either a saved hot slot-block (for non-observed tails) or a miss (for
// observed tails ending in a real access).
//
// Returning HotBlocks rather than a "is pruned?" boolean lets the
// simulator handle three policy families uniformly:
//
//   - NoPrune: HotBlocks = Duration (never demotes)
//   - FixedIdle: HotBlocks = min(Duration, IdleBlocks) (demotes after a
//     fixed wall-clock idle)
//   - Statistical: HotBlocks ∈ {0, Duration} based on the per-interval
//     cost-aware decision
//
// Phase-1's IsPruned / PruneThreshold pair is gone — they could not
// express the statistical case, where the demotion threshold depends on
// covariates evaluated at IntervalStart.
type Policy interface {
	// Name is used for logging, reports, and CLI selection.
	Name() string

	// HotBlocks returns the number of blocks the slot stayed in the hot
	// tier during this interval. Must be in [0, it.Duration].
	HotBlocks(it domain.InterAccessInterval) uint64
}

// NoPrune is the all-hot baseline: slots stay in the hot tier forever.
// Useful as the upper bound on RAM ratio and the lower bound on miss
// rate (zero).
type NoPrune struct{}

func (NoPrune) Name() string                                   { return "no-prune" }
func (NoPrune) HotBlocks(it domain.InterAccessInterval) uint64 { return it.Duration }

// FixedIdle demotes any slot that has been idle for at least IdleBlocks
// blocks. This is the strawman that statistical policies must beat.
type FixedIdle struct {
	Label      string
	IdleBlocks uint64
}

func (f FixedIdle) Name() string { return f.Label }

// HotBlocks: the slot is hot for the first IdleBlocks of the interval
// and cold thereafter. If the interval is shorter than IdleBlocks the
// slot stayed hot for its entire span.
func (f FixedIdle) HotBlocks(it domain.InterAccessInterval) uint64 {
	if it.Duration <= f.IdleBlocks {
		return it.Duration
	}
	return f.IdleBlocks
}

// Phase-1 presets, matching the idle-window values in configs/default.yaml
// (30 and 90 days at 12-second blocks).
var (
	Fixed30d = FixedIdle{Label: "fixed-30d", IdleBlocks: 216_000}
	Fixed90d = FixedIdle{Label: "fixed-90d", IdleBlocks: 648_000}
)

// PolicyDeps bundles optional dependencies that PolicyByName needs to
// resolve certain names. Stateless / no-op policies ignore it; the
// statistical policy requires a pre-built StatisticalPolicy because
// there is no way to materialize a fitted Cox + calibrated model from
// a name alone.
//
// The CLI builds a fresh PolicyDeps per invocation: pre-run the
// statistical fit pipeline via buildStatisticalPolicy if the user
// asked for "statistical", then pass it here; otherwise leave
// Statistical nil and PolicyByName handles fixed / no-prune like
// before.
type PolicyDeps struct {
	Statistical *StatisticalPolicy
}

// PolicyByName resolves a CLI flag or config key into a concrete Policy.
// It is the single entry point for `zksp simulate --policy` so that
// adding a new policy touches one place.
//
// Input is normalized before lookup: the name is lowercased and
// underscores are rewritten to dashes, so CLI idiom ("fixed-30d") and
// YAML idiom ("fixed_30d") both resolve to the same policy. Surrounding
// whitespace is trimmed.
func PolicyByName(name string, deps PolicyDeps) (Policy, error) {
	key := strings.ReplaceAll(strings.ToLower(strings.TrimSpace(name)), "_", "-")
	switch key {
	case "no-prune":
		return NoPrune{}, nil
	case "fixed-30d":
		return Fixed30d, nil
	case "fixed-90d":
		return Fixed90d, nil
	case "statistical", "statistical-robust":
		if deps.Statistical == nil {
			return nil, fmt.Errorf("policy %q: PolicyDeps.Statistical must be a pre-built StatisticalPolicy (use NewStatisticalPolicy after fit + calibrate)", name)
		}
		return deps.Statistical, nil
	default:
		return nil, fmt.Errorf("unknown policy %q", name)
	}
}
