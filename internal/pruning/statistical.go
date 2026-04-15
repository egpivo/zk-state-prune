package pruning

import (
	"fmt"

	"github.com/egpivo/zk-state-prune/internal/model"
)

// CondAccessProbFunc returns p(u) = P(slot is accessed within the next
// τ blocks | it has already been idle for `idle` blocks since the
// previous access) for the given interval. The function owns the
// horizon τ internally (typically it reads `CalibratedModel.Tau` from
// the model it closes over), which keeps the policy API free of Cox /
// calibration-specific types.
//
// Callers that want the robust upper-bound rule should have their
// predict function add the conformal ε to the raw conditional
// probability and clip to [0, 1] before returning.
type CondAccessProbFunc func(it model.InterAccessInterval, idle float64) float64

// StatisticalPolicy is the cost-aware tiering policy. It is the
// plan-faithful realization of
//
//	d_i* = argmin_d  c(d) + ℓ(d) · p_i(τ)
//
// where, for a slot that has been idle for u blocks since its last
// access, p_i(τ) = 1 − S(u+τ) / S(u) is the **conditional** probability
// of an access landing in the next τ blocks. The surrogate reduces to
// a single probability threshold
//
//	p* := (RAMUnitCost · τ) / MissPenalty
//
// and the per-slot decision becomes "demote the slot at the first
// idle duration u where p(u) < p*". HotBlocks(it) searches for that
// crossover T* along the interval and returns min(it.Duration, T*);
// the simulator treats the slot as hot for [0, T*] and cold for
// [T*, Duration], which is the natural mid-interval transition the
// plan asks for. Slots that never cross p* stay hot for the entire
// interval; slots that start below p* (u=0) are demoted immediately.
//
// The per-interval granularity is still a simulation artifact — real
// policies decide at every block — but the search captures the key
// behaviour the binary Phase-1-style rule missed: a slot that starts
// the interval hot can still be demoted later in the same interval
// once its idle duration makes the conditional probability fall
// below p*.
type StatisticalPolicy struct {
	label   string
	predict CondAccessProbFunc
	tau     float64
	costs   CostParams
	pStar   float64
}

// statisticalSearchSamples is the number of uniformly-spaced idle
// durations HotBlocks probes to find the demotion crossover T*.
// Twenty samples per interval keeps the cost bounded (each sample is
// one baseline-hazard lookup) and is fine grained enough to land
// within 5% of Duration for most realistic curves.
const statisticalSearchSamples = 20

// NewStatisticalPolicy wires a conditional prediction function into a
// Policy. Stateless and safe for concurrent use provided the caller's
// predict function is too.
func NewStatisticalPolicy(name string, predict CondAccessProbFunc, tau float64, costs CostParams) (*StatisticalPolicy, error) {
	if name == "" {
		name = "statistical"
	}
	if predict == nil {
		return nil, fmt.Errorf("NewStatisticalPolicy: nil predict function")
	}
	if tau <= 0 {
		return nil, fmt.Errorf("NewStatisticalPolicy: tau must be > 0, got %v", tau)
	}
	if costs.MissPenalty <= 0 {
		return nil, fmt.Errorf("NewStatisticalPolicy: MissPenalty must be > 0, got %v", costs.MissPenalty)
	}
	pStar := costs.RAMUnitCost * tau / costs.MissPenalty
	if pStar < 0 {
		pStar = 0
	}
	if pStar > 1 {
		pStar = 1
	}
	return &StatisticalPolicy{
		label:   name,
		predict: predict,
		tau:     tau,
		costs:   costs,
		pStar:   pStar,
	}, nil
}

// Name reports the policy label, used by SimResult and CLI printers.
func (s *StatisticalPolicy) Name() string { return s.label }

// PStar exposes the demotion threshold for diagnostics and CLI output.
func (s *StatisticalPolicy) PStar() float64 { return s.pStar }

// Tau exposes the configured horizon.
func (s *StatisticalPolicy) Tau() float64 { return s.tau }

// HotBlocks searches the interval [0, Duration] for the first idle
// duration T* at which the conditional access probability drops below
// p*, and returns min(Duration, T*). The search samples
// statisticalSearchSamples + 1 equally-spaced points inclusive of
// both endpoints; the first sample to satisfy p < p* is taken as T*.
//
// If the probability is always above p* (hot-dominant slot) the slot
// stays hot for the full interval. If the probability at u=0 is
// already below p* (cold-dominant slot) the slot is demoted at the
// start — recovering the Phase-2 "binary" shortcut as a special case.
func (s *StatisticalPolicy) HotBlocks(it model.InterAccessInterval) uint64 {
	if s.predict == nil || it.Duration == 0 {
		return it.Duration
	}
	span := float64(it.Duration)
	N := statisticalSearchSamples
	for i := 0; i <= N; i++ {
		u := span * float64(i) / float64(N)
		p := s.predict(it, u)
		if p < s.pStar {
			if u >= span {
				return it.Duration
			}
			return uint64(u)
		}
	}
	return it.Duration
}
