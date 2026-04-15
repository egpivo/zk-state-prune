package extractor

import (
	"context"
	"fmt"
	"math"
	"math/rand/v2"
	"sort"

	"github.com/egpivo/zk-state-prune/internal/model"
	"github.com/egpivo/zk-state-prune/internal/storage"
)

// MockConfig parameterizes the synthetic state-diff generator. All fields
// have sensible defaults via DefaultMockConfig; tests typically shrink
// NumContracts and TotalBlocks to keep runs fast.
type MockConfig struct {
	Seed uint64

	NumContracts int
	// SlotsPerContract is sampled as a Pareto with these parameters; with
	// alpha=2 and xmin=100 the population mean is 200, matching the spec.
	SlotsPerContractXmin  float64
	SlotsPerContractAlpha float64
	SlotsPerContractMax   int

	// AccessRate is per-slot per-block intensity, drawn from a Pareto.
	// Heavy tail (small alpha) means most slots are nearly dead while a
	// handful are extremely hot, mirroring real on-chain access patterns.
	AccessRateXmin  float64
	AccessRateAlpha float64
	MaxEventsPerSlot int

	IntraContractCorrelation float64
	PeriodicContractsRatio   float64
	// PeriodBlocks is the cycle length used by periodic contracts.
	PeriodBlocks uint64

	TotalBlocks uint64

	// Window is the analysis observation window. The mock itself generates
	// the full trace [0, TotalBlocks), but downstream EDA / survival only
	// sees events inside Window, which is where censoring and truncation
	// come from. Keeping it on MockConfig lets the generator hit a target
	// pre-window slot fraction deterministically.
	Window model.ObservationWindow
	// PreWindowSlotFraction controls the share of contracts whose deploy
	// block lands before Window.Start. A value of 0.3 reproduces the plan's
	// "30% slots pre-exist the observation window" assumption; those slots
	// will be flagged as left-truncated by the interval builder.
	PreWindowSlotFraction float64

	ContractTypeDistribution map[model.ContractCategory]float64
}

// Diagnostics is a small report produced by the mock extractor describing how
// many contracts / slots / events landed in emergent buckets like
// "pre-window" or "periodic". The simulator config declares the inputs; these
// are the observed outputs we can later compare against plan assumptions.
type Diagnostics struct {
	Contracts          int
	Slots              int
	Events             int
	PreWindowContracts int
	PreWindowSlots     int
	PeriodicContracts  int
	EventsInWindow     int
}

// DefaultMockConfig returns the canonical Phase-1 generator settings. They
// match configs/default.yaml.
func DefaultMockConfig() MockConfig {
	return MockConfig{
		Seed:                     42,
		NumContracts:             500,
		SlotsPerContractXmin:     100,
		SlotsPerContractAlpha:    2.0, // mean = 2*100/(2-1) = 200
		SlotsPerContractMax:      5000,
		AccessRateXmin:           1e-6,
		AccessRateAlpha:          1.5,
		MaxEventsPerSlot:         2000,
		IntraContractCorrelation: 0.7,
		PeriodicContractsRatio:   0.1,
		PeriodBlocks:             50_000,
		TotalBlocks:              1_000_000,
		Window:                   model.ObservationWindow{Start: 200_000, End: 1_000_000},
		PreWindowSlotFraction:    0.3,
		ContractTypeDistribution: map[model.ContractCategory]float64{
			model.ContractERC20:      0.40,
			model.ContractDEX:        0.15,
			model.ContractNFT:        0.20,
			model.ContractBridge:     0.05,
			model.ContractGovernance: 0.05,
			model.ContractOther:      0.15,
		},
	}
}

// MockExtractor is the deterministic synthetic extractor used by Phase-1
// development and tests.
type MockExtractor struct {
	cfg  MockConfig
	last Diagnostics
}

func NewMockExtractor(cfg MockConfig) *MockExtractor {
	return &MockExtractor{cfg: cfg}
}

// LastDiagnostics returns the diagnostics from the most recent Extract call.
// Zero value if Extract has not been called yet.
func (m *MockExtractor) LastDiagnostics() Diagnostics { return m.last }

// Extract generates contracts/slots/events according to cfg and writes them
// to db. Reusing the same Seed produces byte-identical output.
func (m *MockExtractor) Extract(ctx context.Context, db *storage.DB) error {
	if err := m.cfg.validate(); err != nil {
		return fmt.Errorf("mock config: %w", err)
	}
	// Honour the Extractor idempotency contract: a re-run on the same DB
	// must not duplicate access_events. Truncate up front so the second
	// pass sees a clean slate.
	if err := db.Reset(ctx); err != nil {
		return fmt.Errorf("reset db: %w", err)
	}
	r := rand.New(rand.NewPCG(m.cfg.Seed, m.cfg.Seed^0x9E3779B97F4A7C15))
	catSampler := newCategorySampler(m.cfg.ContractTypeDistribution)
	m.last = Diagnostics{}

	const eventBatch = 10_000
	eventBuf := make([]model.AccessEvent, 0, eventBatch)
	flush := func() error {
		if err := db.InsertAccessEvents(ctx, eventBuf); err != nil {
			return err
		}
		eventBuf = eventBuf[:0]
		return nil
	}

	for i := 0; i < m.cfg.NumContracts; i++ {
		if err := ctx.Err(); err != nil {
			return err
		}
		addr := fmt.Sprintf("0x%040x", i+1)
		category := catSampler.sample(r)
		// Mixture deploy-block sampler: with probability PreWindowSlotFraction
		// the contract is born before Window.Start (→ left-truncated from the
		// analyst's point of view); otherwise it's born inside the window so
		// we see its full history. Drawing from this mixture up-front lets
		// us hit the plan's target pre-window fraction deterministically
		// instead of having it drift with unrelated parameter tweaks.
		var deployBlock uint64
		if m.cfg.Window.Span() > 0 && r.Float64() < m.cfg.PreWindowSlotFraction {
			deployBlock = uint64(r.Float64() * float64(m.cfg.Window.Start))
		} else {
			lo := m.cfg.Window.Start
			hi := m.cfg.TotalBlocks
			if hi <= lo {
				deployBlock = lo
			} else {
				deployBlock = lo + uint64(r.Float64()*float64(hi-lo))
			}
		}
		isPeriodic := r.Float64() < m.cfg.PeriodicContractsRatio
		m.last.Contracts++
		if deployBlock < m.cfg.Window.Start {
			m.last.PreWindowContracts++
		}
		if isPeriodic {
			m.last.PeriodicContracts++
		}

		nSlots := samplePareto(r, m.cfg.SlotsPerContractXmin, m.cfg.SlotsPerContractAlpha)
		numSlots := int(math.Min(math.Round(nSlots), float64(m.cfg.SlotsPerContractMax)))
		if numSlots < 1 {
			numSlots = 1
		}

		slots := make([]model.StateSlot, 0, numSlots)
		// First pass: create slots, generate per-slot event positions.
		perSlotEvents := make([][]uint64, numSlots)
		totalEvents := uint64(0)
		for j := 0; j < numSlots; j++ {
			slotID := fmt.Sprintf("%s:%06d", addr, j)
			slotType := sampleSlotType(r, category)

			rate := samplePareto(r, m.cfg.AccessRateXmin, m.cfg.AccessRateAlpha)
			horizon := float64(m.cfg.TotalBlocks - deployBlock)
			expected := rate * horizon
			n := samplePoisson(r, expected)
			if n > m.cfg.MaxEventsPerSlot {
				n = m.cfg.MaxEventsPerSlot
			}

			blocks := make([]uint64, 0, n)
			for k := 0; k < n; k++ {
				var b uint64
				if isPeriodic {
					b = samplePeriodicBlock(r, deployBlock, m.cfg.TotalBlocks, m.cfg.PeriodBlocks)
				} else {
					span := m.cfg.TotalBlocks - deployBlock
					b = deployBlock + uint64(r.Float64()*float64(span))
				}
				blocks = append(blocks, b)
			}
			perSlotEvents[j] = blocks
			totalEvents += uint64(n)

			// Slot creation block is the *first* event the slot ever
			// receives, not the contract's deploy block — a storage
			// slot only exists once it has been written to. Modelling
			// it this way makes ContractAge and SlotAge naturally
			// distinct (ContractAge ≥ SlotAge), which is what the Cox
			// fit needs to keep both predictors non-collinear.
			createdAt := deployBlock
			lastAccess := deployBlock
			if len(blocks) > 0 {
				createdAt = minU64(blocks)
				lastAccess = maxU64(blocks)
			}
			slots = append(slots, model.StateSlot{
				SlotID:       slotID,
				ContractAddr: addr,
				SlotIndex:    uint64(j),
				SlotType:     slotType,
				CreatedAt:    createdAt,
				LastAccess:   lastAccess,
				AccessCount:  uint64(len(blocks)),
				IsActive:     true,
			})
			m.last.Slots++
			if deployBlock < m.cfg.Window.Start {
				m.last.PreWindowSlots++
			}
		}

		// Second pass: inject co-accesses. With probability rho, every event
		// pulls in a buddy event on a randomly-chosen sibling slot at the
		// same block. This produces the intra-contract clustering that the
		// spatial analysis later looks for.
		if numSlots > 1 && m.cfg.IntraContractCorrelation > 0 {
			for j := 0; j < numSlots; j++ {
				orig := perSlotEvents[j]
				for _, b := range orig {
					if r.Float64() >= m.cfg.IntraContractCorrelation {
						continue
					}
					buddy := r.IntN(numSlots)
					if buddy == j {
						buddy = (buddy + 1) % numSlots
					}
					perSlotEvents[buddy] = append(perSlotEvents[buddy], b)
					slots[buddy].AccessCount++
					if b > slots[buddy].LastAccess {
						slots[buddy].LastAccess = b
					}
				}
			}
		}

		// Persist contract + slots, then stream events in batches.
		if err := db.UpsertContract(ctx, model.ContractMeta{
			Address:      addr,
			ContractType: category,
			DeployBlock:  deployBlock,
			TotalSlots:   uint64(numSlots),
			ActiveSlots:  uint64(numSlots),
		}); err != nil {
			return err
		}
		for _, s := range slots {
			if err := db.UpsertSlot(ctx, s); err != nil {
				return err
			}
		}

		for j := 0; j < numSlots; j++ {
			blocks := perSlotEvents[j]
			sort.Slice(blocks, func(a, b int) bool { return blocks[a] < blocks[b] })
			for idx, b := range blocks {
				at := model.AccessRead
				if idx == 0 {
					at = model.AccessWrite // first touch is the SSTORE
				} else if r.Float64() < 0.2 {
					at = model.AccessWrite
				}
				eventBuf = append(eventBuf, model.AccessEvent{
					SlotID:      slots[j].SlotID,
					BlockNumber: b,
					AccessType:  at,
					TxHash:      fmt.Sprintf("0x%064x", uint64(i)*1_000_000+uint64(j)*1000+uint64(idx)),
				})
				m.last.Events++
				if m.cfg.Window.Contains(b) {
					m.last.EventsInWindow++
				}
				if len(eventBuf) >= eventBatch {
					if err := flush(); err != nil {
						return err
					}
				}
			}
		}
		_ = totalEvents
	}
	if len(eventBuf) > 0 {
		if err := flush(); err != nil {
			return err
		}
	}
	return nil
}

func (c MockConfig) validate() error {
	if c.NumContracts <= 0 {
		return fmt.Errorf("NumContracts must be > 0")
	}
	if c.TotalBlocks == 0 {
		return fmt.Errorf("TotalBlocks must be > 0")
	}
	if c.SlotsPerContractAlpha <= 1 {
		return fmt.Errorf("SlotsPerContractAlpha must be > 1 for finite mean")
	}
	if c.AccessRateAlpha <= 0 {
		return fmt.Errorf("AccessRateAlpha must be > 0")
	}
	if c.Window.End == 0 || c.Window.End > c.TotalBlocks || c.Window.Start >= c.Window.End {
		return fmt.Errorf("Window %v must satisfy 0 <= Start < End <= TotalBlocks(%d)", c.Window, c.TotalBlocks)
	}
	sum := 0.0
	for _, w := range c.ContractTypeDistribution {
		sum += w
	}
	if sum <= 0 {
		return fmt.Errorf("ContractTypeDistribution is empty")
	}
	return nil
}

// samplePareto draws X with P(X >= x) = (x/xmin)^(-alpha), x >= xmin.
func samplePareto(r *rand.Rand, xmin, alpha float64) float64 {
	u := r.Float64()
	if u < 1e-12 {
		u = 1e-12
	}
	return xmin * math.Pow(u, -1.0/alpha)
}

// samplePoisson returns a Poisson(lambda) variate. Knuth's multiplicative
// algorithm is exact for small lambda; for large lambda we fall back to a
// normal approximation, which is plenty for synthetic data.
func samplePoisson(r *rand.Rand, lambda float64) int {
	if lambda <= 0 {
		return 0
	}
	if lambda < 30 {
		L := math.Exp(-lambda)
		k := 0
		p := 1.0
		for {
			k++
			p *= r.Float64()
			if p <= L {
				return k - 1
			}
		}
	}
	n := r.NormFloat64()*math.Sqrt(lambda) + lambda
	if n < 0 {
		return 0
	}
	return int(math.Round(n))
}

// samplePeriodicBlock picks a block number biased toward the peaks of a
// sinusoidal envelope of length periodBlocks. Implemented by rejection
// sampling against the envelope (1 + cos)/2.
func samplePeriodicBlock(r *rand.Rand, start, total, periodBlocks uint64) uint64 {
	span := total - start
	if span == 0 || periodBlocks == 0 {
		return start
	}
	for i := 0; i < 16; i++ {
		b := start + uint64(r.Float64()*float64(span))
		phase := 2 * math.Pi * float64(b%periodBlocks) / float64(periodBlocks)
		envelope := 0.5 * (1 + math.Cos(phase))
		if r.Float64() < envelope {
			return b
		}
	}
	return start + uint64(r.Float64()*float64(span))
}

// sampleSlotType emits a slot-type chosen from a category-specific mix.
// Hard-coded weights keep the table inspectable; tweak as the model demands
// more nuance.
func sampleSlotType(r *rand.Rand, c model.ContractCategory) model.SlotType {
	type w struct {
		t model.SlotType
		p float64
	}
	var dist []w
	switch c {
	case model.ContractERC20:
		dist = []w{{model.SlotTypeBalance, 0.7}, {model.SlotTypeMapping, 0.2}, {model.SlotTypeFixed, 0.1}}
	case model.ContractDEX:
		dist = []w{{model.SlotTypeMapping, 0.5}, {model.SlotTypeArray, 0.2}, {model.SlotTypeBalance, 0.2}, {model.SlotTypeFixed, 0.1}}
	case model.ContractNFT:
		dist = []w{{model.SlotTypeMapping, 0.7}, {model.SlotTypeArray, 0.2}, {model.SlotTypeFixed, 0.1}}
	case model.ContractBridge:
		dist = []w{{model.SlotTypeMapping, 0.5}, {model.SlotTypeBalance, 0.3}, {model.SlotTypeFixed, 0.2}}
	case model.ContractGovernance:
		dist = []w{{model.SlotTypeMapping, 0.4}, {model.SlotTypeArray, 0.3}, {model.SlotTypeFixed, 0.3}}
	default:
		dist = []w{{model.SlotTypeMapping, 0.4}, {model.SlotTypeArray, 0.2}, {model.SlotTypeFixed, 0.2}, {model.SlotTypeBalance, 0.2}}
	}
	u := r.Float64()
	cum := 0.0
	for _, e := range dist {
		cum += e.p
		if u < cum {
			return e.t
		}
	}
	return dist[len(dist)-1].t
}

// categorySampler turns a weight map into an alias-free CDF lookup.
type categorySampler struct {
	cats []model.ContractCategory
	cdf  []float64
}

func newCategorySampler(weights map[model.ContractCategory]float64) *categorySampler {
	cs := &categorySampler{}
	sum := 0.0
	for _, w := range weights {
		sum += w
	}
	// Stable iteration order so the same seed gives the same sequence.
	keys := make([]model.ContractCategory, 0, len(weights))
	for k := range weights {
		keys = append(keys, k)
	}
	sort.Slice(keys, func(i, j int) bool { return keys[i] < keys[j] })
	cum := 0.0
	for _, k := range keys {
		cum += weights[k] / sum
		cs.cats = append(cs.cats, k)
		cs.cdf = append(cs.cdf, cum)
	}
	return cs
}

func (c *categorySampler) sample(r *rand.Rand) model.ContractCategory {
	u := r.Float64()
	for i, p := range c.cdf {
		if u < p {
			return c.cats[i]
		}
	}
	return c.cats[len(c.cats)-1]
}

func maxU64(xs []uint64) uint64 {
	m := xs[0]
	for _, x := range xs[1:] {
		if x > m {
			m = x
		}
	}
	return m
}

func minU64(xs []uint64) uint64 {
	m := xs[0]
	for _, x := range xs[1:] {
		if x < m {
			m = x
		}
	}
	return m
}
