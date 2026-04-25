package extractor

import (
	"context"
	"fmt"
	"log/slog"
	"strconv"
	"strings"

	"github.com/egpivo/zk-state-prune/internal/domain"
	"github.com/egpivo/zk-state-prune/internal/storage"
)

// StateDiffExtractor walks blocks via debug_traceBlockByNumber +
// prestateTracer (diffMode) and lands every storage touch — read AND
// write, Transfer-emitting or not — as an access_event. This is the
// "real" extractor the Transfer-log surrogate (RPCExtractor) was
// always a stand-in for.
//
// Coverage:
//   - prestateTracer's `pre` block lists every slot a tx READ or
//     wrote (it's the state required to replay the tx).
//   - The `post` block lists slots whose value CHANGED.
//   - A slot in `pre` ∩ `post` is a write; a slot in `pre` only is a
//     read; a slot in `post` only doesn't happen (writes always have
//     a pre-state).
//
// The result: ObservesReads and ObservesNonTransferWrite both hold,
// and slot_id is "contract:<32-byte-slotKey>" — the actual EVM
// storage key, not a holder-derived surrogate.
//
// Event granularity: events land per (slot, block), not per (slot,
// tx). Per-block aggregation matches what the tiering policy cares
// about (was this slot touched in this block?) and avoids
// duplicating rows for slots multiple txs in the same block touch.
// Promotion rule: any write in the block supersedes a read; the
// first tx_hash that touched the slot is kept on the persisted row
// for traceability.
//
// Idempotency / resume: same schema_meta high-water mark as
// RPCExtractor (RPCHighWaterKey). They are mutually exclusive on a
// given DB — switching `--source` between rpc and statediff requires
// `--force` (which calls ClearRPCState).
//
// Contract classification: prestateTracer doesn't carry tx input
// data, so the 4-byte function-selector heuristic that drives
// stratification needs a separate signal. We pull it from
// eth_getBlockByNumber(blockHash, true), which returns every tx's
// input in one call — substantially cheaper than the alternative
// (callTracer fan-out, +1 expensive trace per tx). When the selector
// pipeline can't classify a contract, the run falls back to a
// bytecode fingerprint via eth_getCode (cached per address). A
// contract that fails both pipelines lands in ContractOther; the
// otherCategoryWarnRatio guardrail surfaces a high Other-rate as a
// loud warning by default, and when StrictCategories is enabled
// (CLI: --strict-categories), it errors out the run instead — use
// strict in CI / scheduled jobs to catch classifier regressions.
//
// Cost: debug_trace* is the most expensive RPC method by 100×–500×
// vs eth_getBlockByNumber. Per-block, this extractor sends:
//   - 1× debug_traceBlockByNumber (the trace)
//   - 1× eth_getBlockByNumber(true) (txs+inputs for classification)
//   - 0..N× eth_getCode (bytecode fallback, only first time per addr)
//
// Public chain endpoints rarely expose debug_trace*; production
// runs need an archive-capable node (Alchemy Growth / QuickNode
// archive / self-hosted Erigon). The extractor surfaces a clear
// error when the endpoint refuses the method instead of silently
// falling back to the surrogate.
type StateDiffExtractor struct {
	cfg   RPCConfig
	last  StateDiffDiagnostics
	reqID int
}

// StateDiffDiagnostics mirrors RPCDiagnostics' shape so CLI / tests
// have a uniform "what did this run produce" surface, but tracks the
// state-diff-specific counters that don't exist on the surrogate
// (read-vs-write split, Other-category guardrail).
type StateDiffDiagnostics struct {
	BlocksRequested int
	BlocksFetched   int

	// StorageTouches is the union over all tx traces in all blocks
	// of distinct (contract, slot_key, block) slot touches. Equals
	// StorageReads + StorageWrites by construction.
	StorageTouches int
	StorageReads   int
	StorageWrites  int

	ContractsCreated int
	SlotsCreated     int
	EventsAttempted  int
	EventsPersisted  int

	// OtherCategoryContracts is the number of contracts the
	// 4-byte-signature classifier could not place; the run aborts
	// (or warns loudly, depending on policy) if this exceeds
	// otherCategoryWarnRatio of ContractsCreated.
	OtherCategoryContracts int

	StartBlock uint64
	EndBlock   uint64
}

// otherCategoryWarnRatio is the upper bound on (Other / total)
// classification rate before we treat the run as suspect — per
// decision #6 in the Phase 4 plan, statediff must not silently
// default everything to ContractOther because that degrades the
// stratified Cox baseline. 20% is a starting point; tighten or
// loosen with experience.
const otherCategoryWarnRatio = 0.2

// NewStateDiffExtractor validates cfg and returns a ready-to-run
// extractor. Mirrors NewRPCExtractor: same defaults, same validation
// surface, so the CLI doesn't need to special-case which extractor
// kind it built.
func NewStateDiffExtractor(cfg RPCConfig) (*StateDiffExtractor, error) {
	if cfg.Endpoint == "" {
		return nil, fmt.Errorf("StateDiffExtractor: Endpoint is required")
	}
	if cfg.End < cfg.Start {
		return nil, fmt.Errorf("StateDiffExtractor: End(%d) < Start(%d)", cfg.End, cfg.Start)
	}
	if cfg.HTTPClient == nil {
		cfg.HTTPClient = DefaultRPCConfig().HTTPClient
	}
	if cfg.BatchSize <= 0 {
		cfg.BatchSize = 10_000
	}
	return &StateDiffExtractor{cfg: cfg}, nil
}

// LastDiagnostics returns the diagnostics from the most recent
// Extract call. Zero value if Extract has not been called yet.
func (e *StateDiffExtractor) LastDiagnostics() StateDiffDiagnostics { return e.last }

// Capability declares full state-diff coverage: every SLOAD and
// every SSTORE, regardless of whether the writing tx emitted a
// Transfer event. slot_id is "contract:<32-byte-slotKey>", the real
// EVM storage key — collision-free across contracts because we
// concatenate the contract address.
func (*StateDiffExtractor) Capability() Capability {
	return Capability{
		Source:                   "statediff",
		ObservesReads:            true,
		ObservesNonTransferWrite: true,
		SlotIDForm:               "contract:slotkey (real state-diff)",
	}
}

// Extract walks blocks in [Start, End], fetching each block's
// per-tx prestateTracer trace, parsing the (pre, post) state diff,
// and emitting one AccessEvent per (slot, block) touch. Resume is
// driven by RPCHighWaterKey — same as RPCExtractor, so a forced
// re-run goes through `--force` + ClearRPCState.
func (e *StateDiffExtractor) Extract(ctx context.Context, db *storage.DB) error {
	e.last = StateDiffDiagnostics{StartBlock: e.cfg.Start, EndBlock: e.cfg.End}

	start := e.cfg.Start
	if hw, ok, err := readHighWater(ctx, db); err != nil {
		return fmt.Errorf("read high-water mark: %w", err)
	} else if ok && hw+1 > start {
		slog.Info("resuming statediff extract from high-water mark", "from", hw+1)
		start = hw + 1
	}

	state := &statediffRunState{
		contracts: make(map[string]*contractState),
		slots:     make(map[string]*slotState),
		eventBuf:  make([]domain.AccessEvent, 0, e.cfg.BatchSize),
	}
	flush := func() error { return state.flush(ctx, db, &e.last) }

	for blockNum := start; blockNum <= e.cfg.End; blockNum++ {
		if err := ctx.Err(); err != nil {
			return err
		}
		if err := e.processBlock(ctx, db, blockNum, state, flush); err != nil {
			return err
		}
		// Flush before advancing high-water — same invariant as RPC
		// extractor: the mark certifies durability of earlier blocks.
		if err := flush(); err != nil {
			return err
		}
		if err := writeHighWater(ctx, db, blockNum); err != nil {
			return fmt.Errorf("write high-water mark: %w", err)
		}
	}

	// Decision-#6 guardrail. If too many contracts ended up in
	// ContractOther, the stratified Cox baseline downstream will
	// silently degrade to unstratified. Default behaviour is a loud
	// warning so dev iteration on the classifier isn't blocked by
	// every borderline run; opt in to StrictCategories (CLI:
	// --strict-categories) when you want CI / scheduled jobs to
	// fail the run instead of pretending it succeeded.
	if total := e.last.ContractsCreated; total > 0 {
		ratio := float64(e.last.OtherCategoryContracts) / float64(total)
		if ratio > otherCategoryWarnRatio {
			if e.cfg.StrictCategories {
				return fmt.Errorf("statediff: %d/%d (%.0f%%) contracts could not be classified — exceeds threshold %.0f%% (--strict-categories enabled)",
					e.last.OtherCategoryContracts, total, ratio*100, otherCategoryWarnRatio*100)
			}
			slog.Warn("statediff: high ContractOther rate — classifier may need strengthening",
				"other", e.last.OtherCategoryContracts,
				"total", total,
				"ratio", ratio,
				"threshold", otherCategoryWarnRatio)
		}
	}

	return nil
}

// statediffRunState mirrors rpcRunState — same first-seen maps,
// same event buffer. Kept as a separate type only for clarity in
// stack traces; the structure is identical.
type statediffRunState struct {
	contracts map[string]*contractState
	slots     map[string]*slotState
	eventBuf  []domain.AccessEvent
}

func (s *statediffRunState) flush(ctx context.Context, db *storage.DB, diag *StateDiffDiagnostics) error {
	if len(s.eventBuf) == 0 {
		return nil
	}
	inserted, err := db.InsertAccessEvents(ctx, s.eventBuf)
	if err != nil {
		return err
	}
	diag.EventsPersisted += int(inserted)
	diag.EventsAttempted += len(s.eventBuf)
	s.eventBuf = s.eventBuf[:0]
	return nil
}

// processBlock fetches one block's prestateTracer trace and emits
// (slot, event) rows for every storage touch.
func (e *StateDiffExtractor) processBlock(
	ctx context.Context,
	db *storage.DB,
	blockNum uint64,
	state *statediffRunState,
	flush func() error,
) error {
	e.last.BlocksRequested++
	blk, err := e.fetchBlockWithTransactions(ctx, blockNum)
	if err != nil {
		return fmt.Errorf("block %d: %w", blockNum, err)
	}
	traces, err := e.fetchBlockStateDiff(ctx, blockNum)
	if err != nil {
		return fmt.Errorf("block %d: %w", blockNum, err)
	}
	e.last.BlocksFetched++

	// Aggregate the 4-byte signatures seen in this block's txs so
	// we can hand each contract a category at first-seen time.
	sigByContract := collectFunctionSignatures(blk.Transactions)

	// Block-level aggregation: collapse per-tx touches into one
	// (contract, slot_key) entry per block. Tiering policy cares
	// whether a slot was touched in this block, not by which of the
	// (potentially many) txs that touched it. Without this collapse
	// the access_events table inflates by avg-txs-per-touched-slot
	// (~3-5× on Scroll's busier blocks), and downstream BuildIntervals
	// has to dedup anyway. Promotion rule: any write in the block
	// supersedes a read; the first tx_hash that touched the slot is
	// kept on the row for traceability.
	collapsed := map[blockSlotKey]blockTouch{}
	for _, tr := range traces {
		for _, t := range parseStateDiff(tr.Result) {
			key := blockSlotKey{
				contract: strings.ToLower(t.contract),
				slotKey:  strings.ToLower(t.slotKey),
			}
			existing, ok := collapsed[key]
			switch {
			case !ok:
				collapsed[key] = blockTouch{access: t.access, txHash: tr.TxHash}
			case existing.access == domain.AccessRead && t.access == domain.AccessWrite:
				collapsed[key] = blockTouch{access: domain.AccessWrite, txHash: existing.txHash}
			}
		}
	}

	for key, touch := range collapsed {
		e.last.StorageTouches++
		if touch.access == domain.AccessRead {
			e.last.StorageReads++
		} else {
			e.last.StorageWrites++
		}
		if err := e.handleTouch(ctx, db, blockNum, touch.txHash, slotTouch{
			contract: key.contract,
			slotKey:  key.slotKey,
			access:   touch.access,
		}, sigByContract, state, flush); err != nil {
			return err
		}
	}
	return nil
}

// blockSlotKey + blockTouch are the per-block aggregation map's
// key / value types. Defined at package level so processBlock's
// dedup logic doesn't anonymous-struct itself; downstream tests
// can assert behaviour against the same shape.
type blockSlotKey struct {
	contract string
	slotKey  string
}

type blockTouch struct {
	access domain.AccessType
	txHash string
}

// handleTouch upserts the contract + slot (if first-seen this run)
// and appends the AccessEvent to the buffer, flushing on overflow.
// Order matters: contract → slot → event, otherwise the FK on
// access_events fires inside InsertAccessEvents and the whole batch
// rolls back.
func (e *StateDiffExtractor) handleTouch(
	ctx context.Context,
	db *storage.DB,
	blockNum uint64,
	txHash string,
	t slotTouch,
	sigByContract map[string][]string,
	state *statediffRunState,
	flush func() error,
) error {
	contractKey := strings.ToLower(t.contract)
	if err := e.upsertContractIfNew(ctx, db, contractKey, blockNum, sigByContract[contractKey], state); err != nil {
		return err
	}
	if err := e.upsertSlotIfNew(ctx, db, contractKey, strings.ToLower(t.slotKey), blockNum, state); err != nil {
		return err
	}
	state.eventBuf = append(state.eventBuf, domain.AccessEvent{
		SlotID:      slotIDForStateDiff(contractKey, t.slotKey),
		BlockNumber: blockNum,
		AccessType:  t.access,
		TxHash:      txHash,
	})
	if len(state.eventBuf) >= e.cfg.BatchSize {
		if err := flush(); err != nil {
			return err
		}
	}
	return nil
}

func (e *StateDiffExtractor) upsertContractIfNew(
	ctx context.Context,
	db *storage.DB,
	addr string,
	blockNum uint64,
	signatures []string,
	state *statediffRunState,
) error {
	if state.contracts[addr] != nil {
		return nil
	}
	cat := classifyByFunctionSignatures(signatures)
	if cat == domain.ContractOther {
		if bc, ok, err := e.classifyByBytecode(ctx, addr, blockNum); err != nil {
			return err
		} else if ok {
			cat = bc
		}
	}
	c := &contractState{
		address:     addr,
		category:    cat,
		deployBlock: blockNum,
	}
	state.contracts[addr] = c
	e.last.ContractsCreated++
	if cat == domain.ContractOther {
		e.last.OtherCategoryContracts++
	}
	if err := db.UpsertContract(ctx, domain.ContractMeta{
		Address:      c.address,
		ContractType: c.category,
		DeployBlock:  c.deployBlock,
	}); err != nil {
		return fmt.Errorf("upsert contract %s: %w", c.address, err)
	}
	return nil
}

func (e *StateDiffExtractor) upsertSlotIfNew(
	ctx context.Context,
	db *storage.DB,
	contractAddr, slotKey string,
	blockNum uint64,
	state *statediffRunState,
) error {
	slotID := slotIDForStateDiff(contractAddr, slotKey)
	if state.slots[slotID] != nil {
		return nil
	}
	s := &slotState{
		slotID:       slotID,
		contractAddr: contractAddr,
		slotIndex:    slotIndexFromKey(slotKey),
		// Statediff doesn't know the slot's semantic role
		// (balance / mapping / array). Default to Unknown rather
		// than guessing; downstream stratification keys off
		// ContractCategory anyway, which IS classified.
		slotType:  domain.SlotTypeUnknown,
		createdAt: blockNum,
	}
	state.slots[slotID] = s
	e.last.SlotsCreated++
	if err := db.UpsertSlot(ctx, domain.StateSlot{
		SlotID:       s.slotID,
		ContractAddr: s.contractAddr,
		SlotIndex:    s.slotIndex,
		SlotType:     s.slotType,
		CreatedAt:    s.createdAt,
		LastAccess:   blockNum,
		AccessCount:  0,
		IsActive:     true,
	}); err != nil {
		return fmt.Errorf("upsert slot %s: %w", s.slotID, err)
	}
	return nil
}

// ---- slot_id minting -------------------------------------------------

// slotIDForStateDiff mints "<contract>:<slotKey>" — the real EVM
// storage key, lowercased and 0x-prefixed by convention. Different
// shape from the Transfer-log surrogate (contract:holderTopic) on
// purpose: a DB populated from this extractor must NOT collide with
// rows from the surrogate.
func slotIDForStateDiff(contractAddr, slotKey string) string {
	return strings.ToLower(contractAddr) + ":" + strings.ToLower(slotKey)
}

// slotIndexFromKey takes the same last-15-nibbles approach as the
// surrogate's slotIndexFor: SQLite INTEGER is int64, so we drop the
// high bits to keep the column human-readable in DB browsers.
// Downstream analysis treats slot_id as opaque, so the lossy index
// is purely informational.
func slotIndexFromKey(slotKey string) uint64 {
	s := slotKey
	if strings.HasPrefix(s, "0x") {
		s = s[2:]
	}
	if len(s) < 15 {
		v, err := strconv.ParseUint(s, 16, 64)
		if err != nil {
			return 0
		}
		return v
	}
	v, err := strconv.ParseUint(s[len(s)-15:], 16, 64)
	if err != nil {
		return 0
	}
	return v
}

// ---- prestateTracer parsing -----------------------------------------

// debug_traceBlockByNumber with prestateTracer + diffMode returns
// one entry per transaction in the block; each entry has a txHash
// and a result containing pre / post state maps keyed by address.
type traceBlockEntry struct {
	TxHash string         `json:"txHash"`
	Result prestateResult `json:"result"`
}

type rpcTx struct {
	Hash  string `json:"hash"`
	From  string `json:"from"`
	To    string `json:"to"`
	Input string `json:"input"`
}

type rpcBlockWithTx struct {
	Number       string  `json:"number"`
	Hash         string  `json:"hash"`
	Transactions []rpcTx `json:"transactions"`
}

type prestateResult struct {
	Pre  map[string]prestateAccount `json:"pre"`
	Post map[string]prestateAccount `json:"post"`
}

type prestateAccount struct {
	// Storage maps slotKey (32-byte hex) → value (32-byte hex).
	// In diffMode, pre.storage holds every slot the tx read or
	// wrote; post.storage holds only the slots whose value changed.
	Storage map[string]string `json:"storage"`
}

// slotTouch is one per-tx (contract, slot, access type) record the
// extractor will materialise as an AccessEvent.
type slotTouch struct {
	contract string
	slotKey  string
	access   domain.AccessType
}

// parseStateDiff walks a single tx's prestateTracer result and
// returns a slotTouch per accessed slot. Algorithm:
//
//	for each addr in pre:
//	  for each slotKey in pre[addr].storage:
//	    if slotKey ∈ post[addr].storage → write
//	    else                            → read
//
// Reads and writes both materialise; the AccessType field
// distinguishes them.
//
// We do NOT walk post separately: prestateTracer guarantees every
// post key has a corresponding pre key (the EVM cannot SSTORE
// without first having a pre-state for that slot, even if the
// pre-value is implicit zero).
func parseStateDiff(r prestateResult) []slotTouch {
	out := make([]slotTouch, 0, 8)
	for addr, pre := range r.Pre {
		if len(pre.Storage) == 0 {
			continue
		}
		contract := strings.ToLower(addr)
		post, hasPost := r.Post[addr]
		for slotKey := range pre.Storage {
			access := domain.AccessRead
			if hasPost {
				if _, modified := post.Storage[slotKey]; modified {
					access = domain.AccessWrite
				}
			}
			out = append(out, slotTouch{
				contract: contract,
				slotKey:  strings.ToLower(slotKey),
				access:   access,
			})
		}
	}
	return out
}

// fetchBlockStateDiff calls debug_traceBlockByNumber with
// prestateTracer + diffMode for one block. The endpoint must be an
// archive node (or a hosted equivalent like Alchemy Growth /
// QuickNode archive); a "method not found" / "method not supported"
// error gets a hint baked in so users see the cause without
// poring through stack traces.
func (e *StateDiffExtractor) fetchBlockStateDiff(ctx context.Context, blockNum uint64) ([]traceBlockEntry, error) {
	var traces []traceBlockEntry
	params := []any{
		hexU64(blockNum),
		map[string]any{
			"tracer": "prestateTracer",
			"tracerConfig": map[string]any{
				"diffMode": true,
			},
		},
	}
	if err := e.callStateDiff(ctx, "debug_traceBlockByNumber", params, &traces); err != nil {
		return nil, err
	}
	return traces, nil
}

// callStateDiff is a thin wrapper over the package's RPC plumbing
// that translates "method not found" responses into a user-friendly
// error. It mirrors RPCExtractor.call's signature so both extractors
// share a single transport path.
func (e *StateDiffExtractor) callStateDiff(ctx context.Context, method string, params []any, out any) error {
	e.reqID++
	if err := rpcCall(ctx, e.cfg, &e.reqID, method, params, out); err != nil {
		// The "-32601 method not found" hint is the single most
		// common failure mode — public chain endpoints almost
		// universally refuse debug_trace*. Different clients
		// phrase the rejection differently (geth: "does not
		// exist/is not available"; alchemy proxy: "method not
		// supported"; older nodes: "method not found"), so we
		// match on the JSON-RPC code -32601 plus a few common
		// substrings. Either signal triggers the actionable hint.
		// Only wrap on debug_trace* failures. eth_getBlockByNumber and
		// eth_getCode are widely supported on public endpoints; if
		// those fail, bubble the raw error instead of incorrectly
		// blaming archive/debug capability.
		if strings.HasPrefix(method, "debug_trace") {
			msg := err.Error()
			if strings.Contains(msg, "-32601") ||
				strings.Contains(msg, "method not found") ||
				strings.Contains(msg, "method not supported") ||
				strings.Contains(msg, "does not exist") {
				return fmt.Errorf("%s: endpoint does not expose debug_trace* — this source requires an archive-capable node (try Alchemy Growth / QuickNode archive / a self-hosted Erigon). Original error: %w", method, err)
			}
		}
		return err
	}
	return nil
}

func (e *StateDiffExtractor) fetchBlockWithTransactions(ctx context.Context, blockNum uint64) (*rpcBlockWithTx, error) {
	var b rpcBlockWithTx
	if err := e.callStateDiff(ctx, "eth_getBlockByNumber", []any{hexU64(blockNum), true}, &b); err != nil {
		return nil, err
	}
	return &b, nil
}

func (e *StateDiffExtractor) fetchCode(ctx context.Context, addr string, blockNum uint64) (string, error) {
	var code string
	if err := e.callStateDiff(ctx, "eth_getCode", []any{addr, hexU64(blockNum)}, &code); err != nil {
		return "", err
	}
	return code, nil
}

// ---- 4-byte function-signature classifier ---------------------------

// fourByteCategory is the small curated map of well-known function
// selectors → contract category. Hits are by no means exhaustive —
// they're the high-volume signatures that distinguish ERC-20 / NFT
// / DEX / bridge / governance traffic without bytecode analysis.
//
// Selectors are lowercase, 0x-prefixed, exactly 10 chars (4 bytes
// hex). When a contract's tx inputs hit multiple categories, the
// first one wins by classifyByFunctionSignatures' ordering — we
// prefer specific (ERC-721, DEX swap) over generic (ERC-20).
var fourByteCategory = map[string]domain.ContractCategory{
	// ERC-20
	"0xa9059cbb": domain.ContractERC20, // transfer(address,uint256)
	"0x23b872dd": domain.ContractERC20, // transferFrom(address,address,uint256)
	"0x095ea7b3": domain.ContractERC20, // approve(address,uint256)
	// ERC-721 — supersedes ERC-20 if both are present (NFTs have
	// transferFrom too, but only NFTs have safeTransferFrom).
	"0x42842e0e": domain.ContractNFT, // safeTransferFrom(address,address,uint256)
	"0xb88d4fde": domain.ContractNFT, // safeTransferFrom(address,address,uint256,bytes)
	// DEX (Uniswap V2/V3 / SushiSwap variants)
	"0x022c0d9f": domain.ContractDEX, // swap(uint256,uint256,address,bytes) — V2
	"0x128acb08": domain.ContractDEX, // swap (Uniswap V3 IUniswapV3Pool)
	"0x414bf389": domain.ContractDEX, // exactInputSingle (Uniswap V3 SwapRouter)
	// Bridge (canonical L1↔L2 deposit/withdraw signatures vary;
	// the `depositTransaction` selector is widely shared).
	"0xe9e05c42": domain.ContractBridge, // depositTransaction (Optimism portal style)
	// Governance (Compound Governor Bravo)
	"0xda95691a": domain.ContractGovernance, // propose(...)
	"0x15373e3d": domain.ContractGovernance, // castVote(...)
}

// classifyByFunctionSignatures returns the category implied by the
// most specific signature seen. ERC-721 / DEX / Bridge / Governance
// take precedence over ERC-20 because ERC-20 selectors overlap with
// many other contracts' transfer paths.
func classifyByFunctionSignatures(signatures []string) domain.ContractCategory {
	var sawERC20 bool
	for _, sig := range signatures {
		cat, ok := fourByteCategory[sig]
		if !ok {
			continue
		}
		if cat == domain.ContractERC20 {
			sawERC20 = true
			continue
		}
		// First non-ERC-20 specific category wins.
		return cat
	}
	if sawERC20 {
		return domain.ContractERC20
	}
	return domain.ContractOther
}

// collectFunctionSignatures returns a contract → distinct-4-byte
// selectors map for the given block of traces. Keyed by contract
// address (the recipient of the call). We pull selectors from the
// outer tx only — internal calls between contracts would require
// the callTracer, which is a separate (and even more expensive) RPC.
//
// We fetch full transactions via eth_getBlockByNumber(..., true) and
// classify by the first 4 bytes of each tx input. This satisfies the
// decision-#6 requirement: contract categories must not default to Other
// on statediff, otherwise stratified Cox degenerates.
func collectFunctionSignatures(txs []rpcTx) map[string][]string {
	out := make(map[string][]string)
	seen := make(map[string]map[string]bool)
	for _, tx := range txs {
		to := strings.ToLower(tx.To)
		if to == "" || to == "0x" {
			continue
		}
		sel := selectorFromInput(tx.Input)
		if sel == "" {
			continue
		}
		m := seen[to]
		if m == nil {
			m = make(map[string]bool)
			seen[to] = m
		}
		if m[sel] {
			continue
		}
		m[sel] = true
		out[to] = append(out[to], sel)
	}
	return out
}

func selectorFromInput(input string) string {
	s := strings.ToLower(strings.TrimPrefix(input, "0x"))
	if len(s) < 8 {
		return ""
	}
	for _, ch := range s[:8] {
		if (ch < '0' || ch > '9') && (ch < 'a' || ch > 'f') {
			return ""
		}
	}
	return "0x" + s[:8]
}

// ---- bytecode fallback classifier -------------------------------------

// classifyByBytecode attempts to infer a contract category by scanning its
// deployed bytecode for common 4-byte selectors. This is a best-effort
// fallback for contracts that are touched via internal calls (no top-level
// tx selector) or for blocks with atypical traffic.
func (e *StateDiffExtractor) classifyByBytecode(ctx context.Context, addr string, blockNum uint64) (domain.ContractCategory, bool, error) {
	code, err := e.fetchCode(ctx, addr, blockNum)
	if err != nil {
		return domain.ContractOther, false, fmt.Errorf("eth_getCode %s: %w", addr, err)
	}
	if code == "" || code == "0x" {
		return domain.ContractOther, false, nil
	}
	cat := classifyBytecodeHeuristic(code)
	if cat == domain.ContractOther {
		return domain.ContractOther, false, nil
	}
	return cat, true, nil
}

func classifyBytecodeHeuristic(codeHex string) domain.ContractCategory {
	code := strings.ToLower(strings.TrimPrefix(codeHex, "0x"))
	containsSel := func(sel string) bool {
		s := strings.TrimPrefix(strings.ToLower(sel), "0x")
		// Typical Solidity dispatch uses PUSH4 <sel> which encodes as
		// 0x63<8-hex>. Fall back to raw substring too because not every
		// compiler/layout is identical.
		if strings.Contains(code, "63"+s) {
			return true
		}
		return strings.Contains(code, s)
	}
	// Prefer specific categories over generic ERC-20 selectors.
	for _, sel := range []string{"0x42842e0e", "0xb88d4fde"} {
		if containsSel(sel) {
			return domain.ContractNFT
		}
	}
	for _, sel := range []string{"0x022c0d9f", "0x128acb08", "0x414bf389"} {
		if containsSel(sel) {
			return domain.ContractDEX
		}
	}
	for _, sel := range []string{"0xe9e05c42"} {
		if containsSel(sel) {
			return domain.ContractBridge
		}
	}
	for _, sel := range []string{"0xda95691a", "0x15373e3d"} {
		if containsSel(sel) {
			return domain.ContractGovernance
		}
	}
	for _, sel := range []string{"0xa9059cbb", "0x23b872dd", "0x095ea7b3"} {
		if containsSel(sel) {
			return domain.ContractERC20
		}
	}
	return domain.ContractOther
}
