package extractor

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"strconv"
	"strings"
	"time"

	"github.com/egpivo/zk-state-prune/internal/domain"
	"github.com/egpivo/zk-state-prune/internal/storage"
)

// erc20TransferTopic is keccak256("Transfer(address,address,uint256)") — the
// event signature shared by ERC-20 and ERC-721 Transfer events. The two
// standards are distinguished on the wire by how many indexed topics
// they carry: ERC-20 has 3 (sig, from, to) with the value in `data`,
// ERC-721 has 4 (sig, from, to, tokenId) with empty `data`.
const erc20TransferTopic = "0xddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef"

// ScrollPublicRPC is the default Scroll mainnet endpoint. No API key,
// rate-limited, fine for pulling a few thousand blocks for a bench.
const ScrollPublicRPC = "https://rpc.scroll.io"

// RPCConfig configures the RPC extractor. Required fields: Endpoint,
// Start, End. HTTPClient defaults to http.DefaultClient with a
// conservative timeout. BatchSize controls how many events accumulate
// before a SQLite flush.
type RPCConfig struct {
	Endpoint   string
	Start      uint64
	End        uint64
	HTTPClient *http.Client
	BatchSize  int

	// StrictCategories, when true, makes a statediff Extract return
	// an error (instead of a slog.Warn) if more than
	// otherCategoryWarnRatio of contracts the run touched failed
	// classification and landed in ContractOther. The Transfer-log
	// surrogate (RPCExtractor) ignores this flag — its category
	// signal comes from log.Topics, not from the function-selector
	// + bytecode pipeline that this guardrail watches.
	StrictCategories bool

	// MaxRetries is the number of *retries* (not total attempts)
	// rpcCall will perform on transient HTTP-level failures
	// (timeout / connection reset / 5xx). Default (0 in struct, 3
	// after DefaultRPCConfig) is what production runs use; tests
	// often set it to 0 to fail fast on the first error.
	// Protocol-level RPC errors (e.g. -32601 method not found) are
	// never retried — they don't recover by being asked again.
	MaxRetries int

	// RetryBaseDelay is the first-retry backoff. Subsequent
	// retries double the delay (exponential), capped at 10s per
	// attempt. Zero falls back to 200ms in rpcCall.
	RetryBaseDelay time.Duration

	// Limits are per-block guardrails. Zero means "no limit"
	// (preserves existing behaviour for callers that don't opt in).
	// When > 0, processBlock fail-closes with a structured error
	// the moment any tally exceeds the threshold.
	//
	// Calibration on scroll_100k (Transfer-log surrogate, 100k
	// blocks): observed max events/block = 218, max distinct
	// contracts/block = 12, max distinct slots/block = 218.
	// Recommended thresholds are 10× headroom over those — see
	// internal/extractor/EXTRACT_LIMITS.md for the full data,
	// rationale, and the SQL used to derive them.
	MaxEventsPerBlock    uint64
	MaxContractsPerBlock uint64
	MaxSlotsPerBlock     uint64
}

// ExtractLimits is the persisted form of MaxEventsPerBlock /
// MaxContractsPerBlock / MaxSlotsPerBlock. Stamped into
// schema_meta at the end of every successful Extract; the next
// --resume reads it and refuses to continue if the new limits
// don't match (different filtering would silently produce a
// hybrid-DB the analysis layer can't reason about).
type ExtractLimits struct {
	Source               string `json:"source"`
	MaxEventsPerBlock    uint64 `json:"max_events_per_block"`
	MaxContractsPerBlock uint64 `json:"max_contracts_per_block"`
	MaxSlotsPerBlock     uint64 `json:"max_slots_per_block"`
}

// DefaultRPCConfig returns a Scroll-mainnet-friendly default. Callers
// typically override Start/End to point at the block range they want.
//
// Retry defaults (3 retries, 200ms base) mean a single transient
// blip eats ~3.4s before recovering, and a sustained outage takes
// 200+400+800+1600 ≈ 3s of waits before the run fails. Tuned to
// public-RPC reliability — `rpc.scroll.io` drops connections often
// enough that without retry, multi-hour extracts almost never finish.
func DefaultRPCConfig() RPCConfig {
	return RPCConfig{
		Endpoint:  ScrollPublicRPC,
		BatchSize: 10_000,
		HTTPClient: &http.Client{
			Timeout: 30 * time.Second,
		},
		MaxRetries:     3,
		RetryBaseDelay: 200 * time.Millisecond,
	}
}

// RPCExtractor is a TRANSFER-LOG SURROGATE extractor, not a true
// state-diff / storage-access extractor. It walks blocks over a
// JSON-RPC endpoint and synthesizes pseudo "balance slot" rows from
// ERC-20 / ERC-721 Transfer events — the one log signature every
// EVM-compatible chain shares by convention. The downstream analysis
// pipeline treats slot_id as an opaque key, so these rows feed the
// same EDA / survival / tiering passes the mock extractor does, and
// let us run against real on-chain activity without needing trace
// APIs most public endpoints don't expose.
//
// IT DOES NOT CAPTURE:
//   - slot writes that don't emit a Transfer (storage rebalances,
//     admin settings, DEX pool state, governance bookkeeping…),
//   - slot reads at all,
//   - the real EVM storage-slot identifiers (we hash
//     contract||holder as a deterministic surrogate id).
//
// For full state-access traces, the right data source is
// debug_traceBlockByNumber with a prestate/stateDiff tracer on an
// archive node, or a chain's specialized state-diff endpoint. Those
// are a drop-in replacement: the Extractor interface is the only
// contract downstream code depends on.
type RPCExtractor struct {
	cfg  RPCConfig
	last RPCDiagnostics

	reqID int
}

// RPCDiagnostics mirrors the mock extractor's diagnostics shape so
// CLI / tests can print "here's what the run produced" without
// peeking inside the extractor.
type RPCDiagnostics struct {
	BlocksRequested int
	BlocksFetched   int
	ReceiptsFetched int
	LogsSeen        int
	TransferLogs    int
	SlotsCreated    int
	// EventsAttempted is the number of rows passed into
	// InsertAccessEvents across all flushes for this run — i.e.
	// everything the extractor wanted to write. EventsPersisted is
	// the number actually inserted after the unique index dropped
	// duplicates on a resume re-fetch. When Attempted > Persisted,
	// the delta is the number of rows a previous incarnation of
	// this run had already committed.
	EventsAttempted  int
	EventsPersisted  int
	ContractsCreated int
	StartBlock       uint64
	EndBlock         uint64
}

// NewRPCExtractor validates cfg and returns a ready-to-run extractor.
func NewRPCExtractor(cfg RPCConfig) (*RPCExtractor, error) {
	if cfg.Endpoint == "" {
		return nil, fmt.Errorf("RPCExtractor: Endpoint is required")
	}
	if cfg.End < cfg.Start {
		return nil, fmt.Errorf("RPCExtractor: End(%d) < Start(%d)", cfg.End, cfg.Start)
	}
	if cfg.HTTPClient == nil {
		cfg.HTTPClient = &http.Client{Timeout: 30 * time.Second}
	}
	if cfg.BatchSize <= 0 {
		cfg.BatchSize = 10_000
	}
	return &RPCExtractor{cfg: cfg}, nil
}

// LastDiagnostics returns the diagnostics from the most recent Extract.
// Zero value if Extract has not been called yet.
func (e *RPCExtractor) LastDiagnostics() RPCDiagnostics { return e.last }

// Capability declares this extractor as a Transfer-log surrogate:
// it only sees slot touches emitted as ERC-20 / ERC-721 Transfer
// events. Non-Transfer writes (arbitrary SSTORE, DEX pool updates,
// governance state) and all reads are invisible. A full state-diff
// replacement (debug_traceBlockByNumber + prestateTracer) is
// planned — see the statediff extractor when it lands — and will
// differ from this one by both ObservesReads and
// ObservesNonTransferWrite being true.
func (*RPCExtractor) Capability() Capability {
	return Capability{
		Source:                   "rpc",
		ObservesReads:            false,
		ObservesNonTransferWrite: false,
		SlotIDForm:               "contract:holder (Transfer-log surrogate)",
	}
}

// Extract honours the Extractor interface. Walks blocks in [Start, End]
// fetching each block and its receipts, synthesizing (slot, event)
// rows from Transfer logs. Persists contracts/slots/events to db.
//
// Idempotency is driven by the schema_meta high-water mark
// "rpc_high_water_block": Extract skips blocks up to and including
// the stored value, and updates it as it makes progress. Callers that
// want a full refresh should call db.Reset first (via `extract --force`).
func (e *RPCExtractor) Extract(ctx context.Context, db *storage.DB) error {
	e.last = RPCDiagnostics{StartBlock: e.cfg.Start, EndBlock: e.cfg.End}

	// Stamp / verify the extract limits BEFORE any block work, so
	// the DB is self-describing the moment data starts landing
	// and a mid-run crash leaves a coherent record. On resume,
	// reject mismatched limits — silently mixing thresholds
	// produces an analysis-incoherent DB.
	thisRun := ExtractLimits{
		Source:               e.Capability().Source,
		MaxEventsPerBlock:    e.cfg.MaxEventsPerBlock,
		MaxContractsPerBlock: e.cfg.MaxContractsPerBlock,
		MaxSlotsPerBlock:     e.cfg.MaxSlotsPerBlock,
	}
	if stamped, ok, err := readExtractLimits(ctx, db); err != nil {
		return fmt.Errorf("read extract_limits: %w", err)
	} else if ok && stamped != thisRun {
		return fmt.Errorf(
			"extract_limits mismatch on resume: DB stamped %+v, this run %+v "+
				"(rerun with --force to clear, or pass matching --max-* flags)",
			stamped, thisRun,
		)
	}
	if err := writeExtractLimits(ctx, db, thisRun); err != nil {
		return err
	}

	// Resume from the stored high-water mark, if any.
	start := e.cfg.Start
	if hw, ok, err := readHighWater(ctx, db); err != nil {
		return fmt.Errorf("read high-water mark: %w", err)
	} else if ok && hw+1 > start {
		slog.Info("resuming rpc extract from high-water mark", "from", hw+1)
		start = hw + 1
	}

	state := &rpcRunState{
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
		// Flush all buffered events for this block BEFORE advancing the
		// high-water mark. If we crash (or the RPC errors) after the
		// mark is written but before events are persisted, the next
		// resume would skip this block and the events would be lost
		// forever — the mark's only job is to certify that everything
		// from earlier blocks is already durable.
		if err := flush(); err != nil {
			return err
		}
		if err := writeHighWater(ctx, db, blockNum); err != nil {
			return fmt.Errorf("write high-water mark: %w", err)
		}
	}
	return nil
}

// rpcRunState carries the per-run mutable state that used to live as
// closures inside Extract: the first-seen maps for contracts + slots
// (so we only upsert each once) and the event buffer that flushes to
// storage.InsertAccessEvents in BatchSize chunks.
type rpcRunState struct {
	contracts map[string]*contractState
	slots     map[string]*slotState
	eventBuf  []domain.AccessEvent
}

// flush persists the buffered events. INSERT OR IGNORE means resume
// re-fetches collapse silently, so we track persisted separately from
// attempted for the run diagnostics.
func (s *rpcRunState) flush(ctx context.Context, db *storage.DB, diag *RPCDiagnostics) error {
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

// processBlock fetches one block + its receipts, walks every Transfer
// log, and appends the synthesized (slot, event) rows to state. flush
// is called whenever the buffer crosses BatchSize.
//
// Hard-limit enforcement: when MaxEventsPerBlock / MaxContractsPerBlock
// / MaxSlotsPerBlock are non-zero, we tally each block's draft work
// BEFORE writing anything to state and bail with a structured error
// if any tally exceeds its limit. Fail-closed is deliberate: silent
// truncation would corrupt the access_events stream the survival
// model trains on, and a "skip block" mode would leave gaps the
// resume / high-water logic isn't designed to bridge.
func (e *RPCExtractor) processBlock(
	ctx context.Context,
	db *storage.DB,
	blockNum uint64,
	state *rpcRunState,
	flush func() error,
) error {
	e.last.BlocksRequested++
	if _, err := e.fetchBlock(ctx, blockNum); err != nil {
		return fmt.Errorf("block %d: %w", blockNum, err)
	}
	e.last.BlocksFetched++
	receipts, err := e.fetchBlockReceipts(ctx, blockNum)
	if err != nil {
		return fmt.Errorf("receipts for block %d: %w", blockNum, err)
	}
	e.last.ReceiptsFetched += len(receipts)

	// Canonicalise every log's address + topics in place BEFORE
	// any tallying or handling. Production RPC endpoints generally
	// return lowercase, but EIP-55 checksum addresses (and some
	// proxies that re-serialize) can leak through. Without this,
	// "0xAbCd…" and "0xabcd…" would count as two distinct
	// contracts in checkBlockLimits AND produce two slot_ids in
	// slotIDFor — silently doubling our row count for the same
	// underlying entity. Doing it once here guarantees every
	// downstream path keys off the same form.
	for i := range receipts {
		for j := range receipts[i].Logs {
			receipts[i].Logs[j].canonicalize()
		}
	}

	// First pass: pre-tally Transfer logs so we can fail closed
	// without mutating state if any limit is exceeded. Cheap — at
	// most a few hundred logs per block at the calibrated p99.9.
	if err := e.checkBlockLimits(blockNum, receipts); err != nil {
		return err
	}

	for _, rec := range receipts {
		for _, lg := range rec.Logs {
			e.last.LogsSeen++
			if !isTransferLog(lg) {
				continue
			}
			e.last.TransferLogs++
			if err := e.handleTransferLog(ctx, db, blockNum, rec.TransactionHash, lg, state, flush); err != nil {
				return err
			}
		}
	}
	return nil
}

// canonicalize lowercases lg.Address and every entry in lg.Topics
// in place. Topics may include 32-byte left-padded addresses
// (Transfer's from/to) which are case-sensitive at the byte level
// but encode the same Ethereum address regardless of casing —
// canonicalising both fields keeps the slot-id derivation stable
// across hostile / non-conforming RPC responses.
//
// In-place mutation (vs returning a copy) is deliberate: if rpcLog
// gains a field later, copy-construction would silently drop it,
// whereas mutating leaves the rest of the struct untouched.
func (lg *rpcLog) canonicalize() {
	lg.Address = strings.ToLower(lg.Address)
	for i := range lg.Topics {
		lg.Topics[i] = strings.ToLower(lg.Topics[i])
	}
}

// checkBlockLimits walks the block's receipts once, tallies
// Transfer-derived (events, distinct contracts, distinct slots), and
// returns a structured error the moment any tally crosses its limit.
// Returns nil when all limits are 0 (opt-in) or no tally exceeded.
//
// Each Transfer log emits two events (from-side + to-side balance
// slot), matching handleTransferLog's loop over Topics[1:3]. We
// mirror that arithmetic here so the pre-tally and the post-emit
// counts agree.
func (e *RPCExtractor) checkBlockLimits(blockNum uint64, receipts []rpcReceipt) error {
	if e.cfg.MaxEventsPerBlock == 0 && e.cfg.MaxContractsPerBlock == 0 && e.cfg.MaxSlotsPerBlock == 0 {
		return nil
	}
	var events uint64
	contracts := make(map[string]struct{})
	slots := make(map[string]struct{})
	for _, rec := range receipts {
		for _, lg := range rec.Logs {
			if !isTransferLog(lg) {
				continue
			}
			contracts[lg.Address] = struct{}{}
			for _, holderTopic := range lg.Topics[1:3] {
				events++
				slots[slotIDFor(lg.Address, holderTopic)] = struct{}{}
			}
		}
	}
	if e.cfg.MaxEventsPerBlock > 0 && events > e.cfg.MaxEventsPerBlock {
		return reportLimitViolation(blockNum, "events_per_block", events, e.cfg.MaxEventsPerBlock)
	}
	if e.cfg.MaxContractsPerBlock > 0 && uint64(len(contracts)) > e.cfg.MaxContractsPerBlock {
		return reportLimitViolation(blockNum, "contracts_per_block", uint64(len(contracts)), e.cfg.MaxContractsPerBlock)
	}
	if e.cfg.MaxSlotsPerBlock > 0 && uint64(len(slots)) > e.cfg.MaxSlotsPerBlock {
		return reportLimitViolation(blockNum, "slots_per_block", uint64(len(slots)), e.cfg.MaxSlotsPerBlock)
	}
	return nil
}

// reportLimitViolation logs a structured ERROR and returns an error
// the caller can wrap or compare with errors.Is. The slog payload
// gives a downstream alerting consumer everything it needs to act
// without parsing the error string.
func reportLimitViolation(blockNum uint64, kind string, observed, limit uint64) error {
	// Map the structured kind ("events_per_block") to the actual
	// CLI flag name ("--max-events-per-block") so the suggestion
	// is copy-pasteable. Anything off-mapping falls back to the
	// kind itself, which is still useful in the slog payload.
	flag := "--max-" + strings.ReplaceAll(kind, "_", "-")
	slog.Error("extract: per-block limit exceeded — fail-closed",
		"block_number", blockNum,
		"limit_kind", kind,
		"observed", observed,
		"limit", limit,
		"suggestion", "investigate the block (likely token launch / spam) or rerun with a higher "+flag,
	)
	return fmt.Errorf("block %d: %s %d > limit %d", blockNum, kind, observed, limit)
}

// handleTransferLog upserts the contract (if first-seen) plus each
// holder's synthesized balance slot (if first-seen) and appends one
// write event per holder. Both upserts MUST happen before the events
// buffer so a mid-batch flush doesn't violate FK(event → slot).
func (e *RPCExtractor) handleTransferLog(
	ctx context.Context,
	db *storage.DB,
	blockNum uint64,
	txHash string,
	lg rpcLog,
	state *rpcRunState,
	flush func() error,
) error {
	if err := e.upsertContractIfNew(ctx, db, lg, blockNum, state); err != nil {
		return err
	}
	// Synthesize one slot per (contract, holder) pair.
	// Holder = topics[1] (from) first, then topics[2] (to) —
	// each side's balance slot is touched.
	for _, holderTopic := range lg.Topics[1:3] {
		slotID := slotIDFor(lg.Address, holderTopic)
		if err := e.upsertSlotIfNew(ctx, db, lg.Address, holderTopic, blockNum, state); err != nil {
			return err
		}
		state.eventBuf = append(state.eventBuf, domain.AccessEvent{
			SlotID:      slotID,
			BlockNumber: blockNum,
			AccessType:  domain.AccessWrite,
			TxHash:      txHash,
		})
		if len(state.eventBuf) >= e.cfg.BatchSize {
			if err := flush(); err != nil {
				return err
			}
		}
	}
	return nil
}

func (e *RPCExtractor) upsertContractIfNew(
	ctx context.Context, db *storage.DB, lg rpcLog, blockNum uint64, state *rpcRunState,
) error {
	if state.contracts[lg.Address] != nil {
		return nil
	}
	c := &contractState{
		address:     lg.Address,
		category:    classifyFromLog(lg),
		deployBlock: blockNum,
	}
	state.contracts[lg.Address] = c
	e.last.ContractsCreated++
	if err := db.UpsertContract(ctx, domain.ContractMeta{
		Address:      c.address,
		ContractType: c.category,
		DeployBlock:  c.deployBlock,
	}); err != nil {
		return fmt.Errorf("upsert contract %s: %w", c.address, err)
	}
	return nil
}

func (e *RPCExtractor) upsertSlotIfNew(
	ctx context.Context, db *storage.DB, contractAddr, holderTopic string, blockNum uint64, state *rpcRunState,
) error {
	slotID := slotIDFor(contractAddr, holderTopic)
	if state.slots[slotID] != nil {
		return nil
	}
	s := &slotState{
		slotID:       slotID,
		contractAddr: contractAddr,
		slotIndex:    slotIndexFor(contractAddr, holderTopic),
		slotType:     domain.SlotTypeBalance,
		createdAt:    blockNum,
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

// contractState / slotState are extractor-local "have we seen this
// already?" flags. They do NOT carry counter state (access_count /
// last_access / total_slots / active_slots) because mixing per-run
// counters with persisted DB rows is unsound on incremental resume —
// analysis passes already recompute counters from access_events.
type contractState struct {
	address     string
	category    domain.ContractCategory
	deployBlock uint64
}

type slotState struct {
	slotID       string
	contractAddr string
	slotIndex    uint64
	slotType     domain.SlotType
	createdAt    uint64
}

// isTransferLog recognises the Transfer(address,address,uint256)
// event signature used by both ERC-20 and ERC-721. Safe on logs with
// fewer than 3 topics (non-transfer anonymous or partially-indexed
// events).
func isTransferLog(lg rpcLog) bool {
	if len(lg.Topics) < 3 {
		return false
	}
	return strings.EqualFold(lg.Topics[0], erc20TransferTopic)
}

// classifyFromLog infers a contract category from a single Transfer
// log. ERC-20 carries the value in `data` and has exactly 3 topics;
// ERC-721 carries tokenId as topics[3] and has 4. Anything else lands
// in ContractOther for manual follow-up.
func classifyFromLog(lg rpcLog) domain.ContractCategory {
	switch len(lg.Topics) {
	case 3:
		return domain.ContractERC20
	case 4:
		return domain.ContractNFT
	default:
		return domain.ContractOther
	}
}

// slotIDFor mints a deterministic slot identifier for the (contract,
// holder) pair. We don't attempt to compute the real EVM slot hash —
// the downstream analysis treats slot_id as an opaque key, so as long
// as two Transfer events for the same holder on the same contract
// produce the same id we recover the correct access sequence.
func slotIDFor(contractAddr, holderTopic string) string {
	return contractAddr + ":" + holderTopic
}

// slotIndexFor derives a deterministic uint64 slot-index from the
// holder topic so rows can still be compared against the mock
// extractor's SlotIndex field. Takes the last 15 hex nibbles (60 bits)
// to guarantee the result fits in a signed int64 — SQLite (and the
// modernc driver backing it) stores INTEGER as int64 and rejects
// uint64 values with the high bit set.
func slotIndexFor(contractAddr, holderTopic string) uint64 {
	s := holderTopic
	if strings.HasPrefix(s, "0x") {
		s = s[2:]
	}
	if len(s) < 15 {
		return 0
	}
	v, err := strconv.ParseUint(s[len(s)-15:], 16, 64)
	if err != nil {
		return 0
	}
	return v
}

// ---- JSON-RPC client ---------------------------------------------------

type rpcRequest struct {
	JSONRPC string `json:"jsonrpc"`
	Method  string `json:"method"`
	Params  []any  `json:"params"`
	ID      int    `json:"id"`
}

type rpcResponse struct {
	JSONRPC string          `json:"jsonrpc"`
	ID      int             `json:"id"`
	Result  json.RawMessage `json:"result"`
	Error   *rpcError       `json:"error"`
}

type rpcError struct {
	Code    int    `json:"code"`
	Message string `json:"message"`
}

type rpcBlock struct {
	Number       string `json:"number"`
	Hash         string `json:"hash"`
	Timestamp    string `json:"timestamp"`
	Transactions []any  `json:"transactions"`
}

type rpcReceipt struct {
	TransactionHash string   `json:"transactionHash"`
	BlockNumber     string   `json:"blockNumber"`
	Logs            []rpcLog `json:"logs"`
}

type rpcLog struct {
	Address string   `json:"address"`
	Topics  []string `json:"topics"`
	Data    string   `json:"data"`
}

// call performs one JSON-RPC round trip. Thin wrapper over rpcCall
// so RPCExtractor can keep its reqID counter on the receiver.
func (e *RPCExtractor) call(ctx context.Context, method string, params []any, out any) error {
	e.reqID++
	return rpcCall(ctx, e.cfg, &e.reqID, method, params, out)
}

// rpcCall is the shared JSON-RPC transport — used by both the
// Transfer-log surrogate (RPCExtractor) and the real state-diff
// extractor (StateDiffExtractor) so they agree on body shape, error
// translation, and HTTP handling. Wraps rpcCallOnce in
// retry-with-backoff for transient HTTP failures (timeout,
// connection reset, 5xx). Protocol-level errors (-32601, decode
// failures) fail fast — those don't recover on retry.
//
// The reqID pointer is mutated by callers before each invocation;
// retries reuse the same ID, since logically it's the same request.
func rpcCall(ctx context.Context, cfg RPCConfig, reqID *int, method string, params []any, out any) error {
	maxAttempts := cfg.MaxRetries + 1
	if maxAttempts < 1 {
		maxAttempts = 1
	}
	base := cfg.RetryBaseDelay
	if base <= 0 {
		base = 200 * time.Millisecond
	}

	var lastErr error
	for attempt := 1; attempt <= maxAttempts; attempt++ {
		if attempt > 1 {
			backoff := base << (attempt - 2) // exp: base, 2·base, 4·base, …
			if backoff > 10*time.Second {
				backoff = 10 * time.Second
			}
			slog.Warn("rpc transient error, retrying",
				"method", method,
				"attempt", attempt-1,
				"max_retries", cfg.MaxRetries,
				"backoff", backoff,
				"err", lastErr)
			select {
			case <-ctx.Done():
				return ctx.Err()
			case <-time.After(backoff):
			}
		}
		err := rpcCallOnce(ctx, cfg, reqID, method, params, out)
		if err == nil {
			return nil
		}
		if !isTransientHTTPError(err) {
			return err
		}
		lastErr = err
	}
	return fmt.Errorf("%s: gave up after %d transient failures: %w", method, maxAttempts, lastErr)
}

// rpcCallOnce performs exactly one JSON-RPC round trip — no retry.
// Kept separate from rpcCall so the retry wrapper stays readable
// and so tests can target either layer in isolation.
func rpcCallOnce(ctx context.Context, cfg RPCConfig, reqID *int, method string, params []any, out any) error {
	body, err := json.Marshal(rpcRequest{
		JSONRPC: "2.0",
		Method:  method,
		Params:  params,
		ID:      *reqID,
	})
	if err != nil {
		return fmt.Errorf("marshal %s: %w", method, err)
	}
	req, err := http.NewRequestWithContext(ctx, "POST", cfg.Endpoint, bytes.NewReader(body))
	if err != nil {
		return fmt.Errorf("new request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := cfg.HTTPClient.Do(req)
	if err != nil {
		return fmt.Errorf("do %s: %w", method, err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		b, _ := io.ReadAll(io.LimitReader(resp.Body, 4096))
		return fmt.Errorf("%s: HTTP %d: %s", method, resp.StatusCode, string(b))
	}
	var rr rpcResponse
	if err := json.NewDecoder(resp.Body).Decode(&rr); err != nil {
		return fmt.Errorf("decode %s: %w", method, err)
	}
	if rr.Error != nil {
		return fmt.Errorf("%s: rpc error %d: %s", method, rr.Error.Code, rr.Error.Message)
	}
	if out != nil {
		if err := json.Unmarshal(rr.Result, out); err != nil {
			return fmt.Errorf("unmarshal %s: %w", method, err)
		}
	}
	return nil
}

// isTransientHTTPError classifies an rpcCallOnce error as
// transport-level (worth retrying) or protocol-level (fail fast).
// We match on substrings of the error message because Go wraps a
// patchwork of net / context / http errors that don't share a
// single typed root. The patterns cover Scroll public RPC's
// observed failure modes:
//
//   - "context deadline exceeded": HTTP client Timeout fired
//   - "connection reset by peer":  TCP RST mid-response
//   - "connection refused":        endpoint actively rejecting
//   - "broken pipe":               we wrote, then connection died
//   - "i/o timeout":               read or dial timeout
//   - "EOF" / "unexpected EOF":    server closed connection early
//   - "no such host":              DNS NXDOMAIN — macOS resolver
//     flakes briefly, clears on retry
//   - "network is unreachable":    interface flap / VPN reconnect
//   - "host is unreachable":       routing blip
//   - "HTTP 5":                    any 5xx response
//
// Protocol errors ("rpc error -32601 method not found", "decode <m>",
// "unmarshal <m>") return false — they reflect a request the server
// can't fulfil, not a transient transport blip.
func isTransientHTTPError(err error) bool {
	if err == nil {
		return false
	}
	msg := err.Error()
	for _, pat := range []string{
		"deadline exceeded",
		"connection reset",
		"connection refused",
		"broken pipe",
		"i/o timeout",
		"EOF",
		"no such host",
		"network is unreachable",
		"host is unreachable",
		"HTTP 5", // any 5xx
	} {
		if strings.Contains(msg, pat) {
			return true
		}
	}
	return false
}

func (e *RPCExtractor) fetchBlock(ctx context.Context, blockNum uint64) (*rpcBlock, error) {
	var b rpcBlock
	if err := e.call(ctx, "eth_getBlockByNumber", []any{hexU64(blockNum), false}, &b); err != nil {
		return nil, err
	}
	return &b, nil
}

func (e *RPCExtractor) fetchBlockReceipts(ctx context.Context, blockNum uint64) ([]rpcReceipt, error) {
	var recs []rpcReceipt
	if err := e.call(ctx, "eth_getBlockReceipts", []any{hexU64(blockNum)}, &recs); err != nil {
		return nil, err
	}
	return recs, nil
}

func hexU64(v uint64) string {
	return "0x" + strconv.FormatUint(v, 16)
}

// ---- high-water mark ---------------------------------------------------

// RPCHighWaterKey is the schema_meta key the RPC extractor uses to
// track the last fully-flushed block. Exposed so callers (e.g. the
// CLI's `--force` path) can clear it explicitly — storage.Reset()
// intentionally does not touch schema_meta so it remains a pure
// data-table truncation, not an extractor-state wipe.
const RPCHighWaterKey = "rpc_high_water_block"

// ExtractLimitsKey is the schema_meta key holding the JSON-serialised
// ExtractLimits used at extract time. On --resume we read this back
// and refuse to continue if the new run's limits differ — otherwise
// the resulting DB would be a hybrid (some blocks filtered at one
// threshold, the rest at another), which the analysis layer can't
// reason about.
const ExtractLimitsKey = "extract_limits"

// ClearRPCState removes the RPC extractor's schema_meta keys so a
// forced re-extraction starts from the user-supplied --start rather
// than resuming past the old high-water mark. Wipes the stamped
// extract limits too so a follow-up run is free to choose new ones.
func ClearRPCState(ctx context.Context, db *storage.DB) error {
	if _, err := db.SQL().ExecContext(ctx,
		`DELETE FROM schema_meta WHERE key IN (?, ?)`,
		RPCHighWaterKey, ExtractLimitsKey); err != nil {
		return fmt.Errorf("clear rpc state: %w", err)
	}
	return nil
}

// readExtractLimits returns the stamped ExtractLimits, (zero, false,
// nil) when no prior run stamped any. Used by Extract to detect
// limit drift across a resume boundary.
func readExtractLimits(ctx context.Context, db *storage.DB) (ExtractLimits, bool, error) {
	var v string
	err := db.SQL().QueryRowContext(ctx,
		`SELECT value FROM schema_meta WHERE key = ?`, ExtractLimitsKey).Scan(&v)
	if err != nil {
		if err.Error() == "sql: no rows in result set" {
			return ExtractLimits{}, false, nil
		}
		return ExtractLimits{}, false, err
	}
	var lim ExtractLimits
	if jerr := json.Unmarshal([]byte(v), &lim); jerr != nil {
		return ExtractLimits{}, false, fmt.Errorf("decode extract_limits: %w", jerr)
	}
	return lim, true, nil
}

// writeExtractLimits stamps the limits used by this Extract into
// schema_meta. Called once at the end of a successful run (or on the
// first block of a fresh run, before any data is written, so a crash
// mid-run still leaves the DB self-describing).
func writeExtractLimits(ctx context.Context, db *storage.DB, lim ExtractLimits) error {
	b, err := json.Marshal(lim)
	if err != nil {
		return fmt.Errorf("encode extract_limits: %w", err)
	}
	if _, err := db.SQL().ExecContext(ctx,
		`INSERT INTO schema_meta(key, value) VALUES(?, ?)
		 ON CONFLICT(key) DO UPDATE SET value=excluded.value`,
		ExtractLimitsKey, string(b)); err != nil {
		return fmt.Errorf("write extract_limits: %w", err)
	}
	return nil
}

// readHighWater returns the last successfully-extracted block number
// from schema_meta, or (0, false) if the extractor has never run
// against this DB. The third return distinguishes "no prior run" from
// "SQL error" so callers can resume cleanly when the key is missing.
func readHighWater(ctx context.Context, db *storage.DB) (uint64, bool, error) {
	var v string
	err := db.SQL().QueryRowContext(ctx,
		`SELECT value FROM schema_meta WHERE key = ?`, RPCHighWaterKey).
		Scan(&v)
	if err != nil {
		// sql.ErrNoRows shows up as a typed error; anything else is fatal.
		if err.Error() == "sql: no rows in result set" {
			return 0, false, nil
		}
		return 0, false, err
	}
	n, perr := strconv.ParseUint(v, 10, 64)
	if perr != nil {
		return 0, false, fmt.Errorf("high-water value %q: %w", v, perr)
	}
	return n, true, nil
}

func writeHighWater(ctx context.Context, db *storage.DB, block uint64) error {
	_, err := db.SQL().ExecContext(ctx,
		`INSERT INTO schema_meta(key, value) VALUES(?, ?)
		 ON CONFLICT(key) DO UPDATE SET value=excluded.value`,
		RPCHighWaterKey, strconv.FormatUint(block, 10))
	return err
}
