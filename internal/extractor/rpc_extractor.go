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

	"github.com/egpivo/zk-state-prune/internal/model"
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
}

// DefaultRPCConfig returns a Scroll-mainnet-friendly default. Callers
// typically override Start/End to point at the block range they want.
func DefaultRPCConfig() RPCConfig {
	return RPCConfig{
		Endpoint:  ScrollPublicRPC,
		BatchSize: 10_000,
		HTTPClient: &http.Client{
			Timeout: 30 * time.Second,
		},
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
	BlocksRequested  int
	BlocksFetched    int
	ReceiptsFetched  int
	LogsSeen         int
	TransferLogs     int
	SlotsCreated     int
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

	// Resume from the stored high-water mark, if any.
	start := e.cfg.Start
	if hw, ok, err := readHighWater(ctx, db); err != nil {
		return fmt.Errorf("read high-water mark: %w", err)
	} else if ok && hw+1 > start {
		slog.Info("resuming rpc extract from high-water mark", "from", hw+1)
		start = hw + 1
	}

	contracts := make(map[string]*contractState)
	slots := make(map[string]*slotState)
	eventBuf := make([]model.AccessEvent, 0, e.cfg.BatchSize)

	flush := func() error {
		if len(eventBuf) == 0 {
			return nil
		}
		// Count actual inserted rows (INSERT OR IGNORE drops dups on
		// resume re-fetches) so diagnostics report persisted-rather-
		// than-attempted.
		inserted, err := db.InsertAccessEvents(ctx, eventBuf)
		if err != nil {
			return err
		}
		e.last.EventsPersisted += int(inserted)
		e.last.EventsAttempted += len(eventBuf)
		eventBuf = eventBuf[:0]
		return nil
	}

	for blockNum := start; blockNum <= e.cfg.End; blockNum++ {
		if err := ctx.Err(); err != nil {
			return err
		}
		e.last.BlocksRequested++

		block, err := e.fetchBlock(ctx, blockNum)
		if err != nil {
			return fmt.Errorf("block %d: %w", blockNum, err)
		}
		e.last.BlocksFetched++

		receipts, err := e.fetchBlockReceipts(ctx, blockNum)
		if err != nil {
			return fmt.Errorf("receipts for block %d: %w", blockNum, err)
		}
		e.last.ReceiptsFetched += len(receipts)

		for _, rec := range receipts {
			for _, lg := range rec.Logs {
				e.last.LogsSeen++
				if !isTransferLog(lg) {
					continue
				}
				e.last.TransferLogs++

				category := classifyFromLog(lg)
				contract := contracts[lg.Address]
				if contract == nil {
					contract = &contractState{
						address:     lg.Address,
						category:    category,
						deployBlock: blockNum,
					}
					contracts[lg.Address] = contract
					e.last.ContractsCreated++
					// Upsert the parent contract row right now so
					// FK(slot → contract) is satisfied the moment
					// we hit this contract's first slot.
					if err := db.UpsertContract(ctx, model.ContractMeta{
						Address:      contract.address,
						ContractType: contract.category,
						DeployBlock:  contract.deployBlock,
					}); err != nil {
						return fmt.Errorf("upsert contract %s: %w", contract.address, err)
					}
				}
				// Synthesize one slot per (contract, holder) pair.
				// Holder = topics[1] (from) first, then topics[2] (to) —
				// each side's balance slot is touched.
				for _, holderTopic := range lg.Topics[1:3] {
					slotID := slotIDFor(lg.Address, holderTopic)
					s := slots[slotID]
					if s == nil {
						s = &slotState{
							slotID:       slotID,
							contractAddr: lg.Address,
							slotIndex:    slotIndexFor(lg.Address, holderTopic),
							slotType:     model.SlotTypeBalance,
							createdAt:    blockNum,
						}
						slots[slotID] = s
						e.last.SlotsCreated++
						// Upsert the slot before buffering its
						// events so FK(event → slot) is always
						// satisfied when the batch flushes.
						if err := db.UpsertSlot(ctx, model.StateSlot{
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
					}
					// No per-slot counters maintained here: the
					// end-of-loop sweep that used to copy them into
					// the StateSlot row was deleted (it clobbered
					// cumulative counts on incremental runs and used
					// a global slot total instead of a per-contract
					// one). Analysis passes rebuild counts from
					// access_events, so the slot row's counters stay
					// at their initial values.
					eventBuf = append(eventBuf, model.AccessEvent{
						SlotID:      slotID,
						BlockNumber: blockNum,
						AccessType:  model.AccessWrite,
						TxHash:      rec.TransactionHash,
					})
					if len(eventBuf) >= e.cfg.BatchSize {
						if err := flush(); err != nil {
							return err
						}
					}
				}
			}
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
		_ = block // block metadata unused for now; kept for future fields
	}

	// Contract + slot rows were upserted inline when first seen (so
	// FK(event→slot) is always satisfied at flush time). We do NOT
	// do an end-of-loop sweep to backfill access_count / total_slots:
	// those fields would be overwritten with this-run-only values,
	// clobbering the accumulated counts from any earlier incremental
	// run against the same DB. Every analysis pass rebuilds access
	// counts directly from the access_events table, so leaving the
	// slot/contract counters at their initial (0) state is benign
	// for downstream correctness; Phase 4 can add a SQL-level
	// incremental update if a diagnostic surfaces that needs them.
	return nil
}

// contractState / slotState are extractor-local "have we seen this
// already?" flags. They do NOT carry counter state (access_count /
// last_access / total_slots / active_slots) because mixing per-run
// counters with persisted DB rows is unsound on incremental resume —
// analysis passes already recompute counters from access_events.
type contractState struct {
	address     string
	category    model.ContractCategory
	deployBlock uint64
}

type slotState struct {
	slotID       string
	contractAddr string
	slotIndex    uint64
	slotType     model.SlotType
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
func classifyFromLog(lg rpcLog) model.ContractCategory {
	switch len(lg.Topics) {
	case 3:
		return model.ContractERC20
	case 4:
		return model.ContractNFT
	default:
		return model.ContractOther
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
	TransactionHash string  `json:"transactionHash"`
	BlockNumber     string  `json:"blockNumber"`
	Logs            []rpcLog `json:"logs"`
}

type rpcLog struct {
	Address string   `json:"address"`
	Topics  []string `json:"topics"`
	Data    string   `json:"data"`
}

// call performs one JSON-RPC round trip.
func (e *RPCExtractor) call(ctx context.Context, method string, params []any, out any) error {
	e.reqID++
	body, err := json.Marshal(rpcRequest{
		JSONRPC: "2.0",
		Method:  method,
		Params:  params,
		ID:      e.reqID,
	})
	if err != nil {
		return fmt.Errorf("marshal %s: %w", method, err)
	}
	req, err := http.NewRequestWithContext(ctx, "POST", e.cfg.Endpoint, bytes.NewReader(body))
	if err != nil {
		return fmt.Errorf("new request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := e.cfg.HTTPClient.Do(req)
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

// ClearRPCState removes the RPC extractor's schema_meta keys so a
// forced re-extraction starts from the user-supplied --start rather
// than resuming past the old high-water mark.
func ClearRPCState(ctx context.Context, db *storage.DB) error {
	if _, err := db.SQL().ExecContext(ctx,
		`DELETE FROM schema_meta WHERE key = ?`, RPCHighWaterKey); err != nil {
		return fmt.Errorf("clear rpc state: %w", err)
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
