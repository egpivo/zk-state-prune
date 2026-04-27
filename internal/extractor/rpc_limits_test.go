package extractor

import (
	"context"
	"strconv"
	"strings"
	"testing"
)

// TestRPCExtractor_LimitTriggers covers the three per-block limits
// (events / contracts / slots). Each subtest constructs a single
// block whose tally just exceeds the configured threshold and asserts
// that Extract fails with a structured error rather than silently
// truncating or panicking.
//
// All three counts are derived deterministically from a list of
// (contract, from, to) triples so we can dial each axis independently:
//   - 1 transfer log → 2 events, 1 distinct contract, 2 distinct slots
//   - N transfers across N distinct contracts → 2N events, N contracts, ≤ 2N slots
func TestRPCExtractor_LimitTriggers(t *testing.T) {
	holderA := "0x000000000000000000000000000000000000000000000000000000000000aaaa"
	holderB := "0x000000000000000000000000000000000000000000000000000000000000bbbb"
	holderC := "0x000000000000000000000000000000000000000000000000000000000000cccc"

	cases := []struct {
		name     string
		cfgPatch func(*RPCConfig)
		logs     []rpcLog
		wantSubs string // expected substring of the error message
	}{
		{
			name: "events_per_block",
			cfgPatch: func(c *RPCConfig) {
				c.MaxEventsPerBlock = 3
			},
			// 2 transfers → 4 events (each transfer = from+to slot
			// touches), > 3
			logs: []rpcLog{
				transferLog("0x1111111111111111111111111111111111111111", holderA, holderB, "", "erc20"),
				transferLog("0x1111111111111111111111111111111111111111", holderB, holderC, "", "erc20"),
			},
			wantSubs: "events_per_block 4 > limit 3",
		},
		{
			name: "contracts_per_block",
			cfgPatch: func(c *RPCConfig) {
				c.MaxContractsPerBlock = 1
			},
			// 2 contracts in one block, limit=1.
			logs: []rpcLog{
				transferLog("0x1111111111111111111111111111111111111111", holderA, holderB, "", "erc20"),
				transferLog("0x2222222222222222222222222222222222222222", holderA, holderC, "", "erc20"),
			},
			wantSubs: "contracts_per_block 2 > limit 1",
		},
		{
			name: "slots_per_block",
			cfgPatch: func(c *RPCConfig) {
				c.MaxSlotsPerBlock = 2
			},
			// 2 transfers on the same contract with 3 distinct
			// holders → 3 distinct slot_ids (A, B, C), > 2.
			logs: []rpcLog{
				transferLog("0x1111111111111111111111111111111111111111", holderA, holderB, "", "erc20"),
				transferLog("0x1111111111111111111111111111111111111111", holderC, holderB, "", "erc20"),
			},
			wantSubs: "slots_per_block 3 > limit 2",
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			receipts := []rpcReceipt{{
				TransactionHash: "0xaa",
				BlockNumber:     "0x64",
				Logs:            tc.logs,
			}}
			client := newMockRPCClient(t, map[string]func(params []any) any{
				"eth_getBlockByNumber": func(params []any) any {
					return rpcBlock{Number: params[0].(string), Hash: "0xhash", Timestamp: "0x0"}
				},
				"eth_getBlockReceipts": func(params []any) any {
					blockHex := params[0].(string)
					n, _ := strconv.ParseUint(strings.TrimPrefix(blockHex, "0x"), 16, 64)
					if n == 100 {
						return receipts
					}
					return []rpcReceipt{}
				},
			})

			ctx := context.Background()
			db := openRPCTestDB(t)
			cfg := RPCConfig{
				Endpoint:   "http://mock-rpc.invalid",
				HTTPClient: client,
				Start:      100,
				End:        100,
			}
			tc.cfgPatch(&cfg)
			ex, err := NewRPCExtractor(cfg)
			if err != nil {
				t.Fatalf("NewRPCExtractor: %v", err)
			}
			err = ex.Extract(ctx, db)
			if err == nil {
				t.Fatalf("Extract: want error containing %q, got nil", tc.wantSubs)
			}
			if !strings.Contains(err.Error(), tc.wantSubs) {
				t.Fatalf("Extract: error %q does not contain %q", err.Error(), tc.wantSubs)
			}
			// No data should land — fail-closed means the block's
			// events were never emitted.
			events, qerr := db.CountAccessEvents(ctx)
			if qerr != nil {
				t.Fatalf("CountAccessEvents: %v", qerr)
			}
			if events != 0 {
				t.Fatalf("EventsPersisted=%d want 0 (fail-closed must not partially persist)", events)
			}
		})
	}
}

// TestRPCExtractor_StampsExtractLimits — a successful Extract writes
// the configured limits into schema_meta.extract_limits as JSON, so
// any DB consumer (CLI / qa_viz / blog reader) can answer "what was
// filtered out at extract time" by reading one row.
func TestRPCExtractor_StampsExtractLimits(t *testing.T) {
	holderA := "0x000000000000000000000000000000000000000000000000000000000000aaaa"
	holderB := "0x000000000000000000000000000000000000000000000000000000000000bbbb"
	erc20 := "0x1111111111111111111111111111111111111111"

	receipts := []rpcReceipt{{
		TransactionHash: "0xaa",
		BlockNumber:     "0x64",
		Logs:            []rpcLog{transferLog(erc20, holderA, holderB, "", "erc20")},
	}}
	client := newMockRPCClient(t, map[string]func(params []any) any{
		"eth_getBlockByNumber": func(params []any) any {
			return rpcBlock{Number: params[0].(string), Hash: "0xhash", Timestamp: "0x0"}
		},
		"eth_getBlockReceipts": func(params []any) any {
			return receipts
		},
	})

	ctx := context.Background()
	db := openRPCTestDB(t)
	ex, err := NewRPCExtractor(RPCConfig{
		Endpoint:             "http://mock-rpc.invalid",
		HTTPClient:           client,
		Start:                100,
		End:                  100,
		MaxEventsPerBlock:    2200,
		MaxContractsPerBlock: 120,
		MaxSlotsPerBlock:     2200,
	})
	if err != nil {
		t.Fatalf("NewRPCExtractor: %v", err)
	}
	if err := ex.Extract(ctx, db); err != nil {
		t.Fatalf("Extract: %v", err)
	}

	got, ok, err := readExtractLimits(ctx, db)
	if err != nil {
		t.Fatalf("readExtractLimits: %v", err)
	}
	if !ok {
		t.Fatalf("readExtractLimits: stamp missing after successful Extract")
	}
	want := ExtractLimits{
		Source:               "rpc",
		MaxEventsPerBlock:    2200,
		MaxContractsPerBlock: 120,
		MaxSlotsPerBlock:     2200,
	}
	if got != want {
		t.Fatalf("stamped limits = %+v, want %+v", got, want)
	}
}

// TestRPCExtractor_ResumeRejectsMismatchedLimits — once an Extract has
// stamped limits, a second Extract pass against the same DB with
// different limits must refuse to run (otherwise the resulting DB
// would mix two filtering regimes — the analysis layer can't tell
// which blocks were filtered at which threshold).
func TestRPCExtractor_ResumeRejectsMismatchedLimits(t *testing.T) {
	ctx := context.Background()
	db := openRPCTestDB(t)

	// Stamp from a prior run by hand so we don't have to drive a
	// full extract twice (the receipts plumbing isn't what's
	// under test).
	prior := ExtractLimits{
		Source:               "rpc",
		MaxEventsPerBlock:    2200,
		MaxContractsPerBlock: 120,
		MaxSlotsPerBlock:     2200,
	}
	if err := writeExtractLimits(ctx, db, prior); err != nil {
		t.Fatalf("seed writeExtractLimits: %v", err)
	}

	// Mock RPC — irrelevant since Extract should bail before
	// fetching anything.
	client := newMockRPCClient(t, map[string]func(params []any) any{
		"eth_getBlockByNumber": func(params []any) any {
			return rpcBlock{Number: params[0].(string), Hash: "0x", Timestamp: "0x0"}
		},
		"eth_getBlockReceipts": func(params []any) any { return []rpcReceipt{} },
	})

	ex, err := NewRPCExtractor(RPCConfig{
		Endpoint:             "http://mock-rpc.invalid",
		HTTPClient:           client,
		Start:                100,
		End:                  100,
		MaxEventsPerBlock:    4400, // different — must trip the guard
		MaxContractsPerBlock: 120,
		MaxSlotsPerBlock:     2200,
	})
	if err != nil {
		t.Fatalf("NewRPCExtractor: %v", err)
	}
	err = ex.Extract(ctx, db)
	if err == nil {
		t.Fatalf("Extract: want error on mismatched limits, got nil")
	}
	if !strings.Contains(err.Error(), "extract_limits mismatch on resume") {
		t.Fatalf("Extract: error %q lacks expected mismatch hint", err.Error())
	}

	// And after ClearRPCState the new run must be allowed (clean slate).
	if err := ClearRPCState(ctx, db); err != nil {
		t.Fatalf("ClearRPCState: %v", err)
	}
	if _, ok, err := readExtractLimits(ctx, db); err != nil {
		t.Fatalf("readExtractLimits post-clear: %v", err)
	} else if ok {
		t.Fatalf("ClearRPCState should have wiped extract_limits")
	}
}
