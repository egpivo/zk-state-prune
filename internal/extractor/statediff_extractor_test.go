package extractor

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"net/http"
	"strings"
	"testing"

	"github.com/egpivo/zk-state-prune/internal/domain"
	"github.com/egpivo/zk-state-prune/internal/storage"
)

// ---- parseStateDiff -----------------------------------------------------

func TestParseStateDiff_ReadVsWrite(t *testing.T) {
	// Pre lists every slot read OR written. Post lists slots whose
	// value actually changed. So a key only in pre = read; a key in
	// both = write. The order of touches is map-iteration so we
	// compare as a set.
	r := prestateResult{
		Pre: map[string]prestateAccount{
			"0xaaaa": {
				Storage: map[string]string{
					"0xkey1": "0xval1_pre", // read only
					"0xkey2": "0xval2_pre", // written
				},
			},
			"0xbbbb": {
				Storage: map[string]string{
					"0xkey3": "0xval3_pre", // read only (no post entry)
				},
			},
		},
		Post: map[string]prestateAccount{
			"0xaaaa": {
				Storage: map[string]string{
					"0xkey2": "0xval2_post",
				},
			},
		},
	}
	got := parseStateDiff(r)
	gotSet := map[string]domain.AccessType{}
	for _, t := range got {
		gotSet[t.contract+":"+t.slotKey] = t.access
	}
	want := map[string]domain.AccessType{
		"0xaaaa:0xkey1": domain.AccessRead,
		"0xaaaa:0xkey2": domain.AccessWrite,
		"0xbbbb:0xkey3": domain.AccessRead,
	}
	if len(gotSet) != len(want) {
		t.Fatalf("got %d touches, want %d: %v", len(gotSet), len(want), gotSet)
	}
	for k, v := range want {
		if g, ok := gotSet[k]; !ok || g != v {
			t.Errorf("touch %s = %v (present=%v), want %v", k, g, ok, v)
		}
	}
}

func TestParseStateDiff_EmptyPostMeansAllReads(t *testing.T) {
	// A view-only call leaves post untouched; every slot in pre is
	// a pure read.
	r := prestateResult{
		Pre: map[string]prestateAccount{
			"0xaa": {Storage: map[string]string{"0xk1": "0xv1", "0xk2": "0xv2"}},
		},
		Post: map[string]prestateAccount{},
	}
	got := parseStateDiff(r)
	if len(got) != 2 {
		t.Fatalf("got %d, want 2", len(got))
	}
	for _, tt := range got {
		if tt.access != domain.AccessRead {
			t.Errorf("expected all reads, got %v on %s", tt.access, tt.slotKey)
		}
	}
}

func TestParseStateDiff_NoStorageMeansNoTouches(t *testing.T) {
	// An account in pre with only balance/nonce changes but no
	// storage map should produce zero touches.
	r := prestateResult{
		Pre: map[string]prestateAccount{
			"0xaa": {Storage: nil}, // explicit empty
		},
	}
	if got := parseStateDiff(r); len(got) != 0 {
		t.Errorf("got %d touches on no-storage account, want 0: %v", len(got), got)
	}
}

// ---- slot_id minting --------------------------------------------------

func TestSlotIDForStateDiff_LowercaseAndDelimited(t *testing.T) {
	// Mixed-case input must round-trip to lowercase so two extractors
	// that disagree on hex casing still mint the same slot_id.
	got := slotIDForStateDiff("0xAabb", "0xCdEf")
	want := "0xaabb:0xcdef"
	if got != want {
		t.Errorf("slotIDForStateDiff = %q, want %q", got, want)
	}
}

func TestSlotIndexFromKey(t *testing.T) {
	cases := []struct {
		in   string
		want uint64
	}{
		// Short keys parse whole — common in mock fixtures.
		{"0x01", 1},
		{"0xabcdef", 0xabcdef},
		// 32-byte slot key → take last 15 nibbles, fits in int64
		// so SQLite INTEGER doesn't choke.
		{"0xfedcba9876543210ffffffffffffffffaaaaaaaaaaaaaaaa1234567890abcdef", 0x234567890abcdef},
		// Bare hex without 0x prefix is also accepted.
		{"deadbeef", 0xdeadbeef},
	}
	for _, c := range cases {
		if got := slotIndexFromKey(c.in); got != c.want {
			t.Errorf("slotIndexFromKey(%q) = %x, want %x", c.in, got, c.want)
		}
	}
}

// ---- 4-byte signature classifier --------------------------------------

func TestClassifyByFunctionSignatures_PrecedenceAndFallback(t *testing.T) {
	cases := []struct {
		name string
		sigs []string
		want domain.ContractCategory
	}{
		{"empty → other", nil, domain.ContractOther},
		{"unknown only → other", []string{"0xdeadbeef"}, domain.ContractOther},
		{"erc20 transfer", []string{"0xa9059cbb"}, domain.ContractERC20},
		{"erc721 safeTransferFrom wins over erc20",
			[]string{"0xa9059cbb", "0x42842e0e"},
			domain.ContractNFT},
		{"dex swap wins over erc20",
			[]string{"0xa9059cbb", "0x022c0d9f"},
			domain.ContractDEX},
		{"bridge",
			[]string{"0xe9e05c42"},
			domain.ContractBridge},
		{"governance",
			[]string{"0xda95691a"},
			domain.ContractGovernance},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			if got := classifyByFunctionSignatures(c.sigs); got != c.want {
				t.Errorf("got %v, want %v", got, c.want)
			}
		})
	}
}

// ---- StateDiffExtractor.Capability -----------------------------------

func TestStateDiffExtractor_CapabilityIsFullCoverage(t *testing.T) {
	ex, err := NewStateDiffExtractor(RPCConfig{Endpoint: "http://x", Start: 1, End: 1})
	if err != nil {
		t.Fatalf("NewStateDiffExtractor: %v", err)
	}
	c := ex.Capability()
	if c.Source != "statediff" {
		t.Errorf("Source = %q, want statediff", c.Source)
	}
	if !c.ObservesReads || !c.ObservesNonTransferWrite {
		t.Errorf("statediff must claim full coverage, got %+v", c)
	}
	if !strings.Contains(c.SlotIDForm, "slotkey") {
		t.Errorf("SlotIDForm should mention slotkey, got %q", c.SlotIDForm)
	}
}

func TestNewStateDiffExtractor_RejectsBadConfig(t *testing.T) {
	if _, err := NewStateDiffExtractor(RPCConfig{Endpoint: "", Start: 1, End: 1}); err == nil {
		t.Error("empty endpoint should error")
	}
	if _, err := NewStateDiffExtractor(RPCConfig{Endpoint: "x", Start: 5, End: 1}); err == nil {
		t.Error("end < start should error")
	}
}

// ---- end-to-end with fake RoundTripper --------------------------------

func TestStateDiffExtractor_EndToEnd(t *testing.T) {
	// Block 1: one tx with reads + writes on a known ERC-20-ish
	// contract. We fetch tx input via eth_getBlockByNumber(..., true)
	// so the 4-byte classifier assigns the contract to ERC-20 (not
	// Other), satisfying decision-#6.
	client := newMockRPCClient(t, map[string]func(params []any) any{
		"eth_getBlockByNumber": func(params []any) any {
			return map[string]any{
				"number": "0x1",
				"hash":   "0xblock1",
				"transactions": []any{
					map[string]any{
						"hash":  "0xtx1",
						"from":  "0xdead",
						"to":    "0xaaaa",
						"input": "0xa9059cbb00000000",
					},
				},
			}
		},
		"debug_traceBlockByNumber": func(params []any) any {
			return []any{
				map[string]any{
					"txHash": "0xtx1",
					"result": map[string]any{
						"pre": map[string]any{
							"0xaaaa": map[string]any{
								"storage": map[string]any{
									"0xkey1": "0xv1",
									"0xkey2": "0xv2",
								},
							},
						},
						"post": map[string]any{
							"0xaaaa": map[string]any{
								"storage": map[string]any{
									"0xkey2": "0xv2_new",
								},
							},
						},
					},
				},
			}
		},
	})
	ex, err := NewStateDiffExtractor(RPCConfig{
		Endpoint:   "http://x",
		Start:      1,
		End:        1,
		HTTPClient: client,
		BatchSize:  100,
	})
	if err != nil {
		t.Fatalf("NewStateDiffExtractor: %v", err)
	}
	db := openRPCTestDB(t)
	if err := ex.Extract(context.Background(), db); err != nil {
		t.Fatalf("Extract: %v", err)
	}
	d := ex.LastDiagnostics()
	if d.BlocksFetched != 1 {
		t.Errorf("BlocksFetched = %d, want 1", d.BlocksFetched)
	}
	// Two slots touched: key1 (read) + key2 (write).
	if d.StorageTouches != 2 {
		t.Errorf("StorageTouches = %d, want 2", d.StorageTouches)
	}
	if d.StorageReads != 1 || d.StorageWrites != 1 {
		t.Errorf("read/write split = %d/%d, want 1/1", d.StorageReads, d.StorageWrites)
	}
	if d.ContractsCreated != 1 {
		t.Errorf("ContractsCreated = %d, want 1", d.ContractsCreated)
	}
	if d.SlotsCreated != 2 {
		t.Errorf("SlotsCreated = %d, want 2", d.SlotsCreated)
	}
	if d.EventsPersisted != 2 {
		t.Errorf("EventsPersisted = %d, want 2", d.EventsPersisted)
	}
	if d.OtherCategoryContracts != 0 {
		t.Errorf("OtherCategoryContracts = %d, want 0", d.OtherCategoryContracts)
	}
	if ct := stateContractType(t, db, "0xaaaa"); ct != domain.ContractERC20 {
		t.Errorf("contract category = %v, want %v", ct, domain.ContractERC20)
	}
}

// TestStateDiffExtractor_CollapsesPerBlockTouches asserts the
// block-level aggregation invariant: even if a slot is touched by
// multiple txs in the same block, the extractor emits exactly one
// access_events row per (slot, block) pair, and a write in any of
// those txs supersedes a read in another. Regression guard for the
// PR2 follow-up that fixed per-tx event inflation.
func TestStateDiffExtractor_CollapsesPerBlockTouches(t *testing.T) {
	// Block 1 contains:
	//   tx1: reads  0xaaaa.0xkey1 (no post entry)
	//   tx2: writes 0xaaaa.0xkey1 (in both pre and post)
	//   tx3: reads  0xaaaa.0xkey2 (no post entry)
	//   tx4: reads  0xaaaa.0xkey2 again (still no post)
	//
	// Expected: 2 rows in access_events — key1 as Write (tx2 promotes),
	// key2 as Read (kept on tx3, the first toucher).
	client := newMockRPCClient(t, map[string]func(params []any) any{
		"eth_getBlockByNumber": func(params []any) any {
			return map[string]any{
				"number": "0x1",
				"hash":   "0xblock1",
				"transactions": []any{
					map[string]any{"hash": "0xtx1", "to": "0xaaaa", "input": "0x"},
					map[string]any{"hash": "0xtx2", "to": "0xaaaa", "input": "0xa9059cbb"},
					map[string]any{"hash": "0xtx3", "to": "0xaaaa", "input": "0x"},
					map[string]any{"hash": "0xtx4", "to": "0xaaaa", "input": "0x"},
				},
			}
		},
		"debug_traceBlockByNumber": func(params []any) any {
			return []any{
				// tx1: read key1
				map[string]any{
					"txHash": "0xtx1",
					"result": map[string]any{
						"pre": map[string]any{
							"0xaaaa": map[string]any{"storage": map[string]any{"0xkey1": "0xv1_pre"}},
						},
						"post": map[string]any{},
					},
				},
				// tx2: write key1 (pre + post)
				map[string]any{
					"txHash": "0xtx2",
					"result": map[string]any{
						"pre": map[string]any{
							"0xaaaa": map[string]any{"storage": map[string]any{"0xkey1": "0xv1_pre"}},
						},
						"post": map[string]any{
							"0xaaaa": map[string]any{"storage": map[string]any{"0xkey1": "0xv1_post"}},
						},
					},
				},
				// tx3: read key2
				map[string]any{
					"txHash": "0xtx3",
					"result": map[string]any{
						"pre": map[string]any{
							"0xaaaa": map[string]any{"storage": map[string]any{"0xkey2": "0xv2_pre"}},
						},
						"post": map[string]any{},
					},
				},
				// tx4: read key2 again
				map[string]any{
					"txHash": "0xtx4",
					"result": map[string]any{
						"pre": map[string]any{
							"0xaaaa": map[string]any{"storage": map[string]any{"0xkey2": "0xv2_pre"}},
						},
						"post": map[string]any{},
					},
				},
			}
		},
	})
	ex, err := NewStateDiffExtractor(RPCConfig{
		Endpoint: "http://x", Start: 1, End: 1, HTTPClient: client, BatchSize: 100,
	})
	if err != nil {
		t.Fatalf("NewStateDiffExtractor: %v", err)
	}
	db := openRPCTestDB(t)
	if err := ex.Extract(context.Background(), db); err != nil {
		t.Fatalf("Extract: %v", err)
	}

	// Exactly 2 access_events rows (not 4) — one per (slot, block).
	var n int64
	if err := db.SQL().QueryRowContext(context.Background(),
		`SELECT COUNT(*) FROM access_events WHERE block_number = 1`).Scan(&n); err != nil {
		t.Fatalf("count events: %v", err)
	}
	if n != 2 {
		t.Errorf("access_events count for block 1 = %d, want 2 (per-block aggregation)", n)
	}

	// Diagnostic counters must agree with what was persisted: 2
	// touches total = 1 read + 1 write. Pre-fix this test would
	// see StorageTouches=4, StorageReads=3, StorageWrites=1.
	d := ex.LastDiagnostics()
	if d.StorageTouches != 2 {
		t.Errorf("StorageTouches = %d, want 2", d.StorageTouches)
	}
	if d.StorageReads != 1 || d.StorageWrites != 1 {
		t.Errorf("read/write split = %d/%d, want 1/1", d.StorageReads, d.StorageWrites)
	}

	// Per-row access_type assertions: key1 = Write, key2 = Read.
	rows := map[string]string{}
	rs, err := db.SQL().QueryContext(context.Background(),
		`SELECT slot_id, access_type FROM access_events WHERE block_number = 1`)
	if err != nil {
		t.Fatalf("query rows: %v", err)
	}
	defer rs.Close()
	for rs.Next() {
		var slotID, accessType string
		if err := rs.Scan(&slotID, &accessType); err != nil {
			t.Fatalf("scan: %v", err)
		}
		rows[slotID] = accessType
	}
	if got := rows["0xaaaa:0xkey1"]; got != "write" {
		t.Errorf("key1 access_type = %q, want write (tx2 should promote tx1's read)", got)
	}
	if got := rows["0xaaaa:0xkey2"]; got != "read" {
		t.Errorf("key2 access_type = %q, want read (no write in any tx)", got)
	}
}

// TestStateDiffExtractor_StrictCategoriesFailsRun asserts that
// when StrictCategories is on and Other-rate exceeds the threshold,
// Extract returns an error rather than just slog.Warn-ing. The
// fixture below has one tx with no recognised selector + a
// bytecode response that doesn't match any known fingerprint, so
// the single contract lands in ContractOther → 100% Other-rate >
// the 20% threshold → strict mode trips.
func TestStateDiffExtractor_StrictCategoriesFailsRun(t *testing.T) {
	client := newMockRPCClient(t, map[string]func(params []any) any{
		"eth_getBlockByNumber": func(params []any) any {
			return map[string]any{
				"number": "0x1",
				"hash":   "0xblock1",
				"transactions": []any{
					map[string]any{"hash": "0xtx1", "to": "0xaaaa", "input": "0xdeadbeef"},
				},
			}
		},
		"debug_traceBlockByNumber": func(params []any) any {
			return []any{
				map[string]any{
					"txHash": "0xtx1",
					"result": map[string]any{
						"pre": map[string]any{
							"0xaaaa": map[string]any{"storage": map[string]any{"0xkey1": "0xv1"}},
						},
						"post": map[string]any{},
					},
				},
			}
		},
		"eth_getCode": func(params []any) any {
			return "0x6080604052"
		},
	})
	ex, err := NewStateDiffExtractor(RPCConfig{
		Endpoint:         "http://x",
		Start:            1,
		End:              1,
		HTTPClient:       client,
		BatchSize:        100,
		StrictCategories: true,
	})
	if err != nil {
		t.Fatalf("NewStateDiffExtractor: %v", err)
	}
	err = ex.Extract(context.Background(), openRPCTestDB(t))
	if err == nil {
		t.Fatal("strict mode should have failed the run on 100% Other-rate, got nil error")
	}
	if !strings.Contains(err.Error(), "could not be classified") ||
		!strings.Contains(err.Error(), "strict-categories") {
		t.Errorf("error should mention classification + flag, got %q", err.Error())
	}
}

func TestStateDiffExtractor_NonStrictWarnsButPasses(t *testing.T) {
	// Same fixture, StrictCategories=false → run completes; the
	// warning surfaces via slog (which the test runner captures
	// to stderr but doesn't assert against, deliberately).
	client := newMockRPCClient(t, map[string]func(params []any) any{
		"eth_getBlockByNumber": func(params []any) any {
			return map[string]any{
				"number": "0x1", "hash": "0xb1",
				"transactions": []any{
					map[string]any{"hash": "0xtx1", "to": "0xaaaa", "input": "0xdeadbeef"},
				},
			}
		},
		"debug_traceBlockByNumber": func(params []any) any {
			return []any{
				map[string]any{
					"txHash": "0xtx1",
					"result": map[string]any{
						"pre": map[string]any{
							"0xaaaa": map[string]any{"storage": map[string]any{"0xkey1": "0xv1"}},
						},
						"post": map[string]any{},
					},
				},
			}
		},
		"eth_getCode": func(params []any) any { return "0x6080604052" },
	})
	ex, err := NewStateDiffExtractor(RPCConfig{
		Endpoint: "http://x", Start: 1, End: 1, HTTPClient: client, BatchSize: 100,
		// StrictCategories: false (zero value)
	})
	if err != nil {
		t.Fatalf("NewStateDiffExtractor: %v", err)
	}
	if err := ex.Extract(context.Background(), openRPCTestDB(t)); err != nil {
		t.Errorf("non-strict run should not fail on classifier fall-through, got %v", err)
	}
	if d := ex.LastDiagnostics(); d.OtherCategoryContracts != 1 {
		t.Errorf("OtherCategoryContracts = %d, want 1", d.OtherCategoryContracts)
	}
}

func TestStateDiffExtractor_BytecodeFallbackClassifies(t *testing.T) {
	// No top-level tx selector (e.g. contract touched via internal call),
	// but bytecode contains a common ERC-20 selector pattern, so the
	// fallback classifier should still avoid ContractOther.
	client := newMockRPCClient(t, map[string]func(params []any) any{
		"eth_getBlockByNumber": func(params []any) any {
			return map[string]any{
				"number":       "0x1",
				"hash":         "0xblock1",
				"transactions": []any{},
			}
		},
		"eth_getCode": func(params []any) any {
			// PUSH4 a9059cbb ...
			return "0x63a9059cbb00"
		},
		"debug_traceBlockByNumber": func(params []any) any {
			return []any{
				map[string]any{
					"txHash": "0xtx1",
					"result": map[string]any{
						"pre": map[string]any{
							"0xaaaa": map[string]any{
								"storage": map[string]any{
									"0xkey1": "0xv1",
								},
							},
						},
						"post": map[string]any{},
					},
				},
			}
		},
	})
	ex, err := NewStateDiffExtractor(RPCConfig{
		Endpoint:   "http://x",
		Start:      1,
		End:        1,
		HTTPClient: client,
		BatchSize:  100,
	})
	if err != nil {
		t.Fatalf("NewStateDiffExtractor: %v", err)
	}
	db := openRPCTestDB(t)
	if err := ex.Extract(context.Background(), db); err != nil {
		t.Fatalf("Extract: %v", err)
	}
	if ct := stateContractType(t, db, "0xaaaa"); ct != domain.ContractERC20 {
		t.Errorf("contract category = %v, want %v", ct, domain.ContractERC20)
	}
	if d := ex.LastDiagnostics(); d.OtherCategoryContracts != 0 {
		t.Errorf("OtherCategoryContracts = %d, want 0", d.OtherCategoryContracts)
	}
}

func TestStateDiffExtractor_MethodNotFoundGetsHint(t *testing.T) {
	// Public chain endpoints almost always reject debug_trace*.
	// The error message must steer the user to an archive node
	// instead of bouncing them through a generic "rpc error" text.
	client := &http.Client{Transport: methodNotFoundRoundTripper(t)}
	ex, err := NewStateDiffExtractor(RPCConfig{
		Endpoint:   "http://x",
		Start:      1,
		End:        1,
		HTTPClient: client,
		BatchSize:  100,
	})
	if err != nil {
		t.Fatalf("NewStateDiffExtractor: %v", err)
	}
	err = ex.Extract(context.Background(), openRPCTestDB(t))
	if err == nil {
		t.Fatal("expected error from method-not-found endpoint, got nil")
	}
	if !strings.Contains(err.Error(), "archive") {
		t.Errorf("error should hint at archive node, got %q", err.Error())
	}
}

func stateContractType(t *testing.T, db *storage.DB, addr string) domain.ContractCategory {
	t.Helper()
	var ct string
	if err := db.SQL().QueryRowContext(context.Background(),
		`SELECT contract_type FROM contracts WHERE address = ?`, strings.ToLower(addr)).
		Scan(&ct); err != nil {
		t.Fatalf("query contract_type: %v", err)
	}
	return domain.ParseContractCategory(ct)
}

// methodNotFoundRoundTripper serves a minimal eth_getBlockByNumber response
// but rejects debug_traceBlockByNumber with a -32601, exercising the
// archive-node hint path.
func methodNotFoundRoundTripper(t *testing.T) roundTripperFunc {
	return func(r *http.Request) (*http.Response, error) {
		var req rpcRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			return nil, err
		}
		resp := rpcResponse{JSONRPC: "2.0", ID: req.ID}
		switch req.Method {
		case "eth_getBlockByNumber":
			resp.Result = mustJSON(t, map[string]any{
				"number":       "0x1",
				"hash":         "0xblock",
				"transactions": []any{},
			})
		case "debug_traceBlockByNumber":
			resp.Error = &rpcError{Code: -32601, Message: "the method debug_traceBlockByNumber does not exist/is not available"}
		default:
			resp.Error = &rpcError{Code: -32000, Message: "unexpected method " + req.Method}
		}
		return jsonRPCResponse(resp)
	}
}

func jsonRPCResponse(resp rpcResponse) (*http.Response, error) {
	b, err := json.Marshal(resp)
	if err != nil {
		return nil, err
	}
	h := make(http.Header)
	h.Set("Content-Type", "application/json")
	return &http.Response{
		StatusCode: http.StatusOK,
		Body:       io.NopCloser(bytes.NewReader(b)),
		Header:     h,
	}, nil
}

func mustJSON(t *testing.T, v any) json.RawMessage {
	t.Helper()
	b, err := json.Marshal(v)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	return b
}
