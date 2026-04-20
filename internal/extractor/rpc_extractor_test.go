package extractor

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"strconv"
	"strings"
	"testing"

	"github.com/egpivo/zk-state-prune/internal/storage"
)

// newMockRPC returns an httptest server that responds to the two JSON-
// RPC methods our extractor calls (eth_getBlockByNumber,
// eth_getBlockReceipts) using a caller-supplied per-method function.
// The fixture dispatches by method name; tests provide the bodies.
func newMockRPC(t *testing.T, handlers map[string]func(params []any) any) *httptest.Server {
	t.Helper()
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req rpcRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			t.Fatalf("decode request: %v", err)
		}
		fn, ok := handlers[req.Method]
		if !ok {
			http.Error(w, "no mock for "+req.Method, http.StatusInternalServerError)
			return
		}
		result := fn(req.Params)
		resp := rpcResponse{JSONRPC: "2.0", ID: req.ID}
		resultBytes, err := json.Marshal(result)
		if err != nil {
			t.Fatalf("marshal result: %v", err)
		}
		resp.Result = resultBytes
		_ = json.NewEncoder(w).Encode(resp)
	}))
}

// transferLog shorthand builder for tests. kind="erc20" gives 3
// topics; kind="nft" gives 4 topics (tokenId as topics[3]).
func transferLog(contract, from, to, tokenID, kind string) rpcLog {
	topics := []string{erc20TransferTopic, from, to}
	if kind == "nft" {
		topics = append(topics, tokenID)
	}
	return rpcLog{Address: contract, Topics: topics, Data: "0x00"}
}

func openRPCTestDB(t *testing.T) *storage.DB {
	t.Helper()
	db, err := storage.Open(context.Background(), filepath.Join(t.TempDir(), "r.db"))
	if err != nil {
		t.Fatalf("open: %v", err)
	}
	t.Cleanup(func() { _ = db.Close() })
	return db
}

func TestRPCExtractor_HappyPath(t *testing.T) {
	// Two blocks with a mix of ERC-20 and ERC-721 transfers across
	// two contracts. The extractor should:
	//   - fetch each block + its receipts
	//   - synthesize 4 slots (2 contracts × 2 unique holders each)
	//   - persist events at the right block numbers
	holderA := "0x000000000000000000000000000000000000000000000000000000000000aaaa"
	holderB := "0x000000000000000000000000000000000000000000000000000000000000bbbb"
	holderC := "0x000000000000000000000000000000000000000000000000000000000000cccc"
	tokenID := "0x0000000000000000000000000000000000000000000000000000000000000001"
	erc20 := "0x1111111111111111111111111111111111111111"
	nft := "0x2222222222222222222222222222222222222222"

	block100Receipts := []rpcReceipt{{
		TransactionHash: "0xaa",
		BlockNumber:     "0x64",
		Logs: []rpcLog{
			transferLog(erc20, holderA, holderB, "", "erc20"),
		},
	}}
	block101Receipts := []rpcReceipt{{
		TransactionHash: "0xbb",
		BlockNumber:     "0x65",
		Logs: []rpcLog{
			transferLog(nft, holderA, holderC, tokenID, "nft"),
			// A non-Transfer log must be ignored.
			{Address: erc20, Topics: []string{"0xdeadbeef"}, Data: "0x"},
		},
	}}

	srv := newMockRPC(t, map[string]func(params []any) any{
		"eth_getBlockByNumber": func(params []any) any {
			return rpcBlock{
				Number:    params[0].(string),
				Hash:      "0xhash",
				Timestamp: "0x0",
			}
		},
		"eth_getBlockReceipts": func(params []any) any {
			blockHex := params[0].(string)
			n, _ := strconv.ParseUint(strings.TrimPrefix(blockHex, "0x"), 16, 64)
			switch n {
			case 100:
				return block100Receipts
			case 101:
				return block101Receipts
			}
			return []rpcReceipt{}
		},
	})
	defer srv.Close()

	ctx := context.Background()
	db := openRPCTestDB(t)
	ex, err := NewRPCExtractor(RPCConfig{
		Endpoint: srv.URL,
		Start:    100,
		End:      101,
	})
	if err != nil {
		t.Fatalf("NewRPCExtractor: %v", err)
	}
	if err := ex.Extract(ctx, db); err != nil {
		t.Fatalf("Extract: %v", err)
	}
	d := ex.LastDiagnostics()
	if d.BlocksFetched != 2 {
		t.Errorf("BlocksFetched=%d want 2", d.BlocksFetched)
	}
	if d.TransferLogs != 2 {
		t.Errorf("TransferLogs=%d want 2 (non-transfer log must be ignored)", d.TransferLogs)
	}
	// Each transfer synthesizes 2 slot-touches (from + to), so 4 events.
	if d.EventsPersisted != 4 {
		t.Errorf("EventsPersisted=%d want 4", d.EventsPersisted)
	}
	// Unique holders across both contracts: erc20 sees A+B, nft sees A+C → 4 slots.
	if d.SlotsCreated != 4 {
		t.Errorf("SlotsCreated=%d want 4", d.SlotsCreated)
	}
	if d.ContractsCreated != 2 {
		t.Errorf("ContractsCreated=%d want 2", d.ContractsCreated)
	}

	// DB state matches diagnostics.
	slots, _ := db.CountSlots(ctx)
	if slots != 4 {
		t.Errorf("DB CountSlots=%d want 4", slots)
	}
	events, _ := db.CountAccessEvents(ctx)
	if events != 4 {
		t.Errorf("DB CountAccessEvents=%d want 4", events)
	}
}

func TestRPCExtractor_HighWaterResumes(t *testing.T) {
	// First run extracts blocks 200..201, high-water should be 201.
	// Second run with Start=200..End=203 should skip 200-201 and only
	// fetch 202-203, because the high-water mark resumes us.
	srv := newMockRPC(t, map[string]func(params []any) any{
		"eth_getBlockByNumber": func(params []any) any {
			return rpcBlock{Number: params[0].(string)}
		},
		"eth_getBlockReceipts": func(params []any) any {
			return []rpcReceipt{} // no events; we're testing control flow
		},
	})
	defer srv.Close()

	ctx := context.Background()
	db := openRPCTestDB(t)

	ex1, _ := NewRPCExtractor(RPCConfig{Endpoint: srv.URL, Start: 200, End: 201})
	if err := ex1.Extract(ctx, db); err != nil {
		t.Fatalf("Extract 1: %v", err)
	}
	d1 := ex1.LastDiagnostics()
	if d1.BlocksFetched != 2 {
		t.Errorf("run 1 BlocksFetched=%d want 2", d1.BlocksFetched)
	}

	hw, ok, err := readHighWater(ctx, db)
	if err != nil || !ok || hw != 201 {
		t.Fatalf("high-water after run 1: hw=%d ok=%v err=%v", hw, ok, err)
	}

	// Second run: overlapping range should resume.
	ex2, _ := NewRPCExtractor(RPCConfig{Endpoint: srv.URL, Start: 200, End: 203})
	if err := ex2.Extract(ctx, db); err != nil {
		t.Fatalf("Extract 2: %v", err)
	}
	d2 := ex2.LastDiagnostics()
	if d2.BlocksFetched != 2 {
		t.Errorf("run 2 BlocksFetched=%d want 2 (resumed from 202)", d2.BlocksFetched)
	}
	hw2, _, _ := readHighWater(ctx, db)
	if hw2 != 203 {
		t.Errorf("high-water after run 2 = %d, want 203", hw2)
	}
}

func TestRPCExtractor_ClassifyFromLog(t *testing.T) {
	cases := []struct {
		n    int
		want string
	}{
		{3, "erc20"},
		{4, "nft"},
		{2, "other"},
	}
	for _, c := range cases {
		topics := make([]string, c.n)
		for i := range topics {
			topics[i] = erc20TransferTopic
		}
		got := classifyFromLog(rpcLog{Topics: topics}).String()
		if got != c.want {
			t.Errorf("topics=%d: category=%q want %q", c.n, got, c.want)
		}
	}
}

func TestRPCExtractor_RejectsBadConfig(t *testing.T) {
	if _, err := NewRPCExtractor(RPCConfig{}); err == nil {
		t.Error("expected error on empty Endpoint")
	}
	if _, err := NewRPCExtractor(RPCConfig{Endpoint: "http://x", Start: 100, End: 50}); err == nil {
		t.Error("expected error on End < Start")
	}
}

func TestRPCExtractor_PropagatesRPCError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req rpcRequest
		_ = json.NewDecoder(r.Body).Decode(&req)
		resp := rpcResponse{
			JSONRPC: "2.0",
			ID:      req.ID,
			Error:   &rpcError{Code: -32000, Message: "method not supported"},
		}
		_ = json.NewEncoder(w).Encode(resp)
	}))
	defer srv.Close()

	ex, _ := NewRPCExtractor(RPCConfig{Endpoint: srv.URL, Start: 1, End: 1})
	err := ex.Extract(context.Background(), openRPCTestDB(t))
	if err == nil || !strings.Contains(err.Error(), "method not supported") {
		t.Errorf("expected rpc error propagation, got %v", err)
	}
}
