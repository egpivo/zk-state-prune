package main

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"

	"github.com/egpivo/zk-state-prune/internal/extractor"
	"github.com/egpivo/zk-state-prune/internal/storage"
)

// Integration tests for the zksp CLI. We drive newRootCmd()
// end-to-end with cobra.ExecuteContext, capture stdout/stderr via
// cmd.SetOut/SetErr, and assert on the side effects that matter
// (rows in the DB; a saved model file; the high-water mark cleared
// after --force). Snapshot-quality checks on text output are left
// to render/render_test.go; here we only check that the top-level
// section headers show up so a misrouted RunE is caught.

// runCLI executes `zksp <args>...`, captures stdout, and returns
// (stdout, exec-error). Each invocation builds a fresh newRootCmd()
// to avoid test-order dependence via the package-global
// globalCfg. We also chdir into a temp dir up front so the
// configs/default.yaml auto-load at repo root can't leak between
// tests.
func runCLI(t *testing.T, args ...string) (string, error) {
	t.Helper()
	root := newRootCmd()
	var buf bytes.Buffer
	root.SetOut(&buf)
	root.SetErr(&buf)
	root.SetArgs(args)
	err := root.ExecuteContext(context.Background())
	return buf.String(), err
}

// withIsolatedCWD moves the test into a fresh temp dir and restores
// the original CWD on cleanup. Needed because newRootCmd()'s
// PersistentPreRunE auto-loads configs/default.yaml from CWD if no
// --config was passed; from the repo root there IS such a file,
// so without this isolation the tests would pick up real config
// values and deviate from config.Default().
func withIsolatedCWD(t *testing.T) string {
	t.Helper()
	orig, err := os.Getwd()
	if err != nil {
		t.Fatalf("getwd: %v", err)
	}
	dir := t.TempDir()
	if err := os.Chdir(dir); err != nil {
		t.Fatalf("chdir: %v", err)
	}
	t.Cleanup(func() { _ = os.Chdir(orig) })
	return dir
}

// ---- extract ----------------------------------------------------------

func TestCLI_ExtractMock_PopulatesDB(t *testing.T) {
	dir := withIsolatedCWD(t)
	dbPath := filepath.Join(dir, "mock.db")

	if _, err := runCLI(t,
		"extract", "--source", "mock",
		"--output", dbPath,
		"--seed", "7",
		"--num-contracts", "10",
		"--total-blocks", "2000",
	); err != nil {
		t.Fatalf("extract: %v", err)
	}

	db, err := storage.Open(context.Background(), dbPath)
	if err != nil {
		t.Fatalf("open produced DB: %v", err)
	}
	defer db.Close()
	slots, _ := db.CountSlots(context.Background())
	events, _ := db.CountAccessEvents(context.Background())
	if slots == 0 {
		t.Error("extract produced 0 slots")
	}
	if events == 0 {
		t.Error("extract produced 0 events")
	}
}

func TestCLI_ExtractMock_RefusesOverwriteWithoutForce(t *testing.T) {
	dir := withIsolatedCWD(t)
	dbPath := filepath.Join(dir, "mock.db")
	args := []string{
		"extract", "--source", "mock",
		"--output", dbPath,
		"--seed", "7", "--num-contracts", "5", "--total-blocks", "1000",
	}
	if _, err := runCLI(t, args...); err != nil {
		t.Fatalf("first extract: %v", err)
	}
	// Second run without --force must error out rather than silently
	// overwriting.
	if _, err := runCLI(t, args...); err == nil {
		t.Fatal("second extract without --force: want error, got nil")
	} else if !strings.Contains(err.Error(), "refusing to overwrite") {
		t.Errorf("error = %v, want 'refusing to overwrite'", err)
	}
}

func TestCLI_ExtractMock_ForceOverrides(t *testing.T) {
	dir := withIsolatedCWD(t)
	dbPath := filepath.Join(dir, "mock.db")
	seed1 := []string{
		"extract", "--source", "mock",
		"--output", dbPath,
		"--seed", "1", "--num-contracts", "5", "--total-blocks", "1000",
	}
	if _, err := runCLI(t, seed1...); err != nil {
		t.Fatalf("seed1: %v", err)
	}
	seed2 := append([]string{}, seed1...)
	// swap seed so we can tell the second run actually overwrote the
	// data rather than merging or aborting.
	seed2[5] = "2"
	seed2 = append(seed2, "--force")
	if _, err := runCLI(t, seed2...); err != nil {
		t.Fatalf("seed2 --force: %v", err)
	}
}

// TestCLI_ExtractRPC_ForceClearsHighWater is the regression guard for
// the `--force` fix: before it, an RPC re-run with --force kept the
// prior high-water mark and silently skipped blocks the user asked to
// re-fetch. The test plants a high-water of 999 into a populated DB
// and then runs `extract --source rpc --force` over blocks [1, 2]. If
// --force doesn't call ClearRPCState, the new run resumes from 1000
// and this test (which asserts new rows were written AND the stored
// high-water advanced past 999 via fresh progress rather than staying
// at 999 through a no-op run) would fail.
func TestCLI_ExtractRPC_ForceClearsHighWater(t *testing.T) {
	dir := withIsolatedCWD(t)
	dbPath := filepath.Join(dir, "rpc.db")

	// Inject a tiny JSON-RPC client that returns empty blocks/receipts.
	// No Transfer events → no event rows produced, but the high-water mark
	// must still advance. We can't bind an httptest server in the sandbox,
	// so the CLI uses the override hook in main.go.
	rpcHTTPClientOverride = func() *http.Client { return fakeRPCClient(t) }
	t.Cleanup(func() { rpcHTTPClientOverride = nil })

	// Populate the DB with mock data and then plant a stale
	// high-water mark, simulating a previous RPC run that got to
	// block 999.
	if _, err := runCLI(t,
		"extract", "--source", "mock",
		"--output", dbPath,
		"--seed", "1", "--num-contracts", "3", "--total-blocks", "500",
	); err != nil {
		t.Fatalf("seed mock: %v", err)
	}
	db, err := storage.Open(context.Background(), dbPath)
	if err != nil {
		t.Fatalf("open: %v", err)
	}
	if _, err := db.SQL().ExecContext(context.Background(),
		`INSERT INTO schema_meta(key, value) VALUES(?, ?)
		 ON CONFLICT(key) DO UPDATE SET value=excluded.value`,
		extractor.RPCHighWaterKey, "999"); err != nil {
		t.Fatalf("plant high-water: %v", err)
	}
	_ = db.Close()

	// Now run `extract --source rpc --force` over blocks 1..2. If
	// --force didn't clear the high-water, the run would skip both
	// blocks (already "past" 999) — we'd see 0 blocks fetched and
	// the high-water would still read 999. With the fix it's cleared,
	// the run fetches both blocks, and high-water advances to 2.
	if _, err := runCLI(t,
		"extract", "--source", "rpc",
		"--output", dbPath,
		"--rpc", "http://fake-rpc.invalid",
		"--start", "1", "--end", "2",
		"--force",
	); err != nil {
		t.Fatalf("rpc --force: %v", err)
	}

	db, err = storage.Open(context.Background(), dbPath)
	if err != nil {
		t.Fatalf("re-open: %v", err)
	}
	defer db.Close()
	// high-water must be either missing (if the run wrote none) or
	// advanced past the planted 999 — i.e. it's NOT "999" anymore.
	var hw string
	row := db.SQL().QueryRowContext(context.Background(),
		`SELECT value FROM schema_meta WHERE key = ?`, extractor.RPCHighWaterKey)
	_ = row.Scan(&hw)
	if hw == "999" {
		t.Errorf("high-water still at stale value %q — --force did not clear it", hw)
	}
	// Force path also Reset()s the data tables, so the mock slots
	// from the seed run must have been wiped.
	slots, _ := db.CountSlots(context.Background())
	if slots != 0 {
		t.Errorf("slots=%d after --force, want 0 (Reset should have wiped prior mock rows)", slots)
	}
}

// ---- eda / fit / simulate / report -----------------------------------

func TestCLI_EDA_TextAndJSON(t *testing.T) {
	dir := withIsolatedCWD(t)
	dbPath := filepath.Join(dir, "eda.db")
	if _, err := runCLI(t,
		"extract", "--source", "mock",
		"--output", dbPath,
		"--seed", "7", "--num-contracts", "10", "--total-blocks", "5000",
	); err != nil {
		t.Fatalf("extract: %v", err)
	}
	// EDA requires a window that actually has data — our default
	// window-start is 200k which is outside the 5k extract horizon.
	// Use a window the extract produced.
	windowArgs := []string{
		"eda", "--db", dbPath,
		"--window-start", "1000",
		"--window-end", "5000",
	}

	textOut, err := runCLI(t, windowArgs...)
	if err != nil {
		t.Fatalf("eda text: %v\n%s", err, textOut)
	}
	// Section headers we explicitly emit from render.EDA.
	for _, want := range []string{"window", "intervals", "access frequency"} {
		if !strings.Contains(textOut, want) {
			t.Errorf("eda text missing %q\n---\n%s", want, textOut)
		}
	}

	jsonOut, err := runCLI(t, append(windowArgs, "--format", "json")...)
	if err != nil {
		t.Fatalf("eda json: %v\n%s", err, jsonOut)
	}
	var decoded map[string]any
	if err := json.Unmarshal([]byte(jsonOut), &decoded); err != nil {
		t.Fatalf("eda --format json: not valid JSON: %v\n%s", err, jsonOut)
	}
}

func TestCLI_FitKM_PrintsCurves(t *testing.T) {
	dir := withIsolatedCWD(t)
	dbPath := filepath.Join(dir, "km.db")
	if _, err := runCLI(t,
		"extract", "--source", "mock",
		"--output", dbPath,
		"--seed", "7", "--num-contracts", "15", "--total-blocks", "10000",
	); err != nil {
		t.Fatalf("extract: %v", err)
	}
	out, err := runCLI(t,
		"fit", "--db", dbPath, "--model", "km",
		"--stratify", "contract-type",
		"--window-start", "2000", "--window-end", "10000",
	)
	if err != nil {
		t.Fatalf("fit km: %v\n%s", err, out)
	}
	if !strings.Contains(out, "stratum") {
		t.Errorf("fit km output missing 'stratum' header:\n%s", out)
	}
}

func TestCLI_FitCox_SavesModelAndPrintsSections(t *testing.T) {
	dir := withIsolatedCWD(t)
	dbPath := filepath.Join(dir, "cox.db")
	modelPath := filepath.Join(dir, "cox.json")
	if _, err := runCLI(t,
		"extract", "--source", "mock",
		"--output", dbPath,
		"--seed", "7", "--num-contracts", "30", "--total-blocks", "20000",
	); err != nil {
		t.Fatalf("extract: %v", err)
	}
	out, err := runCLI(t,
		"fit", "--db", dbPath, "--model", "cox",
		"--window-start", "4000", "--window-end", "20000",
		"--save", modelPath,
	)
	if err != nil {
		t.Fatalf("fit cox: %v\n%s", err, out)
	}
	for _, want := range []string{"predictor", "Schoenfeld", "calibration"} {
		if !strings.Contains(out, want) {
			t.Errorf("fit cox output missing %q", want)
		}
	}
	if _, err := os.Stat(modelPath); err != nil {
		t.Errorf("--save did not create %s: %v", modelPath, err)
	}
}

func TestCLI_FitCox_UnknownModelErrors(t *testing.T) {
	dir := withIsolatedCWD(t)
	dbPath := filepath.Join(dir, "fit.db")
	if _, err := runCLI(t,
		"extract", "--source", "mock", "--output", dbPath,
		"--seed", "1", "--num-contracts", "5", "--total-blocks", "2000",
	); err != nil {
		t.Fatalf("extract: %v", err)
	}
	_, err := runCLI(t,
		"fit", "--db", dbPath, "--model", "bogus",
		"--window-start", "1000", "--window-end", "2000",
	)
	if err == nil || !strings.Contains(err.Error(), "unknown model") {
		t.Errorf("unknown --model: want 'unknown model' error, got %v", err)
	}
}

func TestCLI_Simulate_FixedPolicy(t *testing.T) {
	dir := withIsolatedCWD(t)
	dbPath := filepath.Join(dir, "sim.db")
	if _, err := runCLI(t,
		"extract", "--source", "mock",
		"--output", dbPath,
		"--seed", "7", "--num-contracts", "10", "--total-blocks", "5000",
	); err != nil {
		t.Fatalf("extract: %v", err)
	}
	out, err := runCLI(t,
		"simulate", "--db", dbPath,
		"--policy", "fixed-30d",
		"--window-start", "1000", "--window-end", "5000",
	)
	if err != nil {
		t.Fatalf("simulate fixed-30d: %v\n%s", err, out)
	}
	if !strings.Contains(out, "fixed-30d") {
		t.Errorf("simulate output missing policy name:\n%s", out)
	}
}

func TestCLI_Simulate_StatisticalRobustCostOverride(t *testing.T) {
	dir := withIsolatedCWD(t)
	dbPath := filepath.Join(dir, "sim.db")
	if _, err := runCLI(t,
		"extract", "--source", "mock",
		"--output", dbPath,
		"--seed", "7", "--num-contracts", "20", "--total-blocks", "10000",
	); err != nil {
		t.Fatalf("extract: %v", err)
	}
	// --ram-unit-cost / --miss-penalty hit costsFromFlags with
	// cmd.Flags().Changed() == true, overriding YAML defaults.
	out, err := runCLI(t,
		"simulate", "--db", dbPath,
		"--policy", "statistical",
		"--robust",
		"--window-start", "2000", "--window-end", "10000",
		"--ram-unit-cost", "1",
		"--miss-penalty", "1000",
		"--holdout", "0.3",
		"--format", "json",
	)
	if err != nil {
		t.Fatalf("simulate statistical --robust: %v\n%s", err, out)
	}
	// JSON output must parse and contain a policy row.
	var envelope struct {
		DataSource *extractor.Capability `json:"data_source"`
		Results    []map[string]any      `json:"results"`
	}
	if err := json.Unmarshal([]byte(out), &envelope); err != nil {
		t.Fatalf("not valid JSON: %v\n%s", err, out)
	}
	if len(envelope.Results) != 1 {
		t.Fatalf("expected 1 result row, got %d", len(envelope.Results))
	}
	if name, _ := envelope.Results[0]["Policy"].(string); name != "statistical-robust" {
		t.Errorf("Policy = %q, want 'statistical-robust'", name)
	}
	// The extract step earlier in this test populated the DB with the
	// mock extractor, so the simulate JSON must round-trip its
	// capability stamp — this is the guardrail that
	// extractor.WriteCapability + ReadCapability stay wired together.
	if envelope.DataSource == nil {
		t.Fatal("data_source missing from simulate JSON envelope")
	}
	if envelope.DataSource.Source != "mock" {
		t.Errorf("DataSource.Source = %q, want mock", envelope.DataSource.Source)
	}
}

func TestCLI_Report_EndToEnd(t *testing.T) {
	dir := withIsolatedCWD(t)
	dbPath := filepath.Join(dir, "report.db")
	if _, err := runCLI(t,
		"extract", "--source", "mock",
		"--output", dbPath,
		"--seed", "7", "--num-contracts", "25", "--total-blocks", "15000",
	); err != nil {
		t.Fatalf("extract: %v", err)
	}
	out, err := runCLI(t,
		"report", "--db", dbPath,
		"--window-start", "3000", "--window-end", "15000",
	)
	if err != nil {
		t.Fatalf("report: %v\n%s", err, out)
	}
	for _, want := range []string{"=== EDA ===", "=== Kaplan–Meier ===", "=== Tiering policies ===", "=== Cox PH"} {
		if !strings.Contains(out, want) {
			t.Errorf("report missing section header %q", want)
		}
	}
}

// ---- tiny glue error paths -------------------------------------------

func TestCLI_Extract_BadSourceErrors(t *testing.T) {
	withIsolatedCWD(t)
	_, err := runCLI(t, "extract", "--source", "nonexistent")
	if err == nil || !strings.Contains(err.Error(), "not supported") {
		t.Errorf("want unsupported-source error, got %v", err)
	}
}

func TestCLI_ExtractRPC_InvalidRangeErrors(t *testing.T) {
	dir := withIsolatedCWD(t)
	dbPath := filepath.Join(dir, "rpc.db")
	_, err := runCLI(t,
		"extract", "--source", "rpc",
		"--output", dbPath,
		"--start", "5", "--end", "0",
	)
	if err == nil {
		t.Fatal("want error for end < start, got nil")
	}
}

// ---- tiny JSON-RPC fake -----------------------------------------------

type rpcReq struct {
	JSONRPC string          `json:"jsonrpc"`
	ID      json.RawMessage `json:"id"`
	Method  string          `json:"method"`
	Params  []any           `json:"params"`
}

type rpcResp struct {
	JSONRPC string          `json:"jsonrpc"`
	ID      json.RawMessage `json:"id"`
	Result  any             `json:"result,omitempty"`
	Error   *rpcErr         `json:"error,omitempty"`
}

type rpcErr struct {
	Code    int    `json:"code"`
	Message string `json:"message"`
}

type roundTripperFunc func(*http.Request) (*http.Response, error)

func (f roundTripperFunc) RoundTrip(r *http.Request) (*http.Response, error) { return f(r) }

func fakeRPCClient(t *testing.T) *http.Client {
	t.Helper()
	return &http.Client{Transport: roundTripperFunc(func(r *http.Request) (*http.Response, error) {
		body, err := io.ReadAll(r.Body)
		if err != nil {
			return nil, err
		}
		var req rpcReq
		if err := json.Unmarshal(body, &req); err != nil {
			return &http.Response{
				StatusCode: http.StatusBadRequest,
				Body:       io.NopCloser(strings.NewReader(err.Error())),
				Header:     make(http.Header),
			}, nil
		}
		resp := rpcResp{JSONRPC: "2.0", ID: req.ID}
		switch req.Method {
		case "eth_getBlockByNumber":
			// Return an empty block with just a number. The extractor doesn't
			// care about transactions when there are no logs.
			blockHex, _ := req.Params[0].(string)
			n, _ := strconv.ParseUint(strings.TrimPrefix(blockHex, "0x"), 16, 64)
			resp.Result = map[string]any{
				"number":       "0x" + strconv.FormatUint(n, 16),
				"hash":         "0xabc",
				"timestamp":    "0x0",
				"transactions": []any{},
			}
		case "eth_getBlockReceipts":
			resp.Result = []any{}
		default:
			resp.Error = &rpcErr{Code: -32601, Message: "method not found"}
		}
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
	})}
}
