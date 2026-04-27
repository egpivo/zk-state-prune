package extractor

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"strings"
	"sync/atomic"
	"testing"
	"time"
)

// rpcCall + isTransientHTTPError tests. We drive the transport layer
// directly via a fake RoundTripper so we don't need a real network or
// httptest.Server — the sandbox can't bind ports anyway.

// Tests need a small RetryBaseDelay so they don't sleep through real
// 200ms backoffs while running. 1ms is plenty for cron-friendly
// `go test -race`.
const testRetryDelay = 1 * time.Millisecond

func TestRPCCall_RetriesOnTransientErrorThenSucceeds(t *testing.T) {
	// Fail twice with "connection reset", succeed on attempt 3.
	// The retry wrapper should report success without surfacing
	// the transient errors to the caller.
	var attempts int32
	transport := roundTripperFunc(func(r *http.Request) (*http.Response, error) {
		n := atomic.AddInt32(&attempts, 1)
		if n < 3 {
			return nil, errors.New("read tcp 1.2.3.4:443: connection reset by peer")
		}
		return jsonOKResponse(t, 1, "ok"), nil
	})
	cfg := RPCConfig{
		Endpoint:       "http://example.invalid",
		HTTPClient:     &http.Client{Transport: transport},
		MaxRetries:     3,
		RetryBaseDelay: testRetryDelay,
	}
	var reqID int
	var out string
	if err := rpcCall(context.Background(), cfg, &reqID, "test", nil, &out); err != nil {
		t.Fatalf("expected success after retries, got %v", err)
	}
	if got := atomic.LoadInt32(&attempts); got != 3 {
		t.Errorf("attempts = %d, want 3 (2 retries + 1 final success)", got)
	}
	if out != "ok" {
		t.Errorf("out = %q, want %q", out, "ok")
	}
}

func TestRPCCall_DoesNotRetryProtocolError(t *testing.T) {
	// JSON-RPC -32601 method-not-found is a protocol error — the
	// server can't fix it on retry, so the wrapper should bail
	// immediately. Counting attempts catches a regression where
	// isTransientHTTPError accidentally classifies "rpc error" as
	// retryable.
	var attempts int32
	transport := roundTripperFunc(func(r *http.Request) (*http.Response, error) {
		atomic.AddInt32(&attempts, 1)
		return jsonRPCError(t, 1, -32601, "the method debug_traceBlockByNumber does not exist/is not available"), nil
	})
	cfg := RPCConfig{
		Endpoint:       "http://example.invalid",
		HTTPClient:     &http.Client{Transport: transport},
		MaxRetries:     5, // generous, just to make a regression count to 6
		RetryBaseDelay: testRetryDelay,
	}
	var reqID int
	err := rpcCall(context.Background(), cfg, &reqID, "test", nil, nil)
	if err == nil {
		t.Fatal("expected error, got nil")
	}
	if !strings.Contains(err.Error(), "-32601") {
		t.Errorf("error should preserve the rpc error code, got %v", err)
	}
	if got := atomic.LoadInt32(&attempts); got != 1 {
		t.Errorf("attempts = %d, want 1 (protocol error must not retry)", got)
	}
}

func TestRPCCall_GivesUpAfterMaxRetries(t *testing.T) {
	// All attempts fail with a transient error. The wrapper should
	// run MaxRetries+1 attempts then return a wrapped error
	// mentioning the give-up.
	var attempts int32
	transport := roundTripperFunc(func(r *http.Request) (*http.Response, error) {
		atomic.AddInt32(&attempts, 1)
		return nil, errors.New("dial tcp: i/o timeout")
	})
	cfg := RPCConfig{
		Endpoint:       "http://example.invalid",
		HTTPClient:     &http.Client{Transport: transport},
		MaxRetries:     2,
		RetryBaseDelay: testRetryDelay,
	}
	var reqID int
	err := rpcCall(context.Background(), cfg, &reqID, "test", nil, nil)
	if err == nil {
		t.Fatal("expected error after exhausting retries, got nil")
	}
	if !strings.Contains(err.Error(), "gave up") {
		t.Errorf("error should mention give-up, got %v", err)
	}
	if got := atomic.LoadInt32(&attempts); got != 3 {
		t.Errorf("attempts = %d, want 3 (1 initial + 2 retries)", got)
	}
}

func TestRPCCall_RetriesOn5xx(t *testing.T) {
	// Some endpoints return 503 mid-overload; should retry.
	var attempts int32
	transport := roundTripperFunc(func(r *http.Request) (*http.Response, error) {
		n := atomic.AddInt32(&attempts, 1)
		if n < 2 {
			return &http.Response{
				StatusCode: 503,
				Body:       io.NopCloser(strings.NewReader("upstream busy")),
				Header:     make(http.Header),
			}, nil
		}
		return jsonOKResponse(t, 1, "ok"), nil
	})
	cfg := RPCConfig{
		Endpoint:       "http://example.invalid",
		HTTPClient:     &http.Client{Transport: transport},
		MaxRetries:     3,
		RetryBaseDelay: testRetryDelay,
	}
	var reqID int
	var out string
	if err := rpcCall(context.Background(), cfg, &reqID, "test", nil, &out); err != nil {
		t.Fatalf("expected eventual success on retry past 503, got %v", err)
	}
	if got := atomic.LoadInt32(&attempts); got != 2 {
		t.Errorf("attempts = %d, want 2", got)
	}
}

func TestRPCCall_HonorsContextCancellationDuringBackoff(t *testing.T) {
	// Always-failing transport, but caller cancels the context.
	// Should return ctx.Err() rather than continuing to retry.
	transport := roundTripperFunc(func(r *http.Request) (*http.Response, error) {
		return nil, errors.New("connection reset by peer")
	})
	// Long backoff so the cancellation can race in.
	cfg := RPCConfig{
		Endpoint:       "http://example.invalid",
		HTTPClient:     &http.Client{Transport: transport},
		MaxRetries:     5,
		RetryBaseDelay: 5 * time.Second,
	}
	ctx, cancel := context.WithCancel(context.Background())
	go func() {
		time.Sleep(50 * time.Millisecond)
		cancel()
	}()
	var reqID int
	err := rpcCall(ctx, cfg, &reqID, "test", nil, nil)
	if !errors.Is(err, context.Canceled) {
		t.Errorf("expected context.Canceled, got %v", err)
	}
}

func TestIsTransientHTTPError_KnownPatterns(t *testing.T) {
	transient := []string{
		"do eth_getBlockByNumber: Post \"https://x\": context deadline exceeded",
		"do eth_getBlockReceipts: Post \"https://x\": read tcp ...: connection reset by peer",
		"do test: Post \"https://x\": dial tcp 1.2.3.4:443: connect: connection refused",
		"do test: Post \"https://x\": write tcp ...: broken pipe",
		"do test: Post \"https://x\": dial tcp: i/o timeout",
		"decode test: unexpected EOF",
		"test: HTTP 503: upstream busy",
		"test: HTTP 502: bad gateway",
		// DNS / network-layer flakes (added after observing
		// `dial tcp: lookup rpc.scroll.io: no such host` on a
		// real run — macOS DNS resolver briefly returns NXDOMAIN
		// then clears on retry).
		"do test: Post \"https://x\": dial tcp: lookup rpc.scroll.io: no such host",
		"do test: Post \"https://x\": dial tcp: connect: network is unreachable",
		"do test: Post \"https://x\": dial tcp: connect: host is unreachable",
	}
	for _, msg := range transient {
		if !isTransientHTTPError(errors.New(msg)) {
			t.Errorf("isTransientHTTPError(%q) = false, want true", msg)
		}
	}
	nonTransient := []string{
		"test: rpc error -32601: method not found",
		"test: rpc error -32602: invalid params",
		"test: HTTP 400: bad request",
		"test: HTTP 404: not found",
		"marshal test: json: unsupported type",
	}
	for _, msg := range nonTransient {
		if isTransientHTTPError(errors.New(msg)) {
			t.Errorf("isTransientHTTPError(%q) = true, want false (protocol-level)", msg)
		}
	}
	if isTransientHTTPError(nil) {
		t.Error("isTransientHTTPError(nil) should be false")
	}
}

// ---- helpers --------------------------------------------------------

func jsonOKResponse(t *testing.T, id int, result any) *http.Response {
	t.Helper()
	resultBytes, err := json.Marshal(result)
	if err != nil {
		t.Fatalf("marshal result: %v", err)
	}
	resp := rpcResponse{JSONRPC: "2.0", ID: id, Result: resultBytes}
	body, err := json.Marshal(resp)
	if err != nil {
		t.Fatalf("marshal response: %v", err)
	}
	h := make(http.Header)
	h.Set("Content-Type", "application/json")
	return &http.Response{
		StatusCode: 200,
		Body:       io.NopCloser(bytes.NewReader(body)),
		Header:     h,
	}
}

func jsonRPCError(t *testing.T, id, code int, message string) *http.Response {
	t.Helper()
	resp := rpcResponse{
		JSONRPC: "2.0",
		ID:      id,
		Error:   &rpcError{Code: code, Message: message},
	}
	body, err := json.Marshal(resp)
	if err != nil {
		t.Fatalf("marshal response: %v", err)
	}
	h := make(http.Header)
	h.Set("Content-Type", "application/json")
	return &http.Response{
		StatusCode: 200,
		Body:       io.NopCloser(bytes.NewReader(body)),
		Header:     h,
	}
}
