package extractor

import (
	"testing"
)

func TestDefaultRPCConfig(t *testing.T) {
	cfg := DefaultRPCConfig()
	if cfg.Endpoint != ScrollPublicRPC {
		t.Errorf("Endpoint=%q, want %q", cfg.Endpoint, ScrollPublicRPC)
	}
	if cfg.BatchSize <= 0 {
		t.Errorf("BatchSize=%d must be positive", cfg.BatchSize)
	}
	if cfg.HTTPClient == nil {
		t.Error("HTTPClient must be set so NewRPCExtractor doesn't fall back")
	} else if cfg.HTTPClient.Timeout == 0 {
		t.Error("HTTPClient.Timeout=0 would hang a mis-configured RPC forever")
	}
}

func TestRPCExtractor_LastDiagnosticsZeroBeforeExtract(t *testing.T) {
	ex, err := NewRPCExtractor(RPCConfig{Endpoint: "http://x", Start: 1, End: 2})
	if err != nil {
		t.Fatalf("NewRPCExtractor: %v", err)
	}
	got := ex.LastDiagnostics()
	if got != (RPCDiagnostics{}) {
		t.Errorf("LastDiagnostics before Extract = %+v, want zero value", got)
	}
}

func TestMockExtractor_LastDiagnosticsZeroBeforeExtract(t *testing.T) {
	ex := NewMockExtractor(DefaultMockConfig())
	if got := ex.LastDiagnostics(); got != (Diagnostics{}) {
		t.Errorf("LastDiagnostics before Extract = %+v, want zero value", got)
	}
}
