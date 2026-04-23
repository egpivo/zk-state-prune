// Package extractor turns external state-diff sources (RPC, files, or
// synthetic generators) into rows in the zksp analysis DB.
//
// All extractors share the Extractor interface so the analysis pipeline can
// be driven by a mock generator today and a real RPC client tomorrow without
// touching downstream code.
package extractor

import (
	"context"
	"encoding/json"
	"fmt"

	"github.com/egpivo/zk-state-prune/internal/storage"
)

// CapabilityKey is the schema_meta row that stores the most-recent
// Extractor.Capability() as a JSON blob. Updated after every
// successful Extract; read by report / simulate so their output
// header can self-document which data source the numbers came from.
const CapabilityKey = "last_extractor_capability"

// Extractor is anything that can populate a zksp DB with contracts, slots
// and access events.
type Extractor interface {
	// Extract is expected to be idempotent: calling it twice on the same DB
	// must not duplicate access events. Mock implementations achieve this by
	// truncating; real RPC implementations should use a high-water mark.
	Extract(ctx context.Context, db *storage.DB) error

	// Capability describes what slot touches the extractor observes
	// and how it mints slot_id. Downstream report / simulate output
	// stamp this so a reader of a Brier score or a cost table knows
	// whether the data was produced by a full state-diff source, a
	// Transfer-log surrogate, or a synthetic generator. Must be
	// stable — i.e. a constant the implementation can return from
	// any state.
	Capability() Capability
}

// Capability is a self-description of what an Extractor sees and how
// it mints slot_id. It is persisted into schema_meta on every
// successful Extract call so downstream consumers can self-document
// their output without re-inspecting the extractor.
//
// Two extractors with different capabilities should produce
// comparable rows (slot_id format differs but the survival pipeline
// treats slot_id as opaque), but downstream comparisons of Hill α,
// censoring rate, or cost regime between sources with different
// capabilities should be flagged — a Transfer-log surrogate will
// systematically under-report writes, for example.
type Capability struct {
	// Source is the --source flag value this extractor answers to
	// ("mock" / "rpc" / "statediff" / …). Carried here so JSON
	// reports don't need to cross-reference the CLI.
	Source string `json:"source"`

	// ObservesReads is true iff the extractor captures SLOAD-level
	// state reads (not just writes). Transfer-log surrogates set this
	// to false because Transfer events are emitted on writes only.
	ObservesReads bool `json:"observes_reads"`

	// ObservesNonTransferWrite is true iff the extractor captures
	// every SSTORE, not just writes that emit an ERC-20 / ERC-721
	// Transfer event. Mock and real state-diff extractors set this
	// to true; the Transfer-log surrogate sets it to false.
	ObservesNonTransferWrite bool `json:"observes_non_transfer_write"`

	// SlotIDForm is a short human-readable description of how the
	// extractor mints slot_id (e.g. "synthetic / deterministic",
	// "contract:holder (surrogate)", "contract:slotkey (real)").
	// Informational only; downstream code never parses it.
	SlotIDForm string `json:"slot_id_form"`
}

// WriteCapability stamps cap into schema_meta under CapabilityKey as
// a JSON blob. Callers typically invoke it from the CLI right after
// a successful Extract, so report / simulate invocations on the same
// DB can later read what data source produced the rows.
func WriteCapability(ctx context.Context, db *storage.DB, cap Capability) error {
	b, err := json.Marshal(cap)
	if err != nil {
		return fmt.Errorf("marshal capability: %w", err)
	}
	_, err = db.SQL().ExecContext(ctx,
		`INSERT INTO schema_meta(key, value) VALUES(?, ?)
		 ON CONFLICT(key) DO UPDATE SET value=excluded.value`,
		CapabilityKey, string(b))
	if err != nil {
		return fmt.Errorf("write capability: %w", err)
	}
	return nil
}

// ReadCapability returns the most recently persisted Capability for
// this DB. The second return distinguishes "no extractor has run yet"
// from a SQL error — a fresh DB returns (zero, false, nil), which the
// CLI can surface as "unknown data source" rather than falsely
// stamping the output.
func ReadCapability(ctx context.Context, db *storage.DB) (Capability, bool, error) {
	var v string
	err := db.SQL().QueryRowContext(ctx,
		`SELECT value FROM schema_meta WHERE key = ?`, CapabilityKey).
		Scan(&v)
	if err != nil {
		if err.Error() == "sql: no rows in result set" {
			return Capability{}, false, nil
		}
		return Capability{}, false, fmt.Errorf("read capability: %w", err)
	}
	var cap Capability
	if err := json.Unmarshal([]byte(v), &cap); err != nil {
		return Capability{}, false, fmt.Errorf("parse capability %q: %w", v, err)
	}
	return cap, true, nil
}
