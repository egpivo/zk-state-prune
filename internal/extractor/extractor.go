// Package extractor turns external state-diff sources (RPC, files, or
// synthetic generators) into rows in the zksp analysis DB.
//
// All extractors share the Extractor interface so the analysis pipeline can
// be driven by a mock generator today and a real RPC client tomorrow without
// touching downstream code.
package extractor

import (
	"context"

	"github.com/egpivo/zk-state-prune/internal/storage"
)

// Extractor is anything that can populate a zksp DB with contracts, slots
// and access events.
type Extractor interface {
	// Extract is expected to be idempotent: calling it twice on the same DB
	// must not duplicate access events. Mock implementations achieve this by
	// truncating; real RPC implementations should use a high-water mark.
	Extract(ctx context.Context, db *storage.DB) error
}
