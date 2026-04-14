package storage

import (
	"context"
	"database/sql"
	"fmt"

	"github.com/egpivo/zk-state-prune/internal/model"
)

// schemaVersion bumps whenever model.Schema changes shape. Phase 1 only
// supports forward, idempotent migrations: we just (re-)apply the current
// schema and stamp the version. When we need destructive migrations later,
// add a versioned switch here.
const schemaVersion = "1"

func migrate(ctx context.Context, db *sql.DB) error {
	if _, err := db.ExecContext(ctx, model.Schema); err != nil {
		return fmt.Errorf("apply schema: %w", err)
	}
	_, err := db.ExecContext(ctx,
		`INSERT INTO schema_meta(key, value) VALUES('version', ?)
		 ON CONFLICT(key) DO UPDATE SET value=excluded.value`,
		schemaVersion,
	)
	if err != nil {
		return fmt.Errorf("stamp schema version: %w", err)
	}
	return nil
}
