package storage

import (
	"context"
	"database/sql"
	"fmt"
	"strconv"
	"strings"

	"github.com/egpivo/zk-state-prune/internal/domain"
)

// schemaVersion is the current on-disk schema version. Bumps whenever
// domain.Schema adds a constraint that could fail on existing rows, so
// the migration path can dedup / backfill before the new DDL runs.
//
// History:
//
//	v1: initial — contracts / state_slots / access_events.
//	v2: unique index (slot_id, block_number, access_type, tx_hash) on
//	    access_events. Existing v1 DBs with duplicate events (possible
//	    if a user ran an old build that used plain INSERT on the RPC
//	    extractor's resume path) would otherwise fail at index
//	    creation; the v1→v2 migration pre-dedups so the CREATE
//	    UNIQUE INDEX below succeeds on any history.
const schemaVersion = "2"

func migrate(ctx context.Context, db *sql.DB) error {
	prior, err := priorSchemaVersion(ctx, db)
	if err != nil {
		return fmt.Errorf("read prior schema version: %w", err)
	}
	// v1 (or prior absent) → v2: drop duplicates so the unique
	// index applied below doesn't collide.
	if prior < 2 {
		if err := dedupAccessEvents(ctx, db); err != nil {
			return fmt.Errorf("dedup access_events before unique index: %w", err)
		}
	}
	if _, err := db.ExecContext(ctx, domain.Schema); err != nil {
		return fmt.Errorf("apply schema: %w", err)
	}
	if _, err := db.ExecContext(ctx,
		`INSERT INTO schema_meta(key, value) VALUES('version', ?)
		 ON CONFLICT(key) DO UPDATE SET value=excluded.value`,
		schemaVersion); err != nil {
		return fmt.Errorf("stamp schema version: %w", err)
	}
	return nil
}

// priorSchemaVersion returns the schema version stamped in schema_meta,
// or 0 if either the table or the row is absent (first-time Open on a
// new DB). A missing table is the brand-new case; a missing row is the
// pre-v1 legacy case. Both want the same treatment: run every
// migration step.
func priorSchemaVersion(ctx context.Context, db *sql.DB) (int, error) {
	var v string
	err := db.QueryRowContext(ctx,
		`SELECT value FROM schema_meta WHERE key = 'version'`).Scan(&v)
	if err != nil {
		if err == sql.ErrNoRows {
			return 0, nil
		}
		if strings.Contains(err.Error(), "no such table") {
			return 0, nil
		}
		return 0, err
	}
	n, perr := strconv.Atoi(v)
	if perr != nil {
		return 0, fmt.Errorf("parse version %q: %w", v, perr)
	}
	return n, nil
}

// dedupAccessEvents removes duplicate rows from access_events so the
// v2 unique index can be created on DBs that predate it. A new DB
// where access_events hasn't been created yet returns quietly — nothing
// to dedup. Deduplication rule: for each
// (slot_id, block_number, access_type, tx_hash) group keep the row
// with the smallest primary-key id, delete the rest.
func dedupAccessEvents(ctx context.Context, db *sql.DB) error {
	_, err := db.ExecContext(ctx, `
		DELETE FROM access_events
		WHERE id NOT IN (
		    SELECT MIN(id) FROM access_events
		    GROUP BY slot_id, block_number, access_type, tx_hash
		)`)
	if err != nil && strings.Contains(err.Error(), "no such table") {
		return nil
	}
	return err
}
