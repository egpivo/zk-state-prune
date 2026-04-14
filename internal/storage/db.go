// Package storage wraps the SQLite analysis DB used by zksp. It owns
// connection lifecycle, schema migration, and the small set of CRUD helpers
// the rest of the codebase needs.
package storage

import (
	"context"
	"database/sql"
	"fmt"

	_ "modernc.org/sqlite" // pure-Go SQLite driver

	"github.com/egpivo/zk-state-prune/internal/model"
)

// DB is a thin wrapper around *sql.DB that exposes typed helpers.
type DB struct {
	sql *sql.DB
}

// Open opens (or creates) the SQLite file at path and applies the current
// schema. The returned DB is safe for concurrent use.
func Open(ctx context.Context, path string) (*DB, error) {
	sqlDB, err := sql.Open("sqlite", path)
	if err != nil {
		return nil, fmt.Errorf("open sqlite %q: %w", path, err)
	}
	// Pragmas: WAL gives us concurrent readers during long extractions;
	// foreign_keys is off by default in SQLite and we want it on.
	pragmas := []string{
		"PRAGMA journal_mode = WAL",
		"PRAGMA synchronous  = NORMAL",
		"PRAGMA foreign_keys = ON",
	}
	for _, p := range pragmas {
		if _, err := sqlDB.ExecContext(ctx, p); err != nil {
			_ = sqlDB.Close()
			return nil, fmt.Errorf("pragma %q: %w", p, err)
		}
	}
	if err := migrate(ctx, sqlDB); err != nil {
		_ = sqlDB.Close()
		return nil, fmt.Errorf("migrate: %w", err)
	}
	return &DB{sql: sqlDB}, nil
}

// Close releases the underlying connection pool.
func (d *DB) Close() error { return d.sql.Close() }

// SQL exposes the raw *sql.DB for callers (analysis passes) that need to
// stream rows directly. Prefer the typed helpers below for writes.
func (d *DB) SQL() *sql.DB { return d.sql }

// UpsertContract inserts a contract row or updates its mutable fields.
func (d *DB) UpsertContract(ctx context.Context, c model.ContractMeta) error {
	_, err := d.sql.ExecContext(ctx,
		`INSERT INTO contracts(address, contract_type, deploy_block, total_slots, active_slots)
		 VALUES(?, ?, ?, ?, ?)
		 ON CONFLICT(address) DO UPDATE SET
		   contract_type = excluded.contract_type,
		   deploy_block  = excluded.deploy_block,
		   total_slots   = excluded.total_slots,
		   active_slots  = excluded.active_slots`,
		c.Address, c.ContractType.String(), c.DeployBlock, c.TotalSlots, c.ActiveSlots,
	)
	if err != nil {
		return fmt.Errorf("upsert contract %s: %w", c.Address, err)
	}
	return nil
}

// UpsertSlot inserts a slot row or refreshes its lifecycle fields.
func (d *DB) UpsertSlot(ctx context.Context, s model.StateSlot) error {
	active := 0
	if s.IsActive {
		active = 1
	}
	_, err := d.sql.ExecContext(ctx,
		`INSERT INTO state_slots(slot_id, contract_addr, slot_index, slot_type,
		                         created_at, last_access, access_count, is_active)
		 VALUES(?, ?, ?, ?, ?, ?, ?, ?)
		 ON CONFLICT(slot_id) DO UPDATE SET
		   last_access  = excluded.last_access,
		   access_count = excluded.access_count,
		   is_active    = excluded.is_active`,
		s.SlotID, s.ContractAddr, s.SlotIndex, s.SlotType.String(),
		s.CreatedAt, s.LastAccess, s.AccessCount, active,
	)
	if err != nil {
		return fmt.Errorf("upsert slot %s: %w", s.SlotID, err)
	}
	return nil
}

// InsertAccessEvents bulk-inserts access events inside a single transaction.
// Extractors should batch into reasonable chunks (~10k) to keep memory bounded
// while still amortizing the per-statement overhead.
func (d *DB) InsertAccessEvents(ctx context.Context, events []model.AccessEvent) error {
	if len(events) == 0 {
		return nil
	}
	tx, err := d.sql.BeginTx(ctx, nil)
	if err != nil {
		return fmt.Errorf("begin tx: %w", err)
	}
	defer func() { _ = tx.Rollback() }()

	stmt, err := tx.PrepareContext(ctx,
		`INSERT INTO access_events(slot_id, block_number, access_type, tx_hash)
		 VALUES(?, ?, ?, ?)`)
	if err != nil {
		return fmt.Errorf("prepare insert: %w", err)
	}
	defer stmt.Close()

	for _, e := range events {
		if _, err := stmt.ExecContext(ctx, e.SlotID, e.BlockNumber, e.AccessType.String(), e.TxHash); err != nil {
			return fmt.Errorf("insert event slot=%s block=%d: %w", e.SlotID, e.BlockNumber, err)
		}
	}
	if err := tx.Commit(); err != nil {
		return fmt.Errorf("commit: %w", err)
	}
	return nil
}

// CountSlots returns the total number of state slot rows. Useful for tests
// and CLI status output.
func (d *DB) CountSlots(ctx context.Context) (int64, error) {
	var n int64
	if err := d.sql.QueryRowContext(ctx, `SELECT COUNT(*) FROM state_slots`).Scan(&n); err != nil {
		return 0, fmt.Errorf("count slots: %w", err)
	}
	return n, nil
}

// CountAccessEvents returns the total number of access event rows.
func (d *DB) CountAccessEvents(ctx context.Context) (int64, error) {
	var n int64
	if err := d.sql.QueryRowContext(ctx, `SELECT COUNT(*) FROM access_events`).Scan(&n); err != nil {
		return 0, fmt.Errorf("count events: %w", err)
	}
	return n, nil
}
