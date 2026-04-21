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

// InsertAccessEvents bulk-inserts access events inside a single
// transaction and returns the number of rows actually inserted.
// Rows dropped by the unique-index INSERT OR IGNORE (duplicates from
// a resume re-fetch) are not counted, so extractor diagnostics can
// report "events persisted" instead of "events attempted".
// Extractors should batch into reasonable chunks (~10k) to keep
// memory bounded while still amortizing the per-statement overhead.
func (d *DB) InsertAccessEvents(ctx context.Context, events []model.AccessEvent) (int64, error) {
	if len(events) == 0 {
		return 0, nil
	}
	tx, err := d.sql.BeginTx(ctx, nil)
	if err != nil {
		return 0, fmt.Errorf("begin tx: %w", err)
	}
	defer func() { _ = tx.Rollback() }()

	// INSERT OR IGNORE + the unique index on
	// (slot_id, block_number, access_type, tx_hash) makes event
	// insertion idempotent: an extractor crash between event flush
	// and its high-water-mark update will re-fetch the block on
	// resume and re-emit the same rows; the unique index drops the
	// duplicates instead of ballooning the event table. The
	// trade-off is that a legitimate repeat of the exact same tuple
	// collapses to one row — but in both the mock and the RPC
	// Transfer-log surrogate every event already carries a distinct
	// synthetic tx_hash, so collisions are limited to the resume
	// path we're explicitly de-duping here.
	stmt, err := tx.PrepareContext(ctx,
		`INSERT OR IGNORE INTO access_events(slot_id, block_number, access_type, tx_hash)
		 VALUES(?, ?, ?, ?)`)
	if err != nil {
		return 0, fmt.Errorf("prepare insert: %w", err)
	}
	defer stmt.Close()

	var inserted int64
	for _, e := range events {
		res, err := stmt.ExecContext(ctx, e.SlotID, e.BlockNumber, e.AccessType.String(), e.TxHash)
		if err != nil {
			return inserted, fmt.Errorf("insert event slot=%s block=%d: %w", e.SlotID, e.BlockNumber, err)
		}
		n, aerr := res.RowsAffected()
		if aerr == nil {
			inserted += n
		}
	}
	if err := tx.Commit(); err != nil {
		return inserted, fmt.Errorf("commit: %w", err)
	}
	return inserted, nil
}

// Reset truncates all data tables (contracts, slots, events) while leaving
// the schema in place. Order matters: children before parents to satisfy the
// foreign keys we enable in Open. Mock extractors call this to honour the
// idempotency contract on the Extractor interface.
func (d *DB) Reset(ctx context.Context) error {
	tables := []string{"access_events", "state_slots", "contracts"}
	tx, err := d.sql.BeginTx(ctx, nil)
	if err != nil {
		return fmt.Errorf("begin tx: %w", err)
	}
	defer func() { _ = tx.Rollback() }()
	for _, t := range tables {
		if _, err := tx.ExecContext(ctx, "DELETE FROM "+t); err != nil {
			return fmt.Errorf("truncate %s: %w", t, err)
		}
	}
	// Reset AUTOINCREMENT counters so a fresh run starts at id=1.
	if _, err := tx.ExecContext(ctx, `DELETE FROM sqlite_sequence WHERE name='access_events'`); err != nil {
		// sqlite_sequence only exists if any AUTOINCREMENT row was ever
		// inserted; missing-table errors are benign here.
		_ = err
	}
	if err := tx.Commit(); err != nil {
		return fmt.Errorf("commit: %w", err)
	}
	return nil
}

// SlotWithMeta bundles a slot with the contract metadata its parent row
// carries. Analysis passes need slot type, contract category, and the
// contract's deploy block (so ContractAge can be distinct from SlotAge),
// so we ship the whole bundle from a single join.
type SlotWithMeta struct {
	Slot        model.StateSlot
	Category    model.ContractCategory
	DeployBlock uint64
}

// IterateSlotEvents streams every slot in the DB together with its full
// access history, ordered by block number. fn is called once per slot;
// returning an error aborts the iteration.
//
// Implementation: slot metadata is loaded into a map once (O(num_slots)),
// then access_events is scanned in a single sorted pass keyed on slot_id.
// We flush whenever slot_id changes, so peak memory is bounded by the
// largest single slot's trace rather than the full event table.
func (d *DB) IterateSlotEvents(
	ctx context.Context,
	fn func(SlotWithMeta, []model.AccessEvent) error,
) error {
	meta, order, err := d.loadSlotMeta(ctx)
	if err != nil {
		return err
	}
	seen, err := d.streamSlotEvents(ctx, meta, fn)
	if err != nil {
		return err
	}
	return emitUnseenSlots(ctx, order, seen, meta, fn)
}

// streamSlotEvents scans access_events in (slot_id, block_number) order,
// buffering rows per slot and flushing each buffer to fn when the slot id
// changes. Returns the set of slot ids that produced at least one event
// so the caller can emit empty callbacks for the rest.
func (d *DB) streamSlotEvents(
	ctx context.Context,
	meta map[string]SlotWithMeta,
	fn func(SlotWithMeta, []model.AccessEvent) error,
) (map[string]bool, error) {
	rows, err := d.sql.QueryContext(ctx, `
		SELECT slot_id, block_number, access_type, tx_hash
		  FROM access_events
		 ORDER BY slot_id, block_number`)
	if err != nil {
		return nil, fmt.Errorf("query events: %w", err)
	}
	defer rows.Close()

	seen := make(map[string]bool, len(meta))
	var (
		curID string
		buf   []model.AccessEvent
	)
	flush := func() error {
		if curID == "" {
			return nil
		}
		m, ok := meta[curID]
		if !ok {
			// Orphan events (FK violation) — skip rather than panic.
			return nil
		}
		seen[curID] = true
		return fn(m, buf)
	}

	for rows.Next() {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		var e model.AccessEvent
		var at string
		if err := rows.Scan(&e.SlotID, &e.BlockNumber, &at, &e.TxHash); err != nil {
			return nil, fmt.Errorf("scan event: %w", err)
		}
		e.AccessType = model.ParseAccessType(at)
		if e.SlotID != curID {
			if err := flush(); err != nil {
				return nil, err
			}
			curID = e.SlotID
			buf = buf[:0]
		}
		buf = append(buf, e)
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("iter events: %w", err)
	}
	if err := flush(); err != nil {
		return nil, err
	}
	return seen, nil
}

// emitUnseenSlots fires fn with an empty event slice for every slot
// that had zero access events, so BuildIntervals can emit a
// fully-censored interval for them.
func emitUnseenSlots(
	ctx context.Context,
	order []string,
	seen map[string]bool,
	meta map[string]SlotWithMeta,
	fn func(SlotWithMeta, []model.AccessEvent) error,
) error {
	for _, id := range order {
		if seen[id] {
			continue
		}
		if err := ctx.Err(); err != nil {
			return err
		}
		if err := fn(meta[id], nil); err != nil {
			return err
		}
	}
	return nil
}

func (d *DB) loadSlotMeta(ctx context.Context) (map[string]SlotWithMeta, []string, error) {
	rows, err := d.sql.QueryContext(ctx, `
		SELECT s.slot_id, s.contract_addr, s.slot_index, s.slot_type,
		       s.created_at, s.last_access, s.access_count, s.is_active,
		       c.contract_type, c.deploy_block
		  FROM state_slots s
		  JOIN contracts    c ON c.address = s.contract_addr
		 ORDER BY s.slot_id`)
	if err != nil {
		return nil, nil, fmt.Errorf("query slots: %w", err)
	}
	defer rows.Close()

	meta := make(map[string]SlotWithMeta)
	order := make([]string, 0)
	for rows.Next() {
		var (
			s             model.StateSlot
			slotType, cat string
			active        int
			deployBlock   uint64
		)
		if err := rows.Scan(
			&s.SlotID, &s.ContractAddr, &s.SlotIndex, &slotType,
			&s.CreatedAt, &s.LastAccess, &s.AccessCount, &active,
			&cat, &deployBlock,
		); err != nil {
			return nil, nil, fmt.Errorf("scan slot: %w", err)
		}
		s.SlotType = model.ParseSlotType(slotType)
		s.IsActive = active != 0
		meta[s.SlotID] = SlotWithMeta{
			Slot:        s,
			Category:    model.ParseContractCategory(cat),
			DeployBlock: deployBlock,
		}
		order = append(order, s.SlotID)
	}
	if err := rows.Err(); err != nil {
		return nil, nil, fmt.Errorf("iter slots: %w", err)
	}
	return meta, order, nil
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
