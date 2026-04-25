package domain

// Schema is the canonical SQLite schema for the zksp analysis DB. It is
// applied verbatim by the storage layer at Open() time; migrations live
// alongside it in internal/storage/migrations.go.
//
// Design notes:
//   - state_slots is keyed by slot_id — an opaque deterministic string
//     each extractor mints in its own format (see StateSlot docs); the
//     storage layer treats it as a black-box primary key.
//   - access_events is append-only and indexed on (slot_id, block_number) so
//     survival analysis can stream inter-access intervals per slot.
//   - Enums are stored as TEXT so the DB stays human-inspectable; the
//     domain package owns the parse/format functions.
//   - schema_meta is a small key/value table for schema version, the
//     RPC high-water mark, and the most-recent extractor capability
//     stamp (so downstream report / simulate output can self-document
//     which data source produced the numbers).
const Schema = `
CREATE TABLE IF NOT EXISTS contracts (
    address       TEXT PRIMARY KEY,
    contract_type TEXT NOT NULL,
    deploy_block  INTEGER NOT NULL,
    total_slots   INTEGER NOT NULL DEFAULT 0,
    active_slots  INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_contracts_type ON contracts(contract_type);

CREATE TABLE IF NOT EXISTS state_slots (
    slot_id       TEXT PRIMARY KEY,
    contract_addr TEXT NOT NULL,
    slot_index    INTEGER NOT NULL,
    slot_type     TEXT NOT NULL,
    created_at    INTEGER NOT NULL,
    last_access   INTEGER NOT NULL,
    access_count  INTEGER NOT NULL DEFAULT 0,
    is_active     INTEGER NOT NULL DEFAULT 1,
    FOREIGN KEY (contract_addr) REFERENCES contracts(address)
);

CREATE INDEX IF NOT EXISTS idx_slots_contract    ON state_slots(contract_addr);
CREATE INDEX IF NOT EXISTS idx_slots_last_access ON state_slots(last_access);
CREATE INDEX IF NOT EXISTS idx_slots_type        ON state_slots(slot_type);
CREATE INDEX IF NOT EXISTS idx_slots_active      ON state_slots(is_active);

CREATE TABLE IF NOT EXISTS access_events (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    slot_id      TEXT NOT NULL,
    block_number INTEGER NOT NULL,
    access_type  TEXT NOT NULL,
    tx_hash      TEXT NOT NULL,
    FOREIGN KEY (slot_id) REFERENCES state_slots(slot_id)
);

CREATE INDEX IF NOT EXISTS idx_events_slot_block ON access_events(slot_id, block_number);
CREATE INDEX IF NOT EXISTS idx_events_block      ON access_events(block_number);

-- Idempotency guard on event insertion. Any re-issued (slot_id,
-- block_number, access_type, tx_hash) tuple collapses to the first
-- surviving row instead of duplicating. This makes the RPC
-- extractor's "crash between event flush and high-water write →
-- resume re-fetches → same events again" path cost-free: on resume
-- every row in the re-fetched block hits this unique index and is
-- silently skipped by the INSERT OR IGNORE in storage.DB.
CREATE UNIQUE INDEX IF NOT EXISTS idx_events_unique
    ON access_events(slot_id, block_number, access_type, tx_hash);

CREATE TABLE IF NOT EXISTS schema_meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
`
