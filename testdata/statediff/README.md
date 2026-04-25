# statediff trace fixtures

Hand-crafted `debug_traceBlockByNumber` (prestateTracer + diffMode)
responses, paired with `<name>.expected.json` files that pin the
exact `(contract, slotKey, access)` triples the parser should emit.

The Go test
[`TestStateDiffExtractor_FixtureRegression`](../../internal/extractor/statediff_fixtures_test.go)
walks this directory at run time — drop in a new
`<name>.json` + `<name>.expected.json` pair and the next
`go test ./internal/extractor/...` exercises it. No code changes.

## When to add a fixture

- A real Geth / Erigon / Reth response surprised the parser at run
  time (e.g. a new field, a renamed key, an unexpected null).
  **Sanitize the addresses / values first** — these files are
  checked in; they should never carry production-private data.
- You're closing a parser bug — add the response shape that
  triggered it so the regression bricks against the same shape.

## File pair contract

```
block_<topic>.json            ← the trace as Geth would return it
block_<topic>.expected.json   ← what parseStateDiff should emit
```

`expected.json` shape:

```json
{
  "description": "one-sentence what this fixture exercises",
  "touches": [
    {"contract": "0x...", "slotKey": "0x...", "access": "read|write"}
  ]
}
```

Touches order doesn't matter — the test compares as a sorted
multiset since `parseStateDiff` walks Go maps.

## Existing fixtures

| File | Exercises |
|------|-----------|
| `block_pure_writes.json`     | one tx, slot in pre+post → write |
| `block_pure_reads.json`      | view-only call, slots only in pre → all reads |
| `block_mixed.json`           | three txs across two contracts (per-tx parser output, not the per-block aggregation) |
| `block_no_storage.json`      | balance-only ETH transfer, no storage maps → zero touches |
| `block_multi_contract.json`  | router-style tx touching three contracts in one go |
