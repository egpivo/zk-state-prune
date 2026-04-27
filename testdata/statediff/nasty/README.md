# Nasty fixtures

Hand-crafted prestateTracer responses that exercise edge / hostile
shapes the regular [`testdata/statediff/`](..) fixtures don't.
Goal: prove the parser **fail-closes** (returns an explicit unmarshal
error or canonicalises the input) rather than panicking, silently
corrupting state, or producing surprising output.

Each fixture comes as a pair:

```
<name>.json            # input (raw JSON, may be malformed)
<name>.expected.json   # expected outcome
```

## Expected-file schema

```json
{
  "description": "human-readable note about what this exercises",
  "expect": "ok" | "unmarshal_error",
  "touches": [ { "contract": "...", "slotKey": "...", "access": "read|write" }, ... ]
}
```

- `expect: "ok"` — the input must successfully `json.Unmarshal` into
  `[]traceBlockEntry`, and `parseStateDiff` over each entry must produce
  exactly the listed `touches` (multiset comparison, lower-cased).
- `expect: "unmarshal_error"` — the input must fail to unmarshal.
  `parseStateDiff` is never called. `touches` must be empty / absent.

## Why this exists

We catch most format drifts via the regular fixtures; nasty fixtures
catch the **other** failure mode: weird-but-not-impossible inputs
that the parser must handle without panicking. Examples included:

- empty payloads (`[]`)
- `pre` / `post` / `storage` set to `null` (Geth occasionally does)
- `storage` typed as a number instead of a map (corrupt RPC payload)
- mixed-case checksum addresses (EIP-55) that must be canonicalised
- oversize / non-hex / unicode slot keys (we don't validate, but the
  pipeline must not blow up downstream)
- top-level garbage (`{"foo": ...}` instead of `[...]`)

Drop a new pair into this directory to extend coverage — the test
walks the directory at runtime, no Go code changes needed.
