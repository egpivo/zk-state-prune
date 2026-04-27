# Extract limits — calibration notes

The RPC extractor supports three opt-in per-block guardrails:

- `--max-events-per-block`
- `--max-contracts-per-block`
- `--max-slots-per-block`

When any tally on the block-under-extraction would exceed its limit,
the extractor logs a structured ERROR and returns a wrapped error
(fail-closed). Nothing for that block is persisted. The defaults
preserve existing behaviour (zero ⇒ no limit), so adopting them is
opt-in via flag.

## Why fail-closed, not truncate

The realistic failure modes that hit these limits are:

- Token launch / airdrop spike (legitimate but unusual)
- Smart-contract spam attack
- Misconfigured RPC returning the wrong block payload
- A parser bug that's now over-counting touches

For all four, **silent truncation corrupts the access_events stream
the survival model trains on**, and a "skip block" mode would leave
gaps the resume / high-water logic isn't designed to bridge. The user
should investigate the specific block and either raise the limit or
exclude the window from the extract run.

## Calibration data — Scroll mainnet, 100k blocks

Source: `testdata/runs/scroll_100k/scroll.db` (rpc Transfer-log
surrogate, blocks `[33,400,000, 33,500,000)`, 26,444 of which had
any Transfer activity).

Per-block percentiles over the 26.4k non-empty blocks:

| Metric | p50 | p90 | p99 | p99.9 | observed max | proposed limit (10×) |
|---|---:|---:|---:|---:|---:|---:|
| events / block | 4 | 10 | 25 | 51 | **218** | **2,200** |
| distinct contracts / block | 2 | 3 | 4 | 5 | **12** | **120** |
| distinct slots / block | 4 | 10 | 22 | 40 | **218** | **2,200** |

The two blocks at observed max:

| block | events |
|---|---:|
| 33,468,137 | 218 |
| 33,422,202 | 199 |

Both are ~50× the median — consistent with token-launch / airdrop
activity and well within the proposed 2200 cap.

## How to reproduce the calibration

```bash
sqlite3 testdata/runs/scroll_100k/scroll.db <<'SQL'
WITH per_block AS (
  SELECT block_number, COUNT(*) AS events
  FROM access_events
  GROUP BY block_number
),
ordered AS (
  SELECT events, ROW_NUMBER() OVER (ORDER BY events) AS rn,
         COUNT(*) OVER () AS n
  FROM per_block
)
SELECT
  (SELECT events FROM ordered WHERE rn = CAST(0.50  * n AS INT)) AS p50,
  (SELECT events FROM ordered WHERE rn = CAST(0.90  * n AS INT)) AS p90,
  (SELECT events FROM ordered WHERE rn = CAST(0.99  * n AS INT)) AS p99,
  (SELECT events FROM ordered WHERE rn = CAST(0.999 * n AS INT)) AS p999,
  (SELECT MAX(events) FROM per_block) AS max,
  (SELECT COUNT(*) FROM per_block) AS n_blocks;
SQL
```

Substitute `COUNT(DISTINCT s.contract_addr)` (joining `state_slots`)
for the contracts axis, and `COUNT(DISTINCT slot_id)` for the slots
axis.

## Recommended invocation

```bash
zksp extract --source rpc \
    --start 33400000 --end 33500000 \
    --output testdata/runs/scroll_100k/scroll.db \
    --max-events-per-block 2200 \
    --max-contracts-per-block 120 \
    --max-slots-per-block 2200
```

## Domain-binding via `schema_meta`

Each successful Extract stamps the limits used into
`schema_meta.extract_limits` as JSON, e.g.:

```json
{
  "source": "rpc",
  "max_events_per_block": 2200,
  "max_contracts_per_block": 120,
  "max_slots_per_block": 2200
}
```

A subsequent `--resume` reads the stamp and **refuses to continue
if the new flags differ** — a hybrid DB (some blocks filtered at
2200, the rest at 4400) is analysis-incoherent. The user gets a
clear error pointing at `--force` (clean slate) or matching flags
(genuine resume).

## Statediff

These limits target the rpc Transfer-log surrogate. The statediff
extractor (`debug_traceBlockByNumber + prestateTracer`) observes
every state read/write, so its per-block tallies will be 1–2 orders
of magnitude higher. Calibration for statediff will land separately
once a statediff DB exists; the limits framework above is
source-agnostic by design and can be re-used as-is.
