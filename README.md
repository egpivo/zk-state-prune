# zk-state-prune

[![CI](https://github.com/egpivo/zk-state-prune/actions/workflows/ci.yml/badge.svg)](https://github.com/egpivo/zk-state-prune/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/egpivo/zk-state-prune/graph/badge.svg?token=siqgL91f4o)](https://codecov.io/gh/egpivo/zk-state-prune)
[![Go Report Card](https://goreportcard.com/badge/github.com/egpivo/zk-state-prune?style=flat-square)](https://goreportcard.com/report/github.com/egpivo/zk-state-prune)
[![Go Version](https://img.shields.io/github/go-mod/go-version/egpivo/zk-state-prune)](https://github.com/egpivo/zk-state-prune/blob/main/go.mod)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Cost-aware hot/cold tiering for ZK rollup state, driven by survival analysis.**

`zk-state-prune` (binary: `zksp`) treats on-chain state access as a survival
process, fits a calibrated per-slot access-probability model, and picks a
tiering decision per slot from the cost surrogate

```
d_i* = argmin_d  c(d) + ℓ(d) · p_i(τ)
```

where `c(d)` is the per-tier holding cost, `ℓ(d)` the miss penalty, and
`p_i(τ) = 1 − S_i(τ)` the calibrated probability of access within horizon `τ`.
The threshold collapses to `p* = (c_ram · τ) / ℓ_miss`: demote a slot at the
first idle duration where its conditional access probability drops below `p*`.

## Pipeline

```
extract → EDA → survival fit → calibration → tiering simulation
```

- **Extract** — mock generator (power-law access, co-access, seasonality) or
  Scroll JSON-RPC (Transfer-log surrogate; see
  [rpc_extractor.go](internal/extractor/rpc_extractor.go) for scope).
- **EDA** — Hill-α tail, per-contract Jaccard co-access, temporal
  autocorrelation (descriptive diagnostics, not covariates).
- **Fit** — Kaplan–Meier, Cox PH with stratified baselines, Schoenfeld PH
  test, isotonic recalibration on 70/30 holdout, split-conformal ε
  (point + conditional).
- **Simulate** — NoPrune / FixedIdle / Statistical (point + robust) compared
  on `RAMRatio`, `HotHitCoverage`, miss count, `TotalCost`.

## CLI

```
zksp extract  --source mock|rpc|statediff  --output <db>
                                          [--rpc <url> --start N --end M]
zksp eda      --db <db>                    [--format text|json]
zksp fit      --db <db> --model km|cox     [--stratify contract-type]
                                          [--save <path.json>]
zksp simulate --db <db> --policy no-prune|fixed-30d|fixed-90d|statistical
                                          [--model <path.json>] [--robust]
                                          [--ram-unit-cost N --miss-penalty N]
zksp report   --db <db>                    [--model <path.json>]
```

`--config configs/default.yaml` overrides hardcoded defaults; flags override
config. `zksp extract --force` wipes the target DB and clears the RPC
high-water mark.

### Data-source capability matrix

Each extractor self-declares what slot touches it observes. After
every `extract`, the capability is stamped into `schema_meta` so
`report --format json` and `simulate --format json` carry a
`data_source` field; the text modes print a one-line header.

| `--source`  | reads | non-Transfer writes | `slot_id` form                |
|-------------|:-----:|:-------------------:|-------------------------------|
| `mock`      |   ✓   |          ✓          | synthetic / deterministic     |
| `rpc`       |   ✗   |          ✗          | `contract:holder` (Transfer-log surrogate) |
| `statediff` |   ✓   |          ✓          | `contract:slotkey` (real `debug_traceBlockByNumber` + `prestateTracer`) |

Reading a Brier score or a cost table without the capability stamp is
an honest-scope violation: a Transfer-log surrogate systematically
under-reports writes, which skews both the heavy-tail diagnostics and
the cost-regime finding.

`--source statediff` requires an archive-capable RPC endpoint that
exposes `debug_traceBlockByNumber` (Alchemy Growth, QuickNode
archive, self-hosted Erigon, …). Public chain endpoints generally
refuse the method; the extractor surfaces a directly actionable
error when that happens instead of silently degrading to the
surrogate.

## QA

Two layers of QA matter in this repo:

- **Deterministic QA (sanity / ingestion correctness)** — validate the
  produced cost tables are internally consistent (schema, arithmetic,
  degenerate "wins", apples-to-oranges comparisons).
- **Probabilistic QA (product claim / risk)** — rolling train→test
  backtests that measure out-of-sample performance and tail risk.

Repro / audit checklist for any quoted numbers:

- Chain + block window `[start,end)` and the run command used.
- `data_source` capability stamp (`rpc` vs `statediff`) next to the
  artifacts (do not compare across stamps).
- Cost parameters (`ram_unit_cost`, `miss_penalty`) and `tau`.
- Any ingestion limits / skips (if present) and the extractor endpoint
  host/provider.
- Repo commit SHA.

Commands:

```
make qa-viz REPORT=testdata/runs/scroll_100k/sweep_v2
make qa-backtest MISS_PENALTY=512
```

## Build

```
make build   # → bin/zksp
make test    # go test -race ./...
make cover   # coverage.out + total
make lint    # golangci-lint
```

Go 1.25+. Pure Go, no C dependencies.

## Known limitations

- The RPC extractor only sees slot touches that emit an ERC-20 / ERC-721
  Transfer event. Non-Transfer writes and all reads are invisible; a full
  state-diff source (`debug_traceBlockByNumber` + prestate tracer) is a
  drop-in Extractor replacement.
- Co-access (per-contract Jaccard) and temporal-autocorrelation
  signals are *descriptive* EDA diagnostics — not fed back to the
  Cox fit as covariates. The pipeline is a **structured survival
  model** (dependence structure as potential covariates), not a
  "spatiotemporal" model in the kernel / covariance sense: slot
  keys have no distance metric and there is no physical space.
  Upgrading the Cox covariates with structural features is a
  future direction.
- Conditional conformal ε has marginal (single-probe) coverage, not
  simultaneous coverage over the T\*-search grid.

## License

MIT — see [LICENSE](LICENSE).
