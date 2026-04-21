# zk-state-prune

[![CI](https://github.com/egpivo/zk-state-prune/actions/workflows/ci.yml/badge.svg)](https://github.com/egpivo/zk-state-prune/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/egpivo/zk-state-prune/graph/badge.svg?token=siqgL91f4o)](https://codecov.io/gh/egpivo/zk-state-prune)
[![Go Report Card](https://goreportcard.com/badge/github.com/egpivo/zk-state-prune)](https://goreportcard.com/report/github.com/egpivo/zk-state-prune)
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
zksp extract  --source mock|rpc          --output <db>
zksp eda      --db <db>                  [--format text|json]
zksp fit      --db <db> --model km|cox   [--stratify contract-type]
                                         [--save <path.json>]
zksp simulate --db <db> --policy no-prune|fixed-30d|fixed-90d|statistical
                                         [--model <path.json>] [--robust]
                                         [--ram-unit-cost N --miss-penalty N]
zksp report   --db <db>                  [--model <path.json>]
```

`--config configs/default.yaml` overrides hardcoded defaults; flags override
config. `zksp extract --force` wipes the target DB and clears the RPC
high-water mark.

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
- Spatial / temporal EDA signals are *descriptive* — not fed back to the
  Cox fit as covariates.
- Conditional conformal ε has marginal (single-probe) coverage, not
  simultaneous coverage over the T\*-search grid.

## License

MIT — see [LICENSE](LICENSE).
