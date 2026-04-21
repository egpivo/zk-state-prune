# zk-state-prune

**Cost-aware hot/cold tiering for ZK rollup state, driven by survival analysis.**

`zk-state-prune` (binary: `zksp`) is a Go framework that treats on-chain
state access as a survival process, fits a calibrated per-slot
access-probability model, and picks a tiering decision per slot from a
cost surrogate. Its purpose is to replace fixed-idle-window "prune every
slot after N blocks" heuristics with a policy that balances hot-state
RAM against cold-tier miss penalty at the scale of a live rollup.

## What it does

For each storage slot the pipeline answers one question: *given the
slot has been idle for `u` blocks, what is the probability it is
accessed within the next `τ` blocks?* That probability `p_i(τ)` is
plugged into the per-slot surrogate

```
d_i* = argmin_d   c(d)  +  ℓ(d) · p_i(τ)
```

where `c(d)` is the per-tier holding cost (RAM / block) and `ℓ(d)` is
the miss penalty. The threshold collapses to a single
`p* = (c_ram · τ) / ℓ_miss`; the policy demotes a slot at the first
idle duration where the conditional access probability drops below
`p*`.

## Pipeline

```
extract  →  EDA  →  survival fit  →  calibration  →  tiering simulation
```

- **Extract** — two sources share an `Extractor` interface:
  - *mock generator* with power-law access rates, intra-contract
    co-access, sinusoidal seasonality, and configurable censoring;
  - *Scroll mainnet RPC* (**Transfer-log surrogate** — ERC-20 / ERC-721
    Transfer events only; not a full state-diff. See
    `internal/extractor/rpc_extractor.go` for the honest scope).
- **EDA** — first-moment + tail summary of access frequency and
  inter-access time (Hill-estimator α for the upper tail), per-contract
  and per-slot-type breakdowns, censoring diagnostics. Also: two
  descriptive signals used as model-calibration sanity checks:
  - *spatial*: per-contract pairwise Jaccard on slot access-block sets
    (intra-contract co-access);
  - *temporal*: per-contract autocorrelation-peak detection on a binned
    event series.
- **Survival fit** — Kaplan–Meier (stratifiable) and Cox PH, with
  Schoenfeld residual PH-assumption check, 70/30 train/holdout split,
  split-conformal ε, and isotonic recalibration. Cox supports
  stratification by contract type so per-category baseline hazards
  absorb non-proportional aging.
- **Calibration** — bin-wise KM for the observed reliability-diagram
  axis (no censored rows dropped); conformal ε fit on a disjoint
  half of the holdout. A second conditional ε is fit on
  `(interval, u, label_at_u)` triples so the robust policy has an
  honest pessimism margin at arbitrary idle.
- **Tiering simulation** — pure function over pre-built inter-access
  intervals. Reports `RAMRatio`, `HotHitCoverage`, miss count, and
  `TotalCost` per policy. Fixed-rule baselines (`fixed-30d`,
  `fixed-90d`) compared head-to-head against `statistical` (point) and
  `statistical-robust` (conformal upper bound).

## CLI

```
zksp extract  --source mock|rpc          --output <db>
zksp eda      --db <db>                  [--format text|json]
zksp fit      --db <db> --model km|cox   [--stratify contract-type]
                                         [--save <path.json>]
zksp simulate --db <db> --policy no-prune|fixed-30d|fixed-90d|statistical
                                         [--model <path.json>]
                                         [--robust]
                                         [--ram-unit-cost N --miss-penalty N]
zksp report   --db <db>                  [--model <path.json>]
                                         [--format text|json]
```

Config via `--config configs/default.yaml` overrides hardcoded
defaults; flags override config. `zksp extract --force` wipes the
target DB and clears the RPC high-water mark.

## Build & test

```
make build   # → bin/zksp
make test    # go test ./...
make lint    # golangci-lint
```

Go 1.25+ (transitively required by `modernc.org/sqlite`). Pure Go, no
C dependencies.

## Status

- **Phase 1** — scaffolding, mock extractor, EDA, Kaplan–Meier, fixed
  tiering baselines, end-to-end tests.
- **Phase 2** — cost-aware surrogate (`c(d) + ℓ(d)·p_i(τ)`), Cox PH +
  Schoenfeld + isotonic, JSON output, YAML config loading, spatial &
  temporal EDA signals, split-conformal point ε.
- **Phase 3** — persisted fitted models, stratified Cox PH, conditional
  split-conformal ε for the robust policy, Scroll mainnet RPC
  extractor (Transfer-log surrogate).

## Known limitations

- The RPC extractor only sees slot touches that emit an ERC-20 / ERC-721
  Transfer event. All reads and non-Transfer writes (DEX pools,
  governance bookkeeping, arbitrary SSTORE, view-only calls) are
  invisible. A full state-diff source would be a drop-in Extractor
  replacement via `debug_traceBlockByNumber` + prestate tracer.
- Spatial / temporal signals are *descriptive*; they are not fed back
  to the Cox fit as covariates. Doing so is a Phase 4 candidate.
- Conditional conformal ε has marginal (single-probe) coverage, not
  simultaneous coverage over the T\*-search grid. The robust policy
  treats it as a principled pessimism margin; a per-`u` or max-over-`u`
  conformal fit is the tighter upgrade.

## License

MIT — see [LICENSE](LICENSE).
