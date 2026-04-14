# zk-state-prune

**Statistical state lifecycle modeling & pruning for ZK Rollups.**

`zk-state-prune` (binary: `zksp`) is a Go framework that uses survival analysis
and spatiotemporal modeling to predict the probability that a state slot in a
ZK rollup will be accessed again, and derives data-driven pruning policies from
those predictions. The goal is to replace fixed-rule state expiry with
statistical models that minimize prover memory, reduce Merkle tree depth, and
keep false-prune rates low.

## Motivation

ZK rollups must commit to the entire state inside a SNARK/STARK circuit. Every
live storage slot inflates the Merkle tree depth and the prover's memory
footprint, even though most slots are touched rarely or never again after their
initial use. Today's mitigations are blunt: drop slots after a fixed idle
window, or never drop them at all. Both are wasteful — the first risks
reactivation cost on slots that are still hot, the second pays for slots that
are statistically dead.

This project treats slot access as a **survival process**: given a slot that
has been idle for `t` blocks, what is the probability it will be touched in the
next `h` blocks? With a calibrated survival function `S(t)` we can prune a slot
exactly when its expected reactivation cost falls below its storage cost.

## Approach

1. **Extract** state-diff streams (or generate realistic synthetic ones) into a
   local SQLite store.
2. **EDA**: characterize access-frequency, inter-access-time, and
   spatial-clustering distributions per contract type and slot type.
3. **Fit** survival models — Kaplan–Meier as a non-parametric baseline, Cox
   proportional-hazards for covariate effects (contract type, slot type, age).
4. **Simulate** several pruning policies (statistical, fixed-30d, fixed-90d,
   no-prune) over historical traces and score them on memory saved, false-prune
   rate, and reactivation cost.
5. **Report** comparative results.

## CLI

```
zksp extract  --source <rpc|file|mock> --output <db_path>
zksp eda      --db <db_path> --output <report_dir>
zksp fit      --db <db_path> --model <km|cox>
zksp simulate --db <db_path> --policy <no-prune|fixed-30d|fixed-90d>   # statistical: Phase 2
zksp report   --db <db_path> --output <report_dir>
```

## Build

```
make build      # → bin/zksp
make test
make lint
```

Requires Go 1.22+. No C dependencies (uses `modernc.org/sqlite`).

## Roadmap

- **Phase 1 (current)**: scaffolding, core types, mock extractor, EDA,
  Kaplan–Meier, fixed-rule pruning baseline.
- **Phase 2**: Cox PH model, spatial correlation analysis, cost-aware
  statistical policy, reporting.
- **Phase 3**: real RPC extractor for an L2 (e.g. zkSync, Scroll), large-scale
  simulation, calibration on real traces.

## License

MIT — see [LICENSE](LICENSE).
