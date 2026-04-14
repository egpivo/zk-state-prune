# zk-state-prune

**Statistical state lifecycle modeling & tiering for ZK Rollups.**

`zk-state-prune` (binary: `zksp`) is a Go framework that uses survival analysis
and spatiotemporal modeling to predict the probability that a state slot in a
ZK rollup will be accessed again, and derives data-driven hot/cold tiering
policies from those predictions. The goal is to replace fixed-rule state
expiry with statistical models that **reduce sequencer lookup latency, witness
generation memory, and hot-state footprint, while keeping miss rates low.**
This is a slot-level surrogate optimization for balancing hot-state pressure
against fetch penalty — not an end-to-end proving cost model.

## Motivation

A live ZK rollup serves reads against a hot working set on the sequencer side
and proves transitions over a witness that materializes whichever slots a
block touches. Both costs scale with the size of the live state: the larger
the hot set, the more sequencer RAM and the bigger the witness for a typical
block. Today's mitigations are blunt: drop (or evict to cold storage) every
slot after a fixed idle window, or keep everything live. The first risks a
miss on slots that are still hot — paying a fetch penalty on the critical
path; the second pays RAM for slots that are statistically dead.

This project treats slot access as a **survival process**: given a slot that
has been idle for `t` blocks, what is the probability it will be touched in
the next `h` blocks? With a calibrated survival function `S(t)` we can demote
a slot exactly when its expected miss penalty falls below the cost of keeping
it hot. The decision rule is a per-slot surrogate
`d_i* = argmin_d c(d) + ℓ(d) · p_i(τ)` where `c(d)` is the per-tier holding
cost, `ℓ(d)` the miss penalty for that tier, and `p_i(τ) = 1 − S_i(τ)` is the
calibrated access-by-horizon probability.

## Approach

1. **Extract** state-diff streams (or generate realistic synthetic ones) into a
   local SQLite store.
2. **EDA**: characterize access-frequency, inter-access-time, and
   spatial-clustering distributions per contract type and slot type.
3. **Fit** survival models — Kaplan–Meier as a non-parametric baseline, Cox
   proportional-hazards for covariate effects (contract type, slot type, age),
   with Schoenfeld-residual PH check and isotonic recalibration on a 70/30
   holdout.
4. **Simulate** several tiering policies (no-prune, fixed-30d, fixed-90d, and
   the statistical surrogate) over historical traces and score them on
   **RAM ratio, hot-hit coverage, miss penalty, and total cost**.
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

Requires Go 1.25+ (transitively pulled in by `modernc.org/sqlite`).
No C dependencies (pure-Go SQLite).

## Roadmap

- **Phase 1 (complete)**: scaffolding, core types, mock extractor, EDA,
  Kaplan–Meier, Cox PH with Schoenfeld PH check and isotonic recalibration,
  fixed-rule tiering baselines.
- **Phase 2**: spatial correlation analysis, cost-aware statistical tiering
  policy wired against `c(d) + ℓ(d) · p_i(τ)`, conformal uncertainty bands.
- **Phase 3**: real RPC extractor for an L2 (e.g. zkSync, Scroll), large-scale
  simulation, calibration on real traces.

## License

MIT — see [LICENSE](LICENSE).
