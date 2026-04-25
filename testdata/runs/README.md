# Experiment runs

Output artifacts from the scripted experiment runners under
[`scripts/`](../../scripts/). Each subdirectory holds the JSON
outputs of one run; the SQLite databases the runs produced are
deliberately `.gitignore`'d (see [.gitignore](.gitignore)) — they
balloon to hundreds of MB at full window sizes.

## Convention

```
testdata/runs/<experiment>/
├── eda.json                       # zksp eda --format json
├── km.json                        # zksp fit --model km --format json
├── cox.json                       # zksp fit --model cox --format json
├── cox.model                      # persisted CalibratedModel (gitignored if large)
├── simulate_<policy>_l<ℓ>.json   # one per (policy, ℓ_miss) cost-sweep cell
├── report.json                    # zksp report bundle
├── scroll.db                      # gitignored — local source of truth
└── extract.log                    # gitignored — extractor stderr
```

## Existing experiments

| Dir | Window | Status |
|-----|--------|--------|
| `scroll_smoke/`  | `[33,400,000, 33,401,000)` (1k blocks) | sanity-check before the full run |
| `scroll_100k/`   | `[33,400,000, 33,500,000)` (100k blocks) | the headline cost-regime test for blog Part 2 |

Reproduce either with:

```bash
make scroll-smoke   # ~5 min
make scroll-100k    # 3-10 h, depends on RPC quota
```
