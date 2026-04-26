#!/usr/bin/env bash
#
# scroll_100k_surrogate.sh — reproducible Transfer-log surrogate run
# over the canonical Scroll mainnet window pinned in
# testdata/scroll_window.yaml.
#
# Goal: answer "does the cost-regime finding from the 1.2k-block
# smoke (statistical wins ℓ ∈ [1, 100], loses at ≥10⁴) survive at
# 100k blocks?" without running the (much more expensive) statediff
# extractor — that's PR2's deliverable, this is the parallel sanity
# check on the surrogate.
#
# Usage:
#   scripts/scroll_100k_surrogate.sh           # full 100k run
#   scripts/scroll_100k_surrogate.sh --smoke   # 1k-block smoke test
#   scripts/scroll_100k_surrogate.sh --rpc <url> --out <dir>
#
# Output (under <out>/, default testdata/runs/scroll_100k/ or
# testdata/runs/scroll_smoke/ for --smoke):
#   - extract.log / extract.json   diagnostics from the extractor run
#   - eda.json                     EDA report (Hill α, censoring, …)
#   - km.json                      stratified Kaplan-Meier curves
#   - cox.json + cox.model         fit + persisted CalibratedModel
#   - simulate_<policy>_l<ℓ>.json  one per (policy, ℓ_miss) combo
#   - report.json                  full report bundle
#
# The SQLite DB is written to <out>/scroll.db and intentionally
# .gitignored — checking it in would balloon the repo (100k blocks
# of Scroll Transfer logs ≈ a few hundred MB of rows).

set -euo pipefail

# ---- canonical window (mirrors testdata/scroll_window.yaml) ---------
# Hard-coded here so the script doesn't take a yaml dep; if you bump
# the window in scroll_window.yaml, bump these constants too. See
# the yaml's comment about why the range is pinned.
START_FULL=33400000
END_FULL=33500000
START_SMOKE=33400000
END_SMOKE=33401000   # 1k blocks — usually <5 min wall clock

# ---- defaults --------------------------------------------------------
RPC="${SCROLL_RPC:-https://rpc.scroll.io}"
OUT_FULL="testdata/runs/scroll_100k"
OUT_SMOKE="testdata/runs/scroll_smoke"
SMOKE=false
OUT=""

# ---- arg parsing -----------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --smoke)  SMOKE=true; shift ;;
        --rpc)    RPC="$2"; shift 2 ;;
        --out)    OUT="$2"; shift 2 ;;
        -h|--help)
            sed -n '2,/^$/p' "$0" | sed 's/^# \{0,1\}//'
            exit 0
            ;;
        *) echo "unknown arg: $1" >&2; exit 1 ;;
    esac
done

if $SMOKE; then
    START=$START_SMOKE
    END=$END_SMOKE
    OUT="${OUT:-$OUT_SMOKE}"
    LABEL="smoke (1k blocks)"
else
    START=$START_FULL
    END=$END_FULL
    OUT="${OUT:-$OUT_FULL}"
    LABEL="full (100k blocks)"
fi
DB="$OUT/scroll.db"
WINDOW_BLOCKS=$((END - START))

# ---- preflight -------------------------------------------------------
ZKSP="${ZKSP:-bin/zksp}"
if [[ ! -x "$ZKSP" ]]; then
    echo "::: building $ZKSP" >&2
    make build >&2
fi
mkdir -p "$OUT"

echo "::: scroll-surrogate run: $LABEL"
echo "    rpc:     $RPC"
echo "    blocks:  $START → $END  ($WINDOW_BLOCKS blocks)"
echo "    output:  $OUT/"
echo

# ---- 1. extract ------------------------------------------------------
echo "::: [1/4] extract --source rpc"
"$ZKSP" extract --source rpc --rpc "$RPC" \
    --start "$START" --end "$END" \
    --output "$DB" --force 2>&1 | tee "$OUT/extract.log"

# ---- 2. EDA ----------------------------------------------------------
echo
echo "::: [2/4] eda"
"$ZKSP" eda --db "$DB" \
    --window-start "$START" --window-end "$END" \
    --format json > "$OUT/eda.json"
echo "    eda.json $(wc -c < "$OUT/eda.json" | xargs) bytes"

# ---- 3. fit + save Cox model (re-used by every simulate run) --------
echo
echo "::: [3/4] fit cox --save"
"$ZKSP" fit --db "$DB" --model cox \
    --window-start "$START" --window-end "$END" \
    --save "$OUT/cox.model" \
    --format json > "$OUT/cox.json"
echo "    cox.model written"

# ---- 4. cost sweep mirroring blog Fig 1b ----------------------------
# Five tiers of ℓ_miss × four policies × {point, robust} for the
# statistical variants. Re-using the saved model keeps each simulate
# under a second; without --model we'd refit Cox per ℓ which is
# wasteful and noisy.
echo
echo "::: [4/4] cost sweep"
LMISS_VALUES=(1 10 100 1000 10000 100000)
POLICIES=("no-prune" "fixed-30d" "fixed-90d")

for L in "${LMISS_VALUES[@]}"; do
    for P in "${POLICIES[@]}"; do
        OUT_FILE="$OUT/simulate_${P//-/_}_l${L}.json"
        "$ZKSP" simulate --db "$DB" --policy "$P" \
            --window-start "$START" --window-end "$END" \
            --ram-unit-cost 1 --miss-penalty "$L" \
            --format json > "$OUT_FILE"
    done
    # statistical (point) — uses saved model, no refit
    "$ZKSP" simulate --db "$DB" --policy statistical \
        --window-start "$START" --window-end "$END" \
        --model "$OUT/cox.model" \
        --ram-unit-cost 1 --miss-penalty "$L" \
        --format json > "$OUT/simulate_statistical_l${L}.json"
    # statistical-robust
    "$ZKSP" simulate --db "$DB" --policy statistical \
        --window-start "$START" --window-end "$END" \
        --model "$OUT/cox.model" --robust \
        --ram-unit-cost 1 --miss-penalty "$L" \
        --format json > "$OUT/simulate_statistical_robust_l${L}.json"
    echo "    ℓ=$L: $((${#POLICIES[@]} + 2)) policies done"
done

# ---- 5. one full report bundle (text + json) ------------------------
echo
echo "::: report bundle"
"$ZKSP" report --db "$DB" \
    --window-start "$START" --window-end "$END" \
    --model "$OUT/cox.model" \
    --format json > "$OUT/report.json"

# Tear off a one-line summary so the operator sees the headline
# numbers without opening the JSONs. Pretty-printed JSON has a
# space after the colon, so we tolerate any whitespace before the
# value and strip the trailing comma/newline.
extract_num() {
    awk -v key="$2" -F: '
        index($0, "\"" key "\"") > 0 {
            v = $2
            gsub(/[ ,\r]/, "", v)
            print v
            exit
        }' "$1"
}
TOTAL_INTERVALS=$(extract_num "$OUT/eda.json" "TotalIntervals")
HILL_ALPHA=$(extract_num "$OUT/eda.json" "PowerLawAlphaMLE")
TOTAL_INTERVALS=${TOTAL_INTERVALS:-?}
HILL_ALPHA=${HILL_ALPHA:-?}

echo
echo "::: done — $LABEL"
echo "    intervals=$TOTAL_INTERVALS  hill_α≈$HILL_ALPHA"
echo "    artifacts in $OUT/"
echo
echo "    next: open $OUT/eda.json + the simulate_*.json files,"
echo "    or write findings into .local/scroll_100k_notes.md"
