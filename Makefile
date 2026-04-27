.PHONY: build test lint tidy clean run cover scroll-smoke scroll-100k qa-viz qa-backtest fuzz-statediff qa-robustness

BIN := bin/zksp
PKG := ./...

build:
	@mkdir -p bin
	go build -o $(BIN) ./cmd/zksp

run: build
	$(BIN) --help

test:
	go test -race -count=1 $(PKG)

cover:
	go test -race -count=1 -coverprofile=coverage.out $(PKG)
	go tool cover -func=coverage.out | tail -n 1

lint:
	golangci-lint run

tidy:
	go mod tidy

clean:
	rm -rf bin coverage.out

# ---- experiment runners ---------------------------------------------
# scroll-smoke: 1k-block sanity run (~5 min) to verify the pipeline
# end-to-end against rpc.scroll.io before committing to scroll-100k.
# scroll-100k:  full 100k-block Transfer-log surrogate run (3-10h
# wall clock; uses the canonical window from
# testdata/scroll_window.yaml). Both write artifacts under
# testdata/runs/scroll_*/ — the SQLite DB is gitignored.
#
# Override the RPC endpoint via SCROLL_RPC=<url> on the make line.
scroll-smoke: build
	./scripts/scroll_100k_surrogate.sh --smoke

scroll-100k: build
	./scripts/scroll_100k_surrogate.sh

# ---- QA / viz -------------------------------------------------------
# qa-viz: stdlib-only Python script that reads either a single
# `zksp report --format json` file OR a directory of simulate_*.json
# files, and emits qa_summary.json, degeneracy_flags.json, and two
# hand-written SVGs under $(QA_OUT) (default
# testdata/runs/scroll_100k/qa/).
#
#   make qa-viz REPORT=testdata/runs/scroll_100k/report.json
#   make qa-viz REPORT=testdata/runs/scroll_100k/sweep_v2 \
#               QA_OUT=testdata/runs/scroll_100k/qa_v2
QA_OUT ?= testdata/runs/scroll_100k/qa
qa-viz:
	@if [ -z "$(REPORT)" ]; then \
	    echo "usage: make qa-viz REPORT=<report.json|sweep-dir> [QA_OUT=<dir>]"; \
	    exit 2; \
	fi
	python3 scripts/qa_viz.py --report "$(REPORT)" --out-dir "$(QA_OUT)"

# qa-backtest: rolling train→fit→simulate product-claim QA.
# Runs multiple folds and writes `backtest_summary.json` +
# `backtest_report.html` under $(BT_OUT).
#
# Required: MISS_PENALTY (ℓ). The other parameters have sensible defaults
# for the canonical scroll_100k run but can be overridden.
BT_DB ?= testdata/runs/scroll_100k/scroll.db
BT_OUT ?= testdata/runs/scroll_100k/backtest
BT_START ?= 33400000
BT_END ?= 33500000
# Defaults are sized so n_folds is large enough for the tail / CVaR
# metrics in Risk QA to be meaningful (n_folds=17 on 100k blocks).
# Tradeoff: shorter train_span = noisier Cox fit per fold. Override on
# the make line if you'd rather have fewer, beefier folds.
BT_TRAIN_SPAN ?= 20000
BT_TEST_SPAN ?= 10000
BT_STEP ?= 5000
BT_TAU ?= 0
qa-backtest: build
	@if [ -z "$(MISS_PENALTY)" ]; then \
	    echo "usage: make qa-backtest MISS_PENALTY=<ℓ> [BT_DB=...] [BT_OUT=...]"; \
	    exit 2; \
	fi
	python3 scripts/backtest.py --db "$(BT_DB)" --zksp "$(BIN)" --out-dir "$(BT_OUT)" \
	    --start "$(BT_START)" --end "$(BT_END)" \
	    --train-span "$(BT_TRAIN_SPAN)" --test-span "$(BT_TEST_SPAN)" --step "$(BT_STEP)" \
	    --tau "$(BT_TAU)" --miss-penalty "$(MISS_PENALTY)"

# ---- Robustness QA fuzzing ------------------------------------------
# fuzz-statediff: smoke-fuzzes the prestateTracer parsing path. Catches
# panics / crashes the hand-written nasty fixtures didn't think of.
#
# This is a SMOKE fuzz at the default 10s — long enough for PR-CI to
# catch obvious regressions, short enough not to dominate the build.
# For real coverage fuzzing run with a much larger budget, e.g.
#
#   make fuzz-statediff FUZZTIME=30m
#
# Go's `-fuzz=` flag runs ONE fuzz function at a time, so add
# FUZZ_FN=<name> when more fuzz targets land.
FUZZTIME ?= 10s
FUZZ_FN ?= FuzzStatediffParse
FUZZ_PKG ?= ./internal/extractor/
fuzz-statediff:
	go test -run=^$$ -fuzz=$(FUZZ_FN) -fuzztime=$(FUZZTIME) $(FUZZ_PKG)

# qa-robustness: runs `go test -v` over the repo, greps the
# `ROBUSTNESS_QA:` tag lines that markRobustness emits, and writes a
# deterministic coverage summary to $(QA_OUT)/robustness_summary.json.
# The JSON is generated from test output — adding a new tagged test
# automatically extends coverage on the next run, so the file can't
# silently drift away from reality.
#
# Pass --run to filter which tests execute (the runner forwards it).
qa-robustness:
	python3 scripts/qa_robustness.py --out-dir "$(QA_OUT)" $(QA_ROBUSTNESS_ARGS)
