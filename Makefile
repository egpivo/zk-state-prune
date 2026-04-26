.PHONY: build test lint tidy clean run cover scroll-smoke scroll-100k qa-viz

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
