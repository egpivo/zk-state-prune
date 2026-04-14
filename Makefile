.PHONY: build test lint tidy clean run cover

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
