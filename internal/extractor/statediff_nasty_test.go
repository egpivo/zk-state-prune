package extractor

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// nastyDir holds hand-crafted "robustness QA" prestateTracer
// responses — see testdata/statediff/nasty/README.md for the schema.
// These exist alongside the regular fixtures because they assert a
// different property: the parser must FAIL CLOSED on hostile or
// malformed input (explicit error or canonicalisation), never panic
// or silently corrupt state.
const nastyFixtureDir = fixtureDir + "/nasty"

// nastyExpected is a superset of expectedTouches that adds an
// `expect` discriminator. "ok" means json.Unmarshal must succeed
// and parseStateDiff produces exactly Touches; "unmarshal_error"
// means json.Unmarshal must fail and Touches must be empty.
type nastyExpected struct {
	Description string          `json:"description"`
	Expect      string          `json:"expect"`
	Touches     []expectedTouch `json:"touches"`
}

// TestStateDiffExtractor_NastyFixtures walks testdata/statediff/nasty
// and asserts each (input, expected) pair behaves according to its
// schema. Adding a new pair extends coverage without code changes.
//
// Every subtest runs in a defer-recover so a panic anywhere in the
// parsing path is reported as a test failure instead of crashing the
// whole package's test run — the safety property we care about.
func TestStateDiffExtractor_NastyFixtures(t *testing.T) {
	entries, err := os.ReadDir(nastyFixtureDir)
	if err != nil {
		t.Fatalf("read nasty fixture dir %s: %v", nastyFixtureDir, err)
	}
	saw := 0
	for _, e := range entries {
		if e.IsDir() || !strings.HasSuffix(e.Name(), ".json") {
			continue
		}
		if strings.HasSuffix(e.Name(), ".expected.json") {
			continue
		}
		saw++
		name := e.Name()
		t.Run(strings.TrimSuffix(name, ".json"), func(t *testing.T) {
			defer func() {
				if r := recover(); r != nil {
					t.Fatalf("panic during parse — robustness invariant broken: %v", r)
				}
			}()

			tracePath := filepath.Join(nastyFixtureDir, name)
			expectedPath := filepath.Join(nastyFixtureDir,
				strings.TrimSuffix(name, ".json")+".expected.json")

			traceBytes, err := os.ReadFile(tracePath)
			if err != nil {
				t.Fatalf("read trace: %v", err)
			}
			expBytes, err := os.ReadFile(expectedPath)
			if err != nil {
				t.Fatalf("read expected: %v", err)
			}
			var exp nastyExpected
			if err := json.Unmarshal(expBytes, &exp); err != nil {
				t.Fatalf("parse expected: %v", err)
			}

			var traces []traceBlockEntry
			unmarshalErr := json.Unmarshal(traceBytes, &traces)

			switch exp.Expect {
			case "unmarshal_error":
				if unmarshalErr == nil {
					t.Fatalf("expected unmarshal error, got none (parsed %d traces)", len(traces))
				}
				if len(exp.Touches) != 0 {
					t.Fatalf("nasty fixture schema bug: expect=unmarshal_error must have empty touches, got %d", len(exp.Touches))
				}
			case "ok":
				if unmarshalErr != nil {
					t.Fatalf("expected ok, got unmarshal error: %v", unmarshalErr)
				}
				var got []slotTouch
				for _, tr := range traces {
					got = append(got, parseStateDiff(tr.Result)...)
				}
				assertTouchesMatch(t, got, exp.Touches)
			default:
				t.Fatalf("invalid expect %q (must be 'ok' or 'unmarshal_error')", exp.Expect)
			}
		})
	}
	if saw == 0 {
		t.Fatalf("no nasty fixtures found under %s — robustness coverage gone", nastyFixtureDir)
	}
}

// FuzzStatediffParse stress-tests the parsing path with arbitrary
// bytes. Goal: parseStateDiff must NEVER panic, regardless of how
// malformed or hostile the input is. Skipping invalid JSON is fair
// game (downstream callers handle the error); panicking is not.
//
// Seeds: every fixture under testdata/statediff/ (regular + nasty),
// plus a handful of synthetic malformed seeds so the fuzzer has
// something interesting to mutate from on a clean checkout.
//
// Invocation:
//
//	go test -run=^$ -fuzz=FuzzStatediffParse -fuzztime=10s ./internal/extractor/
//
// or via the make target: `make fuzz-statediff FUZZTIME=10s`.
func FuzzStatediffParse(f *testing.F) {
	addSeedsFrom := func(dir string) {
		entries, err := os.ReadDir(dir)
		if err != nil {
			return
		}
		for _, e := range entries {
			if e.IsDir() || !strings.HasSuffix(e.Name(), ".json") {
				continue
			}
			if strings.HasSuffix(e.Name(), ".expected.json") {
				continue
			}
			b, err := os.ReadFile(filepath.Join(dir, e.Name()))
			if err == nil {
				f.Add(b)
			}
		}
	}
	addSeedsFrom(fixtureDir)
	addSeedsFrom(nastyFixtureDir)

	// A few synthetic seeds beyond the fixtures so the fuzzer has
	// a starting set even on a fresh checkout where the corpus
	// cache is empty.
	f.Add([]byte(`[]`))
	f.Add([]byte(`null`))
	f.Add([]byte(`{"foo":"bar"}`))
	f.Add([]byte(`[{"txHash":"0x00","result":{"pre":null,"post":null}}]`))

	f.Fuzz(func(t *testing.T, data []byte) {
		// Defensive size cap: real prestateTracer responses are
		// <100 KB per tx; a single fuzz mutation that explodes
		// to MBs (deeply nested arrays, etc.) can blow past the
		// per-iteration deadline without exercising any new code
		// path. Skip outright — we'd rather lose that mutation
		// than have the fuzzer report a transient timeout.
		if len(data) > 256*1024 {
			return
		}
		// We accept any input; invalid JSON is fine — what we
		// forbid is parseStateDiff panicking on whatever did
		// unmarshal successfully.
		var traces []traceBlockEntry
		if err := json.Unmarshal(data, &traces); err != nil {
			return
		}
		for _, tr := range traces {
			_ = parseStateDiff(tr.Result)
		}
	})
}
