package extractor

import (
	"encoding/json"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"testing"

	"github.com/egpivo/zk-state-prune/internal/domain"
)

// fixtureDir holds the canonical, hand-crafted prestateTracer
// responses parseStateDiff is regression-tested against. Path is
// relative to this test file's package; Go's test runner sets the
// CWD to the package dir before running so a plain ../../testdata
// reach-out is what we want.
const fixtureDir = "../../testdata/statediff"

// expectedTouch is the JSON shape of every entry in
// `<fixture>.expected.json`. Mirrors slotTouch but keeps the
// access type as a string so the file stays human-readable.
type expectedTouch struct {
	Contract string `json:"contract"`
	SlotKey  string `json:"slotKey"`
	Access   string `json:"access"`
}

type expectedTouches struct {
	Description string          `json:"description"`
	Touches     []expectedTouch `json:"touches"`
}

// TestStateDiffExtractor_FixtureRegression loads every
// `<name>.json` under testdata/statediff/ as a debug_traceBlockByNumber
// response, runs parseStateDiff over each tx in it, and compares the
// resulting touch set against `<name>.expected.json`. Adding a new
// fixture pair is enough — no test code changes — so the easiest way
// to capture a real Geth response that surprised the parser is to
// drop a sanitised copy into the fixture dir.
//
// What this protects against:
//   - The Geth wire format renaming a field ("storage" → "slots") or
//     restructuring tx entries: the JSON unmarshal into our types
//     drops fields silently, but the touch count comparison would
//     diverge from what the fixture says.
//   - parseStateDiff regressions: the read-vs-write classification
//     gets exercised on shapes that aren't trivially constructed by
//     the inline test fixtures (multi-contract, no-storage, mixed
//     read+write across txs).
func TestStateDiffExtractor_FixtureRegression(t *testing.T) {
	entries, err := os.ReadDir(fixtureDir)
	if err != nil {
		t.Fatalf("read fixture dir %s: %v", fixtureDir, err)
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
			tracePath := filepath.Join(fixtureDir, name)
			expectedPath := filepath.Join(fixtureDir,
				strings.TrimSuffix(name, ".json")+".expected.json")

			traces := loadTraceFixture(t, tracePath)
			expected := loadExpectedFixture(t, expectedPath)

			var got []slotTouch
			for _, tr := range traces {
				got = append(got, parseStateDiff(tr.Result)...)
			}
			assertTouchesMatch(t, got, expected.Touches)
		})
	}
	// Cheap sanity check: if someone deletes every fixture by accident,
	// fail loudly rather than passing trivially.
	if saw == 0 {
		t.Fatalf("no fixtures found under %s — regression coverage gone", fixtureDir)
	}
}

func loadTraceFixture(t *testing.T, path string) []traceBlockEntry {
	t.Helper()
	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read %s: %v", path, err)
	}
	var traces []traceBlockEntry
	if err := json.Unmarshal(b, &traces); err != nil {
		t.Fatalf("parse %s: %v", path, err)
	}
	return traces
}

func loadExpectedFixture(t *testing.T, path string) expectedTouches {
	t.Helper()
	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read %s: %v", path, err)
	}
	var exp expectedTouches
	if err := json.Unmarshal(b, &exp); err != nil {
		t.Fatalf("parse %s: %v", path, err)
	}
	return exp
}

// assertTouchesMatch compares got (from parseStateDiff) against
// expected as a multiset — same (contract, slotKey, access)
// triples regardless of order, since map iteration in
// parseStateDiff is non-deterministic. Sort both sides on a
// composite key before walking so every miss is reported with its
// own line in the test log.
func assertTouchesMatch(t *testing.T, got []slotTouch, want []expectedTouch) {
	t.Helper()

	type key struct{ c, s, a string }
	gotKeys := make([]key, 0, len(got))
	for _, g := range got {
		gotKeys = append(gotKeys, key{
			c: strings.ToLower(g.contract),
			s: strings.ToLower(g.slotKey),
			a: g.access.String(),
		})
	}
	wantKeys := make([]key, 0, len(want))
	for _, w := range want {
		wantKeys = append(wantKeys, key{
			c: strings.ToLower(w.Contract),
			s: strings.ToLower(w.SlotKey),
			a: w.Access,
		})
	}
	sortKeys := func(ks []key) {
		sort.Slice(ks, func(i, j int) bool {
			if ks[i].c != ks[j].c {
				return ks[i].c < ks[j].c
			}
			if ks[i].s != ks[j].s {
				return ks[i].s < ks[j].s
			}
			return ks[i].a < ks[j].a
		})
	}
	sortKeys(gotKeys)
	sortKeys(wantKeys)

	if len(gotKeys) != len(wantKeys) {
		t.Fatalf("touch count: got %d, want %d\ngot:  %v\nwant: %v",
			len(gotKeys), len(wantKeys), gotKeys, wantKeys)
	}
	for i := range gotKeys {
		if gotKeys[i] != wantKeys[i] {
			t.Errorf("touch[%d]: got %+v, want %+v", i, gotKeys[i], wantKeys[i])
		}
	}

	// Cross-check: every "write" in expected must round-trip through
	// our parsed access type — guards against AccessType.String() and
	// the JSON enum drifting apart.
	for i, w := range want {
		if w.Access != "read" && w.Access != "write" {
			t.Errorf("expected.touches[%d].access = %q, want read|write", i, w.Access)
		}
	}
	_ = domain.AccessRead // keep the import live even if assertions short-circuit
}
