package extractor

import (
	"encoding/json"
	"testing"
)

// markRobustness emits a structured tag line on the test's log
// output:
//
//	ROBUSTNESS_QA: {"kind":"...","subject":"...","failure_mode":"...","test":"..."}
//
// The `make qa-robustness` runner greps this prefix out of `go test
// -v` output and aggregates it into a `robustness_summary.json`. The
// goal is to keep the summary GENERATED — adding a new robustness
// test automatically updates the summary on the next run, so the
// JSON can never silently drift out of sync with the actual coverage.
//
//   - kind:         what robustness property the test asserts
//     (e.g. "fail-closed-on-limit", "panic-free-parse",
//     "domain-binding", "schema-drift")
//   - subject:      the code under test, in <package>.<symbol> form
//     (e.g. "rpc_extractor.checkBlockLimits")
//   - failure_mode: the specific hostile / edge scenario this case
//     exercises (e.g. "events-per-block exceeds limit",
//     "storage typed as a number")
//
// All three are required. They're free-form strings — keep the
// vocabulary small so the aggregator's by-kind / by-subject buckets
// stay readable.
func markRobustness(t *testing.T, kind, subject, failureMode string) {
	t.Helper()
	payload := map[string]string{
		"kind":         kind,
		"subject":      subject,
		"failure_mode": failureMode,
		"test":         t.Name(),
	}
	b, err := json.Marshal(payload)
	if err != nil {
		t.Fatalf("markRobustness: marshal payload: %v", err)
	}
	t.Logf("ROBUSTNESS_QA: %s", b)
}
