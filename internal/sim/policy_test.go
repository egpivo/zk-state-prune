package sim

import (
	"strings"
	"testing"
)

// TestPolicyByName_FixedNBlocksPattern verifies the
// `fixed-<N>[k|m]` parser used to spawn ad-hoc FixedIdle baselines
// without adding a named preset for every block count. The named
// presets (fixed-30d, fixed-90d) keep working; the new pattern
// fills the gap when a window is short enough that the named
// presets degenerate to no-prune.
func TestPolicyByName_FixedNBlocksPattern(t *testing.T) {
	cases := []struct {
		in   string
		want uint64 // 0 = expect error
	}{
		// happy path — plain integer, with k / m suffix
		{"fixed-1000", 1000},
		{"fixed-1k", 1000},
		{"fixed-10k", 10_000},
		{"fixed-100k", 100_000},
		{"fixed-1m", 1_000_000},
		// case-insensitive + underscore-tolerant (CLI ↔ YAML idiom)
		{"FIXED-1K", 1000},
		{"fixed_5k", 5000},
		// existing named presets still resolve; this case asserts
		// they are NOT clobbered by the parser.
		{"fixed-30d", 216_000},
		{"fixed-90d", 648_000},
		// rejected: empty, decimal, sign, unknown suffix, zero
		{"fixed-", 0},
		{"fixed-0", 0},
		{"fixed-1.5k", 0},
		{"fixed--5", 0},
		{"fixed-1g", 0},
		{"fixed-k", 0},
		// not the fixed family — these go to the unknown branch.
		{"foo-1k", 0},
	}
	for _, c := range cases {
		t.Run(c.in, func(t *testing.T) {
			p, err := PolicyByName(c.in, PolicyDeps{})
			if c.want == 0 {
				if err == nil {
					t.Errorf("PolicyByName(%q): expected error, got policy %v", c.in, p)
				}
				return
			}
			if err != nil {
				t.Fatalf("PolicyByName(%q): unexpected error %v", c.in, err)
			}
			fi, ok := p.(FixedIdle)
			if !ok {
				t.Fatalf("PolicyByName(%q): got %T, want FixedIdle", c.in, p)
			}
			if fi.IdleBlocks != c.want {
				t.Errorf("PolicyByName(%q): IdleBlocks=%d, want %d", c.in, fi.IdleBlocks, c.want)
			}
			// Label round-trips the normalised key so SimResult.Policy
			// stays human-readable.
			if !strings.Contains(strings.ToLower(fi.Label), strings.ToLower(strings.ReplaceAll(c.in, "_", "-"))) &&
				fi.Label != "fixed-30d" && fi.Label != "fixed-90d" {
				t.Errorf("PolicyByName(%q): label %q lost the original key", c.in, fi.Label)
			}
		})
	}
}

func TestParseFixedBlocks_OverflowGuard(t *testing.T) {
	// A 20-digit number with the `m` suffix would overflow uint64 if
	// we didn't guard. parseFixedBlocks should reject (return false)
	// rather than silently wrap.
	if _, ok := parseFixedBlocks("fixed-99999999999999999999m"); ok {
		t.Error("parseFixedBlocks should reject inputs that overflow uint64 after multiplier")
	}
}
