package model

import (
	"strings"
	"testing"
)

// The enum String / Parse pairs round-trip through SQLite text
// columns — if either side drifts silently, stored rows decode as
// "unknown" / "other" and survival fits stratify wrong. A tiny table
// test pins each declared constant.

func TestSlotType_StringParseRoundTrip(t *testing.T) {
	all := []SlotType{
		SlotTypeUnknown,
		SlotTypeBalance,
		SlotTypeMapping,
		SlotTypeArray,
		SlotTypeFixed,
	}
	for _, s := range all {
		if got := ParseSlotType(s.String()); got != s {
			t.Errorf("round trip: %v -> %q -> %v", s, s.String(), got)
		}
	}
}

func TestSlotType_UnknownStringFallback(t *testing.T) {
	// Out-of-band integer values must stringify to a debuggable token
	// rather than crash or collide with a real name.
	got := SlotType(99).String()
	if !strings.HasPrefix(got, "slot_type(") {
		t.Errorf("unknown SlotType String = %q, want slot_type(...) form", got)
	}
}

func TestParseSlotType_UnknownInputFallsBack(t *testing.T) {
	if got := ParseSlotType("garbage"); got != SlotTypeUnknown {
		t.Errorf("ParseSlotType(garbage) = %v, want SlotTypeUnknown", got)
	}
}

func TestAccessType_StringParseRoundTrip(t *testing.T) {
	for _, a := range []AccessType{AccessRead, AccessWrite} {
		if got := ParseAccessType(a.String()); got != a {
			t.Errorf("round trip: %v -> %q -> %v", a, a.String(), got)
		}
	}
}

func TestAccessType_UnknownStringFallback(t *testing.T) {
	got := AccessType(42).String()
	if !strings.HasPrefix(got, "access_type(") {
		t.Errorf("unknown AccessType String = %q, want access_type(...) form", got)
	}
}

func TestParseAccessType_UnknownDefaultsToRead(t *testing.T) {
	// Intentional: ParseAccessType is lenient — anything other than
	// "write" decodes as Read. This locks that behavior so callers
	// that feed untrusted strings into it stay predictable.
	if got := ParseAccessType("garbage"); got != AccessRead {
		t.Errorf("ParseAccessType(garbage) = %v, want AccessRead (lenient default)", got)
	}
}

func TestContractCategory_StringParseRoundTrip(t *testing.T) {
	all := []ContractCategory{
		ContractOther,
		ContractERC20,
		ContractDEX,
		ContractNFT,
		ContractBridge,
		ContractGovernance,
	}
	for _, c := range all {
		if got := ParseContractCategory(c.String()); got != c {
			t.Errorf("round trip: %v -> %q -> %v", c, c.String(), got)
		}
	}
}

func TestContractCategory_UnknownStringFallback(t *testing.T) {
	got := ContractCategory(99).String()
	if !strings.HasPrefix(got, "contract_category(") {
		t.Errorf("unknown ContractCategory String = %q, want contract_category(...) form", got)
	}
}

func TestParseContractCategory_UnknownFallsBack(t *testing.T) {
	if got := ParseContractCategory("garbage"); got != ContractOther {
		t.Errorf("ParseContractCategory(garbage) = %v, want ContractOther", got)
	}
}

func TestObservationWindow_ContainsAndSpan(t *testing.T) {
	w := ObservationWindow{Start: 100, End: 200}
	if w.Span() != 100 {
		t.Errorf("Span = %d, want 100", w.Span())
	}
	cases := []struct {
		b    uint64
		want bool
	}{
		{99, false},
		{100, true},  // inclusive start
		{150, true},
		{199, true},
		{200, false}, // exclusive end
		{201, false},
	}
	for _, c := range cases {
		if got := w.Contains(c.b); got != c.want {
			t.Errorf("Contains(%d) = %v, want %v", c.b, got, c.want)
		}
	}
}
