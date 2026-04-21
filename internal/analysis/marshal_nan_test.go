package analysis

import (
	"encoding/json"
	"math"
	"strings"
	"testing"
)

// NaN / ±Inf inside a CoxResult, KMResult, or DistributionSummary used to
// crash `json.Marshal` (the encoder refuses non-finite floats by default).
// The three MarshalJSON overrides rewrite them as null so a degenerate
// stratum or a partial fit never kills the report pipeline. These tests
// lock that behavior in.

func TestCoxResult_MarshalJSON_NaNFields(t *testing.T) {
	r := &CoxResult{
		Predictors: []string{"a", "b"},
		Coef:       []float64{0.1, math.NaN()},
		StdErr:     []float64{math.NaN(), math.Inf(1)},
		PValue:     []float64{math.NaN(), 0.05},
		ZScore:     []float64{math.Inf(-1), 1.5},
		LogLike:    math.NaN(),
		NumObs:     10,
		NumEvents:  5,
	}
	b, err := json.Marshal(r)
	if err != nil {
		t.Fatalf("Marshal: %v", err)
	}
	s := string(b)
	// Finite values survive.
	if !strings.Contains(s, "0.1") {
		t.Errorf("expected finite Coef[0]=0.1 in %s", s)
	}
	if !strings.Contains(s, "\"LogLike\":null") {
		t.Errorf("expected LogLike:null for NaN, got %s", s)
	}
	// None of the non-finite tokens leak through.
	for _, bad := range []string{"NaN", "+Inf", "-Inf", "Infinity"} {
		if strings.Contains(s, bad) {
			t.Errorf("non-finite token %q leaked into JSON: %s", bad, s)
		}
	}
	// Round-trip check: the result decodes as valid JSON.
	var back map[string]any
	if err := json.Unmarshal(b, &back); err != nil {
		t.Fatalf("round-trip decode: %v", err)
	}
}

func TestKMResult_MarshalJSON_NaNMedian(t *testing.T) {
	k := &KMResult{
		Label:      "erc20",
		N:          3,
		NumEvents:  2,
		Time:       []float64{1, 2, 3},
		Surv:       []float64{0.9, 0.7, 0.5},
		SE:         []float64{0.01, 0.02, 0.03},
		NumRisk:    []float64{3, 2, 1},
		MedianSurv: math.NaN(),
	}
	b, err := json.Marshal(k)
	if err != nil {
		t.Fatalf("Marshal: %v", err)
	}
	s := string(b)
	if !strings.Contains(s, "\"median_surv\":null") {
		t.Errorf("expected median_surv:null for NaN, got %s", s)
	}
	if strings.Contains(s, "NaN") {
		t.Errorf("NaN leaked into JSON: %s", s)
	}

	// Finite median serializes as a number, not null.
	k.MedianSurv = 2.5
	b, err = json.Marshal(k)
	if err != nil {
		t.Fatalf("Marshal finite: %v", err)
	}
	if !strings.Contains(string(b), "\"median_surv\":2.5") {
		t.Errorf("expected median_surv:2.5, got %s", string(b))
	}
}

func TestDistributionSummary_MarshalJSON_NaNFields(t *testing.T) {
	// Reproduces the NFT-stratum incident: a category with too few
	// intervals returns NaN mean/std from gonum, which would crash
	// the whole EDA JSON output before the MarshalJSON override.
	d := DistributionSummary{
		Count:            2,
		Mean:             math.NaN(),
		StdDev:           math.NaN(),
		Min:              1,
		P50:              math.Inf(1),
		P90:              math.NaN(),
		P99:              math.NaN(),
		Max:              100,
		PowerLawAlphaMLE: 0,
	}
	b, err := json.Marshal(d)
	if err != nil {
		t.Fatalf("Marshal: %v", err)
	}
	s := string(b)
	for _, want := range []string{
		"\"Mean\":null",
		"\"StdDev\":null",
		"\"P50\":null",
		"\"P90\":null",
		"\"P99\":null",
	} {
		if !strings.Contains(s, want) {
			t.Errorf("expected %q in %s", want, s)
		}
	}
	// Finite values still survive as numbers.
	for _, want := range []string{"\"Min\":1", "\"Max\":100", "\"Count\":2"} {
		if !strings.Contains(s, want) {
			t.Errorf("expected %q in %s", want, s)
		}
	}
	if strings.Contains(s, "NaN") || strings.Contains(s, "Inf") {
		t.Errorf("non-finite token leaked: %s", s)
	}
}
