package render

import (
	"bytes"
	"math"
	"strings"
	"testing"

	"github.com/egpivo/zk-state-prune/internal/analysis"
	"github.com/egpivo/zk-state-prune/internal/app"
	"github.com/egpivo/zk-state-prune/internal/model"
	"github.com/egpivo/zk-state-prune/internal/pruning"
)

// Smoke tests. We are not pinning exact byte-for-byte output (that
// would be snapshot churn on every format tweak) — we only assert
// each renderer:
//   1. writes *something* non-empty,
//   2. emits the fixed section headers / column names the CLI
//      promises (so an accidental `fmt.Fprintln` deletion is caught),
//   3. surfaces the numeric values it was handed.

func assertContainsAll(t *testing.T, got string, wants ...string) {
	t.Helper()
	for _, w := range wants {
		if !strings.Contains(got, w) {
			t.Errorf("missing %q\n--- output ---\n%s\n--- end ---", w, got)
		}
	}
}

// ------- EDA ------------------------------------------------------------

func makeEDAReport() *analysis.EDAReport {
	return &analysis.EDAReport{
		Window:             model.ObservationWindow{Start: 100, End: 1000},
		TotalIntervals:     42,
		RightCensoredCount: 7,
		RightCensoredRate:  0.166,
		LeftTruncatedCount: 3,
		LeftTruncatedRate:  0.1,
		SlotsWithNoEvents:  5,
		SlotsSkipped:       1,
		SlotCount:          30,
		Frequency: analysis.DistributionSummary{
			Count: 30, Mean: 1.4, StdDev: 0.7,
			Min: 1, P50: 1, P90: 3, P99: 5, Max: 7,
			PowerLawAlphaMLE: 1.26,
		},
		InterAccessTime: analysis.DistributionSummary{
			Count: 42, Mean: 150, StdDev: 80,
			Min: 1, P50: 100, P90: 300, P99: 900, Max: 1000,
		},
		ByContractType: map[model.ContractCategory]analysis.ContractTypeSummary{
			model.ContractERC20: {
				Slots: 20, Intervals: 30,
				RightCensoredRate: 0.2,
				InterAccessTime:   analysis.DistributionSummary{P50: 80, P99: 700},
			},
		},
	}
}

func TestRenderEDA_HeaderAndRows(t *testing.T) {
	var buf bytes.Buffer
	if err := EDA(&buf, makeEDAReport()); err != nil {
		t.Fatalf("EDA: %v", err)
	}
	out := buf.String()
	assertContainsAll(t, out,
		"[100, 1000)",
		"slots",
		"intervals",
		"right_censored",
		"left_truncated_slots",
		"access frequency per slot",
		"inter-access time",
		"by contract type",
		"category",
		"erc20",
		// A Hill alpha was set, so the optional row must render.
		"power_law_alpha_hill",
		"1.26",
	)
}

func TestRenderEDA_OmitsSpatialTemporalWhenNil(t *testing.T) {
	var buf bytes.Buffer
	if err := EDA(&buf, makeEDAReport()); err != nil {
		t.Fatalf("EDA: %v", err)
	}
	out := buf.String()
	if strings.Contains(out, "intra-contract co-access") {
		t.Error("Spatial section should be omitted when r.Spatial is nil")
	}
	if strings.Contains(out, "temporal periodicity") {
		t.Error("Temporal section should be omitted when r.Temporal is nil")
	}
}

func TestRenderEDA_IncludesSpatialTemporalWhenPresent(t *testing.T) {
	r := makeEDAReport()
	r.Spatial = &analysis.SpatialReport{NumContracts: 12, MeanJaccard: 0.33, MedianJaccard: 0.25}
	r.Temporal = &analysis.TemporalReport{NumContracts: 12, PeriodicContracts: 3, PeriodicFraction: 0.25}

	var buf bytes.Buffer
	if err := EDA(&buf, r); err != nil {
		t.Fatalf("EDA: %v", err)
	}
	out := buf.String()
	assertContainsAll(t, out,
		"intra-contract co-access",
		"contracts_measured=12",
		"temporal periodicity",
		"periodic_fraction=0.250",
	)
}

// ------- KM -------------------------------------------------------------

func TestRenderKM_FiniteMedianRenders(t *testing.T) {
	curve := &analysis.KMResult{
		Label:       "erc20",
		N:           100,
		NumEvents:   80,
		NumCensored: 20,
		Time:        []float64{10, 20, 30},
		Surv:        []float64{0.9, 0.7, 0.5},
		MedianSurv:  30,
	}
	var buf bytes.Buffer
	if err := KM(&buf, []*analysis.KMResult{curve}); err != nil {
		t.Fatalf("KM: %v", err)
	}
	out := buf.String()
	assertContainsAll(t, out, "stratum", "erc20", "100", "80", "30")
	if strings.Contains(out, "NA") {
		t.Errorf("finite median should not emit NA:\n%s", out)
	}
}

func TestRenderKM_NaNMedianEmitsNA(t *testing.T) {
	curve := &analysis.KMResult{
		Label:      "nft",
		N:          4,
		NumEvents:  1,
		MedianSurv: math.NaN(), // never crossed 0.5 under heavy censoring
	}
	var buf bytes.Buffer
	if err := KM(&buf, []*analysis.KMResult{curve}); err != nil {
		t.Fatalf("KM: %v", err)
	}
	if !strings.Contains(buf.String(), "NA") {
		t.Errorf("NaN median should render as NA:\n%s", buf.String())
	}
}

// ------- Cox fit --------------------------------------------------------

func makeCoxFitReport() *app.CoxFitReport {
	return &app.CoxFitReport{
		Tau:              500,
		TrainIntervals:   1000,
		HoldoutIntervals: 400,
		Cox: &analysis.CoxResult{
			Predictors: []string{"access_count", "contract_age"},
			Coef:       []float64{0.42, -0.15},
			StdErr:     []float64{0.05, 0.03},
			ZScore:     []float64{8.4, -5.0},
			PValue:     []float64{0.0001, 0.0002},
			LogLike:    -1234.5,
			NumObs:     1000,
			NumEvents:  700,
		},
		PH: &analysis.PHTestResult{
			Predictors:         []string{"access_count", "contract_age"},
			PerCovariatePValue: map[string]float64{"access_count": 0.12, "contract_age": 0.01},
			GlobalPValue:       0.04,
		},
		RawCurve: &analysis.CalibrationCurve{
			Tau: 500, BrierScore: 0.23,
			NumKept: 400, BrierN: 350,
			Bins: []analysis.CalibrationBin{
				{PredictedMean: 0.1, ObservedRate: 0.12, N: 100},
				{PredictedMean: 0.5, ObservedRate: 0.48, N: 100},
			},
		},
		CalibratedCurve: &analysis.CalibrationCurve{
			Tau: 500, BrierScore: 0.19,
			NumKept: 400, BrierN: 350,
			Bins: []analysis.CalibrationBin{
				{PredictedMean: 0.1, ObservedRate: 0.11, N: 100},
				{PredictedMean: 0.5, ObservedRate: 0.51, N: 100},
			},
		},
		BrierDelta: -0.04,
	}
}

func TestRenderCoxFit_AllSectionsPresent(t *testing.T) {
	var buf bytes.Buffer
	CoxFit(&buf, makeCoxFitReport())
	out := buf.String()
	assertContainsAll(t, out,
		// coxSummary
		"n=1000", "events=700", "loglik=-1234.50",
		"predictor", "access_count", "contract_age",
		// phTest
		"proportional-hazards check (Schoenfeld)",
		"PH ok @ 0.05",
		"NO",  // contract_age p=0.01 < 0.05
		"yes", // access_count p=0.12 >= 0.05
		"global",
		// calibration (τ + both raw and isotonic)
		"calibration @ tau=500",
		"raw Cox:",
		"isotonic-recalibrated:",
		"brier=0.2300",
		"brier=0.1900",
		// Brier delta summary line
		"delta=-0.0400",
	)
}

func TestRenderCoxFit_NilInputIsNoop(t *testing.T) {
	var buf bytes.Buffer
	CoxFit(&buf, nil)
	if buf.Len() != 0 {
		t.Errorf("nil report should produce no output, got %q", buf.String())
	}
}

// ------- SimResults -----------------------------------------------------

func TestRenderSimResults_TableShape(t *testing.T) {
	results := []*pruning.SimResult{
		{
			Policy: "no-prune", TotalSlots: 100, ObservedIntervals: 200,
			RAMRatio: 1.0, HotHitCoverage: 1.0, Reactivations: 0,
			RAMCost: 500, MissPenaltyAgg: 0, TotalCost: 500,
		},
		{
			Policy: "statistical", TotalSlots: 100, ObservedIntervals: 200,
			RAMRatio: 0.15, HotHitCoverage: 0.92, Reactivations: 16,
			RAMCost: 75, MissPenaltyAgg: 160, TotalCost: 235,
		},
	}
	var buf bytes.Buffer
	if err := SimResults(&buf, results); err != nil {
		t.Fatalf("SimResults: %v", err)
	}
	out := buf.String()
	assertContainsAll(t, out,
		"policy", "RAM%", "hot_hit%", "misses", "total_cost",
		"no-prune", "statistical",
		// statistical row values — presence implies the format
		// string isn't silently dropping columns.
		"15.00", "92.00", "16", "235.00",
	)
}
