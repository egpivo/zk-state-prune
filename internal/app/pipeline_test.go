package app

import (
	"context"
	"path/filepath"
	"strings"
	"testing"

	"github.com/egpivo/zk-state-prune/internal/analysis"
	"github.com/egpivo/zk-state-prune/internal/extractor"
	"github.com/egpivo/zk-state-prune/internal/model"
	"github.com/egpivo/zk-state-prune/internal/pruning"
	"github.com/egpivo/zk-state-prune/internal/storage"
)

// ------- pure helpers --------------------------------------------------

func TestCoxStrataColumn(t *testing.T) {
	cases := []struct {
		in      string
		want    string
		wantErr bool
	}{
		{"", "", false},
		{"none", "", false},
		{"contract-type", analysis.ColContractType, false},
		{"contract", analysis.ColContractType, false},
		{"slot-type", analysis.ColSlotType, false},
		{"slot", analysis.ColSlotType, false},
		{"bogus", "", true},
	}
	for _, c := range cases {
		t.Run(c.in, func(t *testing.T) {
			got, err := CoxStrataColumn(c.in)
			if c.wantErr {
				if err == nil {
					t.Errorf("CoxStrataColumn(%q) = %q, want error", c.in, got)
				}
				return
			}
			if err != nil {
				t.Fatalf("CoxStrataColumn(%q) err=%v", c.in, err)
			}
			if got != c.want {
				t.Errorf("CoxStrataColumn(%q) = %q, want %q", c.in, got, c.want)
			}
		})
	}
}

func TestMedianDuration(t *testing.T) {
	mk := func(durs ...uint64) []model.InterAccessInterval {
		out := make([]model.InterAccessInterval, len(durs))
		for i, d := range durs {
			out[i] = model.InterAccessInterval{Duration: d}
		}
		return out
	}
	cases := []struct {
		name string
		in   []model.InterAccessInterval
		want float64
	}{
		// Empty + all-zero + single-element defend the "never return 0"
		// invariant — downstream math (τ denominator) mustn't blow up.
		{"empty", nil, 1},
		{"all zero clamps to 1", mk(0, 0, 0), 1},
		{"single", mk(42), 42},
		{"odd count", mk(1, 5, 10), 5},
		// Go's midpoint is index len/2 — for even counts that picks
		// the upper middle, not the classic arithmetic mean.
		{"even count picks upper middle", mk(1, 5, 10, 100), 10},
		{"unsorted input", mk(100, 1, 5, 10, 3), 5},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			if got := MedianDuration(c.in); got != c.want {
				t.Errorf("MedianDuration = %v, want %v", got, c.want)
			}
		})
	}
}

func TestStatisticalPolicyFromCalibrated_NilModel(t *testing.T) {
	costs := pruning.CostParams{RAMUnitCost: 1, MissPenalty: 10}
	if _, err := StatisticalPolicyFromCalibrated(nil, costs, false); err == nil {
		t.Error("nil model: want error")
	}
	// Non-nil outer, nil Base — still invalid.
	if _, err := StatisticalPolicyFromCalibrated(&analysis.CalibratedModel{}, costs, false); err == nil {
		t.Error("nil Base: want error")
	}
}

func TestCalibrationCurveFromCalibrated_NilModel(t *testing.T) {
	if _, err := CalibrationCurveFromCalibrated(nil, nil, 10); err == nil {
		t.Error("nil model: want error")
	}
	if _, err := CalibrationCurveFromCalibrated(&analysis.CalibratedModel{}, nil, 10); err == nil {
		t.Error("nil Base: want error")
	}
}

// ------- end-to-end pipeline -------------------------------------------
//
// End-to-end on a deterministic mock so the fit/calibrate/policy
// orchestration is exercised together. The hand-written nil-case
// tests above cover the guard clauses; these tests cover the happy path
// that the CLI runs.

func smallMockCfg() extractor.MockConfig {
	c := extractor.DefaultMockConfig()
	c.NumContracts = 30
	c.SlotsPerContractXmin = 5
	c.SlotsPerContractMax = 40
	c.TotalBlocks = 20_000
	c.Window = model.ObservationWindow{Start: 4_000, End: 20_000}
	c.AccessRateXmin = 1e-4
	c.MaxEventsPerSlot = 200
	c.PeriodBlocks = 2_000
	return c
}

func setupMockIntervals(t *testing.T) []model.InterAccessInterval {
	t.Helper()
	ctx := context.Background()
	db, err := storage.Open(ctx, filepath.Join(t.TempDir(), "app.db"))
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	t.Cleanup(func() { _ = db.Close() })

	cfg := smallMockCfg()
	if err := extractor.NewMockExtractor(cfg).Extract(ctx, db); err != nil {
		t.Fatalf("Extract: %v", err)
	}
	built, err := analysis.BuildIntervals(ctx, db, cfg.Window)
	if err != nil {
		t.Fatalf("BuildIntervals: %v", err)
	}
	if len(built.Intervals) == 0 {
		t.Fatal("no intervals produced")
	}
	return built.Intervals
}

func TestBuildStatisticalPolicy_PointAndRobust(t *testing.T) {
	intervals := setupMockIntervals(t)
	costs := pruning.CostParams{RAMUnitCost: 1, MissPenalty: 1_000}

	// Point variant.
	p, err := BuildStatisticalPolicy(intervals, 0.3, 1, 0, costs, false)
	if err != nil {
		t.Fatalf("BuildStatisticalPolicy point: %v", err)
	}
	if p.Name() != "statistical" {
		t.Errorf("Name = %q, want statistical", p.Name())
	}
	if p.Tau() <= 0 {
		t.Errorf("Tau = %v, must be positive", p.Tau())
	}
	if p.PStar() < 0 || p.PStar() > 1 {
		t.Errorf("PStar = %v out of [0,1]", p.PStar())
	}

	// Robust variant.
	pr, err := BuildStatisticalPolicy(intervals, 0.3, 1, 0, costs, true)
	if err != nil {
		t.Fatalf("BuildStatisticalPolicy robust: %v", err)
	}
	if pr.Name() != "statistical-robust" {
		t.Errorf("Name = %q, want statistical-robust", pr.Name())
	}
	// Both variants share the same τ (fit from the same split).
	if p.Tau() != pr.Tau() {
		t.Errorf("tau mismatch between variants: point=%v robust=%v", p.Tau(), pr.Tau())
	}
}

func TestBuildStatisticalPolicy_ExplicitTauOverridesMedian(t *testing.T) {
	intervals := setupMockIntervals(t)
	costs := pruning.CostParams{RAMUnitCost: 1, MissPenalty: 1_000}
	const explicit uint64 = 12345

	p, err := BuildStatisticalPolicy(intervals, 0.3, 1, explicit, costs, false)
	if err != nil {
		t.Fatalf("BuildStatisticalPolicy: %v", err)
	}
	if p.Tau() != float64(explicit) {
		t.Errorf("Tau = %v, want explicit %d", p.Tau(), explicit)
	}
}

func TestBuildCoxFitReport_EndToEnd(t *testing.T) {
	intervals := setupMockIntervals(t)
	fitter := analysis.NewStatmodelFitter()

	r, calib, err := BuildCoxFitReport(fitter, intervals, 0.3, 1, 0, "")
	if err != nil {
		t.Fatalf("BuildCoxFitReport: %v", err)
	}
	if r.Tau <= 0 {
		t.Errorf("Tau = %v, must be positive", r.Tau)
	}
	if r.TrainIntervals == 0 || r.HoldoutIntervals == 0 {
		t.Errorf("split degenerate: train=%d holdout=%d", r.TrainIntervals, r.HoldoutIntervals)
	}
	if r.Cox == nil || r.PH == nil || r.RawCurve == nil || r.CalibratedCurve == nil {
		t.Fatalf("nil section in report: %+v", r)
	}
	if calib == nil || calib.Base == nil {
		t.Fatalf("calib is nil / missing Base")
	}
	if calib.Tau != r.Tau {
		t.Errorf("calib.Tau=%v != report.Tau=%v", calib.Tau, r.Tau)
	}

	// Calibration curve from calibrated model should share τ.
	curve, err := CalibrationCurveFromCalibrated(calib, intervals, 10)
	if err != nil {
		t.Fatalf("CalibrationCurveFromCalibrated: %v", err)
	}
	if curve.BrierScore < 0 || curve.BrierScore > 1 {
		t.Errorf("BrierScore = %v out of [0,1]", curve.BrierScore)
	}
}

func TestBuildCoxFitReport_StratifiedByContractType(t *testing.T) {
	intervals := setupMockIntervals(t)
	fitter := analysis.NewStatmodelFitter()

	r, _, err := BuildCoxFitReport(fitter, intervals, 0.3, 1, 0, "contract-type")
	if err != nil {
		t.Fatalf("BuildCoxFitReport stratified: %v", err)
	}
	if r.Cox.StratumColumn != analysis.ColContractType {
		t.Errorf("StratumColumn=%q, want %q", r.Cox.StratumColumn, analysis.ColContractType)
	}
	if len(r.Cox.StratumLabels) == 0 {
		t.Error("stratified fit: expected non-empty StratumLabels")
	}
}

func TestStatisticalPolicy_ClosuresExerciseOnHotBlocks(t *testing.T) {
	// The closures inside StatisticalPolicyFromCalibrated only run
	// when the policy's T*-search calls them. Construct a policy from
	// a freshly fit CalibratedModel and invoke HotBlocks on a few
	// intervals so the rawCondP / robust branches are hit.
	intervals := setupMockIntervals(t)
	costs := pruning.CostParams{RAMUnitCost: 1, MissPenalty: 1_000}

	for _, robust := range []bool{false, true} {
		p, err := BuildStatisticalPolicy(intervals, 0.3, 1, 0, costs, robust)
		if err != nil {
			t.Fatalf("BuildStatisticalPolicy robust=%v: %v", robust, err)
		}
		// A handful of intervals is plenty to exercise both the
		// idle=0 branch (first search sample) and idle>0 branches
		// (subsequent samples) of the closure.
		for _, it := range intervals[:min(20, len(intervals))] {
			hot := p.HotBlocks(it)
			if hot > it.Duration {
				t.Errorf("HotBlocks=%d > Duration=%d (robust=%v)", hot, it.Duration, robust)
			}
		}
	}
}

func TestBuildCoxFitReport_UnknownStratify(t *testing.T) {
	intervals := setupMockIntervals(t)
	fitter := analysis.NewStatmodelFitter()

	_, _, err := BuildCoxFitReport(fitter, intervals, 0.3, 1, 0, "does-not-exist")
	if err == nil {
		t.Fatal("BuildCoxFitReport with unknown stratify: want error, got nil")
	}
	if !strings.Contains(err.Error(), "unknown cox stratify mode") {
		t.Errorf("error message = %q; want it to mention unknown stratify", err.Error())
	}
}
