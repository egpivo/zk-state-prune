package analysis

import (
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/egpivo/zk-state-prune/internal/model"
)

// osWriteFile is a thin alias so the test body below can write fixture
// files without confusing Go's resolver with an inline closure.
var osWriteFile = os.WriteFile

// roundTripModel constructs a small fitted+calibrated model, saves it,
// reloads it, and asserts that every prediction-surface method returns
// the same value before and after the round trip. Uses synthetic
// PH-holding data so the fit is deterministic.
func TestModelFile_RoundTripPredictions(t *testing.T) {
	ivs := phHoldsIntervals(400)
	fitter := NewStatmodelFitter()
	res, err := fitter.FitCoxPH(ivs, DefaultCoxPredictors)
	if err != nil {
		t.Fatalf("FitCoxPH: %v", err)
	}
	calib, err := fitter.Calibrate(res, ivs)
	if err != nil {
		t.Fatalf("Calibrate: %v", err)
	}

	path := filepath.Join(t.TempDir(), "model.json")
	if err := SaveModelFile(path, calib); err != nil {
		t.Fatalf("SaveModelFile: %v", err)
	}
	loaded, err := LoadModelFile(path)
	if err != nil {
		t.Fatalf("LoadModelFile: %v", err)
	}

	if loaded.Tau != calib.Tau {
		t.Errorf("Tau=%v want %v", loaded.Tau, calib.Tau)
	}
	if loaded.Epsilon != calib.Epsilon {
		t.Errorf("Epsilon=%v want %v", loaded.Epsilon, calib.Epsilon)
	}
	if loaded.CoverageLevel != calib.CoverageLevel {
		t.Errorf("CoverageLevel=%v want %v", loaded.CoverageLevel, calib.CoverageLevel)
	}
	if len(loaded.PredX) != len(calib.PredX) {
		t.Fatalf("PredX len mismatch %d vs %d", len(loaded.PredX), len(calib.PredX))
	}

	// Round-trip predictions on a range of representative intervals.
	probes := []model.InterAccessInterval{
		{SlotID: "p0", AccessCount: 0, SlotAge: 10, ContractAge: 20},
		{SlotID: "p1", AccessCount: 5, SlotAge: 200, ContractAge: 300},
		{SlotID: "p2", AccessCount: 15, SlotAge: 1_000, ContractAge: 1_500},
	}
	for _, it := range probes {
		p1 := calib.PredictAccessProbForInterval(it)
		p2 := loaded.PredictAccessProbForInterval(it)
		if math.Abs(p1-p2) > 1e-12 {
			t.Errorf("PredictAccessProbForInterval(%s): orig=%v loaded=%v", it.SlotID, p1, p2)
		}
		u1 := calib.PredictUpperAccessProbForInterval(it)
		u2 := loaded.PredictUpperAccessProbForInterval(it)
		if math.Abs(u1-u2) > 1e-12 {
			t.Errorf("PredictUpperAccessProbForInterval(%s): orig=%v loaded=%v", it.SlotID, u1, u2)
		}
		// Conditional survival at a non-zero idle (what the
		// statistical policy actually uses in production).
		for _, u := range []float64{0, 100, 1_000} {
			s1 := calib.Base.SurvivalForInterval(it, u)
			s2 := loaded.Base.SurvivalForInterval(it, u)
			if math.Abs(s1-s2) > 1e-12 {
				t.Errorf("SurvivalForInterval(%s, u=%v): orig=%v loaded=%v", it.SlotID, u, s1, s2)
			}
		}
	}
}

func TestModelFile_RejectsSchemaMismatch(t *testing.T) {
	path := filepath.Join(t.TempDir(), "bad.json")
	// Hand-written file with a made-up higher schema version.
	body := `{"schema_version": 999, "tau": 1.0, "epsilon": 0.0, "coverage_level": 0.9, "pred_x": [], "calibrated_y": [], "cox": {}}`
	if err := writeFileBytes(path, body); err != nil {
		t.Fatal(err)
	}
	if _, err := LoadModelFile(path); err == nil {
		t.Error("expected schema_version mismatch error")
	}
}

func TestLoadModelFile_RejectsShapeMismatch(t *testing.T) {
	// Hand-crafted v3 files that violate size invariants Load should
	// reject up front rather than letting SurvivalForInterval
	// index-out-of-range at predict time.
	cases := []struct {
		name string
		body string
	}{
		{
			name: "coef shorter than predictors",
			body: `{"schema_version":3,"tau":1,"epsilon":0,"coverage_level":0.9,
				"pred_x":[0.0],"calibrated_y":[0.0],
				"cox":{"predictors":["A","B"],"coef":[0.1],"scales":[{"Name":"A","Mean":0,"Std":1}],
					"baseline_time":[1],"baseline_cum_haz":[0.1]}}`,
		},
		{
			name: "scales shorter than predictors",
			body: `{"schema_version":3,"tau":1,"epsilon":0,"coverage_level":0.9,
				"pred_x":[0.0],"calibrated_y":[0.0],
				"cox":{"predictors":["A","B"],"coef":[0.1,0.2],"scales":[{"Name":"A","Mean":0,"Std":1}],
					"baseline_time":[1],"baseline_cum_haz":[0.1]}}`,
		},
		{
			name: "baseline time/haz length mismatch",
			body: `{"schema_version":3,"tau":1,"epsilon":0,"coverage_level":0.9,
				"pred_x":[0.0],"calibrated_y":[0.0],
				"cox":{"predictors":["A"],"coef":[0.1],"scales":[{"Name":"A","Mean":0,"Std":1}],
					"baseline_time":[1,2],"baseline_cum_haz":[0.1]}}`,
		},
		{
			name: "stratum labels without baselines",
			body: `{"schema_version":3,"tau":1,"epsilon":0,"coverage_level":0.9,
				"pred_x":[0.0],"calibrated_y":[0.0],
				"cox":{"predictors":["A"],"coef":[0.1],"scales":[{"Name":"A","Mean":0,"Std":1}],
					"baseline_time":[],"baseline_cum_haz":[],
					"stratum_column":"ContractType","stratum_labels":[0,1],
					"stratum_baseline_times":[[1]],"stratum_baseline_cum_haz":[[0.1]]}}`,
		},
		{
			name: "stratum baseline time/haz length mismatch",
			body: `{"schema_version":3,"tau":1,"epsilon":0,"coverage_level":0.9,
				"pred_x":[0.0],"calibrated_y":[0.0],
				"cox":{"predictors":["A"],"coef":[0.1],"scales":[{"Name":"A","Mean":0,"Std":1}],
					"baseline_time":[],"baseline_cum_haz":[],
					"stratum_column":"ContractType","stratum_labels":[0],
					"stratum_baseline_times":[[1,2]],"stratum_baseline_cum_haz":[[0.1]]}}`,
		},
		{
			name: "zero predictors",
			body: `{"schema_version":3,"tau":1,"epsilon":0,"coverage_level":0.9,
				"pred_x":[0.0],"calibrated_y":[0.0],
				"cox":{"predictors":[],"coef":[],"scales":[],"baseline_time":[1],"baseline_cum_haz":[0.1]}}`,
		},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			path := filepath.Join(t.TempDir(), "bad.json")
			if err := osWriteFile(path, []byte(c.body), 0o600); err != nil {
				t.Fatal(err)
			}
			if _, err := LoadModelFile(path); err == nil {
				t.Errorf("expected error for %s", c.name)
			}
		})
	}
}

func TestModelFile_SaveAndLoadErrorPaths(t *testing.T) {
	if err := SaveModelFile("/tmp/never", nil); err == nil {
		t.Error("expected nil-model error")
	}
	if err := SaveModelFile("/tmp/never", &CalibratedModel{}); err == nil {
		t.Error("expected nil-Base error")
	}
	if _, err := LoadModelFile(filepath.Join(t.TempDir(), "nope.json")); err == nil {
		t.Error("expected read error for missing file")
	}
}

func writeFileBytes(path, body string) error {
	return osWriteFile(path, []byte(body), 0o600)
}
