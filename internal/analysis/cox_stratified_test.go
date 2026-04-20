package analysis

import (
	"fmt"
	"math"
	"math/rand/v2"
	"path/filepath"
	"testing"

	"github.com/egpivo/zk-state-prune/internal/model"
)

// twoStrataIntervals synthesizes a dataset where two contract categories
// have dramatically different hazards: ContractERC20 slots get short
// gap times (fast accesses), ContractBridge slots get long ones. This
// is the cleanest way to test that stratified Cox builds visibly
// different per-stratum baseline hazards while sharing a single
// coefficient vector on the non-strata covariates.
func twoStrataIntervals(n int) []model.InterAccessInterval {
	r := rand.New(rand.NewPCG(11, 13))
	out := make([]model.InterAccessInterval, 0, n)
	for i := 0; i < n; i++ {
		ac := uint64(r.IntN(10))
		var cat model.ContractCategory
		var dur uint64
		if i%2 == 0 {
			cat = model.ContractERC20
			// Short gaps, modulated lightly by AccessCount.
			base := 30.0 - float64(ac)*1.5
			dur = uint64(math.Max(1, base+r.NormFloat64()*2))
		} else {
			cat = model.ContractBridge
			// Long gaps, same AccessCount slope but bigger intercept.
			base := 200.0 - float64(ac)*1.5
			dur = uint64(math.Max(1, base+r.NormFloat64()*8))
		}
		out = append(out, model.InterAccessInterval{
			SlotID:       fmt.Sprintf("s%d", i),
			Duration:     dur,
			IsObserved:   true,
			AccessCount:  ac,
			SlotAge:      uint64(i*7 + r.IntN(5)),
			ContractAge:  uint64(i*11 + r.IntN(7)),
			ContractType: cat,
			SlotType:     model.SlotTypeBalance,
		})
	}
	return out
}

func TestFitCoxPHStratified_ProducesPerStratumBaselines(t *testing.T) {
	ivs := twoStrataIntervals(400)

	res, err := NewStatmodelFitter().FitCoxPHStratified(ivs, DefaultCoxPredictors, ColContractType)
	if err != nil {
		t.Fatalf("FitCoxPHStratified: %v", err)
	}
	if res.StratumColumn != ColContractType {
		t.Errorf("StratumColumn=%q want %q", res.StratumColumn, ColContractType)
	}
	if len(res.StratumLabels) != 2 {
		t.Fatalf("StratumLabels=%v want 2 entries", res.StratumLabels)
	}
	if len(res.StratumBaselineTimes) != 2 || len(res.StratumBaselineCumHaz) != 2 {
		t.Fatalf("per-stratum baselines missing: %+v / %+v",
			res.StratumBaselineTimes, res.StratumBaselineCumHaz)
	}
	// Strata labels must be sorted ascending (ERC20=1, Bridge=3 given
	// iota; labels store the raw float values).
	if res.StratumLabels[0] >= res.StratumLabels[1] {
		t.Errorf("StratumLabels not sorted: %v", res.StratumLabels)
	}
	// The two baselines should land on materially different time grids
	// because one stratum's event times are around 30 and the other's
	// around 200.
	for i, times := range res.StratumBaselineTimes {
		if len(times) == 0 {
			t.Errorf("stratum %d has empty baseline grid", i)
		}
	}
	maxERC20 := res.StratumBaselineTimes[0][len(res.StratumBaselineTimes[0])-1]
	maxBridge := res.StratumBaselineTimes[1][len(res.StratumBaselineTimes[1])-1]
	if maxERC20 >= maxBridge {
		t.Errorf("expected Bridge baseline to extend to larger times than ERC20: ERC20_max=%v Bridge_max=%v",
			maxERC20, maxBridge)
	}
}

func TestFitCoxPHStratified_PredictionsDifferByStratum(t *testing.T) {
	ivs := twoStrataIntervals(400)
	res, err := NewStatmodelFitter().FitCoxPHStratified(ivs, DefaultCoxPredictors, ColContractType)
	if err != nil {
		t.Fatalf("FitCoxPHStratified: %v", err)
	}
	// Two probe intervals with identical covariates except for the
	// stratum value. Survival at a moderate horizon must differ — the
	// whole point of stratification is that each category carries its
	// own baseline.
	probeERC := model.InterAccessInterval{
		AccessCount: 3, SlotAge: 100, ContractAge: 100, ContractType: model.ContractERC20,
	}
	probeBridge := probeERC
	probeBridge.ContractType = model.ContractBridge

	for _, t0 := range []float64{20, 50, 100} {
		sE := res.SurvivalForInterval(probeERC, t0)
		sB := res.SurvivalForInterval(probeBridge, t0)
		// With Bridge slots seeing access at t≈200 and ERC20 at t≈30,
		// at t0=50 we should see ERC20's survival much lower than
		// Bridge's. More generally S(ERC20) < S(Bridge) at every
		// probe in this range.
		if !(sE < sB) {
			t.Errorf("t=%v: S(ERC20)=%v !< S(Bridge)=%v (same covariates, diff stratum)", t0, sE, sB)
		}
	}
}

func TestFitCoxPHStratified_RejectsCollidingPredictor(t *testing.T) {
	ivs := twoStrataIntervals(50)
	// Strata column also appears in predictors → error.
	bad := append([]string(nil), DefaultCoxPredictors...)
	bad = append(bad, ColContractType)
	if _, err := NewStatmodelFitter().FitCoxPHStratified(ivs, bad, ColContractType); err == nil {
		t.Error("expected error when strata column appears in predictors")
	}
}

func TestFitCoxPH_UnstratifiedStillWorks(t *testing.T) {
	// Regression guard: the non-stratified path (FitCoxPH) must keep
	// working exactly as before.
	ivs := twoStrataIntervals(200)
	res, err := NewStatmodelFitter().FitCoxPH(ivs, DefaultCoxPredictors)
	if err != nil {
		t.Fatalf("FitCoxPH: %v", err)
	}
	if res.StratumColumn != "" {
		t.Errorf("StratumColumn=%q want empty on unstratified fit", res.StratumColumn)
	}
	if len(res.StratumLabels) != 0 || len(res.StratumBaselineTimes) != 0 {
		t.Errorf("stratum fields should be empty, got labels=%v times=%v", res.StratumLabels, res.StratumBaselineTimes)
	}
	if len(res.BaselineTime) == 0 {
		t.Error("single baseline should be populated")
	}
}

func TestModelFile_StratifiedRoundTrip(t *testing.T) {
	ivs := twoStrataIntervals(300)
	fitter := NewStatmodelFitter()
	res, err := fitter.FitCoxPHStratified(ivs, DefaultCoxPredictors, ColContractType)
	if err != nil {
		t.Fatalf("FitCoxPHStratified: %v", err)
	}
	calib, err := fitter.Calibrate(res, ivs)
	if err != nil {
		t.Fatalf("Calibrate: %v", err)
	}

	path := filepath.Join(t.TempDir(), "strat.json")
	if err := SaveModelFile(path, calib); err != nil {
		t.Fatalf("SaveModelFile: %v", err)
	}
	loaded, err := LoadModelFile(path)
	if err != nil {
		t.Fatalf("LoadModelFile: %v", err)
	}
	if loaded.Base.StratumColumn != res.StratumColumn {
		t.Errorf("StratumColumn: loaded=%q orig=%q", loaded.Base.StratumColumn, res.StratumColumn)
	}
	if len(loaded.Base.StratumLabels) != len(res.StratumLabels) {
		t.Fatalf("StratumLabels len: loaded=%d orig=%d", len(loaded.Base.StratumLabels), len(res.StratumLabels))
	}
	for i := range loaded.Base.StratumLabels {
		if loaded.Base.StratumLabels[i] != res.StratumLabels[i] {
			t.Errorf("StratumLabels[%d]: loaded=%v orig=%v", i, loaded.Base.StratumLabels[i], res.StratumLabels[i])
		}
	}

	// End-to-end: the loaded model must produce identical predictions
	// for a probe interval in each stratum.
	for _, cat := range []model.ContractCategory{model.ContractERC20, model.ContractBridge} {
		probe := model.InterAccessInterval{
			AccessCount: 2, SlotAge: 50, ContractAge: 60, ContractType: cat,
		}
		for _, t0 := range []float64{20, 100} {
			sOrig := res.SurvivalForInterval(probe, t0)
			sLoad := loaded.Base.SurvivalForInterval(probe, t0)
			if math.Abs(sOrig-sLoad) > 1e-12 {
				t.Errorf("cat=%v t=%v: orig=%v loaded=%v", cat, t0, sOrig, sLoad)
			}
		}
	}
}
