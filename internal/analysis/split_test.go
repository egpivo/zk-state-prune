package analysis

import (
	"context"
	"math"
	"math/rand/v2"
	"testing"

	"github.com/egpivo/zk-state-prune/internal/domain"
	"github.com/egpivo/zk-state-prune/internal/extractor"
)

// makeMultiSlotIntervals fabricates `slots` distinct slot ids, each
// contributing 3 intervals. Useful for split tests that need the
// "intervals from one slot stay together" invariant to be measurable.
func makeMultiSlotIntervals(slots int) []domain.InterAccessInterval {
	out := make([]domain.InterAccessInterval, 0, slots*3)
	for i := 0; i < slots; i++ {
		id := stringInt(i)
		for k := 0; k < 3; k++ {
			out = append(out, domain.InterAccessInterval{
				SlotID:   id,
				Duration: uint64(10 + k*5),
			})
		}
	}
	return out
}

func stringInt(i int) string {
	const digits = "0123456789"
	if i == 0 {
		return "s0"
	}
	buf := []byte{'s'}
	rev := []byte{}
	for i > 0 {
		rev = append(rev, digits[i%10])
		i /= 10
	}
	for j := len(rev) - 1; j >= 0; j-- {
		buf = append(buf, rev[j])
	}
	return string(buf)
}

func TestTrainHoldoutSplitBySlot_NoSlotLeakage(t *testing.T) {
	ivs := makeMultiSlotIntervals(50)
	train, holdout, err := TrainHoldoutSplitBySlot(ivs, 0.3, 42)
	if err != nil {
		t.Fatalf("split: %v", err)
	}
	trainSlots := make(map[string]bool)
	for _, it := range train {
		trainSlots[it.SlotID] = true
	}
	for _, it := range holdout {
		if trainSlots[it.SlotID] {
			t.Errorf("slot %q appears in both train and holdout", it.SlotID)
		}
	}
	if got := len(train) + len(holdout); got != len(ivs) {
		t.Errorf("split lost rows: %d + %d != %d", len(train), len(holdout), got)
	}
	// 50 slots × 0.3 = 15 expected in holdout. Each slot has 3 rows, so 45.
	if got := len(holdout); got != 15*3 {
		t.Errorf("holdout size = %d, want 45 (15 slots × 3)", got)
	}
}

func TestTrainHoldoutSplitBySlot_DeterministicWithSeed(t *testing.T) {
	ivs := makeMultiSlotIntervals(80)
	t1, h1, _ := TrainHoldoutSplitBySlot(ivs, 0.3, 7)
	t2, h2, _ := TrainHoldoutSplitBySlot(ivs, 0.3, 7)
	if len(t1) != len(t2) || len(h1) != len(h2) {
		t.Fatalf("non-deterministic sizes")
	}
	for i := range t1 {
		if t1[i].SlotID != t2[i].SlotID {
			t.Errorf("train mismatch at %d: %s vs %s", i, t1[i].SlotID, t2[i].SlotID)
			break
		}
	}
}

func TestTrainHoldoutSplitBySlot_DifferentSeedDifferentPartition(t *testing.T) {
	ivs := makeMultiSlotIntervals(80)
	_, h1, _ := TrainHoldoutSplitBySlot(ivs, 0.3, 1)
	_, h2, _ := TrainHoldoutSplitBySlot(ivs, 0.3, 2)
	a := make(map[string]bool)
	for _, it := range h1 {
		a[it.SlotID] = true
	}
	overlap := 0
	for _, it := range h2 {
		if a[it.SlotID] {
			overlap++
		}
	}
	// Some overlap is fine; we only assert that the partition isn't
	// identical to the seed=1 case.
	if overlap == len(h2) {
		t.Errorf("seeds 1 and 2 produced identical partitions")
	}
}

func TestTrainHoldoutSplitBySlot_RejectsBadFraction(t *testing.T) {
	ivs := makeMultiSlotIntervals(10)
	for _, frac := range []float64{0, 1, -0.1, 1.1} {
		if _, _, err := TrainHoldoutSplitBySlot(ivs, frac, 1); err == nil {
			t.Errorf("expected error for frac=%v", frac)
		}
	}
}

// --- Calibration curve tests --------------------------------------------

// generateCalibratedHoldout fabricates a holdout set whose true access
// probability matches the Cox prediction exactly: for each interval we
// know the model would predict p_i, and we sample a Bernoulli(p_i)
// label. With enough rows the per-bin observed rate should track the
// per-bin predicted mean within a couple of standard errors.
func TestCalibrationCurveFromCox_SmokeOnSyntheticPHData(t *testing.T) {
	// Use the same PH-holding generator from segment 9 to fit Cox.
	ivs := phHoldsIntervals(800)
	res, err := NewStatmodelFitter().FitCoxPH(ivs, DefaultCoxPredictors)
	if err != nil {
		t.Fatalf("FitCoxPH: %v", err)
	}
	// Use a tau near the median observed Duration so labels split.
	durations := make([]float64, 0, len(ivs))
	for _, it := range ivs {
		durations = append(durations, float64(it.Duration))
	}
	median := medianFloat(durations)

	curve, err := CalibrationCurveFromCox(res, ivs, median, 5)
	if err != nil {
		t.Fatalf("CalibrationCurveFromCox: %v", err)
	}
	if curve.NumKept == 0 {
		t.Fatal("no rows kept")
	}
	if got := len(curve.Bins); got != 5 {
		t.Fatalf("len(Bins)=%d want 5", got)
	}
	// Bins must be sorted by predicted mean ascending and observed rate
	// must trend the same direction (positive correlation between
	// predicted and observed across bins). We don't insist on a perfect
	// monotone since synthetic noise can swap adjacent bins.
	rho := pearsonCorrelation(extractPredicted(curve), extractObserved(curve))
	if rho < 0.5 {
		t.Errorf("predicted/observed correlation across bins = %v, want > 0.5", rho)
	}
	if curve.BrierScore < 0 || curve.BrierScore > 1 {
		t.Errorf("BrierScore=%v out of [0,1]", curve.BrierScore)
	}
}

func TestCalibrationCurveFromCox_ErrorPaths(t *testing.T) {
	if _, err := CalibrationCurveFromCox(nil, nil, 1, 5); err == nil {
		t.Error("expected error on nil model")
	}
	res := &CoxResult{}
	if _, err := CalibrationCurveFromCox(res, nil, 1, 5); err == nil {
		t.Error("expected error on empty holdout")
	}
	if _, err := CalibrationCurveFromCox(res, []domain.InterAccessInterval{{Duration: 5, IsObserved: true}}, 0, 5); err == nil {
		t.Error("expected error on tau=0")
	}
}

func TestCalibrationCurveFromCox_BinWiseKMKeepsCensoredRows(t *testing.T) {
	// Model that always predicts 0.5 (no covariate variance).
	res := &CoxResult{
		Predictors:     []string{},
		Coef:           []float64{},
		Scales:         []CovarScale{},
		BaselineTime:   []float64{0, 100},
		BaselineCumHaz: []float64{0, math.Log(2)}, // S(100)=0.5
	}
	// Four holdout rows at τ=100:
	//   (50, censored) — Phase 1 would drop; Phase 2 keeps via KM
	//   (50, observed) — event at t=50, contributes to KM
	//   (200, censored) — censored after τ, unambiguous label 0
	//   (200, observed) — event after τ, unambiguous label 0
	//
	// With one bin covering all four rows, KM processes the event at
	// t=50: at risk = 4, d = 1 → S(50) = 3/4 = 0.75. No events after,
	// so S(100) = 0.75 and the bin's observed rate is 1 − 0.75 = 0.25.
	// BrierN should be 3 (everything except the censored-before-τ row);
	// NumKept should be 4 (bin-wise KM doesn't drop).
	holdout := []domain.InterAccessInterval{
		{SlotID: "a", Duration: 50, IsObserved: false},
		{SlotID: "b", Duration: 50, IsObserved: true},
		{SlotID: "c", Duration: 200, IsObserved: false},
		{SlotID: "d", Duration: 200, IsObserved: true},
	}
	curve, err := CalibrationCurveFromCox(res, holdout, 100, 1)
	if err != nil {
		t.Fatalf("CalibrationCurveFromCox: %v", err)
	}
	if curve.NumDropped != 0 {
		t.Errorf("NumDropped=%d want 0 (bin-wise KM should not drop)", curve.NumDropped)
	}
	if curve.NumKept != 4 {
		t.Errorf("NumKept=%d want 4", curve.NumKept)
	}
	if curve.BrierN != 3 {
		t.Errorf("BrierN=%d want 3 (one censored-before-τ excluded from Brier)", curve.BrierN)
	}
	if got := len(curve.Bins); got != 1 {
		t.Fatalf("len(Bins)=%d want 1", got)
	}
	if got := curve.Bins[0].N; got != 4 {
		t.Errorf("Bins[0].N=%d want 4", got)
	}
	// Bin-wise KM observed rate should be close to 0.25 (see comment).
	if got := curve.Bins[0].ObservedRate; math.Abs(got-0.25) > 1e-9 {
		t.Errorf("Bins[0].ObservedRate=%v want 0.25", got)
	}
}

func TestCalibrationCurve_OnMockTrainHoldoutEndToEnd(t *testing.T) {
	train, holdout := setupMockTrainHoldout(t)
	assertTrainHoldoutSlotDisjoint(t, train, holdout)

	res, err := NewStatmodelFitter().FitCoxPH(train, DefaultCoxPredictors)
	if err != nil {
		t.Fatalf("FitCoxPH: %v", err)
	}
	curve, err := CalibrationCurveFromCox(res, holdout, 2000, 5)
	if err != nil {
		t.Fatalf("CalibrationCurveFromCox: %v", err)
	}
	assertCalibrationCurveSane(t, curve)
}

func setupMockTrainHoldout(t *testing.T) (train, holdout []domain.InterAccessInterval) {
	t.Helper()
	ctx := context.Background()
	db := openDB(t)
	cfg := extractor.DefaultMockConfig()
	cfg.NumContracts = 50
	cfg.SlotsPerContractXmin = 5
	cfg.SlotsPerContractMax = 30
	cfg.TotalBlocks = 20_000
	cfg.Window = domain.ObservationWindow{Start: 4_000, End: 20_000}
	cfg.AccessRateXmin = 1e-4
	cfg.MaxEventsPerSlot = 200
	cfg.PeriodBlocks = 2_000
	if err := extractor.NewMockExtractor(cfg).Extract(ctx, db); err != nil {
		t.Fatalf("Extract: %v", err)
	}
	built, err := BuildIntervals(ctx, db, cfg.Window)
	if err != nil {
		t.Fatalf("BuildIntervals: %v", err)
	}
	train, holdout, err = TrainHoldoutSplitBySlot(built.Intervals, 0.3, 99)
	if err != nil {
		t.Fatalf("split: %v", err)
	}
	return train, holdout
}

func assertTrainHoldoutSlotDisjoint(t *testing.T, train, holdout []domain.InterAccessInterval) {
	t.Helper()
	trainSlots := make(map[string]bool)
	for _, it := range train {
		trainSlots[it.SlotID] = true
	}
	for _, it := range holdout {
		if trainSlots[it.SlotID] {
			t.Fatalf("leakage: slot %q in both", it.SlotID)
		}
	}
}

func assertCalibrationCurveSane(t *testing.T, curve *CalibrationCurve) {
	t.Helper()
	if curve.NumKept == 0 {
		t.Fatal("kept zero rows")
	}
	for i, b := range curve.Bins {
		if b.PredictedMean < 0 || b.PredictedMean > 1 || b.ObservedRate < 0 || b.ObservedRate > 1 {
			t.Errorf("bin[%d] out of [0,1]: %+v", i, b)
		}
	}
	if math.IsNaN(curve.BrierScore) || curve.BrierScore < 0 || curve.BrierScore > 1 {
		t.Errorf("BrierScore=%v out of [0,1]", curve.BrierScore)
	}
}

// --- helpers ------------------------------------------------------------

func medianFloat(xs []float64) float64 {
	c := append([]float64(nil), xs...)
	for i := 1; i < len(c); i++ {
		for j := i; j > 0 && c[j-1] > c[j]; j-- {
			c[j-1], c[j] = c[j], c[j-1]
		}
	}
	return c[len(c)/2]
}

func extractPredicted(c *CalibrationCurve) []float64 {
	out := make([]float64, len(c.Bins))
	for i, b := range c.Bins {
		out[i] = b.PredictedMean
	}
	return out
}
func extractObserved(c *CalibrationCurve) []float64 {
	out := make([]float64, len(c.Bins))
	for i, b := range c.Bins {
		out[i] = b.ObservedRate
	}
	return out
}

// silence unused import warning in tiny test builds
var _ = rand.New
