package analysis

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
)

// modelFileSchemaVersion is bumped whenever the on-disk format changes.
// Writers always emit the current version; readers accept any version
// in modelFileSchemaVersionsSupported. Past additions:
//
//	v1: initial — single baseline hazard, one ε.
//	v2: stratified Cox (StratumColumn, StratumLabels, per-stratum
//	    baselines). Omitempty on the wire so v1 files still parse.
//	v3: conditional split-conformal (ConditionalEpsilon). Omitempty on
//	    the wire so v1/v2 files still parse with ConditionalEpsilon=0
//	    (the "no conditional margin" signal the robust policy already
//	    falls back on).
const (
	modelFileSchemaVersionCurrent = 3
)

var modelFileSchemaVersionsSupported = map[int]bool{1: true, 2: true, 3: true}

// ModelFile is the on-disk representation of a fitted + calibrated
// Cox model. It captures every field prediction needs (coefficients,
// per-predictor scaling, baseline cumulative hazard, isotonic grid,
// conformal ε, horizon τ) plus a schema version for forward
// compatibility. Training diagnostics (StdErr, PValue, ZScore,
// LogLike, training intervals) are intentionally not round-tripped:
// they are meaningful only at fit time and would bloat the on-disk
// format for no prediction-time benefit.
//
// Round-trip semantics: SaveModelFile(path, m); m2 := LoadModelFile(path);
// PredictAccessProb / PredictUpperAccessProb / SurvivalForInterval on m2
// return exactly the same values as the original. CheckPH and Calibrate
// on m2 fail because the training intervals were not persisted.
type ModelFile struct {
	SchemaVersion int `json:"schema_version"`

	Tau                float64 `json:"tau"`
	Epsilon            float64 `json:"epsilon"`
	ConditionalEpsilon float64 `json:"conditional_epsilon,omitempty"`
	CoverageLevel      float64 `json:"coverage_level"`

	PredX       []float64 `json:"pred_x"`
	CalibratedY []float64 `json:"calibrated_y"`

	Cox coxPersist `json:"cox"`
}

// coxPersist is the fields of CoxResult that a loaded model needs for
// prediction. We use explicit *float64 for NaN-bearing diagnostics so
// the JSON encoder doesn't choke on a partial fit's NaN padding.
type coxPersist struct {
	Predictors     []string     `json:"predictors"`
	Coef           []float64    `json:"coef"`
	Scales         []CovarScale `json:"scales"`
	BaselineTime   []float64    `json:"baseline_time"`
	BaselineCumHaz []float64    `json:"baseline_cum_haz"`

	// Stratified-fit extensions (schema v2+). Absent on v1 files.
	StratumColumn         string      `json:"stratum_column,omitempty"`
	StratumLabels         []float64   `json:"stratum_labels,omitempty"`
	StratumBaselineTimes  [][]float64 `json:"stratum_baseline_times,omitempty"`
	StratumBaselineCumHaz [][]float64 `json:"stratum_baseline_cum_haz,omitempty"`

	// Optional diagnostics. Zero-length when the fit didn't produce
	// them (partial fit); non-NaN values in the slice have their
	// real meaning.
	StdErr    []*float64 `json:"std_err,omitempty"`
	PValue    []*float64 `json:"p_value,omitempty"`
	ZScore    []*float64 `json:"z_score,omitempty"`
	LogLike   *float64   `json:"log_like,omitempty"`
	NumObs    int        `json:"num_obs,omitempty"`
	NumEvents int        `json:"num_events,omitempty"`
}

// SaveModelFile serializes m to path as pretty-printed JSON. The file
// is written atomically via a temporary sibling to prevent half-written
// models from being loaded on a subsequent crash.
func SaveModelFile(path string, m *CalibratedModel) error {
	if m == nil {
		return fmt.Errorf("SaveModelFile: nil model")
	}
	if m.Base == nil {
		return fmt.Errorf("SaveModelFile: CalibratedModel.Base is nil")
	}

	cox := m.Base
	file := ModelFile{
		SchemaVersion:      modelFileSchemaVersionCurrent,
		Tau:                m.Tau,
		Epsilon:            m.Epsilon,
		ConditionalEpsilon: m.ConditionalEpsilon,
		CoverageLevel:      m.CoverageLevel,
		PredX:              append([]float64(nil), m.PredX...),
		CalibratedY:        append([]float64(nil), m.CalibratedY...),
		Cox: coxPersist{
			Predictors:            append([]string(nil), cox.Predictors...),
			Coef:                  append([]float64(nil), cox.Coef...),
			Scales:                append([]CovarScale(nil), cox.Scales...),
			BaselineTime:          append([]float64(nil), cox.BaselineTime...),
			BaselineCumHaz:        append([]float64(nil), cox.BaselineCumHaz...),
			StratumColumn:         cox.StratumColumn,
			StratumLabels:         append([]float64(nil), cox.StratumLabels...),
			StratumBaselineTimes:  cloneFloat2D(cox.StratumBaselineTimes),
			StratumBaselineCumHaz: cloneFloat2D(cox.StratumBaselineCumHaz),
			StdErr:                nullableFloats(cox.StdErr),
			PValue:                nullableFloats(cox.PValue),
			ZScore:                nullableFloats(cox.ZScore),
			LogLike:               nullableFloat(cox.LogLike),
			NumObs:                cox.NumObs,
			NumEvents:             cox.NumEvents,
		},
	}

	b, err := json.MarshalIndent(file, "", "  ")
	if err != nil {
		return fmt.Errorf("SaveModelFile: marshal: %w", err)
	}
	tmp := path + ".tmp"
	if err := os.WriteFile(tmp, b, 0o600); err != nil {
		return fmt.Errorf("SaveModelFile: write %s: %w", tmp, err)
	}
	if err := os.Rename(tmp, path); err != nil {
		_ = os.Remove(tmp)
		return fmt.Errorf("SaveModelFile: rename: %w", err)
	}
	return nil
}

// LoadModelFile reads a persisted model from path. The training
// intervals are not restored, so CheckPH and Calibrate are unavailable
// on the returned CalibratedModel; policy prediction (which is what a
// deployed model does) works normally.
func LoadModelFile(path string) (*CalibratedModel, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("LoadModelFile: read %s: %w", path, err)
	}
	var file ModelFile
	if err := json.Unmarshal(b, &file); err != nil {
		return nil, fmt.Errorf("LoadModelFile: parse %s: %w", path, err)
	}
	if !modelFileSchemaVersionsSupported[file.SchemaVersion] {
		return nil, fmt.Errorf("LoadModelFile: schema_version=%d not supported by this reader (current: %d)", file.SchemaVersion, modelFileSchemaVersionCurrent)
	}
	if len(file.PredX) != len(file.CalibratedY) {
		return nil, fmt.Errorf("LoadModelFile: isotonic grid size mismatch (PredX=%d CalibratedY=%d)", len(file.PredX), len(file.CalibratedY))
	}
	cp := file.Cox
	// Every predict-path index lookup expects Predictors, Coef, and
	// Scales to be the same length. Cast a mismatched file out early
	// rather than letting SurvivalForInterval panic later with an
	// opaque index-out-of-range.
	nPred := len(cp.Predictors)
	if nPred == 0 {
		return nil, fmt.Errorf("LoadModelFile: cox has zero predictors")
	}
	if len(cp.Coef) != nPred {
		return nil, fmt.Errorf("LoadModelFile: cox coef len=%d != predictors=%d", len(cp.Coef), nPred)
	}
	if len(cp.Scales) != nPred {
		return nil, fmt.Errorf("LoadModelFile: cox scales len=%d != predictors=%d", len(cp.Scales), nPred)
	}
	// Baseline hazard: either a single grid (unstratified) or a
	// per-stratum map consistent with StratumLabels.
	if cp.StratumColumn == "" {
		if len(cp.BaselineTime) != len(cp.BaselineCumHaz) {
			return nil, fmt.Errorf("LoadModelFile: baseline time/haz len mismatch (%d vs %d)", len(cp.BaselineTime), len(cp.BaselineCumHaz))
		}
		if len(cp.StratumLabels) != 0 || len(cp.StratumBaselineTimes) != 0 || len(cp.StratumBaselineCumHaz) != 0 {
			return nil, fmt.Errorf("LoadModelFile: unstratified model has non-empty stratum fields")
		}
	} else {
		nStrata := len(cp.StratumLabels)
		if nStrata == 0 {
			return nil, fmt.Errorf("LoadModelFile: stratified model (%q) has no stratum labels", cp.StratumColumn)
		}
		if len(cp.StratumBaselineTimes) != nStrata || len(cp.StratumBaselineCumHaz) != nStrata {
			return nil, fmt.Errorf("LoadModelFile: stratified model has %d labels but times=%d, hazards=%d",
				nStrata, len(cp.StratumBaselineTimes), len(cp.StratumBaselineCumHaz))
		}
		for i := 0; i < nStrata; i++ {
			if len(cp.StratumBaselineTimes[i]) != len(cp.StratumBaselineCumHaz[i]) {
				return nil, fmt.Errorf("LoadModelFile: stratum %d baseline time/haz len mismatch (%d vs %d)",
					i, len(cp.StratumBaselineTimes[i]), len(cp.StratumBaselineCumHaz[i]))
			}
		}
	}

	c := file.Cox
	cox := &CoxResult{
		Predictors:            c.Predictors,
		Coef:                  c.Coef,
		Scales:                c.Scales,
		BaselineTime:          c.BaselineTime,
		BaselineCumHaz:        c.BaselineCumHaz,
		StdErr:                denullableFloats(c.StdErr, len(c.Coef)),
		PValue:                denullableFloats(c.PValue, len(c.Coef)),
		ZScore:                denullableFloats(c.ZScore, len(c.Coef)),
		LogLike:               denullableFloat(c.LogLike),
		NumObs:                c.NumObs,
		NumEvents:             c.NumEvents,
		StratumColumn:         c.StratumColumn,
		StratumLabels:         c.StratumLabels,
		StratumBaselineTimes:  c.StratumBaselineTimes,
		StratumBaselineCumHaz: c.StratumBaselineCumHaz,
	}
	return &CalibratedModel{
		Base:               cox,
		Tau:                file.Tau,
		Epsilon:            file.Epsilon,
		ConditionalEpsilon: file.ConditionalEpsilon,
		CoverageLevel:      file.CoverageLevel,
		PredX:              file.PredX,
		CalibratedY:        file.CalibratedY,
	}, nil
}

// cloneFloat2D deep-copies a [][]float64 so the persisted struct
// doesn't retain references into the live model. Returns nil on nil
// input so the JSON encoder's omitempty elides the field.
func cloneFloat2D(in [][]float64) [][]float64 {
	if in == nil {
		return nil
	}
	out := make([][]float64, len(in))
	for i, row := range in {
		out[i] = append([]float64(nil), row...)
	}
	return out
}

// nullableFloats converts a slice with possible NaN / Inf values into
// a slice of *float64 where NaN / Inf entries become nil. Empty input
// returns nil so omitempty elides the JSON field entirely.
func nullableFloats(xs []float64) []*float64 {
	if len(xs) == 0 {
		return nil
	}
	out := make([]*float64, len(xs))
	for i, v := range xs {
		if math.IsNaN(v) || math.IsInf(v, 0) {
			continue
		}
		vv := v
		out[i] = &vv
	}
	return out
}

func nullableFloat(v float64) *float64 {
	if math.IsNaN(v) || math.IsInf(v, 0) {
		return nil
	}
	return &v
}

// denullableFloats reverses nullableFloats, expanding missing entries
// back to NaN so downstream consumers see a consistent slice length.
// Empty input or mismatched length pads with NaN up to `want`.
func denullableFloats(xs []*float64, want int) []float64 {
	out := make([]float64, want)
	for i := range out {
		if i < len(xs) && xs[i] != nil {
			out[i] = *xs[i]
		} else {
			out[i] = math.NaN()
		}
	}
	return out
}

func denullableFloat(v *float64) float64 {
	if v == nil {
		return math.NaN()
	}
	return *v
}
