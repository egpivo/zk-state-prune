// Package app carries the pure-function pipeline helpers that used to
// live in cmd/zksp/main.go. They are split out so they can be unit
// tested without a cobra harness: no flag parsing, no os.Stdout, no
// global config. The CLI layer is now responsible only for
//   - parsing flags into typed arguments,
//   - opening / closing the DB,
//   - choosing between text and JSON rendering.
// Everything statistical — split → fit → PH check → calibrate →
// assemble policy — is here and independently exercisable.
package app

import (
	"fmt"
	"log/slog"
	"sort"

	"github.com/egpivo/zk-state-prune/internal/analysis"
	"github.com/egpivo/zk-state-prune/internal/model"
	"github.com/egpivo/zk-state-prune/internal/pruning"
)

// CoxFitReport is the display-layer bundle produced by BuildCoxFitReport.
// JSON tags match the original shape so `fit --format json` output is
// byte-identical after extraction.
type CoxFitReport struct {
	Tau              float64                    `json:"tau"`
	TrainIntervals   int                        `json:"train_intervals"`
	HoldoutIntervals int                        `json:"holdout_intervals"`
	Cox              *analysis.CoxResult        `json:"cox"`
	PH               *analysis.PHTestResult     `json:"ph_test"`
	RawCurve         *analysis.CalibrationCurve `json:"raw_calibration"`
	CalibratedCurve  *analysis.CalibrationCurve `json:"isotonic_calibration"`
	BrierDelta       float64                    `json:"brier_delta"`
}

// CoxStrataColumn translates the user-facing --stratify flag value
// ("none" / "contract-type" / "slot-type") into a dstream adapter
// column name consumed by FitCoxPHStratified. The empty string means
// unstratified. Unknown values return an error so the CLI can surface
// a helpful message rather than silently fitting an unstratified model.
func CoxStrataColumn(stratify string) (string, error) {
	switch stratify {
	case "", "none":
		return "", nil
	case "contract-type", "contract":
		return analysis.ColContractType, nil
	case "slot-type", "slot":
		return analysis.ColSlotType, nil
	default:
		return "", fmt.Errorf("unknown cox stratify mode %q (use none|contract-type|slot-type)", stratify)
	}
}

// MedianDuration picks a default Cox tau when the user didn't pass one
// explicitly: the median observed interval duration in the training
// set. An empty slice maps to 1 so downstream math doesn't divide by
// zero; a zero median (e.g. all durations in a degenerate fixture)
// also clamps to 1 for the same reason.
func MedianDuration(ivs []model.InterAccessInterval) float64 {
	if len(ivs) == 0 {
		return 1
	}
	xs := make([]float64, len(ivs))
	for i, it := range ivs {
		xs[i] = float64(it.Duration)
	}
	sort.Float64s(xs)
	v := xs[len(xs)/2]
	if v <= 0 {
		v = 1
	}
	return v
}

// CalibrationCurveFromCalibrated routes the calibrated model's
// PredictAccessProbForInterval through the shared
// CalibrationCurveFromPredict helper so the post-isotonic reliability
// diagram uses the exact same bin-wise KM logic as the raw Cox curve.
func CalibrationCurveFromCalibrated(
	calib *analysis.CalibratedModel,
	holdout []model.InterAccessInterval,
	nBins int,
) (*analysis.CalibrationCurve, error) {
	if calib == nil || calib.Base == nil {
		return nil, fmt.Errorf("CalibrationCurveFromCalibrated: nil calibrated model")
	}
	predict := func(it model.InterAccessInterval) float64 {
		return calib.PredictAccessProbForInterval(it)
	}
	return analysis.CalibrationCurveFromPredict(predict, holdout, calib.Tau, nBins)
}

// BuildCoxFitReport drives the Cox PH pipeline — split → fit → PH
// check → raw calibration curve → isotonic recalibration → post
// calibration curve — and returns the stages packed into a
// CoxFitReport plus the CalibratedModel. Keeping the model separate
// lets `fit --save` persist it to disk without dragging the bulky
// training intervals into the sibling JSON report.
//
// tauFlag=0 means "pick a sensible default": MedianDuration(train).
// stratify=""/"none" fits an unstratified Cox; other values route
// through CoxStrataColumn.
func BuildCoxFitReport(
	fitter analysis.StatmodelFitter,
	intervals []model.InterAccessInterval,
	holdoutFrac float64,
	splitSeed uint64,
	tauFlag uint64,
	stratify string,
) (*CoxFitReport, *analysis.CalibratedModel, error) {
	train, holdout, err := analysis.TrainHoldoutSplitBySlot(intervals, holdoutFrac, splitSeed)
	if err != nil {
		return nil, nil, fmt.Errorf("split: %w", err)
	}
	slog.Info("cox split", "train_intervals", len(train), "holdout_intervals", len(holdout), "stratify", stratify)

	strataColumn, err := CoxStrataColumn(stratify)
	if err != nil {
		return nil, nil, err
	}
	res, err := fitter.FitCoxPHStratified(train, analysis.DefaultCoxPredictors, strataColumn)
	if err != nil {
		return nil, nil, fmt.Errorf("fit cox: %w", err)
	}
	ph, err := fitter.CheckPH(res)
	if err != nil {
		return nil, nil, fmt.Errorf("check PH: %w", err)
	}
	tau := float64(tauFlag)
	if tau == 0 {
		tau = MedianDuration(train)
	}
	rawCurve, err := analysis.CalibrationCurveFromCox(res, holdout, tau, 10)
	if err != nil {
		return nil, nil, fmt.Errorf("calibration curve: %w", err)
	}
	calib, err := fitter.CalibrateAt(res, holdout, tau)
	if err != nil {
		return nil, nil, fmt.Errorf("calibrate: %w", err)
	}
	postCurve, err := CalibrationCurveFromCalibrated(calib, holdout, 10)
	if err != nil {
		return nil, nil, fmt.Errorf("post-calibration curve: %w", err)
	}
	return &CoxFitReport{
		Tau:              tau,
		TrainIntervals:   len(train),
		HoldoutIntervals: len(holdout),
		Cox:              res,
		PH:               ph,
		RawCurve:         rawCurve,
		CalibratedCurve:  postCurve,
		BrierDelta:       postCurve.BrierScore - rawCurve.BrierScore,
	}, calib, nil
}

// BuildStatisticalPolicy fits + calibrates a Cox model on the
// provided intervals and wraps the result in a StatisticalPolicy.
// Fit on the train split; calibrate on the disjoint holdout.
//
// robust=true selects the upper-endpoint variant:
//
//	d_i* = argmin_d max_{p ∈ U_i} [c(d) + ℓ(d) · p]
//
// Labels are "statistical" and "statistical-robust" respectively so
// the comparison table in SimResults keeps them distinct.
func BuildStatisticalPolicy(
	intervals []model.InterAccessInterval,
	holdoutFrac float64,
	splitSeed uint64,
	tauFlag uint64,
	costs pruning.CostParams,
	robust bool,
) (*pruning.StatisticalPolicy, error) {
	train, holdout, err := analysis.TrainHoldoutSplitBySlot(intervals, holdoutFrac, splitSeed)
	if err != nil {
		return nil, fmt.Errorf("split: %w", err)
	}
	fitter := analysis.NewStatmodelFitter()
	res, err := fitter.FitCoxPH(train, analysis.DefaultCoxPredictors)
	if err != nil {
		return nil, fmt.Errorf("fit cox: %w", err)
	}
	tau := float64(tauFlag)
	if tau == 0 {
		tau = MedianDuration(train)
	}
	calib, err := fitter.CalibrateAt(res, holdout, tau)
	if err != nil {
		return nil, fmt.Errorf("calibrate: %w", err)
	}
	return StatisticalPolicyFromCalibrated(calib, costs, robust)
}

// StatisticalPolicyFromCalibrated is the shared construction path for
// "turn a CalibratedModel into a StatisticalPolicy". Used by the
// fresh-fit flow in BuildStatisticalPolicy as well as the CLI's
// --model path, which loads a previously persisted model and skips
// the fit pipeline entirely. Keeping the closure logic in one place
// means fresh-fit and loaded-model policies behave identically.
//
// The robust variant uses two separately-fit ε values, each one a
// MARGINAL (not simultaneous) coverage bound on its own target:
//
//   - At idle = 0 we use the calibrated point upper bound
//     p̂(τ) + Epsilon (PredictUpperAccessProbForInterval). Epsilon
//     is the split-conformal quantile fit on PAV residuals at
//     τ / idle = 0, so it covers that exact probe.
//
//   - At idle > 0 we use the raw Cox conditional upper bound
//     (1 − S(u+τ)/S(u)) + ConditionalEpsilon
//     (PredictUpperConditionalAccessProb). ConditionalEpsilon was fit
//     on one deterministically-sampled u per holdout row; it covers
//     a fresh single-u probe marginally, not the worst over the
//     ~21 u values the T*-search scans per interval. Treat it as a
//     principled pessimism margin rather than a simultaneous bound.
//
// Falls back to raw point estimates if the corresponding ε is zero
// (e.g. tiny holdouts where ConditionalEpsilon couldn't be fit on
// 10+ unambiguous rows).
func StatisticalPolicyFromCalibrated(
	calib *analysis.CalibratedModel,
	costs pruning.CostParams,
	robust bool,
) (*pruning.StatisticalPolicy, error) {
	if calib == nil || calib.Base == nil {
		return nil, fmt.Errorf("StatisticalPolicyFromCalibrated: nil model")
	}
	cox := calib.Base
	rawCondP := func(it model.InterAccessInterval, idle float64) float64 {
		sU := cox.SurvivalForInterval(it, idle)
		if sU <= 0 {
			return 1
		}
		sUTau := cox.SurvivalForInterval(it, idle+calib.Tau)
		p := 1 - sUTau/sU
		if p < 0 {
			return 0
		}
		if p > 1 {
			return 1
		}
		return p
	}

	name := "statistical"
	var predict pruning.CondAccessProbFunc = rawCondP
	if robust {
		name = "statistical-robust"
		predict = func(it model.InterAccessInterval, idle float64) float64 {
			if idle == 0 {
				return calib.PredictUpperAccessProbForInterval(it)
			}
			return calib.PredictUpperConditionalAccessProb(it, idle)
		}
	}
	return pruning.NewStatisticalPolicy(name, predict, calib.Tau, costs)
}
