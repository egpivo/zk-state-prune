// Command zksp is the CLI entry point for the zk-state-prune toolkit.
//
// Phase 2 wires every subcommand to the analysis + pruning pipeline and
// supports both the human-readable text output and a `--format json`
// mode for piping into downstream tooling. YAML config loading (viper)
// is deferred to Phase 3 — the current flag set plus hardcoded defaults
// covers every value the project needs.
package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"log/slog"
	"math"
	"os"
	"os/signal"
	"sort"
	"syscall"
	"text/tabwriter"

	"github.com/spf13/cobra"

	"github.com/egpivo/zk-state-prune/internal/analysis"
	"github.com/egpivo/zk-state-prune/internal/config"
	"github.com/egpivo/zk-state-prune/internal/extractor"
	"github.com/egpivo/zk-state-prune/internal/model"
	"github.com/egpivo/zk-state-prune/internal/pruning"
	"github.com/egpivo/zk-state-prune/internal/storage"
)

// globalCfg is the loaded YAML config (or Default() if --config was
// unset). It is populated by the root command's PersistentPreRunE
// hook before any subcommand RunE runs, so subcommands can read
// globalCfg.* freely for flag defaults.
var globalCfg = config.Default()

// outputFormat is the rendering mode shared by every subcommand that
// produces structured output. Defaults to text.
type outputFormat string

const (
	formatText outputFormat = "text"
	formatJSON outputFormat = "json"
)

func (f outputFormat) isJSON() bool { return f == formatJSON }

// addFormatFlag registers the shared --format flag.
func addFormatFlag(cmd *cobra.Command, format *string) {
	cmd.Flags().StringVar(format, "format", "text", "output format: text|json")
}

// emitJSON writes v as pretty-printed JSON.
func emitJSON(w io.Writer, v any) error {
	enc := json.NewEncoder(w)
	enc.SetIndent("", "  ")
	return enc.Encode(v)
}

var version = "0.1.0-dev"

const (
	defaultDBPath      = "zksp.db"
	defaultWindowStart = uint64(200_000)
	defaultWindowEnd   = uint64(1_000_000)
)

func main() {
	logger := slog.New(slog.NewTextHandler(os.Stderr, &slog.HandlerOptions{Level: slog.LevelInfo}))
	slog.SetDefault(logger)

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	if err := newRootCmd().ExecuteContext(ctx); err != nil {
		slog.Error("command failed", "err", err)
		os.Exit(1)
	}
}

func newRootCmd() *cobra.Command {
	var cfgPath string
	root := &cobra.Command{
		Use:           "zksp",
		Short:         "Statistical state lifecycle modeling & pruning for ZK rollups",
		Version:       version,
		SilenceUsage:  true,
		SilenceErrors: true,
		PersistentPreRunE: func(cmd *cobra.Command, _ []string) error {
			// Load --config if set, or try configs/default.yaml as a
			// best-effort fallback so a user with the checked-in
			// default file gets it for free. A missing default file
			// is not an error.
			path := cfgPath
			autoTry := path == ""
			if autoTry {
				path = "configs/default.yaml"
			}
			loaded, err := config.Load(path)
			if err != nil {
				if autoTry && errors.Is(err, fs.ErrNotExist) {
					globalCfg = config.Default()
					return nil
				}
				return fmt.Errorf("load config: %w", err)
			}
			globalCfg = loaded
			slog.Info("config loaded",
				"path", path,
				"ram_unit_cost", globalCfg.Pruning.Cost.RAMUnitCost,
				"miss_penalty", globalCfg.Pruning.Cost.MissPenalty)
			return nil
		},
	}
	root.PersistentFlags().StringVar(&cfgPath, "config", "", "path to YAML config (default: configs/default.yaml if present)")
	root.AddCommand(
		newExtractCmd(),
		newEDACmd(),
		newFitCmd(),
		newSimulateCmd(),
		newReportCmd(),
	)
	return root
}

// openDB is the shared helper every subcommand uses to open the analysis
// store. Centralizing it means changes to pragmas / paths happen once.
func openDB(ctx context.Context, path string) (*storage.DB, error) {
	db, err := storage.Open(ctx, path)
	if err != nil {
		return nil, fmt.Errorf("open db %q: %w", path, err)
	}
	return db, nil
}

// loadIntervals is the common (open → BuildIntervals) preamble for fit /
// simulate / report. It returns the closed-over DB so the caller can use
// it for follow-up queries; callers must defer Close themselves.
func loadIntervals(ctx context.Context, db *storage.DB, w model.ObservationWindow) (analysis.IntervalBuildResult, error) {
	built, err := analysis.BuildIntervals(ctx, db, w)
	if err != nil {
		return analysis.IntervalBuildResult{}, fmt.Errorf("build intervals: %w", err)
	}
	if len(built.Intervals) == 0 {
		return built, fmt.Errorf("no intervals built — is the DB populated for window [%d,%d)?", w.Start, w.End)
	}
	return built, nil
}

// addWindowFlags binds --window-start / --window-end to the given command,
// preloaded with the Phase-1 default observation window.
func addWindowFlags(cmd *cobra.Command, start, end *uint64) {
	cmd.Flags().Uint64Var(start, "window-start", defaultWindowStart, "observation window start block")
	cmd.Flags().Uint64Var(end, "window-end", defaultWindowEnd, "observation window end block")
}

// addCostFlags binds the (RAMUnitCost, MissPenalty) pair shared by every
// command that runs the simulator. The defaults come from globalCfg,
// which the root command's PersistentPreRunE populates from YAML — but
// PersistentPreRunE fires AFTER flag construction, so the flag-default
// baseline is the process-start fallback (config.Default()). The
// real effective value is resolved at RunE time via costsFromFlags.
func addCostFlags(cmd *cobra.Command, ramUnit, missPen *float64) {
	d := globalCfg.Pruning.Cost
	cmd.Flags().Float64Var(ramUnit, "ram-unit-cost", d.RAMUnitCost, "hot-tier cost per slot per block (default: config)")
	cmd.Flags().Float64Var(missPen, "miss-penalty", d.MissPenalty, "cold-tier miss penalty per access (default: config)")
}

// costsFromFlags returns the final CostParams for a RunE invocation.
// If the user did not pass a cost flag explicitly, fall back to
// globalCfg (which PersistentPreRunE has now populated from YAML).
// This is how we honour "YAML overrides hardcoded defaults; CLI flags
// override YAML" with cobra's flag-default-at-construction-time model.
func costsFromFlags(cmd *cobra.Command, ramUnit, missPen float64) pruning.CostParams {
	if !cmd.Flags().Changed("ram-unit-cost") {
		ramUnit = globalCfg.Pruning.Cost.RAMUnitCost
	}
	if !cmd.Flags().Changed("miss-penalty") {
		missPen = globalCfg.Pruning.Cost.MissPenalty
	}
	return pruning.CostParams{RAMUnitCost: ramUnit, MissPenalty: missPen}
}

// ---------------------------------------------------------------- extract

func newExtractCmd() *cobra.Command {
	var (
		source       string
		output       string
		seed         uint64
		numContracts int
		totalBlocks  uint64
		force        bool
		rpcEndpoint  string
		rpcStart     uint64
		rpcEnd       uint64
	)
	cmd := &cobra.Command{
		Use:   "extract",
		Short: "Populate the analysis DB with state-diff data (mock or real RPC)",
		RunE: func(cmd *cobra.Command, _ []string) error {
			ctx := cmd.Context()
			switch source {
			case "mock":
				return runMockExtract(ctx, output, seed, numContracts, totalBlocks, force)
			case "rpc":
				return runRPCExtract(ctx, output, rpcEndpoint, rpcStart, rpcEnd, force)
			default:
				return fmt.Errorf("source %q not supported (use mock|rpc)", source)
			}
		},
	}
	cmd.Flags().StringVar(&source, "source", "mock", "data source: mock|rpc")
	cmd.Flags().StringVar(&output, "output", defaultDBPath, "output SQLite DB path")
	cmd.Flags().Uint64Var(&seed, "seed", 42, "(mock) PRNG seed for the generator")
	cmd.Flags().IntVar(&numContracts, "num-contracts", 0, "(mock) override default contract count (0 = use built-in)")
	cmd.Flags().Uint64Var(&totalBlocks, "total-blocks", 0, "(mock) override default block horizon (0 = use built-in)")
	cmd.Flags().BoolVar(&force, "force", false, "overwrite the output DB even if it already has slots/events")
	cmd.Flags().StringVar(&rpcEndpoint, "rpc", extractor.ScrollPublicRPC, "(rpc) JSON-RPC endpoint URL")
	cmd.Flags().Uint64Var(&rpcStart, "start", 0, "(rpc) first block to extract")
	cmd.Flags().Uint64Var(&rpcEnd, "end", 0, "(rpc) last block to extract (inclusive)")
	return cmd
}

// refuseOverwriteIfPopulated is the shared safety rail for `extract`
// against both the mock and rpc sources: on mock it guards the full
// db.Reset(); on rpc it guards a partial-trace DB that the user may
// want to keep rather than mix with a fresh range.
func refuseOverwriteIfPopulated(ctx context.Context, db *storage.DB, path string, force bool) error {
	slots, err := db.CountSlots(ctx)
	if err != nil {
		return fmt.Errorf("count existing slots: %w", err)
	}
	events, err := db.CountAccessEvents(ctx)
	if err != nil {
		return fmt.Errorf("count existing events: %w", err)
	}
	if (slots > 0 || events > 0) && !force {
		return fmt.Errorf("refusing to overwrite %q: %d slots and %d events already present. Use --force to proceed.",
			path, slots, events)
	}
	return nil
}

func runMockExtract(ctx context.Context, output string, seed uint64, numContracts int, totalBlocks uint64, force bool) error {
	db, err := openDB(ctx, output)
	if err != nil {
		return err
	}
	defer db.Close()
	if err := refuseOverwriteIfPopulated(ctx, db, output, force); err != nil {
		return err
	}

	cfg := extractor.DefaultMockConfig()
	cfg.Seed = seed
	if numContracts > 0 {
		cfg.NumContracts = numContracts
	}
	if totalBlocks > 0 {
		cfg.TotalBlocks = totalBlocks
		if cfg.Window.End > totalBlocks || cfg.Window.End == 0 {
			cfg.Window = model.ObservationWindow{Start: totalBlocks / 5, End: totalBlocks}
		}
	}
	slog.Info("extract begin",
		"source", "mock",
		"output", output,
		"seed", cfg.Seed,
		"contracts", cfg.NumContracts,
		"total_blocks", cfg.TotalBlocks,
		"window_start", cfg.Window.Start,
		"window_end", cfg.Window.End)

	ex := extractor.NewMockExtractor(cfg)
	if err := ex.Extract(ctx, db); err != nil {
		return fmt.Errorf("extract: %w", err)
	}
	d := ex.LastDiagnostics()
	slog.Info("extract complete",
		"contracts", d.Contracts,
		"slots", d.Slots,
		"events", d.Events,
		"pre_window_slots", d.PreWindowSlots,
		"periodic_contracts", d.PeriodicContracts,
		"events_in_window", d.EventsInWindow)
	return nil
}

func runRPCExtract(ctx context.Context, output, endpoint string, start, end uint64, force bool) error {
	if end == 0 || end < start {
		return fmt.Errorf("rpc extract: --end must be > 0 and >= --start (got start=%d end=%d)", start, end)
	}
	db, err := openDB(ctx, output)
	if err != nil {
		return err
	}
	defer db.Close()
	if err := refuseOverwriteIfPopulated(ctx, db, output, force); err != nil {
		return err
	}
	cfg := extractor.DefaultRPCConfig()
	cfg.Endpoint = endpoint
	cfg.Start = start
	cfg.End = end
	slog.Info("extract begin",
		"source", "rpc",
		"endpoint", cfg.Endpoint,
		"start", cfg.Start,
		"end", cfg.End,
		"output", output)
	ex, err := extractor.NewRPCExtractor(cfg)
	if err != nil {
		return err
	}
	if err := ex.Extract(ctx, db); err != nil {
		return fmt.Errorf("extract: %w", err)
	}
	d := ex.LastDiagnostics()
	slog.Info("extract complete",
		"blocks_fetched", d.BlocksFetched,
		"receipts_fetched", d.ReceiptsFetched,
		"logs_seen", d.LogsSeen,
		"transfer_logs", d.TransferLogs,
		"contracts_created", d.ContractsCreated,
		"slots_created", d.SlotsCreated,
		"events_persisted", d.EventsPersisted)
	return nil
}

// -------------------------------------------------------------------- eda

func newEDACmd() *cobra.Command {
	var dbPath, format string
	var startBlock, endBlock uint64
	cmd := &cobra.Command{
		Use:   "eda",
		Short: "Run exploratory data analysis on the populated DB",
		RunE: func(cmd *cobra.Command, _ []string) error {
			ctx := cmd.Context()
			db, err := openDB(ctx, dbPath)
			if err != nil {
				return err
			}
			defer db.Close()
			rep, err := analysis.RunEDAFull(ctx, db, model.ObservationWindow{Start: startBlock, End: endBlock})
			if err != nil {
				return fmt.Errorf("eda: %w", err)
			}
			if outputFormat(format).isJSON() {
				return emitJSON(os.Stdout, rep)
			}
			return printEDAReport(os.Stdout, rep)
		},
	}
	cmd.Flags().StringVar(&dbPath, "db", defaultDBPath, "SQLite DB path")
	addFormatFlag(cmd, &format)
	addWindowFlags(cmd, &startBlock, &endBlock)
	return cmd
}

// -------------------------------------------------------------------- fit

func newFitCmd() *cobra.Command {
	var dbPath, modelKind, stratify, format, savePath string
	var startBlock, endBlock uint64
	var holdoutFrac float64
	var splitSeed uint64
	var tau uint64
	cmd := &cobra.Command{
		Use:   "fit",
		Short: "Fit a survival model: km or cox (with PH check + isotonic calibration)",
		RunE: func(cmd *cobra.Command, _ []string) error {
			ctx := cmd.Context()
			db, err := openDB(ctx, dbPath)
			if err != nil {
				return err
			}
			defer db.Close()
			window := model.ObservationWindow{Start: startBlock, End: endBlock}
			built, err := loadIntervals(ctx, db, window)
			if err != nil {
				return err
			}

			fitter := analysis.NewStatmodelFitter()
			out := os.Stdout
			fmtMode := outputFormat(format)

			switch modelKind {
			case "km":
				if savePath != "" {
					return fmt.Errorf("--save is only supported with --model cox")
				}
				return runKMFit(out, fitter, built.Intervals, stratify, fmtMode)
			case "cox":
				return runCoxFit(out, fitter, built.Intervals, holdoutFrac, splitSeed, tau, fmtMode, savePath, stratify)
			default:
				return fmt.Errorf("unknown model %q (use km or cox)", modelKind)
			}
		},
	}
	cmd.Flags().StringVar(&dbPath, "db", defaultDBPath, "SQLite DB path")
	cmd.Flags().StringVar(&modelKind, "model", "km", "model type: km|cox")
	cmd.Flags().StringVar(&stratify, "stratify", "none", "stratify by: none|contract-type|slot-type (km: one curve per stratum; cox: per-stratum baseline hazard)")
	cmd.Flags().Float64Var(&holdoutFrac, "holdout", 0.3, "(cox only) holdout fraction for calibration split")
	cmd.Flags().Uint64Var(&splitSeed, "split-seed", 1, "(cox only) PRNG seed for the train/holdout split")
	cmd.Flags().Uint64Var(&tau, "tau", 0, "(cox only) calibration horizon in blocks (0 = median training Duration)")
	cmd.Flags().StringVar(&savePath, "save", "", "(cox only) write the fitted + calibrated model to this path as JSON")
	addFormatFlag(cmd, &format)
	addWindowFlags(cmd, &startBlock, &endBlock)
	return cmd
}

func runKMFit(out io.Writer, fitter analysis.SurvivalFitter, intervals []model.InterAccessInterval, stratify string, format outputFormat) error {
	var curves []*analysis.KMResult
	if stratify == "none" || stratify == "" {
		res, err := fitter.FitKaplanMeier(intervals)
		if err != nil {
			return fmt.Errorf("fit km: %w", err)
		}
		res.Label = "all"
		curves = []*analysis.KMResult{res}
	} else {
		var key func(model.InterAccessInterval) string
		switch stratify {
		case "contract-type", "contract":
			key = analysis.StratumByContractType
		case "slot-type", "slot":
			key = analysis.StratumBySlotType
		default:
			return fmt.Errorf("unknown stratify mode %q", stratify)
		}
		strata, err := analysis.FitKaplanMeierStratified(fitter, intervals, key)
		if err != nil {
			return fmt.Errorf("fit km stratified: %w", err)
		}
		labels := make([]string, 0, len(strata))
		for k := range strata {
			labels = append(labels, k)
		}
		sort.Strings(labels)
		curves = make([]*analysis.KMResult, 0, len(strata))
		for _, l := range labels {
			curves = append(curves, strata[l])
		}
	}
	if format.isJSON() {
		return emitJSON(out, curves)
	}
	return printKM(out, curves)
}

// coxFitReport bundles the stages of a Cox fit into one JSON-marshalable
// object so `fit --model cox --format json` can emit the whole pipeline
// in a single payload rather than separate sections of text.
type coxFitReport struct {
	Tau              float64                    `json:"tau"`
	TrainIntervals   int                        `json:"train_intervals"`
	HoldoutIntervals int                        `json:"holdout_intervals"`
	Cox              *analysis.CoxResult        `json:"cox"`
	PH               *analysis.PHTestResult     `json:"ph_test"`
	RawCurve         *analysis.CalibrationCurve `json:"raw_calibration"`
	CalibratedCurve  *analysis.CalibrationCurve `json:"isotonic_calibration"`
	BrierDelta       float64                    `json:"brier_delta"`
}

// buildCoxFitReport drives the Cox PH pipeline — split → fit → PH check
// → raw calibration curve → isotonic recalibration → post calibration
// curve — and returns the stages packed into a coxFitReport. Separated
// from runCoxFit so both the text and JSON paths can share the work,
// and so `report --format json` can embed the Cox section without
// re-routing text output through the CLI.
// coxStrataColumn translates the user-facing --stratify flag value
// ("none" / "contract-type" / …) into a dstream adapter column name
// consumed by FitCoxPHStratified. The empty string means unstratified.
// Kept in main.go so analysis stays free of CLI-idiom string mapping.
func coxStrataColumn(stratify string) (string, error) {
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

// buildCoxFitReport returns the display-layer coxFitReport plus the
// CalibratedModel that produced it. The latter is separate from the
// JSON shape so `fit --save` can round-trip the fitted model to disk
// without including the bulky training intervals in the sibling JSON
// report.
func buildCoxFitReport(
	fitter analysis.StatmodelFitter,
	intervals []model.InterAccessInterval,
	holdoutFrac float64,
	splitSeed uint64,
	tauFlag uint64,
	stratify string,
) (*coxFitReport, *analysis.CalibratedModel, error) {
	train, holdout, err := analysis.TrainHoldoutSplitBySlot(intervals, holdoutFrac, splitSeed)
	if err != nil {
		return nil, nil, fmt.Errorf("split: %w", err)
	}
	slog.Info("cox split", "train_intervals", len(train), "holdout_intervals", len(holdout), "stratify", stratify)

	strataColumn, err := coxStrataColumn(stratify)
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
		tau = medianDuration(train)
	}
	rawCurve, err := analysis.CalibrationCurveFromCox(res, holdout, tau, 10)
	if err != nil {
		return nil, nil, fmt.Errorf("calibration curve: %w", err)
	}
	calib, err := fitter.CalibrateAt(res, holdout, tau)
	if err != nil {
		return nil, nil, fmt.Errorf("calibrate: %w", err)
	}
	postCurve, err := calibrationCurveFromCalibrated(calib, holdout, 10)
	if err != nil {
		return nil, nil, fmt.Errorf("post-calibration curve: %w", err)
	}
	return &coxFitReport{
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

// printCoxFitReport renders a coxFitReport as the text sequence used by
// `fit --model cox` and the Cox section of `report`.
func printCoxFitReport(out io.Writer, r *coxFitReport) {
	if r == nil {
		return
	}
	_ = printCoxSummary(out, r.Cox)
	fmt.Fprintln(out, "\n-- proportional-hazards check (Schoenfeld) --")
	_ = printPHTest(out, r.PH)
	fmt.Fprintf(out, "\n-- calibration @ tau=%.0f blocks --\n", r.Tau)
	fmt.Fprintln(out, "raw Cox:")
	_ = printCalibrationCurve(out, r.RawCurve)
	fmt.Fprintln(out, "\nisotonic-recalibrated:")
	_ = printCalibrationCurve(out, r.CalibratedCurve)
	fmt.Fprintf(out, "\nBrier: raw=%.4f → calibrated=%.4f (delta=%+.4f)\n",
		r.RawCurve.BrierScore, r.CalibratedCurve.BrierScore, r.BrierDelta)
}

// runCoxFit is the `fit --model cox` handler. Thin adapter over
// buildCoxFitReport that chooses the output path based on --format.
func runCoxFit(
	out io.Writer,
	fitter analysis.StatmodelFitter,
	intervals []model.InterAccessInterval,
	holdoutFrac float64,
	splitSeed uint64,
	tauFlag uint64,
	format outputFormat,
	savePath string,
	stratify string,
) error {
	r, calib, err := buildCoxFitReport(fitter, intervals, holdoutFrac, splitSeed, tauFlag, stratify)
	if err != nil {
		return err
	}
	if savePath != "" {
		if err := analysis.SaveModelFile(savePath, calib); err != nil {
			return fmt.Errorf("save model: %w", err)
		}
		slog.Info("model saved", "path", savePath, "tau", calib.Tau, "epsilon", calib.Epsilon)
	}
	if format.isJSON() {
		return emitJSON(out, r)
	}
	printCoxFitReport(out, r)
	return nil
}

// buildStatisticalPolicy fits + calibrates a Cox model on the provided
// intervals and wraps the result in a pruning.StatisticalPolicy. The
// fit uses a 70/30 (or caller-specified) train/holdout split; the
// calibration runs on the holdout side so the isotonic map sees data
// the Cox fit didn't.
//
// When robust is true the policy uses the *upper* endpoint of the
// split-conformal interval around the calibrated probability instead
// of the point estimate. That implements
//
//	d_i* = argmin_d max_{p ∈ U_i} [c(d) + ℓ(d) · p]
//
// which prefers keeping a slot hot under uncertainty: the cold action
// only wins when the high-confidence upper bound is still below p*.
// Label: "statistical-robust" so it shows up distinctly from
// "statistical" in the comparison table.
//
// Phase-2 simplification: we then run the policy against the FULL
// interval set (in the caller), accepting a small leakage on the train
// side, in exchange for a fair side-by-side comparison with the fixed
// baselines that also see every row.
func buildStatisticalPolicy(
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
		tau = medianDuration(train)
	}
	calib, err := fitter.CalibrateAt(res, holdout, tau)
	if err != nil {
		return nil, fmt.Errorf("calibrate: %w", err)
	}
	return statisticalPolicyFromCalibrated(calib, costs, robust)
}

// statisticalPolicyFromCalibrated is the shared construction path for
// "turn a CalibratedModel into a StatisticalPolicy". Used by the
// fit-on-the-fly flow in buildStatisticalPolicy as well as the CLI's
// --model path, which loads a previously persisted model and skips the
// fit pipeline entirely. Keeping the closure logic in one place means
// the fresh-fit and loaded-model policies behave identically.
func statisticalPolicyFromCalibrated(
	calib *analysis.CalibratedModel,
	costs pruning.CostParams,
	robust bool,
) (*pruning.StatisticalPolicy, error) {
	if calib == nil || calib.Base == nil {
		return nil, fmt.Errorf("statisticalPolicyFromCalibrated: nil model")
	}
	// Build the conditional predictor used by the T*-search. The raw
	// Cox survival function is evaluated at (idle) and (idle + τ) to
	// compute the conditional access probability at each search
	// sample. The isotonic calibration only maps single-horizon
	// predictions at τ from idle=0, so we do NOT apply it at u>0 —
	// that's a different quantity the recalibration grid wasn't fit
	// for.
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
		// Robust variant. Two separately-fit conformal quantiles
		// back the policy, each one valid for the exact quantity it
		// was trained on:
		//
		//   - At the idle = 0 sample we use the *calibrated* point
		//     upper bound p̂(τ) + Epsilon (CalibratedModel.
		//     PredictUpperAccessProbForInterval), covered by the
		//     isotonic-grid split-conformal fit in CalibrateAt.
		//
		//   - At idle > 0 samples we use the *raw Cox conditional*
		//     upper bound (1 − S(u+τ)/S(u)) + ConditionalEpsilon
		//     (PredictUpperConditionalAccessProb), covered by the
		//     conditional split-conformal fit on (interval, u,
		//     label_at_u) triples in CalibrateAt's conditional pass.
		//
		// Each sample the T*-search evaluates is now backed by a
		// finite-sample upper bound on its own target quantity, so
		// the demotion rule's "only demote when we're confident"
		// reading is valid at every point on the search grid.
		// Falls back to raw point estimates if the corresponding ε
		// was not fit (e.g., tiny holdouts where
		// ConditionalEpsilon stays zero).
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

// medianDuration is a tiny in-place median for picking a default cox tau.
func medianDuration(ivs []model.InterAccessInterval) float64 {
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

// calibrationCurveFromCalibrated is a CLI-side convenience that routes
// the calibrated model's PredictAccessProbForInterval through the
// shared CalibrationCurveFromPredict helper, so the post-isotonic
// reliability diagram uses the exact same bin-wise KM policy as the
// raw Cox version.
func calibrationCurveFromCalibrated(
	calib *analysis.CalibratedModel,
	holdout []model.InterAccessInterval,
	nBins int,
) (*analysis.CalibrationCurve, error) {
	if calib == nil || calib.Base == nil {
		return nil, fmt.Errorf("calibrationCurveFromCalibrated: nil calibrated model")
	}
	predict := func(it model.InterAccessInterval) float64 {
		return calib.PredictAccessProbForInterval(it)
	}
	return analysis.CalibrationCurveFromPredict(predict, holdout, calib.Tau, nBins)
}

// --------------------------------------------------------------- simulate

func newSimulateCmd() *cobra.Command {
	var dbPath, policyName, format, modelPath string
	var startBlock, endBlock uint64
	var ramUnit, missPen float64
	var holdoutFrac float64
	var splitSeed, tau uint64
	var robust bool
	cmd := &cobra.Command{
		Use:   "simulate",
		Short: "Run a tiering simulation (no-prune | fixed-30d | fixed-90d | statistical)",
		RunE: func(cmd *cobra.Command, _ []string) error {
			ctx := cmd.Context()
			db, err := openDB(ctx, dbPath)
			if err != nil {
				return err
			}
			defer db.Close()
			built, err := loadIntervals(ctx, db, model.ObservationWindow{Start: startBlock, End: endBlock})
			if err != nil {
				return err
			}
			costs := costsFromFlags(cmd, ramUnit, missPen)

			var deps pruning.PolicyDeps
			if policyName == "statistical" || policyName == "statistical-robust" {
				useRobust := robust || policyName == "statistical-robust"
				var p *pruning.StatisticalPolicy
				if modelPath != "" {
					// Load a previously persisted model and wrap it
					// in a statistical policy — skipping the full fit
					// pipeline. This is the "fit once, simulate many
					// times" deployment flow.
					loaded, err := analysis.LoadModelFile(modelPath)
					if err != nil {
						return fmt.Errorf("load model: %w", err)
					}
					p, err = statisticalPolicyFromCalibrated(loaded, costs, useRobust)
					if err != nil {
						return fmt.Errorf("statistical from loaded model: %w", err)
					}
					slog.Info("statistical policy loaded from file",
						"model_path", modelPath, "tau", p.Tau())
				} else {
					p, err = buildStatisticalPolicy(built.Intervals, holdoutFrac, splitSeed, tau, costs, useRobust)
					if err != nil {
						return fmt.Errorf("statistical: %w", err)
					}
				}
				deps.Statistical = p
				slog.Info("statistical policy ready",
					"name", p.Name(), "tau", p.Tau(), "p_star", p.PStar(),
					"ram_unit_cost", costs.RAMUnitCost,
					"miss_penalty", costs.MissPenalty,
					"robust", useRobust)
			} else if modelPath != "" {
				return fmt.Errorf("--model is only meaningful with --policy statistical[-robust]")
			}
			policy, err := pruning.PolicyByName(policyName, deps)
			if err != nil {
				return err
			}

			res, err := pruning.Run(policy, built.Intervals, costs)
			if err != nil {
				return fmt.Errorf("simulate: %w", err)
			}
			if outputFormat(format).isJSON() {
				return emitJSON(os.Stdout, []*pruning.SimResult{res})
			}
			return printSimResults(os.Stdout, []*pruning.SimResult{res})
		},
	}
	cmd.Flags().StringVar(&dbPath, "db", defaultDBPath, "SQLite DB path")
	cmd.Flags().StringVar(&policyName, "policy", "fixed-30d",
		"policy: no-prune|fixed-30d|fixed-90d|statistical")
	cmd.Flags().StringVar(&modelPath, "model", "", "(statistical) load a persisted model from this path instead of refitting")
	cmd.Flags().Float64Var(&holdoutFrac, "holdout", 0.3, "(statistical) holdout fraction for calibration")
	cmd.Flags().Uint64Var(&splitSeed, "split-seed", 1, "(statistical) PRNG seed for the train/holdout split")
	cmd.Flags().Uint64Var(&tau, "tau", 0, "(statistical) horizon in blocks (0 = median training Duration)")
	cmd.Flags().BoolVar(&robust, "robust", false, "(statistical) use split-conformal upper-bound rule (demote only under high confidence)")
	addCostFlags(cmd, &ramUnit, &missPen)
	addFormatFlag(cmd, &format)
	addWindowFlags(cmd, &startBlock, &endBlock)
	return cmd
}

// ----------------------------------------------------------------- report

// fullReport is the JSON shape for `zksp report --format json`: all
// five sections bundled into one payload so downstream scripts can
// parse it without stream-mode parsing the text output.
type fullReport struct {
	EDA      *analysis.EDAReport  `json:"eda"`
	KMCurves []*analysis.KMResult `json:"km_curves"`
	Tiering  []*pruning.SimResult `json:"tiering"`
	Cox      *coxFitReport        `json:"cox,omitempty"`
}

func newReportCmd() *cobra.Command {
	var dbPath, format, modelPath string
	var startBlock, endBlock uint64
	var ramUnit, missPen float64
	cmd := &cobra.Command{
		Use:   "report",
		Short: "End-to-end EDA + KM + tiering comparison + Cox diagnostics",
		RunE: func(cmd *cobra.Command, _ []string) error {
			ctx := cmd.Context()
			db, err := openDB(ctx, dbPath)
			if err != nil {
				return err
			}
			defer db.Close()
			window := model.ObservationWindow{Start: startBlock, End: endBlock}
			out := os.Stdout
			fmtMode := outputFormat(format)

			// 1. EDA (full, with spatial + temporal)
			rep, err := analysis.RunEDAFull(ctx, db, window)
			if err != nil {
				return fmt.Errorf("eda: %w", err)
			}

			// 2. KM (overall + by contract type)
			built, err := loadIntervals(ctx, db, window)
			if err != nil {
				return err
			}
			fitter := analysis.NewStatmodelFitter()
			overall, err := fitter.FitKaplanMeier(built.Intervals)
			if err != nil {
				return fmt.Errorf("fit km: %w", err)
			}
			overall.Label = "all"
			byCat, err := analysis.FitKaplanMeierStratified(fitter, built.Intervals, analysis.StratumByContractType)
			if err != nil {
				return fmt.Errorf("fit km stratified: %w", err)
			}
			curves := []*analysis.KMResult{overall}
			catLabels := make([]string, 0, len(byCat))
			for k := range byCat {
				catLabels = append(catLabels, k)
			}
			sort.Strings(catLabels)
			for _, l := range catLabels {
				curves = append(curves, byCat[l])
			}

			// 3. Baseline + statistical + statistical-robust, all scored on
			// shared CostParams so TotalCost is directly comparable.
			// If --model is set, both statistical variants reuse the
			// same persisted model (the robust/point difference is a
			// policy-layer toggle on top of the same CalibratedModel).
			// Without --model we refit on the fly, once per variant.
			costs := costsFromFlags(cmd, ramUnit, missPen)
			policies := []pruning.Policy{pruning.NoPrune{}, pruning.Fixed30d, pruning.Fixed90d}
			buildStat := func(robust bool) (*pruning.StatisticalPolicy, error) {
				if modelPath != "" {
					loaded, err := analysis.LoadModelFile(modelPath)
					if err != nil {
						return nil, fmt.Errorf("load model: %w", err)
					}
					return statisticalPolicyFromCalibrated(loaded, costs, robust)
				}
				return buildStatisticalPolicy(built.Intervals, 0.3, 1, 0, costs, robust)
			}
			if stat, err := buildStat(false); err != nil {
				slog.Warn("statistical policy unavailable for report", "err", err)
			} else {
				policies = append(policies, stat)
			}
			if statR, err := buildStat(true); err != nil {
				slog.Warn("statistical-robust policy unavailable for report", "err", err)
			} else {
				policies = append(policies, statR)
			}
			results, err := pruning.RunAll(policies, built.Intervals, costs)
			if err != nil {
				return fmt.Errorf("run all policies: %w", err)
			}

			// 4. Cox PH diagnostics. Failure is non-fatal because
			// PH check is expected to reject on realistic traces. The
			// CalibratedModel returned alongside is currently unused
			// here (report doesn't --save); keep it bound to _ so the
			// compiler doesn't prune the return arity by accident.
			coxReport, _, coxErr := buildCoxFitReport(fitter, built.Intervals, 0.3, 1, 0, "")

			if fmtMode.isJSON() {
				return emitJSON(out, fullReport{
					EDA:      rep,
					KMCurves: curves,
					Tiering:  results,
					Cox:      coxReport,
				})
			}

			fmt.Fprintln(out, "=== EDA ===")
			if err := printEDAReport(out, rep); err != nil {
				return err
			}
			fmt.Fprintln(out, "\n=== Kaplan–Meier ===")
			if err := printKM(out, curves); err != nil {
				return err
			}
			fmt.Fprintln(out, "\n=== Tiering policies ===")
			if err := printSimResults(out, results); err != nil {
				return err
			}
			fmt.Fprintln(out, "\n=== Cox PH (70/30 split) ===")
			if coxErr != nil {
				fmt.Fprintf(out, "cox section skipped: %v\n", coxErr)
			} else {
				printCoxFitReport(out, coxReport)
			}
			return nil
		},
	}
	cmd.Flags().StringVar(&dbPath, "db", defaultDBPath, "SQLite DB path")
	cmd.Flags().StringVar(&modelPath, "model", "", "load a persisted statistical model from this path instead of refitting")
	addCostFlags(cmd, &ramUnit, &missPen)
	addFormatFlag(cmd, &format)
	addWindowFlags(cmd, &startBlock, &endBlock)
	return cmd
}

// --------------------------------------------------------------- printers

func printEDAReport(w io.Writer, r *analysis.EDAReport) error {
	tw := tabwriter.NewWriter(w, 0, 0, 2, ' ', 0)
	fmt.Fprintf(tw, "window\t[%d, %d)\n", r.Window.Start, r.Window.End)
	fmt.Fprintf(tw, "slots\t%d\n", r.SlotCount)
	fmt.Fprintf(tw, "intervals\t%d\n", r.TotalIntervals)
	fmt.Fprintf(tw, "right_censored\t%d (%.1f%%)\n", r.RightCensoredCount, 100*r.RightCensoredRate)
	fmt.Fprintf(tw, "left_truncated_slots\t%d (%.1f%%)\n", r.LeftTruncatedCount, 100*r.LeftTruncatedRate)
	fmt.Fprintf(tw, "slots_no_events\t%d\n", r.SlotsWithNoEvents)
	fmt.Fprintf(tw, "slots_skipped\t%d\n", r.SlotsSkipped)
	if err := tw.Flush(); err != nil {
		return err
	}
	fmt.Fprintln(w, "\n-- access frequency per slot --")
	printDistribution(w, r.Frequency)
	fmt.Fprintln(w, "\n-- inter-access time (observed only) --")
	printDistribution(w, r.InterAccessTime)
	fmt.Fprintln(w, "\n-- by contract type --")
	tw = tabwriter.NewWriter(w, 0, 0, 2, ' ', 0)
	fmt.Fprintln(tw, "category\tslots\tintervals\tcensored%\tIAT_p50\tIAT_p99")
	cats := make([]string, 0, len(r.ByContractType))
	idx := make(map[string]model.ContractCategory)
	for k := range r.ByContractType {
		s := k.String()
		cats = append(cats, s)
		idx[s] = k
	}
	sort.Strings(cats)
	for _, name := range cats {
		s := r.ByContractType[idx[name]]
		fmt.Fprintf(tw, "%s\t%d\t%d\t%.1f\t%.0f\t%.0f\n",
			name, s.Slots, s.Intervals, 100*s.RightCensoredRate,
			s.InterAccessTime.P50, s.InterAccessTime.P99)
	}
	if err := tw.Flush(); err != nil {
		return err
	}
	if r.Spatial != nil {
		fmt.Fprintln(w, "\n-- intra-contract co-access (Jaccard) --")
		fmt.Fprintf(w, "contracts_measured=%d  mean=%.3f  median=%.3f\n",
			r.Spatial.NumContracts, r.Spatial.MeanJaccard, r.Spatial.MedianJaccard)
	}
	if r.Temporal != nil {
		fmt.Fprintln(w, "\n-- temporal periodicity (autocorrelation peak) --")
		fmt.Fprintf(w, "contracts_measured=%d  periodic=%d  periodic_fraction=%.3f\n",
			r.Temporal.NumContracts, r.Temporal.PeriodicContracts, r.Temporal.PeriodicFraction)
	}
	return nil
}

func printDistribution(w io.Writer, d analysis.DistributionSummary) {
	tw := tabwriter.NewWriter(w, 0, 0, 2, ' ', 0)
	fmt.Fprintf(tw, "n\t%d\n", d.Count)
	fmt.Fprintf(tw, "mean\t%.2f\n", d.Mean)
	fmt.Fprintf(tw, "stddev\t%.2f\n", d.StdDev)
	fmt.Fprintf(tw, "min/p50/p90/p99/max\t%.0f / %.0f / %.0f / %.0f / %.0f\n",
		d.Min, d.P50, d.P90, d.P99, d.Max)
	if d.PowerLawAlphaMLE > 0 {
		fmt.Fprintf(tw, "power_law_alpha_hill\t%.2f\n", d.PowerLawAlphaMLE)
	}
	_ = tw.Flush()
}

func printKM(w io.Writer, curves []*analysis.KMResult) error {
	tw := tabwriter.NewWriter(w, 0, 0, 2, ' ', 0)
	fmt.Fprintln(tw, "stratum\tn\tevents\tcensored\tmedian\tS(t=1k)\tS(t=10k)\tS(t=100k)")
	for _, c := range curves {
		median := "NA"
		if !math.IsNaN(c.MedianSurv) {
			median = fmt.Sprintf("%.0f", c.MedianSurv)
		}
		fmt.Fprintf(tw, "%s\t%d\t%d\t%d\t%s\t%.3f\t%.3f\t%.3f\n",
			c.Label, c.N, c.NumEvents, c.NumCensored, median,
			c.SurvAt(1_000), c.SurvAt(10_000), c.SurvAt(100_000))
	}
	return tw.Flush()
}

func printCoxSummary(w io.Writer, r *analysis.CoxResult) error {
	fmt.Fprintf(w, "n=%d  events=%d  loglik=%.2f\n", r.NumObs, r.NumEvents, r.LogLike)
	tw := tabwriter.NewWriter(w, 0, 0, 2, ' ', 0)
	fmt.Fprintln(tw, "predictor\tcoef\tstd_err\tz\tp")
	for i, name := range r.Predictors {
		fmt.Fprintf(tw, "%s\t%+.4f\t%.4f\t%+.2f\t%.4f\n",
			name, r.Coef[i], r.StdErr[i], r.ZScore[i], r.PValue[i])
	}
	return tw.Flush()
}

func printPHTest(w io.Writer, r *analysis.PHTestResult) error {
	tw := tabwriter.NewWriter(w, 0, 0, 2, ' ', 0)
	fmt.Fprintln(tw, "predictor\tp_value\tPH ok @ 0.05")
	for _, name := range r.Predictors {
		p := r.PerCovariatePValue[name]
		mark := "yes"
		if p < 0.05 {
			mark = "NO"
		}
		fmt.Fprintf(tw, "%s\t%.4f\t%s\n", name, p, mark)
	}
	fmt.Fprintf(tw, "global\t%.4f\t%s\n", r.GlobalPValue,
		map[bool]string{true: "yes", false: "NO"}[r.GlobalPValue >= 0.05])
	return tw.Flush()
}

func printCalibrationCurve(w io.Writer, c *analysis.CalibrationCurve) error {
	fmt.Fprintf(w, "kept=%d  dropped=%d  brier=%.4f\n", c.NumKept, c.NumDropped, c.BrierScore)
	tw := tabwriter.NewWriter(w, 0, 0, 2, ' ', 0)
	fmt.Fprintln(tw, "bin\tn\tpredicted\tobserved")
	for i, b := range c.Bins {
		fmt.Fprintf(tw, "%d\t%d\t%.3f\t%.3f\n", i+1, b.N, b.PredictedMean, b.ObservedRate)
	}
	return tw.Flush()
}

func printSimResults(w io.Writer, results []*pruning.SimResult) error {
	tw := tabwriter.NewWriter(w, 0, 0, 2, ' ', 0)
	fmt.Fprintln(tw, "policy\tslots\tobs\tRAM%\thot_hit%\tmisses\tRAM_cost\tmiss_cost\ttotal_cost")
	for _, r := range results {
		fmt.Fprintf(tw, "%s\t%d\t%d\t%.2f\t%.2f\t%d\t%.2f\t%.2f\t%.2f\n",
			r.Policy, r.TotalSlots, r.ObservedIntervals,
			100*r.RAMRatio, 100*r.HotHitCoverage, r.Reactivations,
			r.RAMCost, r.MissPenaltyAgg, r.TotalCost)
	}
	return tw.Flush()
}
