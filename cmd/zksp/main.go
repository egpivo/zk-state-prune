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
	"net/http"
	"os"
	"os/signal"
	"sort"
	"syscall"

	"github.com/spf13/cobra"

	"github.com/egpivo/zk-state-prune/internal/analysis"
	"github.com/egpivo/zk-state-prune/internal/app"
	"github.com/egpivo/zk-state-prune/internal/config"
	"github.com/egpivo/zk-state-prune/internal/domain"
	"github.com/egpivo/zk-state-prune/internal/extractor"
	"github.com/egpivo/zk-state-prune/internal/render"
	"github.com/egpivo/zk-state-prune/internal/sim"
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

// rpcHTTPClientOverride is a test hook that replaces the HTTP client used by
// the RPC extractor. The sandbox this repo runs under disallows binding
// httptest servers, so cmd tests inject a custom RoundTripper instead.
var rpcHTTPClientOverride func() *http.Client

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
func loadIntervals(ctx context.Context, db *storage.DB, w domain.ObservationWindow) (analysis.IntervalBuildResult, error) {
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
// override YAML" with cobra's flag-default-at-construction-time domain.
func costsFromFlags(cmd *cobra.Command, ramUnit, missPen float64) sim.CostParams {
	if !cmd.Flags().Changed("ram-unit-cost") {
		ramUnit = globalCfg.Pruning.Cost.RAMUnitCost
	}
	if !cmd.Flags().Changed("miss-penalty") {
		missPen = globalCfg.Pruning.Cost.MissPenalty
	}
	return sim.CostParams{RAMUnitCost: ramUnit, MissPenalty: missPen}
}

// ---------------------------------------------------------------- extract

func newExtractCmd() *cobra.Command {
	var (
		source           string
		output           string
		seed             uint64
		numContracts     int
		totalBlocks      uint64
		force            bool
		rpcEndpoint      string
		rpcStart         uint64
		rpcEnd           uint64
		strictCategories bool
	)
	cmd := &cobra.Command{
		Use:   "extract",
		Short: "Populate the analysis DB with state-diff data (mock, Transfer-log surrogate, or real state-diff)",
		RunE: func(cmd *cobra.Command, _ []string) error {
			ctx := cmd.Context()
			switch source {
			case "mock":
				return runMockExtract(ctx, output, seed, numContracts, totalBlocks, force)
			case "rpc":
				return runRPCExtract(ctx, output, rpcEndpoint, rpcStart, rpcEnd, force)
			case "statediff":
				return runStateDiffExtract(ctx, output, rpcEndpoint, rpcStart, rpcEnd, force, strictCategories)
			default:
				return fmt.Errorf("source %q not supported (use mock|rpc|statediff)", source)
			}
		},
	}
	cmd.Flags().StringVar(&source, "source", "mock", "data source: mock|rpc|statediff (rpc = Transfer-log surrogate; statediff = full debug_traceBlockByNumber + prestateTracer, requires archive node)")
	cmd.Flags().StringVar(&output, "output", defaultDBPath, "output SQLite DB path")
	cmd.Flags().Uint64Var(&seed, "seed", 42, "(mock) PRNG seed for the generator")
	cmd.Flags().IntVar(&numContracts, "num-contracts", 0, "(mock) override default contract count (0 = use built-in)")
	cmd.Flags().Uint64Var(&totalBlocks, "total-blocks", 0, "(mock) override default block horizon (0 = use built-in)")
	cmd.Flags().BoolVar(&force, "force", false, "overwrite the output DB even if it already has slots/events")
	cmd.Flags().StringVar(&rpcEndpoint, "rpc", extractor.ScrollPublicRPC, "(rpc/statediff) JSON-RPC endpoint URL")
	cmd.Flags().Uint64Var(&rpcStart, "start", 0, "(rpc/statediff) first block to extract")
	cmd.Flags().Uint64Var(&rpcEnd, "end", 0, "(rpc/statediff) last block to extract (inclusive)")
	cmd.Flags().BoolVar(&strictCategories, "strict-categories", false, "(statediff) error out instead of warning when more than 20% of contracts can't be classified — use in CI / scheduled jobs to fail the run on classifier regression")
	return cmd
}

// refuseOverwriteIfPopulated is the shared safety rail for `extract`
// against both the mock and rpc sources. On `--force` the mock source
// relies on its internal db.Reset() to wipe rows; the rpc source
// does not Reset internally (it resumes incrementally), so we do the
// wipe here to honour the --force flag's plain-English promise of
// "overwrite". Without that, --force on rpc would only bypass the
// count check and leave stale rows + the rpc_high_water mark intact,
// which is exactly what the user asked not to happen.
func refuseOverwriteIfPopulated(ctx context.Context, db *storage.DB, path string, force, resetOnForce bool) error {
	slots, err := db.CountSlots(ctx)
	if err != nil {
		return fmt.Errorf("count existing slots: %w", err)
	}
	events, err := db.CountAccessEvents(ctx)
	if err != nil {
		return fmt.Errorf("count existing events: %w", err)
	}
	populated := slots > 0 || events > 0
	if populated && !force {
		return fmt.Errorf("refusing to overwrite %q: %d slots and %d events already present. Use --force to proceed.",
			path, slots, events)
	}
	if populated && force && resetOnForce {
		slog.Info("forcing overwrite of existing DB", "path", path, "slots", slots, "events", events)
		if err := db.Reset(ctx); err != nil {
			return fmt.Errorf("reset db before overwrite: %w", err)
		}
	}
	return nil
}

func runMockExtract(ctx context.Context, output string, seed uint64, numContracts int, totalBlocks uint64, force bool) error {
	db, err := openDB(ctx, output)
	if err != nil {
		return err
	}
	defer db.Close()
	// mock extractor calls db.Reset() internally as part of its
	// idempotency contract, so we don't need to Reset here too.
	if err := refuseOverwriteIfPopulated(ctx, db, output, force, false); err != nil {
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
			cfg.Window = domain.ObservationWindow{Start: totalBlocks / 5, End: totalBlocks}
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
	if err := extractor.WriteCapability(ctx, db, ex.Capability()); err != nil {
		return fmt.Errorf("stamp capability: %w", err)
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
	// rpc extractor resumes incrementally from the high-water mark
	// rather than wiping on every run, so --force needs to explicitly
	// Reset() the DB here — otherwise stale rows + the old high-water
	// would survive what the user thought was an overwrite. Note
	// that storage.Reset() deliberately leaves schema_meta alone
	// (it's a generic data-table truncation, not an extractor-state
	// wipe), so we also nuke the RPC extractor's high-water key here
	// or a forced re-run would still resume past the previous mark
	// and silently skip part of the requested range.
	if err := refuseOverwriteIfPopulated(ctx, db, output, force, true); err != nil {
		return err
	}
	if force {
		if err := extractor.ClearRPCState(ctx, db); err != nil {
			return err
		}
	}
	cfg := extractor.DefaultRPCConfig()
	cfg.Endpoint = endpoint
	cfg.Start = start
	cfg.End = end
	if rpcHTTPClientOverride != nil {
		cfg.HTTPClient = rpcHTTPClientOverride()
	}
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
	if err := extractor.WriteCapability(ctx, db, ex.Capability()); err != nil {
		return fmt.Errorf("stamp capability: %w", err)
	}
	d := ex.LastDiagnostics()
	slog.Info("extract complete",
		"blocks_fetched", d.BlocksFetched,
		"receipts_fetched", d.ReceiptsFetched,
		"logs_seen", d.LogsSeen,
		"transfer_logs", d.TransferLogs,
		"contracts_created", d.ContractsCreated,
		"slots_created", d.SlotsCreated,
		"events_attempted", d.EventsAttempted,
		"events_persisted", d.EventsPersisted)
	return nil
}

// runStateDiffExtract drives the real state-diff extractor
// (debug_traceBlockByNumber + prestateTracer). Same `--force`
// + high-water + ClearRPCState discipline as runRPCExtract because
// they share the schema_meta key — switching --source between rpc
// and statediff on the same DB requires --force.
//
// strictCategories propagates to RPCConfig.StrictCategories: when
// true, the extractor fails the whole run if classification
// fall-through to ContractOther exceeds the configured ratio,
// instead of just emitting a slog.Warn. Use in CI / scheduled
// jobs; leave false for dev iteration.
func runStateDiffExtract(ctx context.Context, output, endpoint string, start, end uint64, force, strictCategories bool) error {
	if end == 0 || end < start {
		return fmt.Errorf("statediff extract: --end must be > 0 and >= --start (got start=%d end=%d)", start, end)
	}
	db, err := openDB(ctx, output)
	if err != nil {
		return err
	}
	defer db.Close()
	if err := refuseOverwriteIfPopulated(ctx, db, output, force, true); err != nil {
		return err
	}
	if force {
		if err := extractor.ClearRPCState(ctx, db); err != nil {
			return err
		}
	}
	cfg := extractor.DefaultRPCConfig()
	cfg.Endpoint = endpoint
	cfg.Start = start
	cfg.End = end
	cfg.StrictCategories = strictCategories
	if rpcHTTPClientOverride != nil {
		cfg.HTTPClient = rpcHTTPClientOverride()
	}
	slog.Info("extract begin",
		"source", "statediff",
		"endpoint", cfg.Endpoint,
		"start", cfg.Start,
		"end", cfg.End,
		"strict_categories", cfg.StrictCategories,
		"output", output)
	ex, err := extractor.NewStateDiffExtractor(cfg)
	if err != nil {
		return err
	}
	if err := ex.Extract(ctx, db); err != nil {
		return fmt.Errorf("extract: %w", err)
	}
	if err := extractor.WriteCapability(ctx, db, ex.Capability()); err != nil {
		return fmt.Errorf("stamp capability: %w", err)
	}
	d := ex.LastDiagnostics()
	slog.Info("extract complete",
		"blocks_fetched", d.BlocksFetched,
		"storage_touches", d.StorageTouches,
		"storage_reads", d.StorageReads,
		"storage_writes", d.StorageWrites,
		"contracts_created", d.ContractsCreated,
		"slots_created", d.SlotsCreated,
		"other_category_contracts", d.OtherCategoryContracts,
		"events_attempted", d.EventsAttempted,
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
			rep, err := analysis.RunEDAFull(ctx, db, domain.ObservationWindow{Start: startBlock, End: endBlock})
			if err != nil {
				return fmt.Errorf("eda: %w", err)
			}
			if outputFormat(format).isJSON() {
				return emitJSON(cmd.OutOrStdout(), rep)
			}
			return render.EDA(cmd.OutOrStdout(), rep)
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
			window := domain.ObservationWindow{Start: startBlock, End: endBlock}
			built, err := loadIntervals(ctx, db, window)
			if err != nil {
				return err
			}

			fitter := analysis.NewStatmodelFitter()
			out := cmd.OutOrStdout()
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

func runKMFit(out io.Writer, fitter analysis.SurvivalFitter, intervals []domain.InterAccessInterval, stratify string, format outputFormat) error {
	var curves []*analysis.KMResult
	if stratify == "none" || stratify == "" {
		res, err := fitter.FitKaplanMeier(intervals)
		if err != nil {
			return fmt.Errorf("fit km: %w", err)
		}
		res.Label = "all"
		curves = []*analysis.KMResult{res}
	} else {
		var key func(domain.InterAccessInterval) string
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
	return render.KM(out, curves)
}

// (app.CoxFitReport and the BuildCoxFitReport / CoxStrataColumn
// helpers live in internal/app; the text renderer is in
// internal/render. main.go only chooses between JSON and text.)

// runCoxFit is the `fit --model cox` handler. Thin adapter over
// buildCoxFitReport that chooses the output path based on --format.
func runCoxFit(
	out io.Writer,
	fitter analysis.StatmodelFitter,
	intervals []domain.InterAccessInterval,
	holdoutFrac float64,
	splitSeed uint64,
	tauFlag uint64,
	format outputFormat,
	savePath string,
	stratify string,
) error {
	r, calib, err := app.BuildCoxFitReport(fitter, intervals, holdoutFrac, splitSeed, tauFlag, stratify)
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
	render.CoxFit(out, r)
	return nil
}

// (buildStatisticalPolicy / statisticalPolicyFromCalibrated /
// medianDuration / calibrationCurveFromCalibrated live in
// internal/app. The CLI call sites below invoke them via app.*.)

// --------------------------------------------------------------- simulate

// simulateOutput is the JSON shape for `zksp simulate --format json`.
// Wraps the results in a small envelope so the data-source capability
// travels next to the numbers — a reader of a detached JSON file can
// tell at a glance whether it came from a Transfer-log surrogate or a
// full state-diff source.
type simulateOutput struct {
	DataSource *extractor.Capability `json:"data_source,omitempty"`
	Results    []*sim.Result         `json:"results"`
}

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
			built, err := loadIntervals(ctx, db, domain.ObservationWindow{Start: startBlock, End: endBlock})
			if err != nil {
				return err
			}
			costs := costsFromFlags(cmd, ramUnit, missPen)

			var deps sim.PolicyDeps
			if policyName == "statistical" || policyName == "statistical-robust" {
				useRobust := robust || policyName == "statistical-robust"
				var p *sim.StatisticalPolicy
				if modelPath != "" {
					// Load a previously persisted model and wrap it
					// in a statistical policy — skipping the full fit
					// pipeline. This is the "fit once, simulate many
					// times" deployment flow.
					loaded, err := analysis.LoadModelFile(modelPath)
					if err != nil {
						return fmt.Errorf("load model: %w", err)
					}
					p, err = app.StatisticalPolicyFromCalibrated(loaded, costs, useRobust)
					if err != nil {
						return fmt.Errorf("statistical from loaded model: %w", err)
					}
					slog.Info("statistical policy loaded from file",
						"model_path", modelPath, "tau", p.Tau())
				} else {
					p, err = app.BuildStatisticalPolicy(built.Intervals, holdoutFrac, splitSeed, tau, costs, useRobust)
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
			policy, err := sim.PolicyByName(policyName, deps)
			if err != nil {
				return err
			}

			res, err := sim.Run(policy, built.Intervals, costs)
			if err != nil {
				return fmt.Errorf("simulate: %w", err)
			}
			capStamp, capOK, err := extractor.ReadCapability(ctx, db)
			if err != nil {
				return fmt.Errorf("read capability: %w", err)
			}
			var dataSource *extractor.Capability
			if capOK {
				c := capStamp
				dataSource = &c
			}
			out := cmd.OutOrStdout()
			if outputFormat(format).isJSON() {
				return emitJSON(out, simulateOutput{
					DataSource: dataSource,
					Results:    []*sim.Result{res},
				})
			}
			renderDataSourceHeader(out, dataSource)
			return render.SimResults(out, []*sim.Result{res})
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
	// DataSource stamps the capability of the most recent Extract call
	// that populated this DB. Lets a downstream consumer distinguish
	// "Brier computed on Transfer-only surrogate" from "Brier computed
	// on full state diff" without re-running the pipeline.
	DataSource *extractor.Capability `json:"data_source,omitempty"`
	EDA        *analysis.EDAReport   `json:"eda"`
	KMCurves   []*analysis.KMResult  `json:"km_curves"`
	Tiering    []*sim.Result         `json:"tiering"`
	Cox        *app.CoxFitReport     `json:"cox,omitempty"`
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
			window := domain.ObservationWindow{Start: startBlock, End: endBlock}
			out := cmd.OutOrStdout()
			fmtMode := outputFormat(format)

			// 0. Data-source stamp for self-documenting output.
			capStamp, capOK, err := extractor.ReadCapability(ctx, db)
			if err != nil {
				return fmt.Errorf("read capability: %w", err)
			}

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
			curves, err := buildReportKMCurves(fitter, built.Intervals)
			if err != nil {
				return err
			}

			// 3. Baseline + statistical + statistical-robust, all scored on
			// shared CostParams so TotalCost is directly comparable.
			costs := costsFromFlags(cmd, ramUnit, missPen)
			policies, err := buildReportPolicies(built.Intervals, costs, modelPath)
			if err != nil {
				return err
			}
			results, err := sim.RunAll(policies, built.Intervals, costs)
			if err != nil {
				return fmt.Errorf("run all policies: %w", err)
			}

			// 4. Cox PH diagnostics. Failure is non-fatal because
			// PH check is expected to reject on realistic traces. The
			// CalibratedModel returned alongside is currently unused
			// here (report doesn't --save); keep it bound to _ so the
			// compiler doesn't prune the return arity by accident.
			coxReport, _, coxErr := app.BuildCoxFitReport(fitter, built.Intervals, 0.3, 1, 0, "")

			var dataSource *extractor.Capability
			if capOK {
				c := capStamp
				dataSource = &c
			}
			if fmtMode.isJSON() {
				return emitJSON(out, fullReport{
					DataSource: dataSource,
					EDA:        rep,
					KMCurves:   curves,
					Tiering:    results,
					Cox:        coxReport,
				})
			}
			renderDataSourceHeader(out, dataSource)
			return renderReportText(out, rep, curves, results, coxReport, coxErr)
		},
	}
	cmd.Flags().StringVar(&dbPath, "db", defaultDBPath, "SQLite DB path")
	cmd.Flags().StringVar(&modelPath, "model", "", "load a persisted statistical model from this path instead of refitting")
	addCostFlags(cmd, &ramUnit, &missPen)
	addFormatFlag(cmd, &format)
	addWindowFlags(cmd, &startBlock, &endBlock)
	return cmd
}

// buildReportKMCurves fits the overall Kaplan–Meier curve plus one
// per contract type and returns them in a stable (overall first,
// then strata alphabetically) order for the KM section of `report`.
func buildReportKMCurves(
	fitter analysis.StatmodelFitter,
	intervals []domain.InterAccessInterval,
) ([]*analysis.KMResult, error) {
	overall, err := fitter.FitKaplanMeier(intervals)
	if err != nil {
		return nil, fmt.Errorf("fit km: %w", err)
	}
	overall.Label = "all"
	byCat, err := analysis.FitKaplanMeierStratified(fitter, intervals, analysis.StratumByContractType)
	if err != nil {
		return nil, fmt.Errorf("fit km stratified: %w", err)
	}
	curves := []*analysis.KMResult{overall}
	labels := make([]string, 0, len(byCat))
	for k := range byCat {
		labels = append(labels, k)
	}
	sort.Strings(labels)
	for _, l := range labels {
		curves = append(curves, byCat[l])
	}
	return curves, nil
}

// buildReportPolicies assembles the policy slate `report` scores on
// common CostParams: NoPrune + FixedIdle baselines plus the two
// statistical variants (point + robust). When modelPath is set both
// variants reuse the same persisted model and a load failure is
// fatal; otherwise we refit on the fly and treat per-variant fit
// failures as a warning so the report still renders the baselines
// and the other statistical variant.
func buildReportPolicies(
	intervals []domain.InterAccessInterval,
	costs sim.CostParams,
	modelPath string,
) ([]sim.Policy, error) {
	policies := []sim.Policy{sim.NoPrune{}, sim.Fixed30d, sim.Fixed90d}
	build := func(robust bool) (*sim.StatisticalPolicy, error) {
		if modelPath != "" {
			loaded, err := analysis.LoadModelFile(modelPath)
			if err != nil {
				return nil, fmt.Errorf("load model: %w", err)
			}
			return app.StatisticalPolicyFromCalibrated(loaded, costs, robust)
		}
		return app.BuildStatisticalPolicy(intervals, 0.3, 1, 0, costs, robust)
	}
	if p, err := build(false); err != nil {
		if modelPath != "" {
			return nil, fmt.Errorf("statistical policy from --model: %w", err)
		}
		slog.Warn("statistical policy unavailable for report", "err", err)
	} else {
		policies = append(policies, p)
	}
	if p, err := build(true); err != nil {
		if modelPath != "" {
			return nil, fmt.Errorf("statistical-robust policy from --model: %w", err)
		}
		slog.Warn("statistical-robust policy unavailable for report", "err", err)
	} else {
		policies = append(policies, p)
	}
	return policies, nil
}

// renderDataSourceHeader prints a one-line banner identifying which
// extractor produced the DB rows the report is about to summarise.
// When the DB has never been extracted against (fresh file) the
// banner says "unknown data source" rather than silently printing
// nothing — absence of the stamp is itself information the reader
// should see.
func renderDataSourceHeader(out io.Writer, cap *extractor.Capability) {
	if cap == nil {
		fmt.Fprintln(out, "# data source: unknown (no Extract has populated this DB)")
		return
	}
	fmt.Fprintf(out, "# data source: %s  reads=%v  non-Transfer-writes=%v  slot_id=%s\n",
		cap.Source, cap.ObservesReads, cap.ObservesNonTransferWrite, cap.SlotIDForm)
}

// renderReportText prints the four section blocks of the text-mode
// `report` output in order: EDA → KM → tiering → Cox. The Cox
// section is replaced with a one-line "skipped" message when the
// on-the-fly fit failed upstream, so the rest of the report still
// lands even on degenerate inputs.
func renderReportText(
	out io.Writer,
	eda *analysis.EDAReport,
	curves []*analysis.KMResult,
	results []*sim.Result,
	coxReport *app.CoxFitReport,
	coxErr error,
) error {
	fmt.Fprintln(out, "=== EDA ===")
	if err := render.EDA(out, eda); err != nil {
		return err
	}
	fmt.Fprintln(out, "\n=== Kaplan–Meier ===")
	if err := render.KM(out, curves); err != nil {
		return err
	}
	fmt.Fprintln(out, "\n=== Tiering policies ===")
	if err := render.SimResults(out, results); err != nil {
		return err
	}
	fmt.Fprintln(out, "\n=== Cox PH (70/30 split) ===")
	if coxErr != nil {
		fmt.Fprintf(out, "cox section skipped: %v\n", coxErr)
		return nil
	}
	render.CoxFit(out, coxReport)
	return nil
}
