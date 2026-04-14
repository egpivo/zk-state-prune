// Command zksp is the CLI entry point for the zk-state-prune toolkit.
//
// Phase 1 wires every subcommand to a real implementation backed by the
// internal/* packages. Inputs are minimal flags (no viper/YAML loading
// yet — that's a Phase 2 nicety); outputs go to stdout as plain text and
// to stderr via slog. JSON output and report-directory layouts will land
// alongside the statistical policy in Phase 2.
package main

import (
	"context"
	"fmt"
	"io"
	"log/slog"
	"math"
	"os"
	"os/signal"
	"sort"
	"syscall"
	"text/tabwriter"

	"github.com/spf13/cobra"

	"github.com/egpivo/zk-state-prune/internal/analysis"
	"github.com/egpivo/zk-state-prune/internal/extractor"
	"github.com/egpivo/zk-state-prune/internal/model"
	"github.com/egpivo/zk-state-prune/internal/pruning"
	"github.com/egpivo/zk-state-prune/internal/storage"
)

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
	root := &cobra.Command{
		Use:           "zksp",
		Short:         "Statistical state lifecycle modeling & pruning for ZK rollups",
		Version:       version,
		SilenceUsage:  true,
		SilenceErrors: true,
	}
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

// ---------------------------------------------------------------- extract

func newExtractCmd() *cobra.Command {
	var (
		source       string
		output       string
		seed         uint64
		numContracts int
		totalBlocks  uint64
	)
	cmd := &cobra.Command{
		Use:   "extract",
		Short: "Populate the analysis DB with state-diff data (Phase 1: mock generator only)",
		RunE: func(cmd *cobra.Command, _ []string) error {
			if source != "mock" {
				return fmt.Errorf("source %q not supported in Phase 1 (use mock)", source)
			}
			ctx := cmd.Context()
			db, err := openDB(ctx, output)
			if err != nil {
				return err
			}
			defer db.Close()

			cfg := extractor.DefaultMockConfig()
			cfg.Seed = seed
			if numContracts > 0 {
				cfg.NumContracts = numContracts
			}
			if totalBlocks > 0 {
				cfg.TotalBlocks = totalBlocks
				// Rescale the default window so it still fits the
				// requested horizon. Pre-window fraction stays at 0.2.
				if cfg.Window.End > totalBlocks || cfg.Window.End == 0 {
					cfg.Window = model.ObservationWindow{Start: totalBlocks / 5, End: totalBlocks}
				}
			}

			slog.Info("extract begin",
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
		},
	}
	cmd.Flags().StringVar(&source, "source", "mock", "data source (Phase 1: mock only)")
	cmd.Flags().StringVar(&output, "output", defaultDBPath, "output SQLite DB path")
	cmd.Flags().Uint64Var(&seed, "seed", 42, "PRNG seed for the mock generator")
	cmd.Flags().IntVar(&numContracts, "num-contracts", 0, "override default contract count (0 = use built-in)")
	cmd.Flags().Uint64Var(&totalBlocks, "total-blocks", 0, "override default block horizon (0 = use built-in)")
	return cmd
}

// -------------------------------------------------------------------- eda

func newEDACmd() *cobra.Command {
	var dbPath string
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
			rep, err := analysis.RunEDA(ctx, db, model.ObservationWindow{Start: startBlock, End: endBlock})
			if err != nil {
				return fmt.Errorf("eda: %w", err)
			}
			return printEDAReport(os.Stdout, rep)
		},
	}
	cmd.Flags().StringVar(&dbPath, "db", defaultDBPath, "SQLite DB path")
	addWindowFlags(cmd, &startBlock, &endBlock)
	return cmd
}

// -------------------------------------------------------------------- fit

func newFitCmd() *cobra.Command {
	var dbPath, modelKind, stratify string
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

			switch modelKind {
			case "km":
				return runKMFit(out, fitter, built.Intervals, stratify)
			case "cox":
				return runCoxFit(out, fitter, built.Intervals, holdoutFrac, splitSeed, tau)
			default:
				return fmt.Errorf("unknown model %q (use km or cox)", modelKind)
			}
		},
	}
	cmd.Flags().StringVar(&dbPath, "db", defaultDBPath, "SQLite DB path")
	cmd.Flags().StringVar(&modelKind, "model", "km", "model type: km|cox")
	cmd.Flags().StringVar(&stratify, "stratify", "none", "(km only) stratify by: none|contract-type|slot-type")
	cmd.Flags().Float64Var(&holdoutFrac, "holdout", 0.3, "(cox only) holdout fraction for calibration split")
	cmd.Flags().Uint64Var(&splitSeed, "split-seed", 1, "(cox only) PRNG seed for the train/holdout split")
	cmd.Flags().Uint64Var(&tau, "tau", 0, "(cox only) calibration horizon in blocks (0 = median training Duration)")
	addWindowFlags(cmd, &startBlock, &endBlock)
	return cmd
}

func runKMFit(out io.Writer, fitter analysis.SurvivalFitter, intervals []model.InterAccessInterval, stratify string) error {
	if stratify == "none" || stratify == "" {
		res, err := fitter.FitKaplanMeier(intervals)
		if err != nil {
			return fmt.Errorf("fit km: %w", err)
		}
		res.Label = "all"
		return printKM(out, []*analysis.KMResult{res})
	}
	var key func(model.InterAccessInterval) string
	switch stratify {
	case "contract-type", "contract":
		key = analysis.StratumByContractType
	case "slot-type", "slot":
		key = analysis.StratumBySlotType
	default:
		return fmt.Errorf("unknown stratify mode %q", stratify)
	}
	curves, err := analysis.FitKaplanMeierStratified(fitter, intervals, key)
	if err != nil {
		return fmt.Errorf("fit km stratified: %w", err)
	}
	labels := make([]string, 0, len(curves))
	for k := range curves {
		labels = append(labels, k)
	}
	sort.Strings(labels)
	ordered := make([]*analysis.KMResult, 0, len(curves))
	for _, l := range labels {
		ordered = append(ordered, curves[l])
	}
	return printKM(out, ordered)
}

// runCoxFit drives the Cox PH path: split → fit → PH check → calibration
// curve (raw) → isotonic recalibration → calibration curve (post). The
// before/after Brier comparison is the headline number for whether the
// recalibration actually helps on this dataset.
func runCoxFit(
	out io.Writer,
	fitter analysis.StatmodelFitter,
	intervals []model.InterAccessInterval,
	holdoutFrac float64,
	splitSeed uint64,
	tauFlag uint64,
) error {
	train, holdout, err := analysis.TrainHoldoutSplitBySlot(intervals, holdoutFrac, splitSeed)
	if err != nil {
		return fmt.Errorf("split: %w", err)
	}
	slog.Info("cox split", "train_intervals", len(train), "holdout_intervals", len(holdout))

	res, err := fitter.FitCoxPH(train, analysis.DefaultCoxPredictors)
	if err != nil {
		return fmt.Errorf("fit cox: %w", err)
	}
	if err := printCoxSummary(out, res); err != nil {
		return err
	}

	ph, err := fitter.CheckPH(res)
	if err != nil {
		return fmt.Errorf("check PH: %w", err)
	}
	fmt.Fprintln(out, "\n-- proportional-hazards check (Schoenfeld) --")
	if err := printPHTest(out, ph); err != nil {
		return err
	}

	tau := float64(tauFlag)
	if tau == 0 {
		tau = medianDuration(train)
	}
	fmt.Fprintf(out, "\n-- calibration @ tau=%.0f blocks --\n", tau)
	rawCurve, err := analysis.CalibrationCurveFromCox(res, holdout, tau, 10)
	if err != nil {
		return fmt.Errorf("calibration curve: %w", err)
	}
	fmt.Fprintln(out, "raw Cox:")
	if err := printCalibrationCurve(out, rawCurve); err != nil {
		return err
	}

	calib, err := fitter.CalibrateAt(res, holdout, tau)
	if err != nil {
		return fmt.Errorf("calibrate: %w", err)
	}
	postCurve, err := calibrationCurveFromCalibrated(calib, holdout, tau, 10)
	if err != nil {
		return fmt.Errorf("post-calibration curve: %w", err)
	}
	fmt.Fprintln(out, "\nisotonic-recalibrated:")
	if err := printCalibrationCurve(out, postCurve); err != nil {
		return err
	}
	fmt.Fprintf(out, "\nBrier: raw=%.4f → calibrated=%.4f (delta=%+.4f)\n",
		rawCurve.BrierScore, postCurve.BrierScore, postCurve.BrierScore-rawCurve.BrierScore)
	return nil
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

// calibrationCurveFromCalibrated runs the same drop/bin policy as
// CalibrationCurveFromCox but uses the wrapper's recalibrated
// PredictAccessProb for the predicted side. It lets the CLI report a
// before/after pair without exposing a second analysis-package entry
// point. Definition kept here (CLI-local) because it's only used for
// diagnostic display, not as a reusable analysis API.
func calibrationCurveFromCalibrated(
	calib *analysis.CalibratedModel,
	holdout []model.InterAccessInterval,
	tau float64,
	nBins int,
) (*analysis.CalibrationCurve, error) {
	if calib == nil || calib.Base == nil {
		return nil, fmt.Errorf("calibrationCurveFromCalibrated: nil calibrated model")
	}
	if nBins <= 0 {
		nBins = 10
	}
	type point struct{ p, y float64 }
	pts := make([]point, 0, len(holdout))
	dropped := 0
	for _, it := range holdout {
		var label float64
		switch {
		case it.IsObserved && it.Duration <= uint64(tau):
			label = 1
		case it.Duration > uint64(tau):
			label = 0
		default:
			dropped++
			continue
		}
		p := calib.PredictAccessProb(map[string]float64{
			analysis.ColAccessCount: float64(it.AccessCount),
			analysis.ColSlotAge:     float64(it.SlotAge),
		}, tau)
		pts = append(pts, point{p, label})
	}
	if len(pts) < nBins {
		return nil, fmt.Errorf("calibrationCurveFromCalibrated: only %d non-ambiguous rows for %d bins", len(pts), nBins)
	}
	sort.Slice(pts, func(i, j int) bool { return pts[i].p < pts[j].p })
	bins := make([]analysis.CalibrationBin, 0, nBins)
	binSize := len(pts) / nBins
	brier := 0.0
	for b := 0; b < nBins; b++ {
		lo := b * binSize
		hi := lo + binSize
		if b == nBins-1 {
			hi = len(pts)
		}
		var sumP, sumY float64
		for j := lo; j < hi; j++ {
			sumP += pts[j].p
			sumY += pts[j].y
			d := pts[j].p - pts[j].y
			brier += d * d
		}
		n := hi - lo
		bins = append(bins, analysis.CalibrationBin{
			PredictedMean: sumP / float64(n),
			ObservedRate:  sumY / float64(n),
			N:             n,
		})
	}
	return &analysis.CalibrationCurve{
		Tau:        tau,
		Bins:       bins,
		BrierScore: brier / float64(len(pts)),
		NumKept:    len(pts),
		NumDropped: dropped,
	}, nil
}

// --------------------------------------------------------------- simulate

func newSimulateCmd() *cobra.Command {
	var dbPath, policyName string
	var startBlock, endBlock uint64
	cmd := &cobra.Command{
		Use:   "simulate",
		Short: "Run a pruning simulation",
		RunE: func(cmd *cobra.Command, _ []string) error {
			policy, err := pruning.PolicyByName(policyName)
			if err != nil {
				return err
			}
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
			res, err := pruning.Run(policy, built.Intervals)
			if err != nil {
				return fmt.Errorf("simulate: %w", err)
			}
			return printSimResults(os.Stdout, []*pruning.SimResult{res})
		},
	}
	cmd.Flags().StringVar(&dbPath, "db", defaultDBPath, "SQLite DB path")
	// Default to a Phase-1 baseline. The statistical policy is Phase-2
	// and PolicyByName returns an error for it today, so defaulting to
	// it would break `zksp simulate` the moment this command is wired up.
	cmd.Flags().StringVar(&policyName, "policy", "fixed-30d", "policy: no-prune|fixed-30d|fixed-90d (statistical: Phase 2)")
	addWindowFlags(cmd, &startBlock, &endBlock)
	return cmd
}

// ----------------------------------------------------------------- report

func newReportCmd() *cobra.Command {
	var dbPath string
	var startBlock, endBlock uint64
	cmd := &cobra.Command{
		Use:   "report",
		Short: "End-to-end EDA + KM + baseline pruning comparison",
		RunE: func(cmd *cobra.Command, _ []string) error {
			ctx := cmd.Context()
			db, err := openDB(ctx, dbPath)
			if err != nil {
				return err
			}
			defer db.Close()
			window := model.ObservationWindow{Start: startBlock, End: endBlock}

			// 1. EDA
			rep, err := analysis.RunEDA(ctx, db, window)
			if err != nil {
				return fmt.Errorf("eda: %w", err)
			}
			out := os.Stdout
			fmt.Fprintln(out, "=== EDA ===")
			if err := printEDAReport(out, rep); err != nil {
				return err
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
			fmt.Fprintln(out, "\n=== Kaplan–Meier ===")
			labels := []string{"all"}
			curves := []*analysis.KMResult{overall}
			catLabels := make([]string, 0, len(byCat))
			for k := range byCat {
				catLabels = append(catLabels, k)
			}
			sort.Strings(catLabels)
			for _, l := range catLabels {
				labels = append(labels, l)
				curves = append(curves, byCat[l])
			}
			if err := printKM(out, curves); err != nil {
				return err
			}

			// 3. Baseline pruning policies
			policies := []pruning.Policy{pruning.NoPrune{}, pruning.Fixed30d, pruning.Fixed90d}
			results, err := pruning.RunAll(policies, built.Intervals)
			if err != nil {
				return fmt.Errorf("run all policies: %w", err)
			}
			fmt.Fprintln(out, "\n=== Pruning baselines ===")
			if err := printSimResults(out, results); err != nil {
				return err
			}

			// 4. Cox PH on a 70/30 split, with PH check and isotonic
			// recalibration. Failures here are non-fatal — the PH
			// check is expected to reject for the realistic mock and
			// we still want the EDA / KM / pruning sections to print.
			fmt.Fprintln(out, "\n=== Cox PH (70/30 split) ===")
			if err := runCoxFit(out, fitter, built.Intervals, 0.3, 1, 0); err != nil {
				fmt.Fprintf(out, "cox section skipped: %v\n", err)
			}
			return nil
		},
	}
	cmd.Flags().StringVar(&dbPath, "db", defaultDBPath, "SQLite DB path")
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
	return tw.Flush()
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
	fmt.Fprintln(tw, "policy\tslots\tobs\tsaved%\treactivations\tfalse_prune%\tfinal_pruned")
	for _, r := range results {
		fmt.Fprintf(tw, "%s\t%d\t%d\t%.2f\t%d\t%.2f\t%d\n",
			r.Policy, r.TotalSlots, r.ObservedIntervals,
			100*r.StorageSavedFrac, r.Reactivations, 100*r.FalsePruneRate,
			r.FinalPrunedSlots)
	}
	return tw.Flush()
}
