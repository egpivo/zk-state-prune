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
	cmd := &cobra.Command{
		Use:   "fit",
		Short: "Fit a survival model (Phase 1: km only)",
		RunE: func(cmd *cobra.Command, _ []string) error {
			if modelKind != "km" {
				return fmt.Errorf("model %q not supported in Phase 1 (use km)", modelKind)
			}
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

			if stratify == "none" || stratify == "" {
				res, err := fitter.FitKaplanMeier(built.Intervals)
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
			curves, err := analysis.FitKaplanMeierStratified(fitter, built.Intervals, key)
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
		},
	}
	cmd.Flags().StringVar(&dbPath, "db", defaultDBPath, "SQLite DB path")
	cmd.Flags().StringVar(&modelKind, "model", "km", "model type (Phase 1: km; cox in Phase 2)")
	cmd.Flags().StringVar(&stratify, "stratify", "none", "stratify by: none|contract-type|slot-type")
	addWindowFlags(cmd, &startBlock, &endBlock)
	return cmd
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
			return printSimResults(out, results)
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
