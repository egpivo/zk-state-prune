package main

import (
	"context"
	"fmt"
	"log/slog"
	"os"
	"os/signal"
	"syscall"

	"github.com/spf13/cobra"
)

var version = "0.1.0-dev"

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

func newExtractCmd() *cobra.Command {
	var source, output string
	cmd := &cobra.Command{
		Use:   "extract",
		Short: "Extract state diffs into local DB",
		RunE: func(cmd *cobra.Command, _ []string) error {
			return fmt.Errorf("extract: not implemented (source=%s output=%s)", source, output)
		},
	}
	cmd.Flags().StringVar(&source, "source", "mock", "data source: rpc|file|mock")
	cmd.Flags().StringVar(&output, "output", "zksp.db", "output SQLite DB path")
	return cmd
}

func newEDACmd() *cobra.Command {
	var db, out string
	cmd := &cobra.Command{
		Use:   "eda",
		Short: "Run exploratory data analysis",
		RunE: func(cmd *cobra.Command, _ []string) error {
			return fmt.Errorf("eda: not implemented (db=%s output=%s)", db, out)
		},
	}
	cmd.Flags().StringVar(&db, "db", "zksp.db", "SQLite DB path")
	cmd.Flags().StringVar(&out, "output", "reports/eda", "report output directory")
	return cmd
}

func newFitCmd() *cobra.Command {
	var db, model string
	cmd := &cobra.Command{
		Use:   "fit",
		Short: "Fit survival model",
		RunE: func(cmd *cobra.Command, _ []string) error {
			return fmt.Errorf("fit: not implemented (db=%s model=%s)", db, model)
		},
	}
	cmd.Flags().StringVar(&db, "db", "zksp.db", "SQLite DB path")
	cmd.Flags().StringVar(&model, "model", "km", "model type: km|cox")
	return cmd
}

func newSimulateCmd() *cobra.Command {
	var db, policy string
	cmd := &cobra.Command{
		Use:   "simulate",
		Short: "Run pruning simulation",
		RunE: func(cmd *cobra.Command, _ []string) error {
			return fmt.Errorf("simulate: not implemented (db=%s policy=%s)", db, policy)
		},
	}
	cmd.Flags().StringVar(&db, "db", "zksp.db", "SQLite DB path")
	cmd.Flags().StringVar(&policy, "policy", "statistical", "policy: statistical|fixed-30d|fixed-90d|no-prune")
	return cmd
}

func newReportCmd() *cobra.Command {
	var db, out string
	cmd := &cobra.Command{
		Use:   "report",
		Short: "Generate full comparison report",
		RunE: func(cmd *cobra.Command, _ []string) error {
			return fmt.Errorf("report: not implemented (db=%s output=%s)", db, out)
		},
	}
	cmd.Flags().StringVar(&db, "db", "zksp.db", "SQLite DB path")
	cmd.Flags().StringVar(&out, "output", "reports", "report output directory")
	return cmd
}
