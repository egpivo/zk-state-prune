// Package render contains the human-readable (`--format text`)
// formatters that used to live in cmd/zksp/main.go. Moving them here
// leaves main.go as cobra wiring plus an `if format.isJSON() { emit }
// else { render }` switch, and makes every formatter independently
// testable against a bytes.Buffer.
//
// The helpers deliberately write to an io.Writer rather than return
// a string: an EDA report or a simulation table is cheap to stream
// but expensive to stringify and then re-copy when piped through
// tabwriter. Callers pass os.Stdout in prod and *bytes.Buffer in tests.
package render

import (
	"fmt"
	"io"
	"math"
	"sort"
	"text/tabwriter"

	"github.com/egpivo/zk-state-prune/internal/analysis"
	"github.com/egpivo/zk-state-prune/internal/app"
	"github.com/egpivo/zk-state-prune/internal/domain"
	"github.com/egpivo/zk-state-prune/internal/sim"
)

// EDA renders an EDAReport as the text block `zksp eda` prints. Spatial
// and Temporal sub-sections are omitted if the report didn't compute
// them (omitempty semantics mirror the JSON path).
func EDA(w io.Writer, r *analysis.EDAReport) error {
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
	distribution(w, r.Frequency)
	fmt.Fprintln(w, "\n-- inter-access time (observed only) --")
	distribution(w, r.InterAccessTime)
	fmt.Fprintln(w, "\n-- by contract type --")
	tw = tabwriter.NewWriter(w, 0, 0, 2, ' ', 0)
	fmt.Fprintln(tw, "category\tslots\tintervals\tcensored%\tIAT_p50\tIAT_p99")
	cats := make([]string, 0, len(r.ByContractType))
	idx := make(map[string]domain.ContractCategory)
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

// CoxFit renders a CoxFitReport as the text sequence `fit --model
// cox` and the Cox section of `report` print.
func CoxFit(w io.Writer, r *app.CoxFitReport) {
	if r == nil {
		return
	}
	_ = coxSummary(w, r.Cox)
	fmt.Fprintln(w, "\n-- proportional-hazards check (Schoenfeld) --")
	_ = phTest(w, r.PH)
	fmt.Fprintf(w, "\n-- calibration @ tau=%.0f blocks --\n", r.Tau)
	fmt.Fprintln(w, "raw Cox:")
	_ = calibrationCurve(w, r.RawCurve)
	fmt.Fprintln(w, "\nisotonic-recalibrated:")
	_ = calibrationCurve(w, r.CalibratedCurve)
	fmt.Fprintf(w, "\nBrier: raw=%.4f → calibrated=%.4f (delta=%+.4f)\n",
		r.RawCurve.BrierScore, r.CalibratedCurve.BrierScore, r.BrierDelta)
}

// KM renders one row per Kaplan–Meier curve. Used by both `fit
// --model km` and the KM section of `report`.
func KM(w io.Writer, curves []*analysis.KMResult) error {
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

// SimResults renders the policy comparison table emitted by
// `simulate` and the bottom half of `report`.
func SimResults(w io.Writer, results []*sim.Result) error {
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

// ---- internal helpers -------------------------------------------------

func distribution(w io.Writer, d analysis.DistributionSummary) {
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

func coxSummary(w io.Writer, r *analysis.CoxResult) error {
	fmt.Fprintf(w, "n=%d  events=%d  loglik=%.2f\n", r.NumObs, r.NumEvents, r.LogLike)
	tw := tabwriter.NewWriter(w, 0, 0, 2, ' ', 0)
	fmt.Fprintln(tw, "predictor\tcoef\tstd_err\tz\tp")
	for i, name := range r.Predictors {
		fmt.Fprintf(tw, "%s\t%+.4f\t%.4f\t%+.2f\t%.4f\n",
			name, r.Coef[i], r.StdErr[i], r.ZScore[i], r.PValue[i])
	}
	return tw.Flush()
}

func phTest(w io.Writer, r *analysis.PHTestResult) error {
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

func calibrationCurve(w io.Writer, c *analysis.CalibrationCurve) error {
	fmt.Fprintf(w, "kept=%d  dropped=%d  brier=%.4f\n", c.NumKept, c.NumDropped, c.BrierScore)
	tw := tabwriter.NewWriter(w, 0, 0, 2, ' ', 0)
	fmt.Fprintln(tw, "bin\tn\tpredicted\tobserved")
	for i, b := range c.Bins {
		fmt.Fprintf(tw, "%d\t%d\t%.3f\t%.3f\n", i+1, b.N, b.PredictedMean, b.ObservedRate)
	}
	return tw.Flush()
}
