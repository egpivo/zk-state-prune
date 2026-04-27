#!/usr/bin/env python3
"""backtest — rolling train→fit→simulate evaluation for zksp policies.

Goal: product-claim QA. Unlike qa_viz.py (Layer-0 sanity checks), this
script runs a time-ordered backtest:

  1) Fit Cox(+isotonic) on a train window
  2) Build a statistical policy from that fitted model
  3) Evaluate multiple policies on a *future* test window
  4) Tune a fixed-idle baseline on the train window to match the
     statistical policy's RAMRatio budget (to avoid apples-to-oranges)

Rolling windows:
  train = [t, t+train_span)
  test  = [t+train_span, t+train_span+test_span)
  advance by --step blocks and repeat until the end boundary.

Inputs:
  - a populated SQLite DB (e.g. testdata/runs/scroll_100k/scroll.db)
  - a zksp binary (default: bin/zksp)

Outputs (under --out-dir):
  - backtest_summary.json  machine-readable fold-by-fold results
  - backtest_report.html   one-page human-readable report

Notes / caveats:
  - If your DB was populated via `--source rpc` (Transfer-log surrogate),
    then the "realized misses" are only a lower bound — reads and
    non-Transfer writes are invisible. The report surfaces the stamped
    data_source capability prominently.
  - Baseline tuning is done on the train window only. The matched fixed-N
    is then evaluated on the test window without re-tuning.

No external dependencies: stdlib only.
"""

from __future__ import annotations

import argparse
import html
import json
import math
import sqlite3
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

# Risk QA needs ≥ this many fold samples before percentiles / CVaR are
# considered statistically meaningful. Below this, we still compute the
# numbers but mark them informational so a downstream reader can't
# accidentally cite a fake-precise p99 over n=2.
RISK_MIN_N = 5

# A fold is treated as "in distribution" when each train→test ratio of
# the simple workload summary stats stays inside this band. Picked to
# admit normal week-over-week variation but flag obvious regime shifts;
# tunable via --drift-ratio.
DEFAULT_DRIFT_RATIO = 1.5


def rel_to_cwd(p: Path) -> str:
    try:
        return p.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return p.resolve().as_posix()


def run(cmd: list[str], *, dry_run: bool = False) -> str:
    """Run a zksp subcommand and return its stdout.

    zksp writes its `--format json` payload to stdout, but slog (and
    Cox optimiser progress) goes to stderr. Merging them — what the
    earlier draft did — corrupted the JSON capture on every fold.
    Keep the streams separate; only stdout is parseable.
    """
    if dry_run:
        print("::: dry-run:", " ".join(cmd))
        return ""
    res = subprocess.run(
        cmd, check=False,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    if res.returncode != 0:
        raise RuntimeError(
            f"command failed ({res.returncode}): {' '.join(cmd)}\n"
            f"stderr:\n{res.stderr.strip()}\n"
            f"stdout (first 500 chars):\n{res.stdout[:500]}"
        )
    return res.stdout


def load_simulate_json(output: str) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """Parse `zksp simulate --format json` output.

    Returns (data_source, result_cell).
    """
    data = json.loads(output)
    ds = data.get("data_source")
    results = data.get("results") or []
    if not results:
        raise ValueError("simulate JSON missing results[]")
    if len(results) != 1:
        raise ValueError(f"expected 1 simulate result, got {len(results)}")
    return ds, results[0]


def load_fit_json(output: str) -> dict[str, Any]:
    """Parse `zksp fit --model cox --format json` output (CoxFitReport)."""
    return json.loads(output)


def find_fixed_n_for_ramratio(
    *,
    zksp: Path,
    db: Path,
    window_start: int,
    window_end: int,
    target: float,
    costs: tuple[float, float],
    max_n: int,
    max_steps: int,
    dry_run: bool,
) -> dict[str, Any]:
    """Find fixed-N whose RAMRatio on the given window is closest to target.

    We rely on (approximately) monotonic behaviour: increasing N (idle
    threshold) keeps slots hot longer, increasing RAMRatio.
    """
    ram_unit, miss_pen = costs

    def eval_n(n: int) -> dict[str, Any]:
        out = run(
            [
                str(zksp),
                "simulate",
                "--db",
                str(db),
                "--policy",
                f"fixed-{n}",
                "--window-start",
                str(window_start),
                "--window-end",
                str(window_end),
                "--ram-unit-cost",
                str(ram_unit),
                "--miss-penalty",
                str(miss_pen),
                "--format",
                "json",
            ],
            dry_run=dry_run,
        )
        if dry_run:
            return {"n": n, "ramratio": None, "cell": None}
        _, cell = load_simulate_json(out)
        return {"n": n, "ramratio": float(cell.get("RAMRatio") or 0), "cell": cell}

    if target <= 0:
        r = eval_n(1)
        r["note"] = "target RAMRatio ≤ 0; returning fixed-1"
        return r
    if target >= 1:
        r = eval_n(max_n)
        r["note"] = "target RAMRatio ≥ 1; returning max_n"
        return r

    # Expand upper bound until we cross target or hit max_n.
    lo_n = 1
    lo = eval_n(lo_n)
    hi_n = 1
    hi = lo
    steps = 0
    while steps < max_steps and hi_n < max_n and (hi.get("ramratio") or 0) < target:
        hi_n = min(max_n, hi_n * 2)
        hi = eval_n(hi_n)
        steps += 1

    # If even max_n doesn't reach the target, return max_n.
    if (hi.get("ramratio") or 0) < target:
        hi["note"] = "could not reach target RAMRatio within max_n; returning max_n"
        return hi

    # Binary search for closest.
    best = hi
    best_err = abs((hi.get("ramratio") or 0) - target)
    lo_n = 1
    lo = eval_n(lo_n)
    best2_err = abs((lo.get("ramratio") or 0) - target)
    if best2_err < best_err:
        best, best_err = lo, best2_err

    left, right = lo_n, hi_n
    while steps < max_steps and right - left > 1:
        mid = (left + right) // 2
        m = eval_n(mid)
        steps += 1
        err = abs((m.get("ramratio") or 0) - target)
        if err < best_err:
            best, best_err = m, err
        if (m.get("ramratio") or 0) < target:
            left = mid
        else:
            right = mid

    best["target_ramratio"] = target
    best["abs_error"] = best_err
    best["search_steps"] = steps
    best["max_n"] = max_n
    return best


def mean(xs: list[float]) -> float | None:
    if not xs:
        return None
    return sum(xs) / len(xs)


def stdev(xs: list[float]) -> float | None:
    """Sample stdev (n-1 denominator). Returns None for n < 2."""
    if len(xs) < 2:
        return None
    m = sum(xs) / len(xs)
    var = sum((x - m) ** 2 for x in xs) / (len(xs) - 1)
    return math.sqrt(var)


def percentile(xs: list[float], q: float) -> float | None:
    """Linear-interpolated percentile (NumPy default). q in [0, 100].
    Returns None for empty input. Uses no external deps."""
    if not xs:
        return None
    s = sorted(xs)
    if len(s) == 1:
        return float(s[0])
    pos = (q / 100.0) * (len(s) - 1)
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return float(s[lo])
    return float(s[lo] + (s[hi] - s[lo]) * (pos - lo))


def cvar(xs: list[float], q: float = 95.0) -> float | None:
    """CVaR at q% — mean of the worst (100−q)% of values. For q=95
    this is the mean of the top 5% (the "tail loss"). Falls back to
    max(xs) when n is small enough that ⌈n·(1−q/100)⌉ = 1."""
    if not xs:
        return None
    s = sorted(xs)
    k = max(1, math.ceil(len(s) * (1.0 - q / 100.0)))
    tail = s[-k:]
    return sum(tail) / len(tail)


def tail_summary(xs: list[float], min_n: int = RISK_MIN_N) -> dict[str, Any]:
    """p50/p90/p95 + CVaR95 + max + n. The `informational` flag is set
    when n < min_n so downstream readers know the percentiles are noisy
    by construction (e.g. n=2 ⇒ p95 is just max)."""
    return {
        "n": len(xs),
        "informational": len(xs) < min_n,
        "min": None if not xs else float(min(xs)),
        "p50": percentile(xs, 50),
        "p90": percentile(xs, 90),
        "p95": percentile(xs, 95),
        "max": None if not xs else float(max(xs)),
        "cvar95": cvar(xs, 95),
        "mean": mean(xs),
        "stdev": stdev(xs),
    }


def workload_summary(db: Path, start: int, end: int) -> dict[str, Any]:
    """Cheap per-window workload summary used by the drift detector.
    Pulls four stats directly from the DB (no Go side-channel needed):
      - n_slots:              distinct slot_ids active in [start, end)
      - n_events:             total accesses in [start, end)
      - access_count_p50/p90: per-slot access-count quantiles
      - mean_iat:             mean inter-arrival between consecutive
                              accesses to the same slot, in blocks

    A large train→test ratio in any of these is the signal we use to
    mark a fold OOD.
    """
    # Plain path connect — URI mode requires `file:///abs/path` and is
    # easy to get wrong for negligible benefit. We never write.
    conn = sqlite3.connect(str(db.resolve()))
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT COUNT(DISTINCT slot_id), COUNT(*) FROM access_events "
            "WHERE block_number >= ? AND block_number < ?",
            (start, end),
        )
        n_slots, n_events = cur.fetchone()

        cur.execute(
            "SELECT cnt FROM (SELECT slot_id, COUNT(*) AS cnt FROM access_events "
            "WHERE block_number >= ? AND block_number < ? GROUP BY slot_id) "
            "ORDER BY cnt",
            (start, end),
        )
        per_slot_counts = [int(r[0]) for r in cur.fetchall()]

        # IAT: gap between consecutive accesses to the same slot. We
        # pull (slot_id, block_number) sorted and compute diffs in
        # Python — the dataset for one window fits easily and avoids
        # a SQL window-function dependency on older sqlite builds.
        cur.execute(
            "SELECT slot_id, block_number FROM access_events "
            "WHERE block_number >= ? AND block_number < ? "
            "ORDER BY slot_id, block_number",
            (start, end),
        )
        last_block_by_slot: dict[str, int] = {}
        iats: list[int] = []
        for sid, bn in cur:
            prev = last_block_by_slot.get(sid)
            if prev is not None:
                iats.append(int(bn) - int(prev))
            last_block_by_slot[sid] = int(bn)
    finally:
        conn.close()

    return {
        "n_slots": int(n_slots or 0),
        "n_events": int(n_events or 0),
        "access_count_p50": percentile([float(x) for x in per_slot_counts], 50),
        "access_count_p90": percentile([float(x) for x in per_slot_counts], 90),
        "mean_iat": float(mean([float(x) for x in iats]) or 0) if iats else None,
        "n_iat_samples": len(iats),
    }


def _drift_ratios(train: dict[str, Any], test: dict[str, Any]) -> dict[str, float | None]:
    """train/test ratios for each drift-checked stat. None when either
    side is missing/zero — downstream treats None as 'no signal'."""
    out: dict[str, float | None] = {}
    for key in ("n_slots", "n_events", "access_count_p50", "access_count_p90", "mean_iat"):
        a, b = train.get(key), test.get(key)
        if a is None or b is None or a == 0 or b == 0:
            out[key] = None
        else:
            out[key] = float(max(a, b) / min(a, b))  # ≥ 1 by construction
    return out


def _drift_ok(ratios: dict[str, float | None], thr: float) -> bool:
    """A fold is in-distribution when no drift ratio (skipping None
    placeholders) exceeds the threshold."""
    return all((r is None) or (r <= thr) for r in ratios.values())


def _slim_fit_report(fit_json: dict[str, Any]) -> dict[str, Any]:
    """Drop Cox coefficients/baseline-hazard arrays and isotonic bins
    from the per-fold record. The pieces a reader actually wants —
    sample sizes, calibration Brier delta, PH p-values — stay."""
    if not fit_json:
        return {}
    cox = fit_json.get("cox") or {}
    raw = fit_json.get("raw_calibration") or {}
    iso = fit_json.get("isotonic_calibration") or {}
    return {
        "tau": fit_json.get("tau"),
        "train_intervals": fit_json.get("train_intervals"),
        "holdout_intervals": fit_json.get("holdout_intervals"),
        "cox": {k: cox.get(k) for k in ("Predictors", "Coef", "LogLike", "NumObs", "NumEvents")},
        "ph_test": fit_json.get("ph_test"),
        "raw_calibration_brier": raw.get("BrierScore"),
        "isotonic_calibration_brier": iso.get("BrierScore"),
        "brier_delta": fit_json.get("brier_delta"),
    }


def _slim_cell(cell: dict[str, Any]) -> dict[str, Any]:
    """Pick the tiering metrics a backtest reader actually needs.
    Avoids dragging the full Result struct (TotalSlots, ObservedIntervals,
    etc.) through every fold record."""
    if not cell:
        return {}
    keys = ("RAMRatio", "HotHitCoverage", "FalsePruneRate", "RAMCost",
            "MissPenaltyAgg", "TotalCost", "Reactivations", "StorageSavedFrac")
    return {k: cell.get(k) for k in keys if k in cell}


def render_html(summary: dict[str, Any], out_path: Path) -> None:
    def h(x: Any) -> str:
        return html.escape(str(x), quote=True)

    def fnum(x: Any, fmt: str = ".4f") -> str:
        if x is None:
            return "—"
        try:
            return format(float(x), fmt)
        except (TypeError, ValueError):
            return h(x)

    ds_per_fold: list[dict[str, Any]] = summary.get("data_sources_per_fold") or []
    distinct_sources = sorted({(d or {}).get("source", "?") for d in ds_per_fold if d})
    primary_ds = ds_per_fold[0] if ds_per_fold else {}
    warns = []
    if "rpc" in distinct_sources:
        warns.append(
            "Data source includes <code>rpc</code> (Transfer-log surrogate). "
            "Realized misses / coverage ignore reads and non-Transfer writes — "
            "treat numbers as a lower bound for product validation."
        )
    if len(distinct_sources) > 1:
        sources_html = ", ".join(f"<code>{h(s)}</code>" for s in distinct_sources)
        warns.append(
            f"<b>Mixed data sources across folds:</b> {sources_html}. "
            "Aggregate metrics are not directly comparable; inspect per-fold rows."
        )
    if not primary_ds:
        warns.append("No data_source stamp found; treat results as non-auditable.")

    parts = [
        "<!DOCTYPE html>",
        "<html lang='en'><head><meta charset='utf-8'>",
        f"<title>backtest — {h(summary.get('db'))}</title>",
        "<style>",
        "body { font-family: system-ui, sans-serif; max-width: 1200px; margin: 24px auto; padding: 0 16px; color: #222; }",
        "h1 { font-size: 1.4em; margin-bottom: 6px; }",
        "h2 { font-size: 1.1em; margin-top: 26px; border-bottom: 1px solid #ddd; padding-bottom: 4px; }",
        "table { border-collapse: collapse; margin: 10px 0; font-size: 0.92em; }",
        "th, td { padding: 5px 10px; border-bottom: 1px solid #eee; text-align: left; vertical-align: top; }",
        "th { color: #555; font-weight: 600; background: #fafafa; }",
        "code { background: #f6f6f6; padding: 1px 4px; border-radius: 3px; font-size: 0.92em; }",
        ".warn { padding: 10px 12px; border: 1px solid #f3d19a; background: #fff7e6; border-radius: 6px; margin: 10px 0; }",
        ".muted { color: #888; font-size: 0.92em; }",
        ".bad { color: #b91c1c; font-weight: 600; }",
        ".ok  { color: #1f7a1f; font-weight: 600; }",
        "pre { background: #f6f6f6; padding: 8px 10px; border-radius: 4px; overflow-x: auto; font-size: 0.85em; }",
        "</style></head><body>",
        "<h1>Rolling backtest report</h1>",
        f"<p class='muted'>Input DB: <code>{h(summary.get('db'))}</code>. "
        f"Re-run with <code>make qa-backtest MISS_PENALTY=&lt;ℓ&gt;</code>.</p>",
    ]
    risk = summary.get("risk") or {}
    indep = risk.get("independence_caveat")
    if indep:
        warns.append(indep)
    gates = risk.get("gates") or {}
    if gates.get("violations"):
        for v in gates["violations"]:
            warns.append(f"<b class='bad'>release-gate violation:</b> {h(v)}")

    if warns:
        parts.append("<div class='warn'><b>Caveats:</b><ul>")
        for w in warns:
            parts.append(f"<li>{w}</li>")
        parts.append("</ul></div>")

    parts.append("<h2>Config</h2>")
    cfg = summary.get("config") or {}
    parts.append("<table>")
    for k in ("start", "end", "train_span", "test_span", "step", "n_folds",
              "ram_unit_cost", "miss_penalty", "tau", "holdout", "same_ram_eps"):
        if k in cfg:
            parts.append(f"<tr><th>{h(k)}</th><td>{h(cfg[k])}</td></tr>")
    if primary_ds:
        parts.append(
            f"<tr><th>data_source</th><td><pre>{h(json.dumps(primary_ds, indent=2, sort_keys=True))}</pre></td></tr>"
        )
    parts.append("</table>")

    # Aggregate
    parts.append(f"<h2>Aggregate (n_folds = {h(cfg.get('n_folds', '?'))})</h2>")
    agg = summary.get("aggregate") or {}
    parts.append("<table><tr>"
                 "<th>policy</th><th>RAMRatio (mean ± stdev)</th>"
                 "<th>HotHitCov (mean ± stdev)</th>"
                 "<th>FalsePrune (mean ± stdev)</th>"
                 "<th>TotalCost (mean)</th>"
                 "<th>n</th></tr>")
    pols = agg.get("policies") or {}
    for pol in sorted(pols):
        p = pols[pol]
        parts.append(
            "<tr>"
            f"<td><code>{h(pol)}</code></td>"
            f"<td>{fnum(p.get('ramratio_mean'))} ± {fnum(p.get('ramratio_stdev'))}</td>"
            f"<td>{fnum(p.get('hot_hit_coverage_mean'))} ± {fnum(p.get('hot_hit_coverage_stdev'))}</td>"
            f"<td>{fnum(p.get('false_prune_rate_mean'))} ± {fnum(p.get('false_prune_rate_stdev'))}</td>"
            f"<td>{h(p.get('total_cost_mean'))}</td>"
            f"<td>{h(p.get('n'))}</td>"
            "</tr>"
        )
    parts.append("</table>")

    # In-sample / out-of-sample gap (overfit detector)
    gap = summary.get("statistical_train_test_gap") or {}
    if gap:
        parts.append("<h2>Statistical: train → test gap (overfit detector)</h2>")
        parts.append("<p class='muted'>Difference between in-sample (train) "
                     "and out-of-sample (test) metrics for the <code>statistical</code> "
                     "policy. Large drops on test indicate overfitting; "
                     "stable values across folds indicate the policy generalises.</p>")
        parts.append("<table><tr><th>metric</th>"
                     "<th>train mean</th><th>test mean</th><th>Δ (test - train)</th></tr>")
        for m in ("ramratio", "hot_hit_coverage", "false_prune_rate"):
            tr = gap.get(f"{m}_train_mean")
            te = gap.get(f"{m}_test_mean")
            d = gap.get(f"{m}_delta_mean")
            parts.append(
                "<tr>"
                f"<td>{h(m)}</td>"
                f"<td>{fnum(tr)}</td>"
                f"<td>{fnum(te)}</td>"
                f"<td>{fnum(d, '+.4f')}</td>"
                "</tr>"
            )
        parts.append("</table>")

    # ---- Risk QA section -------------------------------------------------
    parts.append("<h2>Risk QA</h2>")
    parts.append(
        "<p class='muted'>Engineering risk indicators (worst-fold regret, "
        "tail / CVaR over folds). Default scope = <code>all_folds</code>; "
        "the <code>in_distribution</code> view excludes folds where the "
        "train/test workload shifted beyond <code>--drift-ratio</code>. "
        "Release gates evaluate against "
        f"<code>{h(gates.get('scope', 'all_folds'))}</code>.</p>"
    )

    def _regret_block(label: str, blk: dict[str, Any]) -> str:
        n = blk.get("n", 0)
        if n == 0:
            return f"<p class='muted'>{h(label)}: no folds.</p>"
        flag = " <span class='muted'>(informational, n &lt; min)</span>" if blk.get("informational") else ""
        return (
            f"<p><b>{h(label)}</b> — n={h(n)}{flag}</p>"
            f"<table>"
            f"<tr><th>min</th><td>{fnum(blk.get('min'), '+,.0f')}</td>"
            f"<th>p50</th><td>{fnum(blk.get('p50'), '+,.0f')}</td>"
            f"<th>p95</th><td>{fnum(blk.get('p95'), '+,.0f')}</td>"
            f"<th>max</th><td>{fnum(blk.get('max'), '+,.0f')}</td>"
            f"<th>CVaR95</th><td>{fnum(blk.get('cvar95'), '+,.0f')}</td>"
            f"</tr></table>"
            f"<p class='muted'>Definition: {h(blk.get('definition', '—'))}</p>"
        )

    risk_all = risk.get("all_folds") or {}
    risk_in = risk.get("in_distribution") or {}
    parts.append("<h3>Worst-fold regret</h3>")
    parts.append(_regret_block(f"all_folds (n={risk_all.get('n_folds')})", risk_all.get("regret") or {}))
    parts.append(_regret_block(
        f"in_distribution (n={risk_in.get('n_folds')}, "
        f"excluded {risk_in.get('n_excluded_drift_folds', 0)} drift folds)",
        risk_in.get("regret") or {},
    ))

    # Tail metrics for stat policy specifically (the most-watched row).
    def _stat_tail_table(scope_label: str, scope: dict[str, Any]) -> str:
        pols = scope.get("policies") or {}
        stat = pols.get("statistical") or {}
        if not stat:
            return f"<p class='muted'>{h(scope_label)}: no <code>statistical</code> rows.</p>"
        rows = ["<tr><th>metric</th><th>p50</th><th>p90</th><th>p95</th><th>max</th><th>CVaR95</th><th>n</th></tr>"]
        for short in ("hot_hit_coverage", "false_prune_rate", "miss_penalty_agg", "reactivations", "total_cost"):
            t = stat.get(short) or {}
            flag = "*" if t.get("informational") else ""
            fmt = ",.0f" if short in ("miss_penalty_agg", "reactivations", "total_cost") else ".4f"
            rows.append(
                "<tr>"
                f"<td>{h(short)}</td>"
                f"<td>{fnum(t.get('p50'), fmt)}</td>"
                f"<td>{fnum(t.get('p90'), fmt)}</td>"
                f"<td>{fnum(t.get('p95'), fmt)}</td>"
                f"<td>{fnum(t.get('max'), fmt)}</td>"
                f"<td>{fnum(t.get('cvar95'), fmt)}</td>"
                f"<td>{h(t.get('n'))}{h(flag)}</td>"
                "</tr>"
            )
        n_excl = scope.get("n_excluded_drift_folds")
        excl_str = f" (excluded {n_excl} drift folds)" if n_excl else ""
        return (
            f"<p><b>statistical policy on test, {h(scope_label)}{excl_str}</b> "
            "<span class='muted'>(* = informational, n &lt; min)</span></p>"
            "<table>" + "".join(rows) + "</table>"
        )

    parts.append("<h3>Tail metrics — <code>statistical</code> policy on test</h3>")
    parts.append(_stat_tail_table("all_folds", risk_all))
    parts.append(_stat_tail_table("in_distribution", risk_in))

    mns = risk.get("matched_n_stability") or {}
    parts.append("<h3>Matched fixed-N stability</h3>")
    if mns.get("values"):
        parts.append(
            "<p>Per-fold matched N: "
            f"<code>{h(mns['values'])}</code></p>"
            f"<p>n_jumps={h(mns.get('n_jumps'))} · "
            f"max |ΔN|={h(mns.get('max_abs_jump'))} · "
            f"p95 |ΔN|={fnum(mns.get('p95_abs_jump'), '.1f')} · "
            f"mean |ΔN|={fnum(mns.get('mean_abs_jump'), '.1f')}</p>"
        )
    else:
        parts.append("<p class='muted'>No matched-N values.</p>")

    # Per-fold table. The matched fixed-N changes across folds (fold 1
    # might pick fixed-540, fold 2 fixed-764), so we render a generic
    # "fixed-matched" column that pulls from each fold's fixed_match
    # rather than treating "fixed-540" as a fixed column name.
    parts.append("<h2>Folds</h2>")
    static_cols = ["no-prune", "statistical"]
    if "statistical-robust" in (summary.get("policy_columns") or []):
        static_cols.append("statistical-robust")

    header = ["<tr><th>fold</th><th>train</th><th>test</th>",
              "<th>matched fixed_N</th><th>target RAMRatio</th><th>|Δ| on train</th>"]
    # Render one "fixed-matched" group, then one group per static policy.
    cols_to_render = ["fixed-matched"] + static_cols
    for pol in cols_to_render:
        header.append(
            f"<th>{h(pol)}<br>RAMRatio</th>"
            f"<th>{h(pol)}<br>HotHitCov</th>"
            f"<th>{h(pol)}<br>FalsePrune</th>"
        )
    header.append("<th>regret</th>")
    header.append("<th>same-RAM</th>")
    header.append("<th>drift</th>")
    header.append("<th>flags</th></tr>")
    parts.append("<table>" + "".join(header))
    for f in summary.get("folds", []):
        matched = f.get("fixed_match") or {}
        test = f.get("test_results") or {}
        flags = f.get("flags") or []
        same_ram_ok = f.get("same_ram_ok")
        same_label = (
            f'<span class="ok">OK</span>' if same_ram_ok is True else
            f'<span class="bad">drift</span>' if same_ram_ok is False else
            "—"
        )
        drift_ok = f.get("drift_ok")
        drift_label = (
            f'<span class="ok">in</span>' if drift_ok is True else
            f'<span class="bad">OOD</span>' if drift_ok is False else
            "—"
        )
        cells_html = []
        for pol in cols_to_render:
            if pol == "fixed-matched":
                cell = test.get(matched.get("policy_name") or "") or {}
            else:
                cell = test.get(pol) or {}
            cells_html.append(
                f"<td>{fnum(cell.get('RAMRatio'))}</td>"
                f"<td>{fnum(cell.get('HotHitCoverage'))}</td>"
                f"<td>{fnum(cell.get('FalsePruneRate'))}</td>"
            )
        regret = f.get("regret")
        regret_str = "—" if regret is None else f"{regret:+,.0f}"
        regret_class = "bad" if regret is not None and regret > 0 else "ok" if regret is not None else "muted"
        parts.append(
            "<tr>"
            f"<td>{h(f.get('fold'))}</td>"
            f"<td>[{h(f.get('train_start'))}, {h(f.get('train_end'))})</td>"
            f"<td>[{h(f.get('test_start'))}, {h(f.get('test_end'))})</td>"
            f"<td><code>{h(matched.get('policy_name'))}</code></td>"
            f"<td>{fnum(matched.get('target_ramratio'))}</td>"
            f"<td>{fnum(matched.get('abs_error'))}</td>"
            + "".join(cells_html)
            + f"<td><span class='{regret_class}'>{h(regret_str)}</span></td>"
            f"<td>{same_label}</td>"
            f"<td>{drift_label}</td>"
            f"<td>{', '.join(h(x) for x in flags) if flags else '—'}</td>"
            "</tr>"
        )
    parts.append("</table>")

    parts.append("</body></html>\n")
    out_path.write_text("\n".join(parts))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", required=True, help="SQLite DB path")
    ap.add_argument("--zksp", default="bin/zksp", help="Path to zksp binary (default: bin/zksp)")
    ap.add_argument("--out-dir", default="testdata/runs/scroll_100k/backtest", help="Output directory")

    ap.add_argument("--start", type=int, required=True, help="Backtest start block (inclusive)")
    ap.add_argument("--end", type=int, required=True, help="Backtest end block (exclusive)")
    ap.add_argument("--train-span", type=int, required=True, help="Train window size in blocks")
    ap.add_argument("--test-span", type=int, required=True, help="Test window size in blocks")
    ap.add_argument("--step", type=int, required=True, help="Rolling step size in blocks")

    ap.add_argument("--ram-unit-cost", type=float, default=1.0, help="RAM unit cost (passed to simulate)")
    ap.add_argument("--miss-penalty", type=float, required=True, help="Miss penalty ℓ (passed to simulate)")
    ap.add_argument("--tau", type=int, default=0, help="Cox calibration horizon in blocks (0 = median training Duration)")
    ap.add_argument("--holdout", type=float, default=0.3, help="Cox holdout fraction for calibration")
    ap.add_argument("--split-seed-base", type=int, default=1, help="Base PRNG seed; fold index is added to it")

    ap.add_argument("--max-fixed-n", type=int, default=0,
                    help="Max N for fixed-N search (0 = min(train_span, 20000))")
    ap.add_argument("--max-search-steps", type=int, default=18, help="Max steps for fixed-N search per fold")
    ap.add_argument("--same-ram-eps", type=float, default=0.005,
                    help="Per-fold flag: |test_fixed_RAMRatio − test_stat_RAMRatio| ≥ eps marks the comparison non-apples-to-apples")

    ap.add_argument("--include-robust", action="store_true", help="Also evaluate statistical-robust on test windows")

    # Risk QA — Tier 1/2 thresholds. All optional; only activate the
    # fail-closed gates when the user explicitly sets a threshold.
    ap.add_argument("--drift-ratio", type=float, default=DEFAULT_DRIFT_RATIO,
                    help=f"Fold is OOD when any train/test workload-stat ratio "
                         f"exceeds this threshold (default {DEFAULT_DRIFT_RATIO})")
    ap.add_argument("--strict-risk-scope", choices=("all_folds", "in_distribution"),
                    default="all_folds",
                    help="Which fold set the --max-* gates evaluate against (default: all_folds — "
                         "deliberately conservative so OOD folds can't hide bad news)")
    ap.add_argument("--max-regret", type=float, default=None,
                    help="(release gate) fail when any fold's TotalCost(stat) − TotalCost(matched_fixed) "
                         "exceeds this absolute value")
    ap.add_argument("--min-coverage", type=float, default=None,
                    help="(release gate) fail when any fold's stat HotHitCoverage drops below this")
    ap.add_argument("--max-fpr", type=float, default=None,
                    help="(release gate) fail when any fold's stat FalsePruneRate exceeds this")

    ap.add_argument("--strict", action="store_true",
                    help="Exit non-zero if any fold has mixed sources, same-RAM drift, or fixed-N search hit the upper bound")
    ap.add_argument("--dry-run", action="store_true", help="Print commands but do not execute")
    args = ap.parse_args(argv)

    db = Path(args.db)
    zksp = Path(args.zksp)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not args.dry_run:
        if not db.exists():
            print(f"backtest: db does not exist: {db}", file=sys.stderr)
            return 2
        if not zksp.exists():
            print(f"backtest: zksp binary not found: {zksp} (run `make build`)", file=sys.stderr)
            return 2

    if args.end <= args.start:
        print("backtest: --end must be > --start", file=sys.stderr)
        return 2
    if args.train_span <= 0 or args.test_span <= 0 or args.step <= 0:
        print("backtest: spans/step must be positive", file=sys.stderr)
        return 2

    costs = (args.ram_unit_cost, args.miss_penalty)
    # The fixed-N search doubles up to max_n. Letting it run to the full
    # train span (60k) on each fold is ~16 extra simulate calls per fold;
    # cap it tighter by default and let the user override when they need to.
    max_fixed_n = args.max_fixed_n if args.max_fixed_n > 0 else min(args.train_span, 20000)

    folds: list[dict[str, Any]] = []
    data_sources_per_fold: list[dict[str, Any] | None] = []

    fold = 0
    t = args.start
    while True:
        train_start = t
        train_end = t + args.train_span
        test_start = train_end
        test_end = test_start + args.test_span
        if test_end > args.end:
            break

        fold += 1
        seed = args.split_seed_base + fold
        fold_dir = out_dir / f"fold_{fold:03d}"
        model_path = fold_dir / "cox.model.json"
        fold_dir.mkdir(parents=True, exist_ok=True)

        print(
            f"::: fold {fold}: train=[{train_start},{train_end}) "
            f"test=[{test_start},{test_end}) seed={seed}",
            flush=True,
        )
        fold_t0 = time.time()

        # 1) Fit Cox on train and save model.
        fit_out = run(
            [
                str(zksp), "fit", "--db", str(db), "--model", "cox",
                "--window-start", str(train_start), "--window-end", str(train_end),
                "--format", "json",
                "--holdout", str(args.holdout),
                "--split-seed", str(seed),
                "--tau", str(args.tau),
                "--save", str(model_path),
            ],
            dry_run=args.dry_run,
        )
        fit_json = load_fit_json(fit_out) if not args.dry_run else {}

        # 2) Simulate statistical on train to define a RAMRatio budget AND
        #    capture the in-sample baseline for the train→test gap report.
        stat_train_out = run(
            [
                str(zksp), "simulate", "--db", str(db),
                "--policy", "statistical", "--model", str(model_path),
                "--window-start", str(train_start), "--window-end", str(train_end),
                "--ram-unit-cost", str(costs[0]), "--miss-penalty", str(costs[1]),
                "--format", "json",
            ],
            dry_run=args.dry_run,
        )
        ds, stat_train_cell = (None, {}) if args.dry_run else load_simulate_json(stat_train_out)
        data_sources_per_fold.append(ds)

        target_ram = float(stat_train_cell.get("RAMRatio") or 0) if not args.dry_run else 0.0

        # 3) Match a fixed-N baseline on the train window to that budget.
        fixed_match = find_fixed_n_for_ramratio(
            zksp=zksp, db=db,
            window_start=train_start, window_end=train_end,
            target=target_ram, costs=costs,
            max_n=max_fixed_n, max_steps=args.max_search_steps,
            dry_run=args.dry_run,
        )
        fixed_n = int(fixed_match.get("n") or 0)
        fixed_policy = f"fixed-{fixed_n}" if fixed_n > 0 else "fixed-?"
        fixed_match["policy_name"] = fixed_policy

        # 4) Evaluate on the test window.
        policies = ["no-prune", fixed_policy, "statistical"]
        if args.include_robust:
            policies.append("statistical-robust")

        test_results: dict[str, Any] = {}
        for pol in policies:
            cmd = [
                str(zksp), "simulate", "--db", str(db),
                "--policy", pol,
                "--window-start", str(test_start), "--window-end", str(test_end),
                "--ram-unit-cost", str(costs[0]), "--miss-penalty", str(costs[1]),
                "--format", "json",
            ]
            if pol.startswith("statistical"):
                cmd += ["--model", str(model_path)]
                if pol == "statistical-robust":
                    cmd += ["--robust"]
            out = run(cmd, dry_run=args.dry_run)
            if args.dry_run:
                continue
            _, cell = load_simulate_json(out)
            test_results[pol] = cell

        # 5) Per-fold integrity flags. Honest reporting only — the script
        #    still produces artifacts; --strict gates a non-zero exit.
        flags: list[str] = []
        same_ram_ok: bool | None = None
        regret = None  # TotalCost(stat) − TotalCost(matched_fixed); negative = stat wins
        drift_summary: dict[str, Any] | None = None
        drift_ok: bool | None = None

        if not args.dry_run:
            test_fixed = test_results.get(fixed_policy) or {}
            test_stat = test_results.get("statistical") or {}
            ram_drift = abs(
                (test_fixed.get("RAMRatio") or 0) - (test_stat.get("RAMRatio") or 0)
            )
            same_ram_ok = ram_drift < args.same_ram_eps
            if not same_ram_ok:
                flags.append(f"same-RAM drift on test ({ram_drift:.4f} ≥ eps {args.same_ram_eps})")
            if fixed_match.get("note"):
                flags.append(fixed_match["note"])
            # Compare this fold's source against fold 1's (we treat fold 1
            # as the reference; mismatches are flagged but don't abort).
            ref_ds = next((d for d in data_sources_per_fold if d), None)
            if ref_ds and ds and ds.get("source") != ref_ds.get("source"):
                flags.append(f"mixed source: {ds.get('source')} vs {ref_ds.get('source')}")

            # Regret on test: stat − matched_fixed. If matched_fixed is
            # missing or stat is missing, leave regret None.
            tc_stat = test_stat.get("TotalCost")
            tc_fixed = test_fixed.get("TotalCost")
            if tc_stat is not None and tc_fixed is not None:
                regret = float(tc_stat) - float(tc_fixed)

            # Drift summary: cheap workload stats on train vs test windows.
            train_w = workload_summary(db, train_start, train_end)
            test_w = workload_summary(db, test_start, test_end)
            ratios = _drift_ratios(train_w, test_w)
            drift_ok = _drift_ok(ratios, args.drift_ratio)
            drift_summary = {"train": train_w, "test": test_w, "ratios": ratios}
            if not drift_ok:
                worst = max(
                    ((k, v) for k, v in ratios.items() if v is not None),
                    key=lambda kv: kv[1],
                    default=(None, None),
                )
                if worst[0]:
                    flags.append(
                        f"OOD: train/test {worst[0]} ratio {worst[1]:.2f} > {args.drift_ratio}"
                    )

        # Wall time is per-run noise, not a canonical artifact: report
        # it on stdout but keep it OUT of the persisted JSON so diffs
        # against a prior run stay clean.
        wall = time.time() - fold_t0
        print(
            f":::   fold {fold} done in {wall:.1f}s "
            f"(matched={fixed_policy}, same_ram_ok={same_ram_ok}, drift_ok={drift_ok})",
            flush=True,
        )

        fold_record = {
            "fold": fold,
            "train_start": train_start,
            "train_end": train_end,
            "test_start": test_start,
            "test_end": test_end,
            "split_seed": seed,
            "model_path": rel_to_cwd(model_path),
            "fit_report": _slim_fit_report(fit_json),
            "stat_train": _slim_cell(stat_train_cell),
            "fixed_match": fixed_match,
            "test_results": {pol: _slim_cell(c) for pol, c in test_results.items()},
            "flags": flags,
            "same_ram_ok": same_ram_ok,
            "regret": regret,
            "drift_ok": drift_ok,
            "drift_summary": drift_summary,
        }
        folds.append(fold_record)

        t += args.step

    if args.dry_run:
        return 0

    # Aggregate per-policy: mean + stdev + n (so reader can see how
    # thin the evidence is — a single mean over 2 folds is misleadingly
    # precise).
    policy_names: set[str] = set()
    for f in folds:
        policy_names |= set((f.get("test_results") or {}).keys())
    policy_names = {p for p in policy_names if isinstance(p, str)}

    def _agg_one(cells: list[dict[str, Any]]) -> dict[str, Any]:
        ramratio = [float(c["RAMRatio"]) for c in cells if "RAMRatio" in c]
        cov = [float(c["HotHitCoverage"]) for c in cells if "HotHitCoverage" in c]
        fpr = [float(c["FalsePruneRate"]) for c in cells if "FalsePruneRate" in c]
        tc = [float(c["TotalCost"]) for c in cells if "TotalCost" in c]
        return {
            "n": len(ramratio),
            "ramratio_mean":  None if mean(ramratio) is None else round(mean(ramratio) or 0, 6),
            "ramratio_stdev": None if stdev(ramratio) is None else round(stdev(ramratio) or 0, 6),
            "hot_hit_coverage_mean":  None if mean(cov) is None else round(mean(cov) or 0, 6),
            "hot_hit_coverage_stdev": None if stdev(cov) is None else round(stdev(cov) or 0, 6),
            "false_prune_rate_mean":  None if mean(fpr) is None else round(mean(fpr) or 0, 6),
            "false_prune_rate_stdev": None if stdev(fpr) is None else round(stdev(fpr) or 0, 6),
            "total_cost_mean": None if mean(tc) is None else int(round(mean(tc) or 0)),
        }

    agg_pols: dict[str, dict[str, Any]] = {}
    for pol in sorted(policy_names):
        cells = [(f.get("test_results") or {}).get(pol) or {} for f in folds]
        agg_pols[pol] = _agg_one([c for c in cells if c])

    # Virtual "fixed-matched" row: pull each fold's matched fixed-N
    # cell (the actual N differs across folds, but the *role* — matched
    # baseline at the statistical RAM budget — is the same).
    matched_cells = []
    matched_names = []
    for f in folds:
        name = (f.get("fixed_match") or {}).get("policy_name")
        if not name:
            continue
        cell = (f.get("test_results") or {}).get(name) or {}
        if cell:
            matched_cells.append(cell)
            matched_names.append(name)
    if matched_cells:
        agg_pols["fixed-matched"] = _agg_one(matched_cells)
        agg_pols["fixed-matched"]["fold_policies"] = matched_names

    # In-sample (train) vs out-of-sample (test) gap for the statistical
    # policy. Big drops on test ⇒ overfit; tiny drops ⇒ generalises.
    train_metrics: dict[str, list[float]] = {"ramratio": [], "hot_hit_coverage": [], "false_prune_rate": []}
    test_metrics:  dict[str, list[float]] = {"ramratio": [], "hot_hit_coverage": [], "false_prune_rate": []}
    delta_metrics: dict[str, list[float]] = {"ramratio": [], "hot_hit_coverage": [], "false_prune_rate": []}
    metric_keys = {"ramratio": "RAMRatio", "hot_hit_coverage": "HotHitCoverage", "false_prune_rate": "FalsePruneRate"}
    for f in folds:
        train_cell = f.get("stat_train") or {}
        test_cell = (f.get("test_results") or {}).get("statistical") or {}
        for short, full in metric_keys.items():
            tr = train_cell.get(full)
            te = test_cell.get(full)
            if tr is None or te is None:
                continue
            train_metrics[short].append(float(tr))
            test_metrics[short].append(float(te))
            delta_metrics[short].append(float(te) - float(tr))

    gap = {}
    for short in metric_keys:
        gap[f"{short}_train_mean"] = None if mean(train_metrics[short]) is None else round(mean(train_metrics[short]) or 0, 6)
        gap[f"{short}_test_mean"]  = None if mean(test_metrics[short]) is None else round(mean(test_metrics[short]) or 0, 6)
        gap[f"{short}_delta_mean"] = None if mean(delta_metrics[short]) is None else round(mean(delta_metrics[short]) or 0, 6)

    distinct_sources = sorted({(d or {}).get("source") for d in data_sources_per_fold if d})
    pol_columns = ["no-prune"]
    if folds:
        pol_columns.append((folds[0].get("fixed_match") or {}).get("policy_name") or "fixed-?")
    pol_columns.append("statistical")
    if args.include_robust:
        pol_columns.append("statistical-robust")

    # ---- Risk QA --------------------------------------------------------
    # The rolling design overlaps test windows when step < test_span, so
    # folds aren't statistically independent. Treat percentile / CVaR
    # numbers as engineering risk indicators, not strict statistical
    # inference. We surface the same caveat in the persisted summary.
    risk_independence_caveat = (
        "Rolling folds with step < test_span overlap and are not "
        "statistically independent. Treat percentile / CVaR figures as "
        "engineering risk indicators, not strict statistical inference."
    ) if args.step < args.test_span else (
        "Folds do not overlap (step ≥ test_span); percentile / CVaR "
        "figures are over near-independent samples but n_folds is still "
        "the binding constraint."
    )

    def _per_policy_tails(fold_subset: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
        """For each (policy, metric) pair, summarise the per-fold values
        as a tail-stat dict — used by both the all-folds and the
        in-distribution risk blocks."""
        out: dict[str, dict[str, Any]] = {}
        names: set[str] = set()
        for f in fold_subset:
            names |= set((f.get("test_results") or {}).keys())
        for pol in sorted(n for n in names if isinstance(n, str)):
            block: dict[str, Any] = {}
            for short, full in (
                ("ramratio", "RAMRatio"),
                ("hot_hit_coverage", "HotHitCoverage"),
                ("false_prune_rate", "FalsePruneRate"),
                ("total_cost", "TotalCost"),
                ("miss_penalty_agg", "MissPenaltyAgg"),
                ("reactivations", "Reactivations"),
            ):
                xs = [
                    float((f.get("test_results") or {}).get(pol, {}).get(full))
                    for f in fold_subset
                    if (f.get("test_results") or {}).get(pol, {}).get(full) is not None
                ]
                block[short] = tail_summary(xs)
            out[pol] = block
        return out

    def _regret_tails(fold_subset: list[dict[str, Any]]) -> dict[str, Any]:
        regrets = [float(f["regret"]) for f in fold_subset if f.get("regret") is not None]
        return {
            **tail_summary(regrets),
            "definition": "TotalCost(statistical) − TotalCost(matched_fixed) per fold; negative = stat wins",
        }

    in_dist_folds = [f for f in folds if f.get("drift_ok") is True]
    risk_all = {
        "scope": "all_folds",
        "n_folds": len(folds),
        "policies": _per_policy_tails(folds),
        "regret": _regret_tails(folds),
    }
    risk_in_dist = {
        "scope": "in_distribution",
        "n_folds": len(in_dist_folds),
        "n_excluded_drift_folds": len(folds) - len(in_dist_folds),
        "policies": _per_policy_tails(in_dist_folds),
        "regret": _regret_tails(in_dist_folds),
    }

    # Tier 1 D — matched-N stability across folds. Captures whether the
    # tuned baseline is itself a stable target or itself drifting.
    matched_ns: list[int] = []
    for f in folds:
        n = (f.get("fixed_match") or {}).get("n")
        if isinstance(n, int):
            matched_ns.append(n)
    matched_jumps = [
        abs(matched_ns[i] - matched_ns[i - 1]) for i in range(1, len(matched_ns))
    ]
    matched_n_stability = {
        "values": matched_ns,
        "n_jumps": len(matched_jumps),
        "max_abs_jump": max(matched_jumps) if matched_jumps else None,
        "p95_abs_jump": percentile([float(x) for x in matched_jumps], 95),
        "mean_abs_jump": mean([float(x) for x in matched_jumps]),
    }

    # Tier 1 F — fail-closed gates. We evaluate against the fold subset
    # the user asked for via --strict-risk-scope. Defaults to all_folds
    # so OOD folds can't quietly hide bad news in releases.
    gate_scope_folds = in_dist_folds if args.strict_risk_scope == "in_distribution" else folds
    risk_violations: list[str] = []
    if args.max_regret is not None:
        bad = [
            f for f in gate_scope_folds
            if f.get("regret") is not None and float(f["regret"]) > args.max_regret
        ]
        if bad:
            risk_violations.append(
                f"max-regret: {len(bad)}/{len(gate_scope_folds)} folds exceed "
                f"{args.max_regret} (worst fold {max(int(f['fold']) for f in bad)})"
            )
    if args.min_coverage is not None:
        bad = [
            f for f in gate_scope_folds
            if (f.get("test_results", {}).get("statistical") or {}).get("HotHitCoverage")
            is not None
            and float(f["test_results"]["statistical"]["HotHitCoverage"]) < args.min_coverage
        ]
        if bad:
            risk_violations.append(
                f"min-coverage: {len(bad)}/{len(gate_scope_folds)} folds drop stat "
                f"HotHitCoverage below {args.min_coverage}"
            )
    if args.max_fpr is not None:
        bad = [
            f for f in gate_scope_folds
            if (f.get("test_results", {}).get("statistical") or {}).get("FalsePruneRate")
            is not None
            and float(f["test_results"]["statistical"]["FalsePruneRate"]) > args.max_fpr
        ]
        if bad:
            risk_violations.append(
                f"max-fpr: {len(bad)}/{len(gate_scope_folds)} folds exceed stat "
                f"FalsePruneRate {args.max_fpr}"
            )

    summary = {
        # Deliberately no "generated_at": we want byte-identical re-runs
        # so a reader can `git diff` two backtests from the same DB
        # without being distracted by a wall-clock timestamp.
        "db": rel_to_cwd(db),
        "zksp": rel_to_cwd(zksp),
        "data_sources_per_fold": data_sources_per_fold,
        "data_sources_distinct": list(distinct_sources),
        "data_sources_mixed": len(distinct_sources) > 1,
        "policy_columns": pol_columns,
        "config": {
            "start": args.start,
            "end": args.end,
            "train_span": args.train_span,
            "test_span": args.test_span,
            "step": args.step,
            "n_folds": len(folds),
            "ram_unit_cost": args.ram_unit_cost,
            "miss_penalty": args.miss_penalty,
            "tau": args.tau,
            "holdout": args.holdout,
            "same_ram_eps": args.same_ram_eps,
            "drift_ratio": args.drift_ratio,
            "strict_risk_scope": args.strict_risk_scope,
            "max_fixed_n": max_fixed_n,
            "risk_min_n": RISK_MIN_N,
        },
        "aggregate": {"policies": agg_pols},
        "statistical_train_test_gap": gap,
        "risk": {
            "independence_caveat": risk_independence_caveat,
            "all_folds": risk_all,
            "in_distribution": risk_in_dist,
            "matched_n_stability": matched_n_stability,
            "gates": {
                "scope": args.strict_risk_scope,
                "max_regret": args.max_regret,
                "min_coverage": args.min_coverage,
                "max_fpr": args.max_fpr,
                "violations": risk_violations,
            },
        },
        "folds": folds,
    }

    (out_dir / "backtest_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    render_html(summary, out_dir / "backtest_report.html")

    # Concise stdout summary.
    n_drift = sum(1 for f in folds if f.get("same_ram_ok") is False)
    n_search_capped = sum(1 for f in folds if (f.get("fixed_match") or {}).get("note"))
    n_ood = sum(1 for f in folds if f.get("drift_ok") is False)
    print(
        f"::: backtest: folds={len(folds)} sources={list(distinct_sources)} "
        f"same-RAM-drift={n_drift} OOD={n_ood} fixed-N-search-capped={n_search_capped} "
        f"risk-violations={len(risk_violations)} out={rel_to_cwd(out_dir)}"
    )
    for v in risk_violations:
        print(f":::   risk gate failed: {v}")

    exit_code = 0
    if args.strict and (
        (len(distinct_sources) > 1) or n_drift > 0 or n_search_capped > 0
    ):
        exit_code = 3
    if risk_violations:
        # Risk gate failures are independent of --strict and always
        # surface a distinct exit code so CI can tell them apart from
        # data-integrity failures.
        exit_code = 4
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
