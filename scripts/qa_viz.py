#!/usr/bin/env python3
"""qa_viz — minimal QA visualisations over zksp simulate / report output.

Reads either:
  - a single `zksp report --format json` file (5 policies at one cost cell), or
  - a directory of `*_l<N>.json` envelopes (one cell per file, ℓ encoded
    in the stem; matches both `simulate_<policy>_l<N>.json` (top-level)
    and the unprefixed `<policy>_l<N>.json` shape used under sweep_v2/).

Writes deterministic artifacts to --out-dir (default
`testdata/runs/scroll_100k/qa/`):
  - qa_summary.json            machine-readable summary, all checks rolled up
  - degeneracy_flags.json      cells flagged as degenerate, with reasons
  - schema_issues.json         cells with missing fields, wrong types, out-of-range values
  - grid_gaps.json             missing or duplicated (policy, ℓ) cells
  - monotonic_violations.json  statistical-policy ℓ-trend violations
  - same_ram_matches.json      auto-paired (policy_a, policy_b, ℓ) where |ΔRAMRatio| < eps
  - cost_decomposition.svg     stacked RAMCost + MissPenaltyAgg per cell
  - pareto_ramratio_coverage.svg  RAMRatio (x) vs HotHitCoverage (y)
  - qa_report.html             one-page dashboard (embeds the two SVGs)

Determinism: all loops sort by (ℓ, policy); JSON dumps with sort_keys.
SVG/HTML is hand-written (no matplotlib / Jinja), so output bytes are
stable across hosts as long as the Python decimal formatter is stable.

No external dependencies: stdlib only (json, argparse, html, itertools,
math, pathlib, re).
"""

from __future__ import annotations

import argparse
import html
import itertools
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

# ----- thresholds (also exposed via CLI) -------------------------------

DEFAULT_RAMRATIO_DEGEN = 0.001  # < 0.1% → "demote almost everything"
DEFAULT_PRUNED_FRAC_DEGEN = 0.99  # > 99% of TotalExposure pruned
DEFAULT_HOT_HIT_DEGEN = 0.30  # < 30% → "miss-heavy"
DEFAULT_SAME_RAM_EPS = 0.005  # |ΔRAMRatio| < 0.5pp → comparable RAM budget

# Numeric fields whose value should always lie in [0, 1]. Drift here is
# usually a units/scale bug (e.g. percent vs fraction).
UNIT_RANGE_FIELDS = ("RAMRatio", "HotHitCoverage", "FalsePruneRate", "StorageSavedFrac")
# Numeric fields that must be ≥ 0 but have no upper bound.
NON_NEGATIVE_FIELDS = (
    "RAMCost", "MissPenaltyAgg", "TotalCost",
    "TotalSlots", "TotalExposure", "SlotBlocksPruned", "SlotBlocksHot",
    "ObservedIntervals", "CensoredIntervals",
    "Reactivations", "FinalPrunedSlots",
)
# Required fields per cell.
REQUIRED_FIELDS = ("Policy",) + UNIT_RANGE_FIELDS + NON_NEGATIVE_FIELDS
# Policies whose ℓ-monotonicity properties we sanity-check.
THRESHOLD_POLICIES = ("statistical", "statistical-robust")

# ----- policies + colours (deterministic) ------------------------------

# Hand-picked palette so each policy maps to the same colour across runs.
# Hexes, no random / hash-based assignments.
POLICY_PALETTE: dict[str, str] = {
    "no-prune": "#7a7a7a",
    "fixed-30d": "#a3a3a3",
    "fixed-90d": "#bdbdbd",
    "fixed-1k": "#5da95d",
    "fixed-10k": "#3e7e3e",
    "fixed-100k": "#274f27",
    "statistical": "#4f7ec3",
    "statistical-robust": "#9d6cc0",
}
FALLBACK_PALETTE = [
    "#4f7ec3", "#e89853", "#5da95d", "#c45a5a", "#9d6cc0",
    "#7a7a7a", "#3e7e3e", "#9b6a31", "#274f27", "#5a5a5a",
]


def colour_for(policy: str, idx: int) -> str:
    return POLICY_PALETTE.get(policy, FALLBACK_PALETTE[idx % len(FALLBACK_PALETTE)])


# ----- input loading ---------------------------------------------------

ELL_RE = re.compile(r"_l(\d+)$")


def parse_ell(stem: str) -> int | None:
    m = ELL_RE.search(stem)
    return int(m.group(1)) if m else None


def rel_to_cwd(p: Path) -> str:
    """Return p as a repo-relative POSIX path when it sits under cwd, else
    the absolute path. Keeps qa_summary.json diffs portable across hosts
    without losing pointer fidelity for out-of-tree inputs."""
    try:
        return p.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return p.resolve().as_posix()


def load_results(input_path: Path) -> list[dict[str, Any]]:
    """Returns a flat list of cell dicts. Each cell carries the original
    fields from sim.Result plus internal `_source` and `_ell` keys.
    `_ell` is None when ℓ cannot be inferred (e.g. single report.json
    where the cost params live in the run config, not the filename).
    """
    if input_path.is_file():
        return _load_single_report(input_path)
    if input_path.is_dir():
        return _load_sweep_dir(input_path)
    raise SystemExit(f"qa_viz: input does not exist: {input_path}")


def _load_single_report(p: Path) -> list[dict[str, Any]]:
    with p.open() as f:
        data = json.load(f)
    tiering = data.get("tiering") or data.get("Tiering") or []
    ds = data.get("data_source")
    src = rel_to_cwd(p)
    return [{**r, "_source": src, "_ell": None, "_data_source": ds} for r in tiering]


def _load_sweep_dir(d: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    # Sweep cells live under two naming conventions in this repo:
    #   testdata/runs/scroll_100k/simulate_<policy>_l<ℓ>.json   (top-level)
    #   testdata/runs/scroll_100k/sweep_v2/<policy>_l<ℓ>.json   (re-runs)
    # Both encode ℓ as `_l<N>` in the stem, so gate inclusion on the
    # ℓ-pattern instead of a name prefix. This also screens out
    # qa_summary.json, report.json, and any other envelope that
    # happens to carry a `results` key but isn't a sweep cell.
    for child in sorted(d.glob("*.json")):
        if parse_ell(child.stem) is None:
            continue
        try:
            with child.open() as f:
                data = json.load(f)
        except json.JSONDecodeError:
            continue
        results = data.get("results")
        if not isinstance(results, list):
            continue
        ell = parse_ell(child.stem)
        ds = data.get("data_source")
        src = rel_to_cwd(child)
        for r in results:
            out.append({**r, "_source": src, "_ell": ell, "_data_source": ds})
    return out


# ----- QA logic --------------------------------------------------------


def verify_cost_consistency(results: list[dict[str, Any]], rel_tol: float = 1e-9) -> list[dict[str, Any]]:
    """For each cell, check TotalCost == RAMCost + MissPenaltyAgg within
    `rel_tol` relative tolerance. Returns list of issue dicts; empty on
    healthy data."""
    issues = []
    for r in results:
        rc = r.get("RAMCost", 0)
        mp = r.get("MissPenaltyAgg", 0)
        tc = r.get("TotalCost", 0)
        expected = rc + mp
        if expected == 0 and tc == 0:
            continue
        rel = abs(tc - expected) / expected if expected else float("inf")
        if rel > rel_tol:
            issues.append({
                "source": r.get("_source"),
                "policy": r.get("Policy"),
                "ell": r.get("_ell"),
                "ramcost": rc,
                "miss_penalty_agg": mp,
                "expected_total": expected,
                "reported_total": tc,
                "relative_error": rel,
            })
    return issues


def flag_degenerate(
    results: list[dict[str, Any]],
    ramratio_thr: float,
    pruned_frac_thr: float,
    hot_hit_thr: float,
) -> list[dict[str, Any]]:
    """Flag a cell only when ALL THREE degenerate-corner conditions hold
    simultaneously. The three metrics co-vary by construction (kept-few
    ↔ pruned-most ↔ miss-heavy), so requiring all three avoids false
    positives from a single extreme metric while still catching the
    "demote everything and pay miss penalty" regime."""
    flags = []
    for r in results:
        ramratio = r.get("RAMRatio") or 0
        pruned = r.get("SlotBlocksPruned") or 0
        exposure = r.get("TotalExposure") or 0
        hot_hit = r.get("HotHitCoverage") or 0
        pruned_frac = (pruned / exposure) if exposure else 0
        reasons = []
        if ramratio < ramratio_thr:
            reasons.append(
                f"RAMRatio {ramratio:.6f} < {ramratio_thr:.6f} "
                f"(demote almost everything — TotalCost minimisation but no useful prediction)"
            )
        if pruned_frac > pruned_frac_thr:
            reasons.append(
                f"SlotBlocksPruned/TotalExposure {pruned_frac:.6f} > {pruned_frac_thr:.6f} "
                f"(prunes almost all exposure — same regime as above)"
            )
        if hot_hit < hot_hit_thr:
            reasons.append(
                f"HotHitCoverage {hot_hit:.6f} < {hot_hit_thr:.6f} "
                f"(miss-heavy — most accesses fall on demoted slots)"
            )
        if len(reasons) == 3:
            flags.append({
                "source": r.get("_source"),
                "policy": r.get("Policy"),
                "ell": r.get("_ell"),
                "ramratio": ramratio,
                "pruned_frac": pruned_frac,
                "hot_hit_coverage": hot_hit,
                "false_prune_rate": r.get("FalsePruneRate"),
                "reasons": reasons,
            })
    return flags


def find_same_ram_matches(results: list[dict[str, Any]], eps: float) -> list[dict[str, Any]]:
    """For every ℓ, emit one match per (policy_a, policy_b) pair whose
    RAMRatio differs by less than `eps`. Replaces the old fixed-band
    comparison: instead of forcing the reader to hand-pick a band that
    happens to contain a comparable pair, surface every comparable pair
    the sweep contains.

    Lexicographic ordering on policy names makes (a, b) appear once
    (no mirror duplicates), and the output is sorted by (ell,
    policy_a, policy_b) for byte-stable diffs."""
    by_ell: dict[int, list[dict[str, Any]]] = {}
    for r in results:
        ell = r.get("_ell")
        if ell is None or r.get("RAMRatio") is None:
            continue
        by_ell.setdefault(ell, []).append(r)

    matches = []
    for ell in sorted(by_ell):
        cells = sorted(by_ell[ell], key=lambda r: r.get("Policy") or "")
        for a, b in itertools.combinations(cells, 2):
            ra = a.get("RAMRatio") or 0
            rb = b.get("RAMRatio") or 0
            if abs(ra - rb) >= eps:
                continue
            pol_a = a.get("Policy") or "?"
            pol_b = b.get("Policy") or "?"
            matches.append({
                "ell": ell,
                "policy_a": pol_a,
                "policy_b": pol_b,
                "ramratio_a": ra,
                "ramratio_b": rb,
                "delta_ramratio": ra - rb,
                "delta_hot_hit_coverage": (a.get("HotHitCoverage", 0) - b.get("HotHitCoverage", 0)),
                "delta_false_prune_rate": (a.get("FalsePruneRate", 0) - b.get("FalsePruneRate", 0)),
                "delta_total_cost": (a.get("TotalCost", 0) - b.get("TotalCost", 0)),
                "details": {pol_a: _slice(a), pol_b: _slice(b)},
            })
    return matches


def _slice(r: dict[str, Any]) -> dict[str, Any]:
    return {
        "ramratio": r.get("RAMRatio"),
        "hot_hit_coverage": r.get("HotHitCoverage"),
        "false_prune_rate": r.get("FalsePruneRate"),
        "ramcost": r.get("RAMCost"),
        "miss_penalty_agg": r.get("MissPenaltyAgg"),
        "total_cost": r.get("TotalCost"),
        "reactivations": r.get("Reactivations"),
    }


def check_grid_completeness(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Detect missing or duplicated (policy, ℓ) cells in a sweep.

    Missing = cells absent from the cartesian product `policies × ells`,
    where both axes are derived from what's actually in the input. (We
    do not know the *intended* grid; the closest honest proxy is
    "everything we'd expect if the sweep was a complete cartesian
    product of what we observed").

    Duplicate = a (policy, ℓ) pair that appears more than once across
    files — usually a leftover from a half-cleaned re-run."""
    counts: dict[tuple[str, int], int] = {}
    sources: dict[tuple[str, int], list[str]] = {}
    for r in results:
        ell = r.get("_ell")
        pol = r.get("Policy")
        if ell is None or pol is None:
            continue
        key = (pol, ell)
        counts[key] = counts.get(key, 0) + 1
        sources.setdefault(key, []).append(r.get("_source") or "")

    policies = sorted({pol for (pol, _) in counts})
    ells = sorted({ell for (_, ell) in counts})
    expected = {(pol, ell) for pol in policies for ell in ells}
    present = set(counts)
    missing = sorted(expected - present)
    duplicates = sorted([k for k, c in counts.items() if c > 1])

    return {
        "policies": policies,
        "ells": ells,
        "n_expected": len(expected),
        "n_present": len(present),
        "missing": [{"policy": p, "ell": e} for (p, e) in missing],
        "duplicates": [
            {"policy": p, "ell": e, "count": counts[(p, e)], "sources": sources[(p, e)]}
            for (p, e) in duplicates
        ],
    }


def check_data_sources(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Capability/data-source guardrail. Mixing `rpc` (Transfer-log
    surrogate) with `statediff` (true read+write trace) in the same
    sweep would silently produce apples-to-oranges costs, since the
    extractors observe different slot universes — flag that loudly."""
    by_source: dict[str, int] = {}
    full_capabilities: dict[str, dict[str, Any]] = {}
    for r in results:
        ds = r.get("_data_source")
        if not isinstance(ds, dict):
            continue
        src = ds.get("source") or "?"
        by_source[src] = by_source.get(src, 0) + 1
        # Keep one full capability dict per source kind so the reader
        # can spot drift even within a single source family.
        full_capabilities.setdefault(src, ds)

    sources = sorted(by_source)
    return {
        "sources": sources,
        "n_distinct": len(sources),
        "cells_per_source": {s: by_source[s] for s in sources},
        "capabilities": {s: full_capabilities[s] for s in sources},
        "mixed": len(sources) > 1,
    }


def check_schema_sanity(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Per-cell schema/type/range checks. Catches drift earlier than the
    cost-arithmetic verifier does — e.g. a renamed field, a percent that
    snuck in where a fraction was expected, or a negative count."""
    issues = []
    for r in results:
        cell_issues: list[str] = []
        for f in REQUIRED_FIELDS:
            if f not in r:
                cell_issues.append(f"missing field {f!r}")
                continue
            v = r.get(f)
            if f == "Policy":
                if not isinstance(v, str) or not v:
                    cell_issues.append(f"{f}: expected non-empty str, got {type(v).__name__} {v!r}")
                continue
            if not isinstance(v, (int, float)) or isinstance(v, bool):
                cell_issues.append(f"{f}: expected number, got {type(v).__name__} {v!r}")
                continue
            if math.isnan(v) or math.isinf(v):
                cell_issues.append(f"{f}: non-finite ({v!r})")
                continue
            if f in UNIT_RANGE_FIELDS and not (0.0 <= v <= 1.0):
                cell_issues.append(f"{f}: {v!r} not in [0, 1]")
            elif f in NON_NEGATIVE_FIELDS and v < 0:
                cell_issues.append(f"{f}: {v!r} < 0")
        if cell_issues:
            issues.append({
                "source": r.get("_source"),
                "policy": r.get("Policy"),
                "ell": r.get("_ell"),
                "issues": cell_issues,
            })
    return issues


def check_monotonic_trends(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """For each threshold-based policy in `THRESHOLD_POLICIES`, walk
    cells in increasing ℓ order and flag violations of:
      - RAMRatio non-decreasing in ℓ (larger ℓ → higher c·τ/ℓ-derived
        threshold p* → keep more slots hot → RAMRatio ↑)
      - Reactivations non-increasing in ℓ (more slots kept hot → fewer
        demoted → fewer reactivations needed)

    We deliberately don't run this for `fixed-*` (idle threshold is in
    blocks, not ℓ-driven) or `no-prune` (RAMRatio≡1, Reactivations≡0).
    A small numeric tolerance avoids spurious flags from rounding."""
    tol = 1e-9
    violations = []
    by_policy: dict[str, list[dict[str, Any]]] = {}
    for r in results:
        pol = r.get("Policy")
        ell = r.get("_ell")
        if pol not in THRESHOLD_POLICIES or ell is None:
            continue
        by_policy.setdefault(pol, []).append(r)

    for pol in sorted(by_policy):
        cells = sorted(by_policy[pol], key=lambda r: r.get("_ell") or 0)
        for prev, cur in zip(cells, cells[1:]):
            ramratio_drop = (prev.get("RAMRatio") or 0) - (cur.get("RAMRatio") or 0)
            if ramratio_drop > tol:
                violations.append({
                    "policy": pol,
                    "metric": "RAMRatio",
                    "expected": "non-decreasing in ℓ",
                    "ell_prev": prev.get("_ell"),
                    "ell_cur": cur.get("_ell"),
                    "value_prev": prev.get("RAMRatio"),
                    "value_cur": cur.get("RAMRatio"),
                    "delta": -(ramratio_drop),  # cur - prev (negative = bad)
                })
            react_jump = (cur.get("Reactivations") or 0) - (prev.get("Reactivations") or 0)
            if react_jump > tol:
                violations.append({
                    "policy": pol,
                    "metric": "Reactivations",
                    "expected": "non-increasing in ℓ",
                    "ell_prev": prev.get("_ell"),
                    "ell_cur": cur.get("_ell"),
                    "value_prev": prev.get("Reactivations"),
                    "value_cur": cur.get("Reactivations"),
                    "delta": react_jump,  # cur - prev (positive = bad)
                })
    return violations


# ----- SVG rendering ---------------------------------------------------
# Hand-written so we have no matplotlib / numpy dependency. Coordinates
# are floats rounded to 1 decimal so the bytes are deterministic across
# Python versions (avoiding repr float quirks).


def _f(x: float) -> str:
    """Deterministic short float format for SVG attributes."""
    return f"{x:.1f}"


def _esc(s: str) -> str:
    """Tiny XML escaper — we only have to handle the chars likely in
    policy names / numbers, but be safe with &/</>"""
    return (
        str(s)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def render_cost_decomposition(results: list[dict[str, Any]], out_path: Path) -> None:
    """Stacked-bar SVG: per (ℓ, policy) cell, RAMCost (bottom) +
    MissPenaltyAgg (top). When ℓ is None for every cell (single-report
    input), x-axis groups by policy alone."""
    cells = sorted(
        results,
        key=lambda r: ((r.get("_ell") if r.get("_ell") is not None else -1), r.get("Policy") or ""),
    )
    if not cells:
        out_path.write_text(_empty_svg("cost_decomposition (no data)"))
        return

    W, H = 1100, 540
    margin = {"left": 90, "right": 220, "top": 60, "bottom": 110}
    plot_w = W - margin["left"] - margin["right"]
    plot_h = H - margin["top"] - margin["bottom"]

    n = len(cells)
    cell_w = plot_w / n
    bar_w = max(cell_w * 0.7, 6)
    max_total = max((c.get("TotalCost", 0) for c in cells), default=1) or 1

    parts = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
        f'viewBox="0 0 {W} {H}" font-family="sans-serif" font-size="11">',
        f'<text x="{W//2}" y="28" text-anchor="middle" font-size="14" font-weight="bold">'
        f'Cost decomposition: RAMCost + MissPenaltyAgg = TotalCost</text>',
        f'<text x="{W//2}" y="46" text-anchor="middle" font-size="10" fill="#555">'
        f'one bar per (policy, ℓ) cell — sorted by ℓ then policy</text>',
    ]

    # bars
    for i, c in enumerate(cells):
        x = margin["left"] + i * cell_w + (cell_w - bar_w) / 2
        ram = c.get("RAMCost", 0) or 0
        miss = c.get("MissPenaltyAgg", 0) or 0
        ram_h = (ram / max_total) * plot_h
        miss_h = (miss / max_total) * plot_h
        bottom = margin["top"] + plot_h
        # RAM (bottom)
        parts.append(
            f'<rect x="{_f(x)}" y="{_f(bottom - ram_h)}" '
            f'width="{_f(bar_w)}" height="{_f(ram_h)}" fill="#4f7ec3"/>'
        )
        # Miss (stacked on top)
        parts.append(
            f'<rect x="{_f(x)}" y="{_f(bottom - ram_h - miss_h)}" '
            f'width="{_f(bar_w)}" height="{_f(miss_h)}" fill="#e89853"/>'
        )
        # x-axis label (rotated)
        ell = c.get("_ell")
        label = f"{c.get('Policy')}@ℓ={ell}" if ell is not None else c.get("Policy")
        anchor_x = x + bar_w / 2
        anchor_y = bottom + 12
        parts.append(
            f'<text x="{_f(anchor_x)}" y="{_f(anchor_y)}" text-anchor="end" '
            f'transform="rotate(-45 {_f(anchor_x)} {_f(anchor_y)})" font-size="9">'
            f'{_esc(label)}</text>'
        )

    # axes
    parts += _axis_frame(margin, plot_w, plot_h)
    # y ticks (5 ticks)
    for tick in (0, 0.25, 0.5, 0.75, 1.0):
        y = margin["top"] + plot_h - tick * plot_h
        parts.append(
            f'<line x1="{_f(margin["left"]-5)}" y1="{_f(y)}" '
            f'x2="{_f(margin["left"])}" y2="{_f(y)}" stroke="black"/>'
        )
        parts.append(
            f'<text x="{_f(margin["left"]-8)}" y="{_f(y+3)}" text-anchor="end">'
            f'{tick * max_total:.2e}</text>'
        )

    # legend
    lx = W - margin["right"] + 20
    parts.append(f'<rect x="{_f(lx)}" y="80" width="14" height="14" fill="#4f7ec3"/>')
    parts.append(f'<text x="{_f(lx+20)}" y="92">RAMCost</text>')
    parts.append(f'<rect x="{_f(lx)}" y="105" width="14" height="14" fill="#e89853"/>')
    parts.append(f'<text x="{_f(lx+20)}" y="117">MissPenaltyAgg</text>')
    parts.append(
        f'<text x="{_f(lx)}" y="150" font-size="9" fill="#555">'
        f'TotalCost = RAMCost + MissPenaltyAgg</text>'
    )
    parts.append(
        f'<text x="{_f(lx)}" y="165" font-size="9" fill="#555">'
        f'(verified to {1e-9:.0e} relative tolerance)</text>'
    )

    parts.append("</svg>")
    out_path.write_text("\n".join(parts))


def render_pareto(results: list[dict[str, Any]], out_path: Path) -> None:
    """Scatter SVG: RAMRatio (x, [0,1]) vs HotHitCoverage (y, [0,1]).
    Each cell is a coloured circle. Top-left = ideal (low RAM, high
    coverage)."""
    points = sorted(
        (r for r in results if r.get("RAMRatio") is not None and r.get("HotHitCoverage") is not None),
        key=lambda r: (r.get("Policy") or "", r.get("_ell") if r.get("_ell") is not None else -1),
    )
    if not points:
        out_path.write_text(_empty_svg("pareto (no data)"))
        return

    W, H = 760, 540
    margin = {"left": 80, "right": 220, "top": 60, "bottom": 70}
    plot_w = W - margin["left"] - margin["right"]
    plot_h = H - margin["top"] - margin["bottom"]

    parts = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
        f'viewBox="0 0 {W} {H}" font-family="sans-serif" font-size="11">',
        f'<text x="{W//2}" y="28" text-anchor="middle" font-size="14" font-weight="bold">'
        f'RAMRatio vs HotHitCoverage (lower-left = cheap-but-low-coverage)</text>',
        f'<text x="{W//2}" y="46" text-anchor="middle" font-size="10" fill="#555">'
        f'top-left ↖ is ideal: low RAM, high coverage</text>',
    ]

    parts += _axis_frame(margin, plot_w, plot_h)

    # ticks + axis labels
    for tick in (0, 0.25, 0.5, 0.75, 1.0):
        x = margin["left"] + tick * plot_w
        y = margin["top"] + plot_h - tick * plot_h
        parts.append(
            f'<line x1="{_f(x)}" y1="{_f(margin["top"]+plot_h)}" '
            f'x2="{_f(x)}" y2="{_f(margin["top"]+plot_h+5)}" stroke="black"/>'
        )
        parts.append(
            f'<text x="{_f(x)}" y="{_f(margin["top"]+plot_h+18)}" text-anchor="middle">'
            f'{tick:.2f}</text>'
        )
        parts.append(
            f'<line x1="{_f(margin["left"]-5)}" y1="{_f(y)}" '
            f'x2="{_f(margin["left"])}" y2="{_f(y)}" stroke="black"/>'
        )
        parts.append(
            f'<text x="{_f(margin["left"]-8)}" y="{_f(y+3)}" text-anchor="end">'
            f'{tick:.2f}</text>'
        )

    # axis titles
    parts.append(
        f'<text x="{_f(margin["left"]+plot_w/2)}" y="{_f(H-25)}" text-anchor="middle">'
        f'RAMRatio</text>'
    )
    parts.append(
        f'<text x="20" y="{_f(margin["top"]+plot_h/2)}" text-anchor="middle" '
        f'transform="rotate(-90 20 {_f(margin["top"]+plot_h/2)})">'
        f'HotHitCoverage</text>'
    )

    # ideal-corner annotation
    parts.append(
        f'<text x="{_f(margin["left"]+8)}" y="{_f(margin["top"]+12)}" '
        f'font-size="9" fill="#888" font-style="italic">↖ ideal</text>'
    )

    # diagonal "do nothing" reference line: RAMRatio = HotHitCoverage
    # (a totally uninformative policy with random demote would land near it).
    parts.append(
        f'<line x1="{_f(margin["left"])}" y1="{_f(margin["top"]+plot_h)}" '
        f'x2="{_f(margin["left"]+plot_w)}" y2="{_f(margin["top"])}" '
        f'stroke="#bbb" stroke-dasharray="3,3"/>'
    )

    # points
    policies_seen = sorted({p.get("Policy") for p in points if p.get("Policy")})
    for p in points:
        policy = p.get("Policy") or "?"
        idx = policies_seen.index(policy) if policy in policies_seen else 0
        col = colour_for(policy, idx)
        x = margin["left"] + (p.get("RAMRatio") or 0) * plot_w
        y = margin["top"] + plot_h - (p.get("HotHitCoverage") or 0) * plot_h
        parts.append(
            f'<circle cx="{_f(x)}" cy="{_f(y)}" r="4" fill="{col}" '
            f'fill-opacity="0.75" stroke="{col}" stroke-width="0.5"/>'
        )

    # legend (one entry per policy, in deterministic order)
    lx = W - margin["right"] + 20
    ly = margin["top"] + 20
    for i, pol in enumerate(policies_seen):
        col = colour_for(pol, i)
        parts.append(
            f'<circle cx="{_f(lx+5)}" cy="{_f(ly+i*18)}" r="4" fill="{col}"/>'
        )
        parts.append(
            f'<text x="{_f(lx+15)}" y="{_f(ly+i*18+4)}">{_esc(pol)}</text>'
        )

    parts.append("</svg>")
    out_path.write_text("\n".join(parts))


def _axis_frame(margin: dict[str, int], plot_w: float, plot_h: float) -> list[str]:
    return [
        f'<line x1="{_f(margin["left"])}" y1="{_f(margin["top"])}" '
        f'x2="{_f(margin["left"])}" y2="{_f(margin["top"]+plot_h)}" stroke="black"/>',
        f'<line x1="{_f(margin["left"])}" y1="{_f(margin["top"]+plot_h)}" '
        f'x2="{_f(margin["left"]+plot_w)}" y2="{_f(margin["top"]+plot_h)}" stroke="black"/>',
    ]


def _empty_svg(title: str) -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<svg xmlns="http://www.w3.org/2000/svg" width="400" height="120" '
        'viewBox="0 0 400 120" font-family="sans-serif">'
        f'<text x="200" y="60" text-anchor="middle" font-size="14" fill="#888">{_esc(title)}</text>'
        '</svg>'
    )


# ----- HTML dashboard --------------------------------------------------


def render_html_dashboard(summary: dict[str, Any], out_path: Path) -> None:
    """Single self-contained HTML page summarising every check, with the
    two SVGs referenced from the same out_dir. Hand-written so we don't
    pick up a Jinja / Mustache dependency."""

    def h(s: Any) -> str:
        return html.escape(str(s), quote=True)

    def status(ok: bool) -> str:
        return '<span class="ok">OK</span>' if ok else '<span class="bad">FAIL</span>'

    cost = summary["cost_consistency"]
    schema = summary["schema_sanity"]
    grid = summary["grid_completeness"]
    sources = summary["data_sources"]
    degen = summary["degenerate_cells"]
    mono = summary["monotonic_trends"]
    sram = summary["same_ram_matches"]

    def kv_table(rows: list[tuple[str, Any]]) -> str:
        body = "".join(
            f"<tr><th>{h(k)}</th><td>{h(v)}</td></tr>" for k, v in rows
        )
        return f"<table class='kv'>{body}</table>"

    cost_ok = cost["n_issues"] == 0
    schema_ok = schema["n_issues"] == 0
    grid_ok = len(grid["missing"]) == 0 and len(grid["duplicates"]) == 0
    sources_ok = not sources["mixed"] and sources["n_distinct"] >= 1
    mono_ok = mono["n_violations"] == 0

    summary_rows = [
        ("input", summary["input"]),
        ("cells", summary["n_cells"]),
        ("policies", ", ".join(summary["policies"]) or "—"),
        ("ℓ values seen", ", ".join(str(e) for e in summary["ells_seen"]) or "—"),
        ("data sources", ", ".join(sources["sources"]) or "—"),
    ]

    check_rows = [
        ("Schema sanity",
         f"{status(schema_ok)} ({schema['n_issues']} cells with issues / {summary['n_cells']})"),
        ("Cost arithmetic",
         f"{status(cost_ok)} ({cost['n_issues']} of {cost['n_checked']} cells failed)"),
        ("Data-source guardrail",
         f"{status(sources_ok)} ({sources['n_distinct']} distinct, "
         f"{'mixed!' if sources['mixed'] else 'consistent'})"),
        ("Grid completeness",
         f"{status(grid_ok)} ({grid['n_present']}/{grid['n_expected']} cells, "
         f"{len(grid['missing'])} missing, {len(grid['duplicates'])} duplicate)"),
        ("Degenerate cells (info)",
         f"{degen['count']} flagged (RAMRatio &lt; {h(degen['thresholds']['ramratio_degen'])}, "
         f"pruned_frac &gt; {h(degen['thresholds']['pruned_frac_degen'])}, "
         f"HotHitCoverage &lt; {h(degen['thresholds']['hot_hit_degen'])})"),
        ("Monotonic trends (statistical)",
         f"{status(mono_ok)} ({mono['n_violations']} violations across "
         f"{', '.join(THRESHOLD_POLICIES)})"),
        ("Same-RAM matches",
         f"{sram['n_matches']} pairs with |ΔRAMRatio| &lt; {h(sram['epsilon'])}"),
    ]

    def check_table(rows: list[tuple[str, str]]) -> str:
        body = "".join(f"<tr><th>{h(k)}</th><td>{v}</td></tr>" for k, v in rows)
        return f"<table class='checks'>{body}</table>"

    same_ram_table = _html_same_ram_table(sram["matches"], h)
    degen_table = _html_degen_table(summary.get("_degen_full", []), h)
    grid_block = _html_grid_block(grid, h)
    mono_block = _html_mono_block(mono["violations"], h)

    parts = [
        "<!DOCTYPE html>",
        "<html lang='en'><head><meta charset='utf-8'>",
        f"<title>qa_viz — {h(summary['input'])}</title>",
        "<style>",
        "body { font-family: system-ui, sans-serif; max-width: 1100px; margin: 24px auto; padding: 0 16px; color: #222; }",
        "h1 { font-size: 1.4em; margin-bottom: 4px; }",
        "h2 { font-size: 1.1em; margin-top: 28px; border-bottom: 1px solid #ddd; padding-bottom: 4px; }",
        "table { border-collapse: collapse; margin: 8px 0; }",
        "th, td { padding: 4px 10px; border-bottom: 1px solid #eee; text-align: left; vertical-align: top; }",
        "table.kv th { width: 180px; color: #555; font-weight: 500; }",
        "table.checks th { width: 260px; color: #555; font-weight: 500; }",
        "table.data th { background: #f6f6f6; font-weight: 600; }",
        "table.data { font-size: 0.9em; }",
        "code { background: #f6f6f6; padding: 1px 4px; border-radius: 3px; font-size: 0.92em; }",
        ".ok  { color: #1f7a1f; font-weight: 600; }",
        ".bad { color: #b91c1c; font-weight: 600; }",
        ".muted { color: #888; font-size: 0.9em; }",
        "img { max-width: 100%; border: 1px solid #eee; padding: 4px; background: white; }",
        "</style></head><body>",
        f"<h1>qa_viz dashboard</h1>",
        f"<p class='muted'>Auto-generated from <code>{h(summary['input'])}</code>. "
        f"Re-run with <code>make qa-viz REPORT=&lt;path&gt;</code> to refresh.</p>",
        "<h2>Run summary</h2>",
        kv_table(summary_rows),
        "<h2>Checks</h2>",
        check_table(check_rows),
        "<h2>Cost decomposition</h2>",
        "<img src='cost_decomposition.svg' alt='RAMCost + MissPenaltyAgg per cell'>",
        "<h2>RAMRatio × HotHitCoverage</h2>",
        "<img src='pareto_ramratio_coverage.svg' alt='RAMRatio vs HotHitCoverage'>",
        "<h2>Same-RAM matches</h2>",
        same_ram_table,
        "<h2>Degenerate cells</h2>",
        degen_table,
        "<h2>Grid gaps</h2>",
        grid_block,
        "<h2>Monotonic-trend violations</h2>",
        mono_block,
        "<p class='muted'>Detail JSON: "
        "<code>qa_summary.json</code>, <code>schema_issues.json</code>, "
        "<code>grid_gaps.json</code>, <code>same_ram_matches.json</code>, "
        "<code>monotonic_violations.json</code>, <code>degeneracy_flags.json</code>.</p>",
        "</body></html>",
    ]
    out_path.write_text("\n".join(parts) + "\n")


def _html_same_ram_table(matches: list[dict[str, Any]], h) -> str:
    if not matches:
        return "<p class='muted'>No comparable pairs at the current epsilon.</p>"
    rows = ["<table class='data'><tr>"
            "<th>ℓ</th><th>policy A</th><th>policy B</th>"
            "<th>RAMRatio A</th><th>RAMRatio B</th>"
            "<th>ΔHotHitCov</th><th>ΔFalsePrune</th><th>ΔTotalCost</th></tr>"]
    for m in matches:
        rows.append(
            f"<tr>"
            f"<td>{h(m['ell'])}</td>"
            f"<td>{h(m['policy_a'])}</td>"
            f"<td>{h(m['policy_b'])}</td>"
            f"<td>{m['ramratio_a']:.4f}</td>"
            f"<td>{m['ramratio_b']:.4f}</td>"
            f"<td>{m['delta_hot_hit_coverage']:+.4f}</td>"
            f"<td>{m['delta_false_prune_rate']:+.4f}</td>"
            f"<td>{m['delta_total_cost']:+,.0f}</td>"
            f"</tr>"
        )
    rows.append("</table>")
    return "".join(rows)


def _html_degen_table(degen: list[dict[str, Any]], h) -> str:
    if not degen:
        return "<p class='muted'>None.</p>"
    rows = ["<table class='data'><tr>"
            "<th>policy</th><th>ℓ</th>"
            "<th>RAMRatio</th><th>pruned_frac</th><th>HotHitCov</th>"
            "<th>FalsePrune</th></tr>"]
    for d in degen:
        rows.append(
            f"<tr>"
            f"<td>{h(d.get('policy'))}</td>"
            f"<td>{h(d.get('ell'))}</td>"
            f"<td>{(d.get('ramratio') or 0):.6f}</td>"
            f"<td>{(d.get('pruned_frac') or 0):.6f}</td>"
            f"<td>{(d.get('hot_hit_coverage') or 0):.4f}</td>"
            f"<td>{(d.get('false_prune_rate') or 0):.4f}</td>"
            f"</tr>"
        )
    rows.append("</table>")
    return "".join(rows)


def _html_grid_block(grid: dict[str, Any], h) -> str:
    if not grid["missing"] and not grid["duplicates"]:
        return "<p class='muted'>Complete cartesian grid — no missing or duplicate cells.</p>"
    parts: list[str] = []
    if grid["missing"]:
        parts.append("<p><b>Missing cells:</b></p><ul>")
        for m in grid["missing"]:
            parts.append(f"<li>{h(m['policy'])} @ ℓ={h(m['ell'])}</li>")
        parts.append("</ul>")
    if grid["duplicates"]:
        parts.append("<p><b>Duplicate cells:</b></p><ul>")
        for d in grid["duplicates"]:
            parts.append(
                f"<li>{h(d['policy'])} @ ℓ={h(d['ell'])} — appears {h(d['count'])} times "
                f"({', '.join(h(s) for s in d['sources'])})</li>"
            )
        parts.append("</ul>")
    return "".join(parts)


def _html_mono_block(violations: list[dict[str, Any]], h) -> str:
    if not violations:
        return "<p class='muted'>None.</p>"
    rows = ["<table class='data'><tr>"
            "<th>policy</th><th>metric</th><th>expected</th>"
            "<th>ℓ prev → cur</th><th>value prev → cur</th><th>Δ</th></tr>"]
    for v in violations:
        rows.append(
            f"<tr>"
            f"<td>{h(v['policy'])}</td>"
            f"<td>{h(v['metric'])}</td>"
            f"<td>{h(v['expected'])}</td>"
            f"<td>{h(v['ell_prev'])} → {h(v['ell_cur'])}</td>"
            f"<td>{h(v['value_prev'])} → {h(v['value_cur'])}</td>"
            f"<td>{v['delta']:+}</td>"
            f"</tr>"
        )
    rows.append("</table>")
    return "".join(rows)


# ----- main ------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__.splitlines()[0] if __doc__ else None,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--report", required=True,
        help="Path to a single zksp report.json OR directory of simulate_*.json",
    )
    ap.add_argument(
        "--out-dir", default="testdata/runs/scroll_100k/qa",
        help="Output directory for QA artifacts (default: %(default)s)",
    )
    ap.add_argument(
        "--ramratio-degen", type=float, default=DEFAULT_RAMRATIO_DEGEN,
        help=f"RAMRatio threshold below which a cell is degenerate (default {DEFAULT_RAMRATIO_DEGEN})",
    )
    ap.add_argument(
        "--pruned-frac-degen", type=float, default=DEFAULT_PRUNED_FRAC_DEGEN,
        help=f"SlotBlocksPruned/TotalExposure threshold above which a cell is degenerate "
             f"(default {DEFAULT_PRUNED_FRAC_DEGEN})",
    )
    ap.add_argument(
        "--hot-hit-degen", type=float, default=DEFAULT_HOT_HIT_DEGEN,
        help=f"HotHitCoverage threshold below which a cell is degenerate (default {DEFAULT_HOT_HIT_DEGEN})",
    )
    ap.add_argument(
        "--same-ram-eps", type=float, default=DEFAULT_SAME_RAM_EPS,
        help=f"Pair (policy_a, policy_b) at the same ℓ are 'same RAM' when "
             f"|ΔRAMRatio| < eps (default {DEFAULT_SAME_RAM_EPS})",
    )
    ap.add_argument(
        "--strict", action="store_true",
        help="Exit non-zero on any cost-arithmetic / schema-sanity / "
             "data-source-mixing / monotonic-trend failure. "
             "Default is to emit artifacts and exit 0 even with findings.",
    )
    args = ap.parse_args(argv)

    in_path = Path(args.report).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    results = load_results(in_path)
    if not results:
        print(f"qa_viz: no cells found in {in_path}", file=sys.stderr)
        return 1

    schema_issues = check_schema_sanity(results)
    cost_issues = verify_cost_consistency(results)
    degen = flag_degenerate(
        results, args.ramratio_degen, args.pruned_frac_degen, args.hot_hit_degen,
    )
    grid = check_grid_completeness(results)
    sources = check_data_sources(results)
    mono = check_monotonic_trends(results)
    same_ram = find_same_ram_matches(results, args.same_ram_eps)

    summary = {
        "input": rel_to_cwd(in_path),
        "out_dir": rel_to_cwd(out_dir),
        "n_cells": len(results),
        "policies": sorted({r.get("Policy") for r in results if r.get("Policy")}),
        "ells_seen": sorted({r["_ell"] for r in results if r.get("_ell") is not None}),
        "schema_sanity": {
            "n_checked": len(results),
            "n_issues": len(schema_issues),
            "issues": schema_issues,
        },
        "cost_consistency": {
            "n_checked": len(results),
            "n_issues": len(cost_issues),
            "issues": cost_issues,
        },
        "data_sources": sources,
        "grid_completeness": grid,
        "degenerate_cells": {
            "count": len(degen),
            "thresholds": {
                "ramratio_degen": args.ramratio_degen,
                "pruned_frac_degen": args.pruned_frac_degen,
                "hot_hit_degen": args.hot_hit_degen,
            },
        },
        "monotonic_trends": {
            "policies_checked": list(THRESHOLD_POLICIES),
            "n_violations": len(mono),
            "violations": mono,
        },
        "same_ram_matches": {
            "epsilon": args.same_ram_eps,
            "n_matches": len(same_ram),
            "matches": same_ram,
        },
        # Pull-through for the HTML renderer; not consumed by JSON readers.
        "_degen_full": degen,
    }

    (out_dir / "qa_summary.json").write_text(
        json.dumps({k: v for k, v in summary.items() if not k.startswith("_")},
                   indent=2, sort_keys=True) + "\n"
    )
    (out_dir / "degeneracy_flags.json").write_text(json.dumps(degen, indent=2, sort_keys=True) + "\n")
    (out_dir / "schema_issues.json").write_text(json.dumps(schema_issues, indent=2, sort_keys=True) + "\n")
    (out_dir / "grid_gaps.json").write_text(json.dumps(grid, indent=2, sort_keys=True) + "\n")
    (out_dir / "monotonic_violations.json").write_text(json.dumps(mono, indent=2, sort_keys=True) + "\n")
    (out_dir / "same_ram_matches.json").write_text(json.dumps(same_ram, indent=2, sort_keys=True) + "\n")
    render_cost_decomposition(results, out_dir / "cost_decomposition.svg")
    render_pareto(results, out_dir / "pareto_ramratio_coverage.svg")
    render_html_dashboard(summary, out_dir / "qa_report.html")

    # tiny stdout report
    print(f"::: qa_viz: input={rel_to_cwd(in_path)}")
    print(f"::: cells: {len(results)}")
    print(f"::: policies: {summary['policies']}")
    if summary["ells_seen"]:
        print(f"::: ells_seen: {summary['ells_seen']}")
    print(f"::: data sources: {sources['sources']} (mixed={sources['mixed']})")
    print(f"::: schema issues:           {len(schema_issues)} of {len(results)}")
    print(f"::: cost-arithmetic issues:  {len(cost_issues)} of {len(results)}")
    print(f"::: grid: {grid['n_present']}/{grid['n_expected']} cells "
          f"(missing {len(grid['missing'])}, duplicate {len(grid['duplicates'])})")
    print(f"::: degenerate cells:        {len(degen)}")
    print(f"::: monotonic violations:    {len(mono)}")
    print(f"::: same-RAM matches (|Δ|<{args.same_ram_eps}): {len(same_ram)}")
    print(f"::: artifacts under {rel_to_cwd(out_dir)}")
    # Suppress unused warnings; math is used implicitly via floats.
    _ = math

    if args.strict:
        if (schema_issues or cost_issues or sources["mixed"] or mono):
            return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
