#!/usr/bin/env python3
"""qa_viz — minimal QA visualisations over zksp simulate / report output.

Reads either:
  - a single `zksp report --format json` file (5 policies at one cost cell), or
  - a directory of `simulate_*.json` files (one cell per file, ℓ encoded
    in the filename as `..._l<N>.json`).

Writes deterministic artifacts to --out-dir (default
`testdata/runs/scroll_100k/qa/`):
  - qa_summary.json        machine-readable summary
  - degeneracy_flags.json  cells flagged as degenerate, with reasons
  - cost_decomposition.svg stacked RAMCost + MissPenaltyAgg per cell
  - pareto_ramratio_coverage.svg  RAMRatio (x) vs HotHitCoverage (y)

Determinism: all loops sort by (ℓ, policy); JSON dumps with sort_keys.
SVG is hand-written (no matplotlib), so output bytes are stable across
hosts as long as the Python decimal formatter is stable.

No external dependencies: stdlib only (json, argparse, math, pathlib, re).
"""

from __future__ import annotations

import argparse
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
DEFAULT_SAME_RAM_LO = 0.060  # 6.0%
DEFAULT_SAME_RAM_HI = 0.062  # 6.2%

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
    return [{**r, "_source": str(p), "_ell": None} for r in tiering]


def _load_sweep_dir(d: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for child in sorted(d.glob("*.json")):
        if child.name == "qa_summary.json" or child.name.endswith(".expected.json"):
            continue
        try:
            with child.open() as f:
                data = json.load(f)
        except json.JSONDecodeError:
            continue
        # `simulate_*` JSONs are wrapped: {data_source, results: [...]}
        # `report.json` is a different shape; we only treat envelope here.
        results = data.get("results")
        if not isinstance(results, list):
            continue
        ell = parse_ell(child.stem)
        for r in results:
            out.append({**r, "_source": str(child), "_ell": ell})
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
    """Tag any cell that hits one of the three degenerate corners and
    record why. Each reason carries the actual value + the threshold so
    a reader can re-tune the policy without rerunning the QA tool."""
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
        if reasons:
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


def same_ram_pairs(
    results: list[dict[str, Any]],
    lo: float,
    hi: float,
    policy_a: str,
    policy_b: str,
) -> list[dict[str, Any]]:
    """For every ℓ where both policies have RAMRatio in [lo, hi], emit a
    side-by-side comparison. Lets a reader spot prediction-quality
    differences without the cost-arithmetic confound."""
    by_ell: dict[int, dict[str, dict[str, Any]]] = {}
    for r in results:
        ell = r.get("_ell")
        if ell is None:
            continue
        if not (lo <= (r.get("RAMRatio") or 0) <= hi):
            continue
        by_ell.setdefault(ell, {})[r.get("Policy")] = r

    pairs = []
    for ell in sorted(by_ell):
        plc = by_ell[ell]
        a, b = plc.get(policy_a), plc.get(policy_b)
        if a is None or b is None:
            continue
        pairs.append({
            "ell": ell,
            policy_a: _slice(a),
            policy_b: _slice(b),
            "delta": {
                "hot_hit_coverage": (a.get("HotHitCoverage", 0) - b.get("HotHitCoverage", 0)),
                "false_prune_rate": (a.get("FalsePruneRate", 0) - b.get("FalsePruneRate", 0)),
            },
        })
    return pairs


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
        f'RAMRatio vs HotHitCoverage (lower-left = noisier policy)</text>',
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
        "--same-ram-lo", type=float, default=DEFAULT_SAME_RAM_LO,
        help=f"Lower bound of the same-RAM-budget band (default {DEFAULT_SAME_RAM_LO})",
    )
    ap.add_argument(
        "--same-ram-hi", type=float, default=DEFAULT_SAME_RAM_HI,
        help=f"Upper bound of the same-RAM-budget band (default {DEFAULT_SAME_RAM_HI})",
    )
    ap.add_argument(
        "--compare", default="fixed-1k:statistical",
        help="Two policy names separated by ':' for the same-RAM comparison "
             "(default fixed-1k:statistical)",
    )
    args = ap.parse_args(argv)

    in_path = Path(args.report).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    results = load_results(in_path)
    if not results:
        print(f"qa_viz: no cells found in {in_path}", file=sys.stderr)
        return 1

    cost_issues = verify_cost_consistency(results)
    degen = flag_degenerate(
        results, args.ramratio_degen, args.pruned_frac_degen, args.hot_hit_degen,
    )
    pol_a, _, pol_b = args.compare.partition(":")
    if not pol_b:
        print(f"qa_viz: --compare must be 'policyA:policyB', got {args.compare!r}", file=sys.stderr)
        return 2
    same_ram = same_ram_pairs(
        results, args.same_ram_lo, args.same_ram_hi, pol_a, pol_b,
    )

    summary = {
        "input": str(in_path),
        "out_dir": str(out_dir),
        "n_cells": len(results),
        "policies": sorted({r.get("Policy") for r in results if r.get("Policy")}),
        "ells_seen": sorted({r["_ell"] for r in results if r.get("_ell") is not None}),
        "cost_consistency": {
            "n_checked": len(results),
            "n_issues": len(cost_issues),
            "issues": cost_issues,
        },
        "degenerate_cells": {
            "count": len(degen),
            "thresholds": {
                "ramratio_degen": args.ramratio_degen,
                "pruned_frac_degen": args.pruned_frac_degen,
                "hot_hit_degen": args.hot_hit_degen,
            },
        },
        "same_ram_compare": {
            "policy_a": pol_a,
            "policy_b": pol_b,
            "ramratio_band": [args.same_ram_lo, args.same_ram_hi],
            "n_pairs": len(same_ram),
            "pairs": same_ram,
        },
    }

    (out_dir / "qa_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    (out_dir / "degeneracy_flags.json").write_text(json.dumps(degen, indent=2, sort_keys=True) + "\n")
    render_cost_decomposition(results, out_dir / "cost_decomposition.svg")
    render_pareto(results, out_dir / "pareto_ramratio_coverage.svg")

    # tiny stdout report
    print(f"::: qa_viz: input={in_path}")
    print(f"::: cells: {len(results)}")
    print(f"::: policies: {summary['policies']}")
    if summary["ells_seen"]:
        print(f"::: ells_seen: {summary['ells_seen']}")
    print(f"::: cost-arithmetic issues: {len(cost_issues)} of {len(results)}")
    print(f"::: degenerate cells:        {len(degen)}")
    print(f"::: same-RAM pairs ({pol_a} vs {pol_b}, "
          f"[{args.same_ram_lo:.3f}, {args.same_ram_hi:.3f}]): {len(same_ram)}")
    print(f"::: artifacts under {out_dir}")
    # Suppress unused warnings; math is used implicitly via floats.
    _ = math
    return 0


if __name__ == "__main__":
    sys.exit(main())
