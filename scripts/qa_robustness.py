#!/usr/bin/env python3
"""qa_robustness — Layer 0.5 robustness QA runner.

Runs `go test -v ./...` (or a caller-supplied subset), greps the
`ROBUSTNESS_QA:` tag lines that the markRobustness helper emits, and
aggregates them into a deterministic summary JSON.

The summary is GENERATED, not curated: adding a new robustness test
that calls markRobustness automatically updates the next refresh —
the JSON cannot silently drift out of sync with the actual coverage.

Outputs to --out-dir (default testdata/runs/scroll_100k/qa/):
  - robustness_summary.json   coverage matrix + raw tags
  - robustness_run.log        full `go test -v` stdout/stderr (debug)

Exit codes:
  0  `go test` passed. Tags may be 0 — only --strict treats that as
     a failure.
  1  `go test` itself failed (compile or test failure). The captured
     output lands in robustness_run.log for diagnosis.
  2  --strict was passed AND zero ROBUSTNESS_QA tags were found —
     either markRobustness rotted out of every tagged test, or the
     --run filter excluded all of them.

No external dependencies: stdlib only.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

TAG_RE = re.compile(r"ROBUSTNESS_QA:\s*(\{.*\})")
SUMMARY_TAG_RE = re.compile(r"^(=== RUN|--- (PASS|FAIL|SKIP))")


def rel_to_cwd(p: Path) -> str:
    try:
        return p.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return p.resolve().as_posix()


def run_go_test(pkg: str, run_filter: str | None) -> tuple[int, str]:
    """Run `go test -v <pkg>` (with optional -run filter) and capture
    combined stdout+stderr. We want stderr too because go test sends
    build errors there.

    Returns (returncode, combined_output).
    """
    cmd = ["go", "test", "-count=1", "-v"]
    if run_filter:
        cmd += ["-run", run_filter]
    cmd.append(pkg)
    res = subprocess.run(cmd, check=False, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    return res.returncode, res.stdout


def parse_tags(output: str) -> list[dict[str, Any]]:
    """Extract every ROBUSTNESS_QA payload as a dict. Skips lines
    where the JSON suffix doesn't parse (catches accidental
    ROBUSTNESS_QA: mentions in code comments / log strings)."""
    tags: list[dict[str, Any]] = []
    for line in output.splitlines():
        m = TAG_RE.search(line)
        if not m:
            continue
        try:
            payload = json.loads(m.group(1))
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict):
            continue
        tags.append(payload)
    return tags


def aggregate(tags: list[dict[str, Any]]) -> dict[str, Any]:
    """Deterministic aggregation: sort tags, build by-kind / by-subject
    counters, surface the unique vocabulary so a reader can spot drift
    (e.g. "fail-closed-on-limit" vs "fail_closed_on_limit") at a glance."""
    sorted_tags = sorted(
        tags,
        key=lambda t: (
            t.get("kind", ""),
            t.get("subject", ""),
            t.get("test", ""),
            t.get("failure_mode", ""),
        ),
    )
    kinds = Counter(t.get("kind", "") for t in sorted_tags)
    subjects = Counter(t.get("subject", "") for t in sorted_tags)
    return {
        "n_tags": len(sorted_tags),
        "kinds": dict(sorted(kinds.items())),
        "subjects": dict(sorted(subjects.items())),
        "tags": sorted_tags,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__.splitlines()[0] if __doc__ else None,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--pkg",
        default="./...",
        help="Go package selector for `go test` (default: ./...)",
    )
    ap.add_argument(
        "--run",
        default=None,
        help="Optional -run regex passed straight to `go test` to limit which tests execute",
    )
    ap.add_argument(
        "--out-dir",
        default="testdata/runs/scroll_100k/qa",
        help="Output directory for robustness_summary.json + robustness_run.log",
    )
    ap.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero (2) if zero tagged tests ran — useful in CI to catch a rotted helper",
    )
    args = ap.parse_args(argv)

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"::: qa_robustness: running `go test -v` over {args.pkg}"
          f"{f' filter={args.run!r}' if args.run else ''}")
    rc, output = run_go_test(args.pkg, args.run)
    log_path = out_dir / "robustness_run.log"
    log_path.write_text(output)

    if rc != 0:
        print(f"::: go test failed (exit {rc}) — see {rel_to_cwd(log_path)}", file=sys.stderr)
        return 1

    tags = parse_tags(output)
    if not tags and args.strict:
        print("::: no ROBUSTNESS_QA tags found — markRobustness helper may have rotted", file=sys.stderr)
        return 2

    summary = {
        "pkg": args.pkg,
        "run_filter": args.run,
        **aggregate(tags),
    }
    summary_path = out_dir / "robustness_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    print(f"::: tags emitted:    {summary['n_tags']}")
    print(f"::: kinds:           {summary['kinds']}")
    print(f"::: subjects:        {len(summary['subjects'])} distinct")
    print(f"::: artifacts under  {rel_to_cwd(out_dir)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
