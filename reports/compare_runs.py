from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reports.make_table import resolve_run_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare two FD-SET runs.")
    parser.add_argument("--baseline-run", type=Path, required=True)
    parser.add_argument("--candidate-run", type=Path, required=True)
    parser.add_argument("--artifacts-root", type=Path, default=Path("artifacts"))
    return parser.parse_args()


def render_run_comparison(baseline_dir: Path, candidate_dir: Path) -> str:
    baseline_summary = _load_json(baseline_dir / "summary.json")
    candidate_summary = _load_json(candidate_dir / "summary.json")

    lines = [
        f"Baseline: {baseline_dir.name}",
        f"Candidate: {candidate_dir.name}",
        "",
        "Metric Deltas:",
    ]
    lines.extend(_render_metric_deltas(baseline_summary, candidate_summary))
    lines.append("")
    lines.append("Failure Type Delta:")
    lines.extend(_render_counter_delta(_count_failures(baseline_dir), _count_failures(candidate_dir)))
    lines.append("")
    lines.append("Interaction Pattern Delta:")
    lines.extend(_render_counter_delta(_count_patterns(baseline_dir), _count_patterns(candidate_dir)))
    return "\n".join(lines)


def _render_metric_deltas(baseline_summary: dict[str, Any], candidate_summary: dict[str, Any]) -> list[str]:
    metrics = candidate_summary.get("metrics_keys") or baseline_summary.get("metrics_keys") or []
    if not isinstance(metrics, list):
        metrics = []

    baseline_last = _last_metrics(baseline_summary)
    candidate_last = _last_metrics(candidate_summary)

    lines: list[str] = []
    for metric in metrics:
        metric_name = str(metric)
        base = float(baseline_last.get(metric_name, 0.0))
        cand = float(candidate_last.get(metric_name, 0.0))
        lines.append(f"- {metric_name}: {base:.2f} -> {cand:.2f} ({cand - base:+.2f})")
    return lines or ["- No metric keys available."]


def _render_counter_delta(baseline: Counter[str], candidate: Counter[str]) -> list[str]:
    keys = sorted(set(baseline) | set(candidate))
    if not keys:
        return ["- None"]

    lines = []
    for key in keys:
        base = baseline.get(key, 0)
        cand = candidate.get(key, 0)
        lines.append(f"- {key}: {base} -> {cand} ({cand - base:+d})")
    return lines


def _last_metrics(summary: dict[str, Any]) -> dict[str, Any]:
    rounds = summary.get("rounds", [])
    if isinstance(rounds, list) and rounds:
        last = rounds[-1]
        if isinstance(last, dict):
            metrics = last.get("metrics", {})
            if isinstance(metrics, dict):
                return metrics
    return {}


def _count_failures(run_dir: Path) -> Counter[str]:
    payload = _load_json(run_dir / "failures_raw_all.json", default=[])
    counter: Counter[str] = Counter()
    for row in payload:
        if isinstance(row, dict):
            counter[str(row.get("failure_type", "unknown"))] += 1
    return counter


def _count_patterns(run_dir: Path) -> Counter[str]:
    payload = _load_json(run_dir / "failure_db.json", default=[])
    counter: Counter[str] = Counter()
    for row in payload:
        if isinstance(row, dict):
            counter[str(row.get("interaction_pattern", "unknown"))] += 1
    return counter


def _load_json(path: Path, *, default: Any | None = None) -> Any:
    if not path.exists():
        return [] if default is None else default
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    baseline_dir = resolve_run_dir(args.artifacts_root, args.baseline_run)
    candidate_dir = resolve_run_dir(args.artifacts_root, args.candidate_run)
    print(render_run_comparison(baseline_dir, candidate_dir))


if __name__ == "__main__":
    main()
