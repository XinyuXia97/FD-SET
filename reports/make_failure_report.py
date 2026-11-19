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
    parser = argparse.ArgumentParser(description="Render a failure-centric report for a completed run.")
    parser.add_argument("--artifacts-root", type=Path, default=Path("artifacts"))
    parser.add_argument("--run-dir", type=Path, help="Use a specific run directory instead of the latest completed run")
    return parser.parse_args()


def render_failure_report(run_dir: Path) -> str:
    summary = _load_json(run_dir / "summary.json")
    raw_failures = _load_json(run_dir / "failures_raw_all.json", default=[])
    structured_failures = _load_json(run_dir / "failure_db.json", default=[])
    evolved_scenarios = _load_json(run_dir / "evolved_scenarios_all.json", default=[])

    lines = [f"Run: {run_dir.name}", ""]
    lines.extend(_render_round_summary(summary))
    lines.append("")
    lines.extend(_render_counter_section("Failure Types", _count_failures_by_type(raw_failures)))
    lines.append("")
    lines.extend(_render_counter_section("Interaction Patterns", _count_interaction_patterns(structured_failures)))
    lines.append("")
    lines.extend(_render_counter_section("Top Anchor Sources", _count_anchor_sources(evolved_scenarios)))
    return "\n".join(lines)


def _render_round_summary(summary: dict[str, Any]) -> list[str]:
    rounds = summary.get("rounds", [])
    if not isinstance(rounds, list) or not rounds:
        return ["No round summary available."]

    header = "| Round | Failures | Structured | Evolved | Unique Anchors | Avg Anchor Score | Bank Size |"
    divider = "|---:|---:|---:|---:|---:|---:|---:|"
    rows = [header, divider]
    for row in rounds:
        if not isinstance(row, dict):
            continue
        rows.append(
            "| "
            + " | ".join(
                [
                    str(row.get("round", "?")),
                    str(row.get("num_failures", 0)),
                    str(row.get("num_structured", 0)),
                    str(row.get("num_evolved", 0)),
                    str(row.get("num_unique_anchor_sources", 0)),
                    f"{float(row.get('avg_anchor_score', 0.0)):.2f}",
                    f"{row.get('scenario_bank_size_before', 0)} -> {row.get('scenario_bank_size_after', 0)}",
                ]
            )
            + " |"
        )
    return rows


def _render_counter_section(title: str, counter: Counter[str]) -> list[str]:
    lines = [f"{title}:"]
    if not counter:
        lines.append("- None")
        return lines

    for key, value in counter.most_common():
        lines.append(f"- {key}: {value}")
    return lines


def _count_failures_by_type(raw_failures: list[dict[str, Any]]) -> Counter[str]:
    counter: Counter[str] = Counter()
    for failure in raw_failures:
        if isinstance(failure, dict):
            counter[str(failure.get("failure_type", "unknown"))] += 1
    return counter


def _count_interaction_patterns(structured_failures: list[dict[str, Any]]) -> Counter[str]:
    counter: Counter[str] = Counter()
    for failure in structured_failures:
        if isinstance(failure, dict):
            counter[str(failure.get("interaction_pattern", "unknown"))] += 1
    return counter


def _count_anchor_sources(evolved_scenarios: list[dict[str, Any]]) -> Counter[str]:
    counter: Counter[str] = Counter()
    for scenario in evolved_scenarios:
        if isinstance(scenario, dict):
            counter[str(scenario.get("source_scenario_id", "unknown"))] += 1
    return counter


def _load_json(path: Path, *, default: Any | None = None) -> Any:
    if not path.exists():
        return [] if default is None else default
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    run_dir = resolve_run_dir(args.artifacts_root, args.run_dir)
    print(render_failure_report(run_dir))


if __name__ == "__main__":
    main()
