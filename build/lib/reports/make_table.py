from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a markdown table for a completed run summary.")
    parser.add_argument("--artifacts-root", type=Path, default=Path("artifacts"))
    parser.add_argument("--run-dir", type=Path, help="Use a specific run directory instead of the latest completed run")
    return parser.parse_args()


def resolve_run_dir(artifacts_root: Path, run_dir: Path | None) -> Path:
    if run_dir is not None:
        candidates = [run_dir]
        if not run_dir.is_absolute():
            candidates.append(Path.cwd() / run_dir)
            candidates.append(artifacts_root / run_dir)
            candidates.append(Path.cwd() / artifacts_root / run_dir)

        for candidate in candidates:
            if candidate.exists():
                return candidate.resolve()

        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    runs = sorted([path for path in artifacts_root.glob("*") if path.is_dir()])
    if not runs:
        raise FileNotFoundError(f"No runs found under: {artifacts_root}")

    for run in reversed(runs):
        if (run / "summary.json").exists():
            return run.resolve()

    raise FileNotFoundError("No completed runs found (missing summary.json).")


def render_table(summary: dict[str, object]) -> str:
    rounds = summary.get("rounds", [])
    if not isinstance(rounds, list) or not rounds:
        raise ValueError("Summary has no rounds.")

    metric_keys = summary.get("metrics_keys", [])
    if not isinstance(metric_keys, list) or not metric_keys:
        raise ValueError("Summary has no metrics_keys.")

    headers = ["Round", *[str(metric) for metric in metric_keys]]
    header_row = "| " + " | ".join(headers) + " |"
    separator_row = "|" + "|".join(["---:" for _ in headers]) + "|"

    rows = [header_row, separator_row]
    for round_row in rounds:
        if not isinstance(round_row, dict):
            continue
        metrics = round_row.get("metrics", {})
        values = [str(round_row.get("round", "?"))]
        for metric_key in metric_keys:
            value = float(metrics.get(metric_key, 0.0)) if isinstance(metrics, dict) else 0.0
            values.append(f"{value:.2f}")
        rows.append("| " + " | ".join(values) + " |")

    improvement = summary.get("improvement", {})
    rows.append("")
    rows.append("Improvement:")
    for metric_key in metric_keys:
        value = float(improvement.get(metric_key, 0.0)) if isinstance(improvement, dict) else 0.0
        rows.append(f"- {metric_key}: {value:+.2f}")

    return "\n".join(rows)


def main() -> None:
    args = parse_args()
    run_dir = resolve_run_dir(args.artifacts_root, args.run_dir)
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    print(render_table(summary))


if __name__ == "__main__":
    main()
