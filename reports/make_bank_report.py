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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize an FD-SET scenario bank.")
    parser.add_argument("--scenario-bank", type=Path, required=True)
    return parser.parse_args()


def render_bank_report(rows: list[dict[str, Any]]) -> str:
    pattern_counter: Counter[str] = Counter()
    tag_counter: Counter[str] = Counter()
    route_spec_count = 0

    for row in rows:
        if not isinstance(row, dict):
            continue
        pattern_counter[str(row.get("pattern", "unknown"))] += 1
        tags = row.get("tags", [])
        if isinstance(tags, list):
            tag_counter.update(str(tag) for tag in tags)
        config = row.get("config", {})
        if isinstance(config, dict) and isinstance(config.get("bench2drive_spec"), dict):
            route_spec_count += 1

    lines = [
        f"Scenario Count: {len(rows)}",
        f"Bench2Drive-Spec Count: {route_spec_count}",
        "",
        "Patterns:",
    ]
    lines.extend(_render_counter(pattern_counter))
    lines.append("")
    lines.append("Top Tags:")
    lines.extend(_render_counter(tag_counter, limit=10))
    return "\n".join(lines)


def _render_counter(counter: Counter[str], *, limit: int | None = None) -> list[str]:
    items = counter.most_common(limit)
    if not items:
        return ["- None"]
    return [f"- {key}: {value}" for key, value in items]


def main() -> None:
    args = parse_args()
    rows = json.loads(args.scenario_bank.read_text(encoding="utf-8"))
    print(render_bank_report(rows))


if __name__ == "__main__":
    main()
