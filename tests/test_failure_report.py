from __future__ import annotations

import json
from pathlib import Path

from reports.make_failure_report import render_failure_report


def test_render_failure_report_includes_breakdowns(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_001"
    run_dir.mkdir(parents=True)
    (run_dir / "summary.json").write_text(
        json.dumps(
            {
                "rounds": [
                    {
                        "round": 0,
                        "num_failures": 2,
                        "num_structured": 2,
                        "num_evolved": 3,
                        "num_unique_anchor_sources": 2,
                        "avg_anchor_score": 4.5,
                        "scenario_bank_size_before": 3,
                        "scenario_bank_size_after": 6,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "failures_raw_all.json").write_text(
        json.dumps(
            [
                {"failure_type": "collision"},
                {"failure_type": "collision"},
                {"failure_type": "timeout"},
            ]
        ),
        encoding="utf-8",
    )
    (run_dir / "failure_db.json").write_text(
        json.dumps(
            [
                {"interaction_pattern": "adversarial_merge"},
                {"interaction_pattern": "late_response"},
                {"interaction_pattern": "adversarial_merge"},
            ]
        ),
        encoding="utf-8",
    )
    (run_dir / "evolved_scenarios_all.json").write_text(
        json.dumps(
            [
                {"source_scenario_id": "seed_merge_01"},
                {"source_scenario_id": "seed_merge_01"},
                {"source_scenario_id": "seed_static_03"},
            ]
        ),
        encoding="utf-8",
    )

    report = render_failure_report(run_dir)

    assert "Failure Types:" in report
    assert "- collision: 2" in report
    assert "Interaction Patterns:" in report
    assert "- adversarial_merge: 2" in report
    assert "Top Anchor Sources:" in report
    assert "- seed_merge_01: 2" in report
