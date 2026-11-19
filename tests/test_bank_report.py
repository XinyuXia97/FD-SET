from __future__ import annotations

from reports.make_bank_report import render_bank_report


def test_render_bank_report_summarizes_patterns_and_tags() -> None:
    rows = [
        {
            "scenario_id": "a",
            "tags": ["bench2drive", "Town12", "HighwayCutIn"],
            "pattern": "adversarial_merge",
            "config": {"bench2drive_spec": {}},
        },
        {
            "scenario_id": "b",
            "tags": ["bench2drive", "Town12", "PedestrianCrossing"],
            "pattern": "pedestrian_intrusion",
            "config": {},
        },
    ]

    report = render_bank_report(rows)

    assert "Scenario Count: 2" in report
    assert "Bench2Drive-Spec Count: 1" in report
    assert "- adversarial_merge: 1" in report
    assert "- bench2drive: 2" in report
