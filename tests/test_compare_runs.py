from __future__ import annotations

import json
from pathlib import Path

from reports.compare_runs import render_run_comparison


def test_render_run_comparison_includes_metric_and_failure_deltas(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    baseline.mkdir()
    candidate.mkdir()

    baseline_summary = {
        "metrics_keys": ["DS", "SR"],
        "rounds": [{"round": 0, "metrics": {"DS": 34.0, "SR": 12.0}}],
    }
    candidate_summary = {
        "metrics_keys": ["DS", "SR"],
        "rounds": [{"round": 0, "metrics": {"DS": 36.0, "SR": 13.0}}],
    }

    (baseline / "summary.json").write_text(json.dumps(baseline_summary), encoding="utf-8")
    (candidate / "summary.json").write_text(json.dumps(candidate_summary), encoding="utf-8")
    (baseline / "failures_raw_all.json").write_text(json.dumps([{"failure_type": "collision"}]), encoding="utf-8")
    (candidate / "failures_raw_all.json").write_text(json.dumps([{"failure_type": "timeout"}]), encoding="utf-8")
    (baseline / "failure_db.json").write_text(json.dumps([{"interaction_pattern": "late_response"}]), encoding="utf-8")
    (candidate / "failure_db.json").write_text(json.dumps([{"interaction_pattern": "adversarial_merge"}]), encoding="utf-8")

    report = render_run_comparison(baseline, candidate)

    assert "- DS: 34.00 -> 36.00 (+2.00)" in report
    assert "- collision: 1 -> 0 (-1)" in report
    assert "- timeout: 0 -> 1 (+1)" in report
