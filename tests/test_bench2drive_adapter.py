from __future__ import annotations

import json
from pathlib import Path
import pytest

from core.bench2drive_adapter import calculate_official_motion_metrics, parse_metrics


def test_parse_metrics_from_bench2drive_checkpoint(tmp_path: Path) -> None:
    metrics_path = tmp_path / "ckpt.json"
    metrics_path.write_text(
        json.dumps(
            {
                "_checkpoint": {
                    "global_record": {
                        "scores_mean": {
                            "score_composed": 34.2,
                            "score_route": 50.0,
                            "score_penalty": 0.7,
                        }
                    },
                    "records": [
                        {
                            "status": "Perfect",
                            "infractions": {
                                "collisions_vehicle": [],
                                "collisions_pedestrian": [],
                                "collisions_layout": [],
                                "min_speed_infractions": [],
                            },
                        },
                        {
                            "status": "Failed - Agent timed out",
                            "infractions": {
                                "collisions_vehicle": ["hit vehicle"],
                                "collisions_pedestrian": [],
                                "collisions_layout": [],
                                "min_speed_infractions": [],
                            },
                        },
                    ],
                }
            }
        ),
        encoding="utf-8",
    )

    metrics = parse_metrics(metrics_path, keys=["DS", "SR", "CR"])

    assert metrics["DS"] == 34.2
    assert metrics["SR"] == 50.0
    assert metrics["CR"] == 50.0


def test_strict_official_metrics_reject_missing_efficiency_and_comfort(tmp_path: Path) -> None:
    metrics_path = tmp_path / "ckpt.json"
    metrics_path.write_text(
        json.dumps({
            "_checkpoint": {
                "global_record": {"scores_mean": {"score_composed": 34.2}},
                "records": [{"status": "Perfect", "infractions": {}}],
            }
        }),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Eff, Comf"):
        parse_metrics(metrics_path, keys=["DS", "SR", "Eff", "Comf", "CR"], strict=True)


def test_official_motion_metric_tool_reads_recorded_agent_motion(tmp_path: Path) -> None:
    bench_root = Path("../Bench2Drive")
    if not (bench_root / "tools" / "efficiency_smoothness_benchmark.py").is_file():
        pytest.skip("Official Bench2Drive metric tool is not installed")
    route_name = "RouteScenario_1_rep0"
    evidence_root = tmp_path / "evidence"
    route_dir = evidence_root / route_name
    route_dir.mkdir(parents=True)
    stationary = {
        "acceleration": [0.0, 0.0, 0.0],
        "angular_velocity": [0.0, 0.0, 0.0],
        "forward_vector": [1.0, 0.0, 0.0],
        "right_vector": [0.0, 1.0, 0.0],
        "location": [0.0, 0.0, 0.0],
        "rotation": [0.0, 0.0, 0.0],
    }
    (route_dir / "metric_info.json").write_text(
        json.dumps({str(index): stationary for index in range(20)}), encoding="utf-8",
    )
    checkpoint = tmp_path / "ckpt.json"
    checkpoint.write_text(
        json.dumps({"_checkpoint": {"records": [{
            "save_name": route_name,
            "infractions": {"min_speed_infractions": [
                "Average speed is 40% of traffic", "Average speed is 80% of traffic",
            ]},
        }]}}), encoding="utf-8",
    )
    assert calculate_official_motion_metrics(checkpoint, evidence_root, bench_root) == {
        "Eff": 60.0, "Comf": 100.0,
    }
