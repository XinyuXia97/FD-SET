from __future__ import annotations

import json
from pathlib import Path

from core.failure_extract import extract_failures_from_log


def test_extract_failures_from_nested_json_payload(tmp_path: Path) -> None:
    log_path = tmp_path / "events.json"
    log_path.write_text(
        json.dumps(
            {
                "data": {
                    "records": [
                        {
                            "meta": {"route": "route_001", "timestamp": 12.5},
                            "event": {"type": "collision", "note": "cut-in vehicle"},
                            "ego": {"speed": 7.2},
                            "scene": {"tags": "urban,merge,dense_traffic"},
                        },
                        {
                            "meta": {"route": "route_002", "timestamp": 15.0},
                            "event": {"type": "info", "note": "ignored"},
                            "ego": {"speed": 5.0},
                            "scene": {"tags": "urban"},
                        },
                    ]
                }
            }
        ),
        encoding="utf-8",
    )

    failures = extract_failures_from_log(
        log_path,
        {
            "format": "json",
            "records_path": "data.records",
            "accepted_failure_types": ["collision", "offroad"],
            "field_map": {
                "route_id": "meta.route",
                "timestamp_s": "meta.timestamp",
                "event_type": "event.type",
                "ego_speed": "ego.speed",
                "description": "event.note",
                "scene_tags": "scene.tags",
            },
            "scene_tags_separator": ",",
        },
    )

    assert len(failures) == 1
    assert failures[0].route_id == "route_001"
    assert failures[0].scene_tags == ["urban", "merge", "dense_traffic"]
    assert failures[0].risk_level == "high"


def test_extract_failures_from_bench2drive_checkpoint(tmp_path: Path) -> None:
    log_path = tmp_path / "ckpt.json"
    metric_info_dir = tmp_path / "metric_info_root" / "route_001"
    metric_info_dir.mkdir(parents=True)
    (metric_info_dir / "frame_000.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    (metric_info_dir / "frame_001.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    (metric_info_dir / "metric_info.json").write_text(
        json.dumps(
            {
                "0": {
                    "location": [0.0, 0.0, 0.0],
                    "acceleration": [0.0, 0.0, 0.0],
                    "angular_velocity": [0.0, 0.0, 0.0],
                },
                "1": {
                    "location": [1.0, 0.0, 0.0],
                    "acceleration": [0.5, 0.0, 0.0],
                    "angular_velocity": [0.0, 0.0, 0.1],
                },
            }
        ),
        encoding="utf-8",
    )
    log_path.write_text(
        json.dumps(
            {
                "_checkpoint": {
                    "records": [
                        {
                            "route_id": "RouteScenario_001_rep0",
                            "scenario_name": "HighwayCutIn",
                            "town_name": "Town04",
                            "save_name": "route_001",
                            "status": "Failed - Agent timed out",
                            "meta": {"duration_game": 42.0},
                            "infractions": {
                                "collisions_vehicle": ["Collision with vehicle"],
                                "red_light": [],
                                "route_timeout": ["Route timeout."],
                            },
                        }
                    ]
                }
            }
        ),
        encoding="utf-8",
    )

    failures = extract_failures_from_log(
        log_path,
        {
            "format": "bench2drive_checkpoint",
            "records_path": "_checkpoint.records",
            "metric_info_root": str(tmp_path / "metric_info_root"),
            "frame_glob": "frame_*.png",
            "frame_limit": 2,
        },
    )

    assert len(failures) == 2
    assert failures[0].failure_type == "collision"
    assert failures[0].scene_tags[:3] == ["bench2drive", "HighwayCutIn", "Town04"]
    assert failures[0].metadata is not None
    assert failures[0].metadata["context_summary"]["path_length_m"] == 1.0
    assert len(failures[0].metadata["context_summary"]["evidence_frames"]) == 2
    assert failures[1].failure_type == "timeout"


def test_bench2drive_front_camera_frames_are_localized_to_failure_time(tmp_path: Path) -> None:
    evaluation_dir = tmp_path / "benchmark_evaluation"
    frame_dir = evaluation_dir / "evidence" / "scene_1711" / "rgb"
    frame_dir.mkdir(parents=True)
    for frame in (101, 3951, 3981, 3991):
        (frame_dir / f"{frame:04}.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    checkpoint = evaluation_dir / "ckpt.json"
    checkpoint.write_text(json.dumps({
        "_checkpoint": {"records": [{
            "route_id": "RouteScenario_1711_rep0",
            "save_name": "scene_1711",
            "meta": {"duration_game": 200.0},
            "infractions": {"collisions_vehicle": ["Collision with vehicle"]},
        }]},
    }), encoding="utf-8")

    failures = extract_failures_from_log(checkpoint, {
        "format": "bench2drive_checkpoint",
        "__log_file__": str(checkpoint),
        "frame_dir_template": "{log_dir}/evidence/{save_name}/rgb",
        "frame_rate": 20.0,
        "frame_limit": 2,
        "evidence_window_before": 2,
        "evidence_window_after": 0,
    })
    frames = failures[0].metadata["context_summary"]["evidence_frames"]
    assert [Path(frame).stem for frame in frames] == ["3951", "3991"]
