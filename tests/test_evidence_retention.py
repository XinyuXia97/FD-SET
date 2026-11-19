from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.evidence_retention import compact_new_route_evidence
from core.failure_extract import extract_failures_from_log


def _checkpoint(path: Path, save_name: str) -> dict:
    record = {
        "route_id": "RouteScenario_12_rep0",
        "save_name": save_name,
        "scenario_name": "CrossingBicycleFlow",
        "town_name": "Town12",
        "status": "Failed - TickRuntime",
        "infractions": {"collisions_vehicle": ["vehicle collision"]},
        "scores": {"score_composed": 19.0, "score_route": 19.0, "score_penalty": 1.0},
        "meta": {"duration_game": 0.8, "failure_frame": 8},
    }
    path.write_text(json.dumps({"_checkpoint": {"progress": [1, 1], "records": [record]}}))
    return record


def test_compact_new_evidence_keeps_vlm_frames_motion_data_and_older_route(tmp_path):
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    name = "RouteScenario_12_rep0_Town12_CrossingBicycleFlow_01"
    older = evidence / "RouteScenario_11_rep0_Town12_older"
    (older / "rgb").mkdir(parents=True)
    (older / "rgb" / "0001.png").write_bytes(b"prior original frame")
    route = evidence / name
    for folder in ("rgb", "bev", "meta"):
        (route / folder).mkdir(parents=True)
    trajectory = route / "metric_info.json"
    trajectory.write_text(json.dumps({"0001": {"location": [0, 0, 0], "acceleration": [0, 0, 0], "angular_velocity": [0, 0, 0]}}))
    for frame in range(1, 13):
        stem = f"{frame:04d}"
        (route / "rgb" / f"{stem}.png").write_bytes(f"rgb {stem}".encode())
        (route / "bev" / f"{stem}.png").write_bytes(f"bev {stem}".encode())
        (route / "meta" / f"{stem}.json").write_text("{}")
    checkpoint = _checkpoint(tmp_path / "route.json", name)
    config = {"format": "bench2drive_checkpoint", "frame_limit": 8, "frame_rate": 10.0,
              "evidence_window_before": 4, "evidence_window_after": 4,
              "frame_dir_template": "{log_dir}/evidence/{save_name}/rgb"}
    before_config = {**config, "frame_dir_template": str(evidence / "{save_name}" / "rgb"),
                     "metric_info_root": str(evidence)}
    failures = extract_failures_from_log(tmp_path / "route.json", before_config)
    consumed_before = failures[0].metadata["context_summary"]["evidence_frames"][:3]

    receipt = compact_new_route_evidence(
        evidence_root=evidence, route_id="12", checkpoint=tmp_path / "route.json",
        record=checkpoint, failure_config=config, preexisting_names={older.name}, max_images=3,
    )
    assert receipt[0]["removed_frame_files"] > 20
    assert trajectory.exists()
    assert (older / "rgb" / "0001.png").read_bytes() == b"prior original frame"
    assert sorted(str(path) for path in (route / "rgb").iterdir()) == sorted(consumed_before)
    assert len(list((route / "bev").iterdir())) == 0
    assert len(list((route / "meta").iterdir())) == len(consumed_before)
    after = extract_failures_from_log(tmp_path / "route.json", before_config)
    assert after[0].metadata["context_summary"]["evidence_frames"] == consumed_before


def test_missing_motion_info_stops_without_erasing_frames(tmp_path):
    evidence = tmp_path / "evidence"
    route = evidence / "RouteScenario_12_rep0_Town12_broken"
    (route / "rgb").mkdir(parents=True)
    frame = route / "rgb" / "0001.png"
    frame.write_bytes(b"only failure image")
    checkpoint = _checkpoint(tmp_path / "route.json", route.name)
    with pytest.raises(RuntimeError, match="lacks official motion trajectory"):
        compact_new_route_evidence(
            evidence_root=evidence, route_id="12", checkpoint=tmp_path / "route.json",
            record=checkpoint, failure_config={"format": "bench2drive_checkpoint"}, preexisting_names=set(),
        )
    assert frame.read_bytes() == b"only failure image"
