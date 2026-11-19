from __future__ import annotations

import gzip
import json
from pathlib import Path

from scripts.audit_scenic_bank import _audit_output, _valid_windows


def _run(tmp_path: Path, family: str, measurements: list[dict], boxes: list[list[dict]],
         *, execution_status: str = "completed", collisions: int = 0, program_id: str = "sample") -> dict:
    output = tmp_path / "run"
    pdm = output / "pdm_lite"
    route = pdm / "pdm_lite_dataset" / "FDSET_SCENIC_Rep0_0"
    for folder in ("measurements", "boxes"):
        (route / folder).mkdir(parents=True)
    for frame, (measurement, actors) in enumerate(zip(measurements, boxes)):
        for folder, payload in (("measurements", measurement), ("boxes", actors)):
            with gzip.open(route / folder / f"{frame:04d}.json.gz", "wt", encoding="utf-8") as stream:
                json.dump(payload, stream)
    (pdm / "native_scenic_policy_results.json").write_text(json.dumps({
        "programs": [{"status": execution_status, "steps": len(measurements) * 5, "collisions": collisions}],
    }), encoding="utf-8")
    return _audit_output(
        pdm, {"program_id": program_id, "interaction_family": family}, 2026, [],
        0 if execution_status == "completed" else 1,
        tmp_path / "collector.log", tmp_path / "manifest.json",
    )


def _measurement(frame: int, *, command: int = 4, light: bool = False,
                 speed: float = 5.0, reduced: str | None = None) -> dict:
    return {
        "timestamp_s": frame * 0.5, "command": command, "next_command": 1,
        "pos_global": [frame * 2.5, 0.0], "theta": 0.0, "speed": speed,
        "acceleration": [0.0, 0.0, 0.0], "light_hazard": light,
        "speed_reduced_by_obj_type": reduced,
    }


def test_navigation_requires_the_sampled_near_command_not_distant_command(tmp_path: Path) -> None:
    samples = [_measurement(i, command=2) for i in range(11)]
    wrong = _run(tmp_path / "wrong", "left_turn_navigation", samples, [[] for _ in samples])
    assert wrong["status"] == "completed"
    assert wrong["interaction_status"] == "unobserved"
    assert wrong["trainable_contiguous_0_5s_windows"] == 1
    samples[5]["command"] = 1
    matching = _run(tmp_path / "matching", "left_turn_navigation", samples, [[] for _ in samples])
    assert matching["interaction_status"] == "navigation_command_observed"
    samples[5]["timestamp_s"] += 0.3
    assert _valid_windows(samples) == 0

def test_straight_connector_requires_measured_travel_without_a_turn(tmp_path: Path) -> None:
    samples = [_measurement(i) for i in range(16)]
    straight = _run(tmp_path / "straight", "straight_junction_navigation", samples, [[] for _ in samples])
    assert straight["interaction_status"] == "observed"
    assert straight["navigation_command_observable"] is False
    assert straight["ego_start_to_end_displacement_m"] == 37.5

    too_short = [_measurement(i) for i in range(8)]
    assert _run(tmp_path / "short", "straight_junction_navigation", too_short,
                [[] for _ in too_short])["interaction_status"] == "unobserved"

    samples[-1]["theta"] = 1.2
    assert _run(tmp_path / "turned", "straight_junction_navigation", samples,
                [[] for _ in samples])["interaction_status"] == "unobserved"


def test_signal_requires_both_red_stop_and_green_departure(tmp_path: Path) -> None:
    boxes = [[{"class": "traffic_light", "id": 19, "affects_ego": True}] for _ in range(4)]
    red_only = [_measurement(i, light=True, speed=0.0) for i in range(4)]
    assert _run(tmp_path / "red", "signal_stop_restart", red_only, boxes)["interaction_status"] == "unobserved"
    released = red_only[:2] + [_measurement(i, light=False, speed=4.0) for i in (2, 3)]
    assert _run(tmp_path / "released", "signal_stop_restart", released, boxes)["interaction_status"] == "observed"
    wrong_order = [_measurement(0, speed=4.0), _measurement(1, speed=4.0)] + red_only[2:]
    assert _run(tmp_path / "wrong_order", "signal_stop_restart", wrong_order, boxes)["interaction_status"] == "unobserved"
    assert _run(tmp_path / "not_green", "green_light_start", red_only, boxes)["interaction_status"] == "unobserved"

def test_green_start_reports_raw_windows_but_excludes_stop_sign_contamination(tmp_path: Path) -> None:
    samples = [_measurement(i) for i in range(16)]
    samples[5]["stop_sign_hazard"] = True
    result = _run(tmp_path, "green_light_start", samples, [[] for _ in samples],
                  program_id="fdset_green_light_start")
    assert result["measurement_contiguous_0_5s_windows"] == 6
    assert result["trainable_contiguous_0_5s_windows"] == 0
    assert result["hazard_frame_counts"]["stop_sign"] == 1


def test_narrow_passage_requires_passage_without_collision_and_windows_exclude_crashes(tmp_path: Path) -> None:
    samples = [_measurement(i) for i in range(16)]
    obstacle = [{"class": "static", "id": 7, "type_id": "static.prop.bin",
                 "distance": 12.0, "matrix": [[1, 0, 0, 30], [0, 1, 0, 0], [0, 0, 1, 0]]}]
    boxes = [obstacle for _ in samples]
    passed = _run(tmp_path / "passed", "narrow_passage", samples, boxes)
    assert passed["interaction_status"] == "observed"
    assert passed["trainable_contiguous_0_5s_windows"] == 6

    crashed = _run(tmp_path / "crashed", "narrow_passage", samples, boxes, collisions=2)
    assert crashed["interaction_status"] == "unobserved"
    assert crashed["measurement_contiguous_0_5s_windows"] == 6
    assert crashed["trainable_contiguous_0_5s_windows"] == 0

    short = [_measurement(i) for i in range(6)]
    assert _run(tmp_path / "short", "narrow_passage", short,
                [obstacle for _ in short])["interaction_status"] == "unobserved"


def test_bicycle_requires_world_motion_near_ego_and_an_expert_response(tmp_path: Path) -> None:
    bike = "vehicle.gazelle.omafiets"
    measures = [_measurement(i, reduced=bike if i >= 2 else None) for i in range(4)]

    def cyclist(x: float) -> list[dict]:
        return [{"class": "ego_car", "id": 1, "type_id": "", "distance": -1,
                 "speed": 0.0, "matrix": [[1, 0, 0, x * 10], [0, 1, 0, 0.0], [0, 0, 1, 0.0]]},
                {"class": "car", "id": 7, "type_id": bike, "distance": 12.0,
                 "speed": 3.0, "matrix": [[1, 0, 0, x], [0, 1, 0, 0.0], [0, 0, 1, 0.0]]}]

    stationary = [cyclist(0.0) for _ in measures]
    assert _run(tmp_path / "stationary", "bicycle_crossing", measures, stationary)["interaction_status"] == "unobserved"
    moving = [cyclist(i * 1.5) for i in range(len(measures))]
    observed = _run(tmp_path / "moving", "bicycle_crossing", measures, moving)
    assert observed["interaction_status"] == "observed"
    assert observed["actor_box_speed_displacement_mae_mps_by_id"]["7"] == 0.0
    assert observed["actor_box_total_displacement_m"] == 4.5
    assert set(observed["actor_box_speed_displacement_mae_mps_by_id"]) == {"7"}
    failed = _run(tmp_path / "failed", "bicycle_crossing", measures, moving, execution_status="error")
    assert failed["status"] == "failed"
    assert failed["interaction_status"] == "unknown_due_to_run_failure"
