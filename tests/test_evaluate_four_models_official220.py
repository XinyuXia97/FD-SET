from __future__ import annotations

import json
import threading
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from scripts.evaluate_four_models_official220 import (
    CRASH_TRUNCATED_STATUS, TIMEOUT_SCORED_STATUS, UNSCORED_CRASH_STATUS, UNSTARTED_STATUS,
    accept_crash_snapshot, accept_timeout_snapshot, archive_vad_adapted_for_live_timeouts,
    checkpoint_progress, coverage, mark_checkpoint_ineligible, prune_completed_evidence,
    skip_unscored_crash, skip_unstarted_route,
)


def _checkpoint(path: Path, route_name: str, save_name: str) -> None:
    path.write_text(json.dumps({"_checkpoint": {"progress": [1, 2],
        "records": [{"route_id": route_name, "save_name": save_name}]}}), encoding="utf-8")


def test_cleanup_preserves_scored_motion_data_and_unscored_route_images(tmp_path: Path) -> None:
    checkpoint = tmp_path / "ckpt.json"
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    scored = "RouteScenario_1711_rep0_Town12_Test_1"
    interrupted = "RouteScenario_1712_rep0_Town12_Test_2"
    _checkpoint(checkpoint, "RouteScenario_1711_rep0", scored)
    for name in (scored, interrupted):
        route = evidence / name
        (route / "rgb_front").mkdir(parents=True)
        (route / "rgb_front" / "0001.jpg").write_bytes(b"image")
        (route / "bev").mkdir()
        (route / "bev" / "0001.png").write_bytes(b"image")
        (route / "metric_info.json").write_text('{"0": {}}', encoding="utf-8")
    (evidence / scored / "unrelated").mkdir()
    (evidence / scored / "unrelated" / "keep.txt").write_text("keep", encoding="utf-8")
    prune_completed_evidence(checkpoint, evidence, "uniad", {"1711", "1712"})
    assert (evidence / scored / "metric_info.json").is_file()
    assert not (evidence / scored / "rgb_front").exists()
    assert not (evidence / scored / "bev").exists()
    assert (evidence / scored / "unrelated" / "keep.txt").read_text() == "keep"
    assert (evidence / interrupted / "rgb_front" / "0001.jpg").is_file()
    assert coverage(checkpoint, {"1711", "1712"}) == (1, False)


def test_vad_unmodified_agent_prefix_keeps_official_motion_tool_path(tmp_path: Path) -> None:
    checkpoint = tmp_path / "ckpt.json"
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    name = "RouteScenario_1711_rep0_Town12_Test_1"
    _checkpoint(checkpoint, "RouteScenario_1711_rep0", name)
    actual = evidence / ("bench2drive220_" + name)
    (actual / "rgb_back").mkdir(parents=True)
    (actual / "rgb_back" / "0001.jpg").write_bytes(b"image")
    (actual / "metric_info.json").write_text('{"measured": true}', encoding="utf-8")
    prune_completed_evidence(checkpoint, evidence, "vad", {"1711", "1712"})
    assert json.loads((evidence / name / "metric_info.json").read_text()) == {"measured": True}
    assert not (actual / "rgb_back").exists()
    assert (actual / "metric_info.json").is_file()


def test_complete_coverage_requires_finished_scored_routes(tmp_path: Path) -> None:
    checkpoint = tmp_path / "ckpt.json"
    route_ids = {"1711", "1712"}
    payload = {
        "entry_status": "Finished",
        "_checkpoint": {
            "progress": [2, 2],
            "records": [
                {"route_id": "RouteScenario_1711_rep0", "status": "Completed"},
                {"route_id": "RouteScenario_1712_rep0", "status": "Failed - TickRuntime"},
            ],
        },
    }

    def result() -> tuple[int, bool]:
        checkpoint.write_text(json.dumps(payload), encoding="utf-8")
        return coverage(checkpoint, route_ids)

    assert result() == (2, True)  # A scored route timeout is not an execution crash.
    payload["entry_status"] = "Started"
    assert result() == (2, False)
    payload["entry_status"] = "Finished"
    payload["_checkpoint"]["records"][1]["status"] = "Failed - Agent crashed"
    assert result() == (2, False)
    payload["_checkpoint"]["records"][1]["status"] = "Completed"
    payload["_checkpoint"]["progress"] = [1, 2]
    assert result() == (2, False)


def test_crash_point_uses_live_criteria_and_is_not_an_official_completion(tmp_path: Path) -> None:
    checkpoint = tmp_path / "ckpt.json"
    snapshot = tmp_path / "live.json"
    evidence = tmp_path / "evidence"
    save_name = "RouteScenario_3785_rep0_Town13_MergerIntoSlowTraffic_1_10"
    (evidence / save_name).mkdir(parents=True)
    (evidence / save_name / "metric_info.json").write_text('{"1971": {}}', encoding="utf-8")
    checkpoint.write_text(json.dumps({
        "entry_status": "Started", "eligible": False,
        "_checkpoint": {"progress": [1, 3], "global_record": {}, "records": [
            {"route_id": "RouteScenario_1711_rep0", "status": "Completed"},
        ]},
    }), encoding="utf-8")
    measured = {
        "index": 1, "route_id": "RouteScenario_3785_rep0", "save_name": save_name,
        "status": "Failed", "num_infractions": 1,
        "infractions": {"collisions_vehicle": ["observed collision"], "red_light": []},
        "scores": {"score_route": 43.21, "score_penalty": 0.6, "score_composed": 25.926},
        "meta": {"route_length": 200.0, "duration_game": 98.7, "duration_system": 355.5},
    }
    snapshot.write_text(json.dumps(measured), encoding="utf-8")

    accepted = accept_crash_snapshot(
        checkpoint, snapshot, evidence, ["1711", "3785", "3800"], "3785", 139)
    payload = json.loads(checkpoint.read_text(encoding="utf-8"))
    assert accepted["status"] == CRASH_TRUNCATED_STATUS
    assert accepted["scores"] == measured["scores"]
    assert accepted["infractions"] == measured["infractions"]
    assert payload["_checkpoint"]["progress"] == [2, 3]
    assert payload["_checkpoint"]["records"][1] == accepted
    assert payload["eligible"] is False
    assert json.loads((tmp_path / "crash_truncation_3785.json").read_text())["record"] == accepted
    second_name = "RouteScenario_3800_rep0_Town13_MergerIntoSlowTraffic_1_25"
    (evidence / second_name).mkdir()
    (evidence / second_name / "metric_info.json").write_text('{"1000": {}}', encoding="utf-8")
    second = {**measured, "index": 2, "route_id": "RouteScenario_3800_rep0",
              "save_name": second_name,
              "scores": {"score_route": 5.52, "score_penalty": 1.0, "score_composed": 5.52}}
    snapshot.write_text(json.dumps(second), encoding="utf-8")
    accept_crash_snapshot(checkpoint, snapshot, evidence, ["1711", "3785", "3800"], "3800", 139)
    payload = json.loads(checkpoint.read_text(encoding="utf-8"))
    assert payload["_checkpoint"]["progress"] == [3, 3]
    assert payload["_checkpoint"]["records"][2]["scores"] == second["scores"]
    payload["entry_status"] = "Finished"
    payload["eligible"] = True
    checkpoint.write_text(json.dumps(payload), encoding="utf-8")
    mark_checkpoint_ineligible(checkpoint)
    assert json.loads(checkpoint.read_text(encoding="utf-8"))["eligible"] is False
    assert coverage(checkpoint, {"1711", "3785", "3800"}) == (3, False)
    assert coverage(checkpoint, {"1711", "3785", "3800"}, allow_non_official=True) == (3, True)
    payload["_checkpoint"]["records"][2]["status"] = "Failed - Agent crashed"
    checkpoint.write_text(json.dumps(payload), encoding="utf-8")
    assert coverage(checkpoint, {"1711", "3785", "3800"}, allow_non_official=True) == (3, False)


def test_crash_point_rejects_wrong_route_without_advancing_checkpoint(tmp_path: Path) -> None:
    checkpoint = tmp_path / "ckpt.json"
    snapshot = tmp_path / "live.json"
    checkpoint.write_text(json.dumps({"_checkpoint": {"progress": [0, 1], "records": []}}),
                          encoding="utf-8")
    snapshot.write_text(json.dumps({"route_id": "RouteScenario_1711_rep0"}), encoding="utf-8")
    original = checkpoint.read_bytes()

    import pytest
    with pytest.raises(ValueError, match="genuine live criteria"):
        accept_crash_snapshot(checkpoint, snapshot, tmp_path, ["3785"], "3785", 139)
    assert checkpoint.read_bytes() == original


def test_unstarted_skip_assigns_audited_policy_zero_and_preserves_resume_order(tmp_path: Path) -> None:
    checkpoint = tmp_path / "ckpt.json"
    snapshot = tmp_path / "crash_route_snapshot.json"
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    routes = ET.fromstring("""<routes>
        <route id="1711" town="Town01"><scenarios><scenario name="First"/></scenarios></route>
        <route id="24206" town="Town03"><scenarios><scenario name="Pedestrian"/></scenarios></route>
        <route id="24211" town="Town01"><scenarios><scenario name="Next"/></scenarios></route>
    </routes>""").findall("route")
    checkpoint.write_text(json.dumps({
        "entry_status": "Started", "eligible": True,
        "_checkpoint": {"progress": [1, 3], "records": [
            {"route_id": "RouteScenario_1711_rep0", "status": "Completed",
             "infractions": {"collisions_vehicle": [], "route_timeout": []}},
        ]},
    }), encoding="utf-8")
    original = checkpoint.read_bytes()
    with pytest.raises(ValueError, match="next route"):
        skip_unstarted_route(checkpoint, snapshot, evidence, routes, "24211", "wrong route")
    assert checkpoint.read_bytes() == original
    (evidence / "RouteScenario_24206_rep0_Town03_Pedestrian_1").mkdir()
    with pytest.raises(ValueError, match="has evidence"):
        skip_unstarted_route(checkpoint, snapshot, evidence, routes, "24206", "has data")
    assert checkpoint.read_bytes() == original
    (evidence / "RouteScenario_24206_rep0_Town03_Pedestrian_1").rmdir()
    record = skip_unstarted_route(checkpoint, snapshot, evidence, routes, "24206", "no world tick")
    payload = json.loads(checkpoint.read_text(encoding="utf-8"))
    assert payload["_checkpoint"]["progress"] == [2, 3]
    assert payload["_checkpoint"]["records"][1] == record
    assert record["status"] == UNSTARTED_STATUS
    assert record["scores"] == {"score_route": 0, "score_penalty": 1, "score_composed": 0}
    assert record["infractions"] == {"collisions_vehicle": [], "route_timeout": []}
    assert payload["eligible"] is False
    audit = json.loads((tmp_path / "skipped_unstarted_24206.json").read_text())
    assert audit["score_source"].startswith("explicit policy zero")
    assert audit["motion_metrics_available"] is False
    assert audit["record"] == record
    assert not (evidence / record["save_name"] / "metric_info.json").exists()
    with pytest.raises(ValueError, match="next route"):
        skip_unstarted_route(checkpoint, snapshot, evidence, routes, "24206", "duplicate")
    next_record = skip_unstarted_route(checkpoint, snapshot, evidence, routes, "24211", "next failed")
    assert next_record["route_id"] == "RouteScenario_24211_rep0"
    assert json.loads(checkpoint.read_text())["_checkpoint"]["progress"] == [3, 3]
    payload = json.loads(checkpoint.read_text())
    payload["entry_status"] = "Finished"
    checkpoint.write_text(json.dumps(payload), encoding="utf-8")
    assert coverage(checkpoint, {"1711", "24206", "24211"}) == (3, False)
    assert coverage(checkpoint, {"1711", "24206", "24211"}, allow_non_official=True) == (3, True)


def test_vad_timeout_uses_matching_live_criteria_and_preserves_measured_infractions(tmp_path: Path) -> None:
    checkpoint = tmp_path / "ckpt.json"
    snapshot = tmp_path / "live.json"
    evidence = tmp_path / "evidence"
    name = "RouteScenario_23659_rep0_Town12_Obstacle_1"
    path = evidence / ("bench2drive220_" + name)
    path.mkdir(parents=True)
    checkpoint.write_text(json.dumps({"entry_status": "Started", "eligible": True,
        "_checkpoint": {"progress": [1, 2], "records": [
            {"route_id": "RouteScenario_1711_rep0", "status": "Completed"},
        ]},
    }), encoding="utf-8")
    measured = {
        "index": 1, "route_id": "RouteScenario_23659_rep0", "save_name": name,
        "num_infractions": 1, "infractions": {"outside_route_lanes": ["37.745%"]},
        "scores": {"score_route": 62.21, "score_penalty": 0.62255,
                   "score_composed": 38.728821},
        "meta": {"route_length": 200.0, "duration_game": 88.5, "duration_system": 61.0},
    }
    snapshot.write_text(json.dumps(measured), encoding="utf-8")
    original = checkpoint.read_bytes()
    route_ids = ["1711", "23659"]
    with pytest.raises(ValueError, match="elapsed wall-clock timeout"):
        accept_timeout_snapshot(checkpoint, snapshot, evidence, route_ids, "23659", 59, 60)
    with pytest.raises(ValueError, match="next route"):
        accept_timeout_snapshot(checkpoint, snapshot, evidence, route_ids, "1711", 61, 60)
    with pytest.raises(ValueError, match="genuine live criteria"):
        accept_timeout_snapshot(checkpoint, snapshot, evidence, route_ids, "23659", 61, 60)
    assert checkpoint.read_bytes() == original
    (path / "metric_info.json").write_text('{"3": {"speed": 1}}', encoding="utf-8")
    snapshot.write_text(json.dumps({**measured, "scores": {
        **measured["scores"], "score_composed": 38.73}}), encoding="utf-8")
    with pytest.raises(ValueError, match="genuine live criteria"):
        accept_timeout_snapshot(checkpoint, snapshot, evidence, route_ids, "23659", 61, 60)
    assert checkpoint.read_bytes() == original
    snapshot.write_text(json.dumps(measured), encoding="utf-8")
    row = accept_timeout_snapshot(checkpoint, snapshot, evidence, route_ids, "23659", 61, 60)
    assert row["status"] == TIMEOUT_SCORED_STATUS
    assert row["scores"] == measured["scores"]
    assert row["infractions"] == measured["infractions"]
    state = json.loads(checkpoint.read_text())
    assert state["_checkpoint"]["progress"] == [2, 2]
    assert state["eligible"] is False
    audit = json.loads((tmp_path / "timeout_partial_23659.json").read_text())
    assert audit["elapsed_wall_seconds"] == 61
    assert audit["record"] == row
    with pytest.raises(ValueError, match="next route"):
        accept_timeout_snapshot(checkpoint, snapshot, evidence, route_ids, "23659", 62, 60)
    state["entry_status"] = "Finished"
    checkpoint.write_text(json.dumps(state), encoding="utf-8")
    assert coverage(checkpoint, set(route_ids)) == (2, False)
    assert coverage(checkpoint, set(route_ids), allow_non_official=True) == (2, True)

def test_uniad_adapted_timeout_uses_measured_partial_distance_not_policy_zero(tmp_path: Path) -> None:
    checkpoint = tmp_path / "ckpt.json"
    snapshot = tmp_path / "live.json"
    evidence = tmp_path / "evidence"
    name = "RouteScenario_1711_rep0_Town12_ParkingCutIn_1"
    route = evidence / name
    route.mkdir(parents=True)
    checkpoint.write_text(json.dumps({"entry_status": "Started", "eligible": True,
        "_checkpoint": {"progress": [0, 1], "records": []}}), encoding="utf-8")
    measured = {
        "index": 0, "route_id": "RouteScenario_1711_rep0", "save_name": name,
        "num_infractions": 1, "infractions": {"outside_route_lanes": ["10%"]},
        "scores": {"score_route": 45.6, "score_penalty": 0.5, "score_composed": 22.8},
        "meta": {"route_length": 180, "duration_game": 41, "duration_system": 4200},
    }
    snapshot.write_text(json.dumps(measured), encoding="utf-8")
    original = checkpoint.read_bytes()
    with pytest.raises(ValueError, match="genuine live criteria"):
        accept_timeout_snapshot(checkpoint, snapshot, evidence, ["1711"], "1711",
                                14401, 14400, family="uniad")
    assert checkpoint.read_bytes() == original
    (route / "metric_info.json").write_text('{"measured": true}', encoding="utf-8")
    snapshot.write_text(json.dumps({**measured, "scores": {
        "score_route": 0, "score_penalty": 0.5, "score_composed": 0}}), encoding="utf-8")
    with pytest.raises(ValueError, match="no measured route completion"):
        accept_timeout_snapshot(checkpoint, snapshot, evidence, ["1711"], "1711",
                                14401, 14400, family="uniad")
    assert checkpoint.read_bytes() == original
    snapshot.write_text(json.dumps(measured), encoding="utf-8")
    row = accept_timeout_snapshot(checkpoint, snapshot, evidence, ["1711"], "1711",
                                  14401, 14400, family="uniad")
    assert row["scores"]["score_route"] == 45.6
    assert row["scores"]["score_composed"] == 22.8
    assert row["infractions"] == measured["infractions"]
    assert json.loads(checkpoint.read_text())["_checkpoint"]["progress"] == [1, 1]
    assert json.loads(checkpoint.read_text())["eligible"] is False

def test_vad_adapted_replay_archives_old_scores_without_touching_baseline(tmp_path: Path) -> None:
    vad = tmp_path / "vad"
    baseline = vad / "baseline"
    adapted = vad / "adapted"
    baseline.mkdir(parents=True)
    adapted.mkdir()
    completed = {"entry_status": "Finished", "_checkpoint": {
        "progress": [2, 2], "records": [
            {"route_id": "RouteScenario_1711_rep0", "status": "Completed"},
            {"route_id": "RouteScenario_23659_rep0", "status": "Completed"},
        ]}}
    (baseline / "ckpt.json").write_text(json.dumps(completed), encoding="utf-8")
    original_baseline = (baseline / "ckpt.json").read_bytes()
    previous = {"entry_status": "Started", "_checkpoint": {
        "progress": [2, 2], "records": [
            {"route_id": "RouteScenario_1711_rep0", "status": "Completed"},
            {"route_id": "RouteScenario_23659_rep0", "status":
             "Failed - Wall-timeout skipped (non-official)"},
        ]}}
    (adapted / "ckpt.json").write_text(json.dumps(previous), encoding="utf-8")
    archive = archive_vad_adapted_for_live_timeouts(tmp_path, {"1711", "23659"})
    assert json.loads((archive / "ckpt.json").read_text()) == previous
    assert not adapted.exists()
    assert (baseline / "ckpt.json").read_bytes() == original_baseline
    with pytest.raises(ValueError, match="Cannot safely archive"):
        archive_vad_adapted_for_live_timeouts(tmp_path, {"1711", "23659"})



def test_checkpoint_progress_waits_for_in_place_rewrite(tmp_path: Path) -> None:
    checkpoint = tmp_path / "ckpt.json"
    checkpoint.write_text('{"_checkpoint": {"progress": [151,', encoding="utf-8")
    writer = threading.Timer(
        0.05, lambda: checkpoint.write_text(
            json.dumps({"_checkpoint": {"progress": [152, 220]}}), encoding="utf-8"))
    writer.start()
    try:
        assert checkpoint_progress(checkpoint) == 152
    finally:
        writer.join()


def test_unscored_crash_skips_partial_images_but_never_real_scores(tmp_path: Path) -> None:
    checkpoint = tmp_path / "ckpt.json"
    snapshot = tmp_path / "snapshot.json"
    evidence = tmp_path / "evidence"
    partial = evidence / "RouteScenario_26966_rep0_Town05_Junction_1"
    partial.mkdir(parents=True)
    (partial / "rgb_front").mkdir()
    routes = ET.fromstring("""<routes>
        <route id="1711" town="Town01"><scenarios><scenario name="First"/></scenarios></route>
        <route id="26966" town="Town05"><scenarios><scenario name="Junction"/></scenarios></route>
    </routes>""").findall("route")
    checkpoint.write_text(json.dumps({"entry_status": "Started", "eligible": True,
        "_checkpoint": {"progress": [1, 2], "records": [
            {"route_id": "RouteScenario_1711_rep0", "status": "Completed",
             "infractions": {"collisions_vehicle": [], "route_timeout": []}},
        ]}}), encoding="utf-8")
    original = checkpoint.read_bytes()
    (partial / "metric_info.json").write_text('{"live": true}', encoding="utf-8")
    with pytest.raises(ValueError, match="genuine score evidence"):
        skip_unscored_crash(checkpoint, snapshot, evidence, routes, "26966", "CARLA exited")
    (partial / "metric_info.json").unlink()
    snapshot.write_text('{"route_id": "RouteScenario_26966_rep0"}', encoding="utf-8")
    with pytest.raises(ValueError, match="genuine score evidence"):
        skip_unscored_crash(checkpoint, snapshot, evidence, routes, "26966", "CARLA exited")
    snapshot.unlink()
    assert checkpoint.read_bytes() == original
    row = skip_unscored_crash(checkpoint, snapshot, evidence, routes, "26966", "3 CARLA exits")
    state = json.loads(checkpoint.read_text())
    assert state["_checkpoint"]["progress"] == [2, 2]
    assert state["_checkpoint"]["records"][-1] == row
    assert state["eligible"] is False
    assert row["status"] == UNSCORED_CRASH_STATUS
    assert row["scores"] == {"score_route": 0, "score_penalty": 1, "score_composed": 0}
    audit = json.loads((tmp_path / "skipped_unscored_crash_26966.json").read_text())
    assert audit["preserved_evidence"] == [str(partial)]
    assert audit["record"] == row
    assert (partial / "rgb_front").is_dir()
    with pytest.raises(ValueError, match="next route"):
        skip_unscored_crash(checkpoint, snapshot, evidence, routes, "26966", "duplicate")
    state["entry_status"] = "Finished"
    checkpoint.write_text(json.dumps(state), encoding="utf-8")
    assert coverage(checkpoint, {"1711", "26966"}) == (2, False)
    assert coverage(checkpoint, {"1711", "26966"}, allow_non_official=True) == (2, True)
