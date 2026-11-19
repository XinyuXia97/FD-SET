from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from core.scenic_policy_adapter import (
    build_native_policy_plan,
    run_native_policy_manifest,
    _collision_failure_record,
    _prepare_scene_objects_for_carla,
    _agent_config_with_evidence_save_name,
    validate_native_policy_config,
    _build_route,
)


def _manifest(tmp_path: Path) -> Path:
    program = tmp_path / "native" / "example.scenic"
    program.parent.mkdir()
    program.write_text("model scenic.domains.driving.model\n", encoding="utf-8")
    manifest = tmp_path / "scenic_execution_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": "fdset.scenic_execution_manifest.v1",
                "programs": [
                    {
                        "scenario_id": "scenic_example",
                        "program_path": str(program),
                        "native_execution_ready": True,
                        "map": "Town_Test",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return manifest


def test_native_policy_plan_is_side_effect_free_and_dry_run_does_not_import_carla(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    cfg = {
        "bench2drive": {"agent": "uniad"},
        "materialization": {
            "scenic": {
                "policy_execution": {
                    "enabled": True,
                    "max_programs": 1,
                    "max_steps": 7,
                    "timestep": 0.05,
                }
            }
        },
    }

    plan = build_native_policy_plan(manifest, cfg)
    assert plan["status"] == "planned"
    assert plan["program_count"] == 1
    assert plan["programs"][0]["program_exists"] is True

    output_dir = tmp_path / "round0"
    result = run_native_policy_manifest(manifest, cfg, output_dir, dry_run=True)
    assert result["status"] == "dry_run"
    assert (output_dir / "native_scenic_policy_results.json").exists()
    assert not (output_dir / "native_scenic_policy_failures.jsonl").exists()


def test_disabled_native_policy_is_explicit_in_result(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    cfg = {
        "materialization": {
            "scenic": {
                "policy_execution": {"enabled": False},
            }
        }
    }
    validate_native_policy_config(cfg)
    result = run_native_policy_manifest(manifest, cfg, tmp_path / "round0")
    assert result["status"] == "disabled"
    assert "enabled=true" in result["note"]


def test_native_resume_preserves_completed_exposure_and_retries_only_failed_scenarios(
    tmp_path: Path, monkeypatch,
) -> None:
    manifest = _manifest(tmp_path)
    data = json.loads(manifest.read_text(encoding="utf-8"))
    second = tmp_path / "native" / "second.scenic"
    second.write_text("model scenic.domains.driving.model\n", encoding="utf-8")
    data["programs"].append({
        "scenario_id": "scenic_second",
        "program_path": str(second),
        "native_execution_ready": True,
    })
    manifest.write_text(json.dumps(data), encoding="utf-8")
    cfg = {"materialization": {"scenic": {"policy_execution": {"enabled": True}}}}
    agent = tmp_path / "original_agent.py"
    monkeypatch.setattr(
        "core.scenic_policy_adapter._prepare_live_runtime",
        lambda config: {
            "bench_root": tmp_path,
            "agent_path": agent,
            "description": {"agent_path": str(agent)},
        },
    )
    calls = []
    fail_second = True

    def execute(program, config, policy, runtime, output_dir, *, index):
        nonlocal fail_second
        scenario_id = program["scenario_id"]
        calls.append(scenario_id)
        if scenario_id == "scenic_second" and fail_second:
            raise RuntimeError("CARLA timed out")
        trace = output_dir / f"{scenario_id}.jsonl"
        trace.write_text('{"measured": true}\n', encoding="utf-8")
        return (
            {
                "scenario_id": scenario_id,
                "program_path": program["program_path"],
                "status": "completed",
                "collisions": 1,
                "trace_file": str(trace),
            },
            [{"route_id": scenario_id, "event_type": "collision"}],
        )

    monkeypatch.setattr("core.scenic_policy_adapter._run_one_program", execute)
    output_dir = tmp_path / "results"
    first = run_native_policy_manifest(manifest, cfg, output_dir)
    assert first["completed"] == 1 and first["errors"] == 1
    fail_second = False
    completed = run_native_policy_manifest(manifest, cfg, output_dir, resume_completed=True)
    assert calls == ["scenic_example", "scenic_second", "scenic_second"]
    assert completed["completed"] == 2 and completed["errors"] == 0
    assert completed["collisions"] == 2
    failures = [
        json.loads(line) for line in (output_dir / "native_scenic_policy_failures.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    assert {row["route_id"] for row in failures} == {"scenic_example", "scenic_second"}


def test_native_resume_reuses_measured_exposure_after_directory_move(
    tmp_path: Path, monkeypatch,
) -> None:
    from core.closed_loop_runner import execute_native_policy_stage

    old_root = tmp_path / "old"
    old_root.mkdir()
    old_manifest = _manifest(old_root)
    old_output = old_root / "results"
    cfg = {"materialization": {"scenic": {"policy_execution": {"enabled": True}}}}
    agent = tmp_path / "agent.py"
    monkeypatch.setattr(
        "core.scenic_policy_adapter._prepare_live_runtime",
        lambda config: {
            "bench_root": tmp_path,
            "agent_path": agent,
            "description": {"agent_path": str(agent)},
        },
    )
    calls = []

    def execute(program, config, policy, runtime, output_dir, *, index):
        calls.append(program["scenario_id"])
        trace = output_dir / "exposure.jsonl"
        trace.write_text('{"collision": true}\n', encoding="utf-8")
        return (
            {
                "scenario_id": program["scenario_id"],
                "program_path": program["program_path"],
                "status": "completed",
                "collisions": 1,
                "trace_file": str(trace),
            },
            [{"route_id": program["scenario_id"], "event_type": "collision"}],
        )

    monkeypatch.setattr("core.scenic_policy_adapter._run_one_program", execute)
    first = run_native_policy_manifest(old_manifest, cfg, old_output)
    assert first["collisions"] == 1

    new_root = tmp_path / "data"
    old_root.rename(new_root)
    old_root.symlink_to(new_root, target_is_directory=True)
    new_manifest = new_root / old_manifest.name
    manifest_data = json.loads(new_manifest.read_text(encoding="utf-8"))
    manifest_data["programs"][0]["program_path"] = str(new_root / "native" / "example.scenic")
    new_manifest.write_text(json.dumps(manifest_data), encoding="utf-8")
    new_output = new_root / "results"

    reused = execute_native_policy_stage(
        new_manifest, new_manifest, cfg, new_output, dry_run=False, resume=True,
    )
    assert reused["collisions"] == 1
    resumed = run_native_policy_manifest(new_manifest, cfg, new_output, resume_completed=True)
    assert resumed["collisions"] == 1
    assert resumed["completed"] == 1
    assert calls == ["scenic_example"]
    failures = (new_output / "native_scenic_policy_failures.jsonl").read_text(encoding="utf-8")
    assert len(failures.splitlines()) == 1


def test_native_policy_config_rejects_invalid_limits() -> None:
    try:
        validate_native_policy_config(
            {
                "materialization": {
                    "scenic": {
                        "policy_execution": {"enabled": True, "route_points": 1}
                    }
                }
            }
        )
    except ValueError as exc:
        assert "route_points" in str(exc)
    else:
        raise AssertionError("invalid native policy limits should be rejected")


def test_agent_config_for_evidence_keeps_load_paths_and_sanitizes_save_name() -> None:
    assert _agent_config_with_evidence_save_name(
        "/opt/checkpoints/tcp_b2d.ckpt"
    ) == "/opt/checkpoints/tcp_b2d.ckpt+tcp_b2d.ckpt"
    assert _agent_config_with_evidence_save_name(
        "/opt/config.py+/opt/checkpoints/vad_b2d.pth"
    ) == "/opt/config.py+/opt/checkpoints/vad_b2d.pth+vad_b2d.pth"
    assert _agent_config_with_evidence_save_name("") == ""


def test_legacy_scenic_mercedes_blueprint_spawns_as_carla_0915_coupe() -> None:
    class Car:
        pass

    ego = Car()
    ego.blueprint = "vehicle.lincoln.mkz_2017"
    lead = Car()
    lead.blueprint = "vehicle.mercedes-benz.coupe"
    scene = SimpleNamespace(
        objects=[ego, lead],
        dynamicScenario=SimpleNamespace(_agents=[ego, lead]),
    )

    _prepare_scene_objects_for_carla(scene, ego)

    assert ego.blueprint == "vehicle.lincoln.mkz_2017"
    assert lead.blueprint == "vehicle.mercedes.coupe"
    assert scene.dynamicScenario._agents == [lead]


def test_native_collision_uses_observed_ego_speed_for_failure_constraints(tmp_path: Path) -> None:
    from core.failure_extract import extract_failures_from_log
    from core.failure_struct import structure_failures

    record = _collision_failure_record(
        {"scenario_id": "scenic_intersection"},
        {"frame": 51, "timestamp_s": 5.1, "other_actor": "vehicle"},
        timestep=0.1,
        scene_tags=["junction"],
        ego_speed=11.5,
    )
    path = tmp_path / "failures.jsonl"
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    event = extract_failures_from_log(path, {"format": "jsonl"})[0]
    structured = structure_failures([event])[0]
    assert structured.event.ego_speed == 11.5
    assert structured.constraints["ego_speed_max"] == 14.5


def test_scenic_goal_offroad_fails_before_expert_can_train_on_wrong_route() -> None:
    import pytest

    from core.scenic_policy_adapter import NativePolicyExecutionError

    class Location:
        def __init__(self, x, y, z=0.0):
            self.x, self.y, self.z = x, y, z

    class Waypoint:
        def __init__(self, x, y):
            self.transform = SimpleNamespace(location=Location(x, y))

    start = Location(1.0, -2.0)
    actor = SimpleNamespace(get_location=lambda: start)

    class Map:
        def get_waypoint(self, location, **kwargs):
            if location is start:
                return Waypoint(1.0, -2.0)
            return Waypoint(location.x, location.y - 4.0)

    world = SimpleNamespace(get_map=lambda: Map())
    carla = SimpleNamespace(LaneType=SimpleNamespace(Driving=object()), Location=Location)
    with pytest.raises(NativePolicyExecutionError, match="not on a driving lane"):
        _build_route(
            actor, world, 120, SimpleNamespace(LANEFOLLOW=object()), carla,
            route_goal=SimpleNamespace(x=20.0, y=3.0),
        )
