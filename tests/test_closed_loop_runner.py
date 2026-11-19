from __future__ import annotations

import json
import sys
from collections import namedtuple
from pathlib import Path

import pytest

from core.closed_loop_runner import (
    build_bench2drive_runtime_env,
    extract_round_failures,
    main,
    normalize_bench2drive_root,
    resolve_agent_path,
    resolve_agent_config_value,
    resolve_bench2drive_zoo_root,
    resolve_carla_runtime_source,
    select_carla_python_artifact,
    validate_agent_config_value,
    validate_agent_runtime_dependencies,
    native_policy_metrics,
    validate_native_execution_complete,
)


VersionInfo = namedtuple("VersionInfo", "major minor micro releaselevel serial")


def test_dry_run_end_to_end(tmp_path, monkeypatch) -> None:
    scenario_bank_path = tmp_path / "seed_scenarios.json"
    scenario_bank_path.write_text(
        json.dumps(
            [
                {
                    "scenario_id": "seed_merge_01",
                    "tags": ["urban", "merge"],
                    "pattern": "adversarial_merge",
                    "config": {
                        "ego_speed": 7.5,
                        "merge_offset": 16.0,
                        "bench2drive_spec": {
                            "route": {
                                "id": "1",
                                "town": "Town12",
                                "waypoints": [
                                    {"x": 1.0, "y": 2.0, "z": 3.0},
                                    {"x": 4.0, "y": 5.0, "z": 6.0},
                                ],
                            },
                            "scenario": {
                                "name": "HighwayCutIn_1",
                                "type": "HighwayCutIn",
                                "trigger_point": {"x": 1.0, "y": 2.0, "z": 3.0, "yaw": 90.0},
                                "parameters": [
                                    {"tag": "flow_speed", "attrs": {"value": "8.3"}},
                                    {"tag": "source_dist_interval", "attrs": {"from": "15", "to": "30"}},
                                ],
                            },
                        },
                    },
                }
            ]
        ),
        encoding="utf-8",
    )

    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "output_root": str(tmp_path / "artifacts"),
                "num_rounds": 2,
                "scenario_bank_path": str(scenario_bank_path),
                "metrics": {"keys": ["DS", "SR", "CR", "Comf", "L2m"]},
                "retrieval": {"top_k": 1},
                "evolution": {"escalation_strength": 0.2},
                "bench2drive": {"root": "Bench2Drive", "eval_cmd": "", "metrics_file": "metrics.json"},
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "fd-set",
            "--config",
            str(config_path),
            "--dry-run",
            "--mode",
            "full",
            "--run-id",
            "test_run",
        ],
    )
    main()

    run_root = tmp_path / "artifacts" / "test_run"
    summary = json.loads((run_root / "summary.json").read_text(encoding="utf-8"))

    assert summary["num_rounds"] == 2
    assert len(summary["rounds"]) == 2
    assert (run_root / "run_manifest.json").exists()
    assert (run_root / "failures_raw_all.json").exists()
    assert (run_root / "round0" / "scenarios_bench2drive.xml").exists()
    assert (run_root / "round0" / "materialization_trace.json").exists()
    assert (run_root / "round0" / "round_summary.json").exists()
    assert (run_root / "round0" / "failures_raw.json").exists()
    assert (run_root / "round0" / "failures_semantics.json").exists()
    assert (run_root / "round0" / "retrieval_trace.json").exists()
    assert (run_root / "scenario_bank_final.json").exists()

    manifest = json.loads((run_root / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["failure_extract"]["format"] == "jsonl"


def test_resume_run_uses_existing_round_outputs(tmp_path, monkeypatch) -> None:
    scenario_bank_path = tmp_path / "seed_scenarios.json"
    scenario_bank_path.write_text(
        json.dumps(
            [
                {
                    "scenario_id": "seed_merge_01",
                    "tags": ["urban", "merge"],
                    "pattern": "adversarial_merge",
                    "config": {
                        "ego_speed": 7.5,
                        "merge_offset": 16.0,
                        "bench2drive_spec": {
                            "route": {
                                "id": "1",
                                "town": "Town12",
                                "waypoints": [
                                    {"x": 1.0, "y": 2.0, "z": 3.0},
                                    {"x": 4.0, "y": 5.0, "z": 6.0},
                                ],
                            },
                            "scenario": {
                                "name": "HighwayCutIn_1",
                                "type": "HighwayCutIn",
                                "trigger_point": {"x": 1.0, "y": 2.0, "z": 3.0, "yaw": 90.0},
                                "parameters": [
                                    {"tag": "flow_speed", "attrs": {"value": "8.3"}},
                                    {"tag": "source_dist_interval", "attrs": {"from": "15", "to": "30"}},
                                ],
                            },
                        },
                    },
                }
            ]
        ),
        encoding="utf-8",
    )

    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "output_root": str(tmp_path / "artifacts"),
                "num_rounds": 1,
                "scenario_bank_path": str(scenario_bank_path),
                "metrics": {"keys": ["DS", "SR", "CR", "Comf", "L2m"]},
                "retrieval": {"top_k": 1},
                "evolution": {"escalation_strength": 0.2},
                "bench2drive": {"root": "Bench2Drive", "eval_cmd": "", "metrics_file": "metrics.json"},
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "fd-set",
            "--config",
            str(config_path),
            "--dry-run",
            "--run-id",
            "resume_target",
        ],
    )
    main()

    config_path.write_text(
        json.dumps(
            {
                "output_root": str(tmp_path / "artifacts"),
                "num_rounds": 2,
                "scenario_bank_path": str(scenario_bank_path),
                "metrics": {"keys": ["DS", "SR", "CR", "Comf", "L2m"]},
                "retrieval": {"top_k": 1},
                "evolution": {"escalation_strength": 0.2},
                "bench2drive": {"root": "Bench2Drive", "eval_cmd": "", "metrics_file": "metrics.json"},
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "fd-set",
            "--config",
            str(config_path),
            "--dry-run",
            "--resume-run",
            str(tmp_path / "artifacts" / "resume_target"),
        ],
    )
    main()

    run_root = tmp_path / "artifacts" / "resume_target"
    summary = json.loads((run_root / "summary.json").read_text(encoding="utf-8"))
    assert summary["num_rounds"] == 2
    assert (run_root / "round1" / "round_summary.json").exists()





def test_build_bench2drive_runtime_env_applies_model_runtime_overrides(tmp_path, monkeypatch) -> None:
    bench_root = tmp_path / "Bench2Drive"
    (bench_root / "leaderboard").mkdir(parents=True)
    (bench_root / "leaderboard" / "team_code").mkdir(parents=True)
    (bench_root / "scenario_runner").mkdir(parents=True)
    extra_root = tmp_path / "Bench2DriveZoo_1"
    extra_root.mkdir()
    monkeypatch.setattr("importlib.util.find_spec", lambda name: object() if name == "carla" else None)

    env = build_bench2drive_runtime_env(
        {
            "bench2drive": {
                "zoo_root": str(extra_root),
                "extra_pythonpath": [str(extra_root)],
                "environment": {"PLANNER_TYPE": "merge_ctrl_traj"},
            },
            "carla": {},
        },
        bench_root,
    )

    pythonpath_entries = env["PYTHONPATH"].split(":")
    assert str(extra_root) in pythonpath_entries
    assert env["BENCH2DRIVE_ZOO_ROOT"] == str(extra_root)
    assert env["PLANNER_TYPE"] == "merge_ctrl_traj"


def test_extract_round_failures_reads_live_agent_evidence_root(tmp_path) -> None:
    run_root = tmp_path / "run"
    round_dir = run_root / "round0"
    evidence_dir = round_dir / "evidence" / "route_001"
    evidence_dir.mkdir(parents=True)
    (evidence_dir / "0000.png").write_bytes(b"frame")
    (evidence_dir / "metric_info.json").write_text(
        json.dumps({"0": {"location": [0.0, 0.0, 0.0]}}),
        encoding="utf-8",
    )
    (round_dir / "ckpt.json").write_text(
        json.dumps(
            {
                "_checkpoint": {
                    "records": [
                        {
                            "route_id": "route_001",
                            "scenario_name": "HighwayCutIn",
                            "town_name": "Town04",
                            "save_name": "route_001",
                            "status": "Failed",
                            "meta": {"duration_game": 1.0},
                            "infractions": {"collisions_vehicle": ["vehicle collision"]},
                        }
                    ]
                }
            }
        ),
        encoding="utf-8",
    )

    failures = extract_round_failures(
        0,
        dry_run=False,
        cfg={
            "round_log_template": "{round_dir}/ckpt.json",
            "failure_extract": {
                "format": "bench2drive_checkpoint",
                "records_path": "_checkpoint.records",
                "capture_agent_evidence": True,
                "evidence_root_template": "{round_dir}/evidence",
                "frame_limit": 1,
            },
        },
        run_root=run_root,
        round_dir=round_dir,
    )

    assert len(failures) == 1
    assert failures[0].metadata["context_summary"]["evidence_frames"] == [str(evidence_dir / "0000.png")]


def test_resolve_agent_path_discovers_official_candidate(tmp_path) -> None:
    bench_root = tmp_path / "Bench2Drive"
    agent_path = bench_root / "team_code" / "uniad_b2d_agent.py"
    agent_path.parent.mkdir(parents=True)
    agent_path.write_text("class Agent:\n    pass\n", encoding="utf-8")

    resolved = resolve_agent_path({"bench2drive": {}}, bench_root)

    assert resolved == agent_path


def test_resolve_agent_path_falls_back_to_builtin_autoagent(tmp_path) -> None:
    bench_root = tmp_path / "Bench2Drive"
    agent_path = bench_root / "leaderboard" / "leaderboard" / "autoagents" / "npc_agent.py"
    agent_path.parent.mkdir(parents=True)
    agent_path.write_text("class Agent:\n    pass\n", encoding="utf-8")

    resolved = resolve_agent_path({"bench2drive": {}}, bench_root)

    assert resolved == agent_path


def test_select_carla_python_artifact_rejects_incompatible_binary_abi(tmp_path, monkeypatch) -> None:
    dist_dir = tmp_path / "dist"
    dist_dir.mkdir()
    py37 = dist_dir / "carla-0.9.15-cp37-cp37m-manylinux_2_27_x86_64.whl"
    py38 = dist_dir / "carla-0.9.15-cp38-cp38-manylinux_2_27_x86_64.whl"
    py37.write_bytes(b"")
    monkeypatch.setattr("sys.version_info", VersionInfo(3, 8, 0, "final", 0))

    with pytest.raises(RuntimeError, match="compatible with Python 3.8"):
        select_carla_python_artifact(dist_dir)
    py38.write_bytes(b"")
    assert select_carla_python_artifact(dist_dir) == py38


def test_build_bench2drive_runtime_env_prefers_installed_carla(tmp_path, monkeypatch) -> None:
    bench_root = tmp_path / "Bench2Drive"
    (bench_root / "leaderboard").mkdir(parents=True)
    (bench_root / "leaderboard" / "team_code").mkdir(parents=True)
    (bench_root / "scenario_runner").mkdir(parents=True)
    (tmp_path / "Bench2DriveZoo").mkdir()
    carla_root = tmp_path / "carla"
    (carla_root / "PythonAPI" / "carla").mkdir(parents=True)

    monkeypatch.setattr("importlib.util.find_spec", lambda name: object() if name == "carla" else None)

    env = build_bench2drive_runtime_env({"carla": {"pythonapi_root": str(carla_root)}}, bench_root)
    pythonpath_entries = env["PYTHONPATH"].split(":")

    assert env["CARLA_PYTHON_SOURCE"] == "installed-package"
    assert "CARLA_PYTHON_ARTIFACT" not in env
    assert env["BENCH2DRIVE_ZOO_ROOT"] == str(tmp_path / "Bench2DriveZoo")
    assert env["BENCH2DRIVE_ZOO_PARENT"] == str(tmp_path)
    assert str(tmp_path / "Bench2DriveZoo") in pythonpath_entries
    assert str(carla_root / "PythonAPI") in pythonpath_entries
    assert str(carla_root / "PythonAPI" / "carla") in pythonpath_entries
    assert str(tmp_path) in pythonpath_entries


def test_resolve_carla_runtime_source_uses_pythonapi_when_package_missing(tmp_path, monkeypatch) -> None:
    carla_root = tmp_path / "carla"
    dist_dir = carla_root / "PythonAPI" / "carla" / "dist"
    dist_dir.mkdir(parents=True)
    artifact = dist_dir / "carla-0.9.14-py3.8-linux-x86_64.egg"
    artifact.write_text("", encoding="utf-8")

    monkeypatch.setattr("importlib.util.find_spec", lambda name: None)
    monkeypatch.setattr("sys.version_info", VersionInfo(3, 8, 0, "final", 0))

    source = resolve_carla_runtime_source({"carla": {"pythonapi_root": str(carla_root)}})

    assert source["source"] == "pythonapi_artifact"
    assert source["artifact"] == artifact


def test_normalize_bench2drive_root_accepts_nested_checkout(tmp_path) -> None:
    outer = tmp_path / "Bench2Drive"
    inner = outer / "Bench2Drive"
    (inner / "leaderboard").mkdir(parents=True)
    (inner / "scenario_runner").mkdir(parents=True)

    resolved = normalize_bench2drive_root(outer)

    assert resolved == inner


def test_resolve_bench2drive_zoo_root_prefers_sibling_checkout(tmp_path) -> None:
    bench_root = tmp_path / "Bench2Drive"
    zoo_root = tmp_path / "Bench2DriveZoo"
    bench_root.mkdir()
    zoo_root.mkdir()

    resolved = resolve_bench2drive_zoo_root({}, bench_root)

    assert resolved == zoo_root


def test_resolve_agent_path_supports_uniad_alias(tmp_path) -> None:
    bench_root = tmp_path / "Bench2Drive"
    agent_path = bench_root / "leaderboard" / "team_code" / "uniad_b2d_agent.py"
    agent_path.parent.mkdir(parents=True)
    agent_path.write_text("class Agent:\n    pass\n", encoding="utf-8")

    resolved = resolve_agent_path({"bench2drive": {"agent": "uniad"}}, bench_root)

    assert resolved == agent_path


def test_resolve_agent_config_value_supports_uniad_alias(tmp_path) -> None:
    bench_root = tmp_path / "Bench2Drive"
    config_path = tmp_path / "Bench2DriveZoo" / "adzoo" / "uniad" / "configs" / "stage2_e2e" / "base_e2e_b2d.py"
    checkpoint_path = tmp_path / "weights" / "uniad_base_b2d.pth"
    config_path.parent.mkdir(parents=True)
    checkpoint_path.parent.mkdir(parents=True)
    config_path.write_text("cfg = {}", encoding="utf-8")
    checkpoint_path.write_text("", encoding="utf-8")

    value = resolve_agent_config_value(
        {
            "bench2drive": {
                "agent": "uniad",
                "agent_config": "../Bench2DriveZoo/adzoo/uniad/configs/stage2_e2e/base_e2e_b2d.py+../weights/uniad_base_b2d.pth",
            }
        },
        bench_root,
    )

    assert value == f"{config_path}+{checkpoint_path}"


def test_validate_agent_config_value_rejects_uniad_placeholder(tmp_path) -> None:
    bench_root = tmp_path / "Bench2Drive"
    with pytest.raises(ValueError, match="requires a real checkpoint path"):
        validate_agent_config_value({"bench2drive": {"agent": "uniad"}}, bench_root)


def test_resolve_agent_config_value_expands_user_home(tmp_path, monkeypatch) -> None:
    bench_root = tmp_path / "Bench2Drive"
    home_dir = tmp_path / "home"
    config_path = home_dir / "cfg.py"
    checkpoint_path = home_dir / "weights" / "uniad.pth"
    config_path.parent.mkdir(parents=True)
    checkpoint_path.parent.mkdir(parents=True)
    config_path.write_text("cfg = {}", encoding="utf-8")
    checkpoint_path.write_text("", encoding="utf-8")
    monkeypatch.setenv("HOME", str(home_dir))

    value = resolve_agent_config_value(
        {
            "bench2drive": {
                "agent_config": "~/cfg.py+~/weights/uniad.pth",
            }
        },
        bench_root,
    )

    assert value == f"{config_path.resolve()}+{checkpoint_path.resolve()}"


def test_validate_agent_runtime_dependencies_requires_mmcv_for_uniad(monkeypatch, tmp_path) -> None:
    bench_root = tmp_path / "Bench2Drive"
    bench_root.mkdir()

    def fake_find_spec(name: str):
        if name in {"mmcv", "Bench2DriveZoo"}:
            return None
        return object()

    monkeypatch.setattr("importlib.util.find_spec", fake_find_spec)

    with pytest.raises(RuntimeError, match="missing model runtime dependencies"):
        validate_agent_runtime_dependencies({"bench2drive": {"agent": "uniad"}}, bench_root)


def test_validate_agent_runtime_dependencies_accepts_source_checkout(tmp_path, monkeypatch) -> None:
    bench_root = tmp_path / "Bench2Drive"
    bench_root.mkdir()
    zoo_root = tmp_path / "Bench2DriveZoo"
    (zoo_root / "mmcv").mkdir(parents=True)
    (zoo_root / "adzoo").mkdir(parents=True)
    monkeypatch.setattr("importlib.util.find_spec", lambda name: None)

    validate_agent_runtime_dependencies(
        {"bench2drive": {"agent": "uniad", "zoo_root": str(zoo_root)}},
        bench_root,
    )
def test_native_policy_metrics_excludes_failed_and_skipped_runs() -> None:
    execution = {
        "programs": [
            {"status": "completed", "collisions": 2},
            {"status": "completed", "collisions": 0},
            {"status": "error", "collisions": 1},
            {"status": "skipped", "collisions": 1},
        ]
    }

    assert native_policy_metrics(execution, collision_rate_scale="percent") == {"CR": 50.0}
    assert native_policy_metrics(execution, collision_rate_scale="fraction") == {"CR": 0.5}


def test_fixed_policy_protocol_runs_native_bank_not_official_routes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bank_path = tmp_path / "seed_bank.json"
    bank_path.write_text(json.dumps([{"scenario_id": "seed", "tags": ["seed"], "config": {}}]))
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "protocol": "fixed_policy_evolution",
                "output_root": str(tmp_path / "runs"),
                "scenario_bank_path": str(bank_path),
                "num_rounds": 2,
                "metrics": {"keys": ["DS", "CR"]},
                "retrieval": {"top_k": 1},
                "evolution": {"escalation_strength": 0.2},
                "materialization": {
                    "scenic": {"enabled": True, "policy_execution": {"enabled": False}},
                    "bench2drive": {"enabled": False},
                },
                "adaptation": {"enabled": False},
            }
        ),
        encoding="utf-8",
    )
    run_root = tmp_path / "runs" / "fixed"
    native_bank_sizes: list[int] = []

    def materialize(bank, round_dir, cfg):
        native_bank_sizes.append(len(bank))
        manifest = round_dir / "scenic_execution_manifest.json"
        manifest.write_text(json.dumps({"programs": []}), encoding="utf-8")
        scenarios = round_dir / "scenarios.json"
        scenarios.write_text(json.dumps(bank), encoding="utf-8")
        return {"scenic_execution_manifest": manifest, "json": scenarios}

    def native_run(manifest_path, cfg, output_dir, *, dry_run=False):
        assert dry_run
        return {
            "status": "dry_run",
            "programs": [
                {"scenario_id": "seed", "status": "planned"},
            ],
        }

    def append_one(bank, candidates):
        return bank + [
            {
                "scenario_id": f"evolved_{len(bank)}",
                "tags": ["native"],
                "config": {"scenic_program_format": "scenic_native_v1"},
            }
        ]

    monkeypatch.setattr("core.closed_loop_runner.materialize_round_inputs", materialize)
    monkeypatch.setattr("core.closed_loop_runner.run_native_policy_manifest", native_run)
    monkeypatch.setattr("core.closed_loop_runner.structure_failures", lambda *args, **kwargs: [])
    monkeypatch.setattr("core.closed_loop_runner.append_evolved_to_bank", append_one)
    monkeypatch.setattr(
        "core.closed_loop_runner.run_eval_round",
        lambda *args, **kwargs: pytest.fail("fixed protocol must not call official-route evaluation"),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["fd-set", "--config", str(config_path), "--dry-run", "--run-id", "fixed"],
    )

    main()

    assert native_bank_sizes == [1, 2]
    summary = json.loads((run_root / "summary.json").read_text(encoding="utf-8"))
    first_round = summary["rounds"][0]
    assert first_round["protocol"] == "fixed_policy_evolution"
    assert first_round["evaluation_synthetic"] is True
    assert first_round["official_benchmark_routes"] is None
    assert first_round["metrics"] == {"CR": 0.0}
    assert first_round["adaptation_executed"] is False
    assert (run_root / "round0" / "native_policy_evaluation" / "evaluation_contract.json").exists()


def test_fixed_policy_reingests_native_collision_failures(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    bank_path = tmp_path / "bank.json"
    bank_path.write_text(json.dumps([{"scenario_id": "seed", "config": {}}]), encoding="utf-8")
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "protocol": "fixed_policy_evolution",
                "output_root": str(tmp_path / "runs"),
                "scenario_bank_path": str(bank_path),
                "num_rounds": 1,
                "retrieval": {"top_k": 1},
                "evolution": {"escalation_strength": 0.2},
                "materialization": {
                    "scenic": {"enabled": True, "policy_execution": {"enabled": True}},
                    "bench2drive": {"enabled": False},
                },
            }
        ),
        encoding="utf-8",
    )
    failures_seen = []

    def materialize(bank, round_dir, cfg):
        manifest = round_dir / "manifest.json"
        manifest.write_text(json.dumps({"programs": []}), encoding="utf-8")
        scenario_file = round_dir / "scenarios.json"
        scenario_file.write_text(json.dumps(bank), encoding="utf-8")
        return {"scenic_execution_manifest": manifest, "json": scenario_file}

    def execute(manifest_path, config_path, cfg, output_dir, *, dry_run, resume):
        assert not dry_run
        output_dir.mkdir(parents=True, exist_ok=True)
        failure_file = output_dir / "native_failures.jsonl"
        failure_file.write_text(
            json.dumps(
                {
                    "route_id": "seed",
                    "timestamp_s": 4.2,
                    "event_type": "collision",
                    "ego_speed": 2.0,
                    "note": "native Scenic collision",
                    "scene_tags": ["native"],
                }
            )
            + "\n",
            encoding="utf-8",
        )
        return {
            "status": "completed",
            "failure_file": str(failure_file),
            "programs": [
                {"scenario_id": "seed", "status": "completed", "collisions": 1},
            ],
        }

    def structure(failures, semantics, *, use_structure):
        failures_seen.extend(failures)
        return []

    monkeypatch.setattr("core.closed_loop_runner.materialize_round_inputs", materialize)
    monkeypatch.setattr("core.closed_loop_runner.execute_native_policy_stage", execute)
    monkeypatch.setattr("core.closed_loop_runner.structure_failures", structure)
    monkeypatch.setattr(
        "core.closed_loop_runner.run_eval_round",
        lambda *args, **kwargs: pytest.fail("fixed protocol must not call Bench2Drive evaluation"),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["fd-set", "--config", str(config_path), "--run-id", "live"],
    )

    main()

    assert len(failures_seen) == 1
    assert failures_seen[0].route_id == "seed"
    summary = json.loads((tmp_path / "runs" / "live" / "summary.json").read_text(encoding="utf-8"))
    round_row = summary["rounds"][0]
    assert round_row["metrics"] == {"CR": 100.0}
    assert round_row["native_execution_counts"] == {
        "total": 1,
        "completed": 1,
        "errors": 0,
        "skipped": 0,
        "collision_scenarios": 1,
    }
def test_native_policy_execution_rejects_errors_skips_and_empty_results() -> None:
    for status in ("error", "skipped"):
        with pytest.raises(RuntimeError, match="not successes"):
            validate_native_execution_complete({"programs": [{"status": status}]}, 0)
    with pytest.raises(RuntimeError, match="no native Scenic execution outcomes"):
        validate_native_execution_complete({"programs": []}, 0)
