from __future__ import annotations

import json
from pathlib import Path
import pytest

import core.closed_loop_runner as closed_loop_runner
from core.bench2drive_adapter import parse_metrics
from core.closed_loop_runner import prepare_adaptation_artifacts
from core.policy_adaptation import build_adaptation_manifest, select_adaptation_scenarios
from core.scenario_materialize import build_bench2drive_evaluation_routes_artifacts, materialize_round_inputs
from core.scenario_retrieve import retrieve_anchor_scenarios
from core.scenario_evolve import evolve_from_anchors
from core.models import FailureEvent, StructuredFailure


def _route_spec(route_id: str = "1") -> dict:
    return {
        "route": {
            "id": route_id,
            "town": "Town12",
            "waypoints": [{"x": 0.0, "y": 0.0, "z": 0.0}],
        },
        "scenario": {
            "name": "HighwayCutIn_1",
            "type": "HighwayCutIn",
            "trigger_point": {"x": 0.0, "y": 0.0, "z": 0.0, "yaw": 0.0},
            "parameters": [{"tag": "flow_speed", "attrs": {"value": "8"}}],
        },
    }


def test_collision_rate_is_collision_rate_not_collision_free_rate(tmp_path: Path) -> None:
    path = tmp_path / "ckpt.json"
    path.write_text(
        json.dumps(
            {
                "_checkpoint": {
                    "records": [
                        {"status": "Completed", "infractions": {}},
                        {"status": "Completed", "infractions": {"collisions_vehicle": ["hit"]}},
                        {"status": "Completed", "infractions": {}},
                    ]
                }
            }
        ),
        encoding="utf-8",
    )
    assert parse_metrics(path, keys=["CR"], collision_rate_scale="fraction")["CR"] == 1.0 / 3.0


def test_official_benchmark_rejects_partial_or_crashed_route_scores(tmp_path: Path) -> None:
    routes = tmp_path / "two_routes.xml"
    routes.write_text("<routes><route id='1'/><route id='2'/></routes>", encoding="utf-8")
    checkpoint = tmp_path / "ckpt.json"
    payload = {
        "entry_status": "Started",
        "_checkpoint": {"progress": [1, 2], "records": [{"route_id": "1"}]},
    }
    checkpoint.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="expected 2 finished routes"):
        closed_loop_runner.validate_official_benchmark_complete(checkpoint, routes)

    payload["entry_status"] = "Finished"
    payload["_checkpoint"]["progress"] = [2, 2]
    with pytest.raises(RuntimeError, match="records=1"):
        checkpoint.write_text(json.dumps(payload), encoding="utf-8")
        closed_loop_runner.validate_official_benchmark_complete(checkpoint, routes)

    payload["_checkpoint"]["records"].append({"route_id": "2"})
    checkpoint.write_text(json.dumps(payload), encoding="utf-8")
    assert closed_loop_runner.validate_official_benchmark_complete(checkpoint, routes) == 2
    payload["_checkpoint"]["records"][1]["status"] = "Failed - TickRuntime"
    checkpoint.write_text(json.dumps(payload), encoding="utf-8")
    assert closed_loop_runner.validate_official_benchmark_complete(checkpoint, routes) == 2
    payload["_checkpoint"]["records"][1]["status"] = "Failed - Simulation crashed"
    checkpoint.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="route execution failed"):
        closed_loop_runner.validate_official_benchmark_complete(checkpoint, routes)


def test_embedding_retrieval_accepts_frozen_encoder_contract() -> None:
    class FakeEncoder:
        def encode(self, texts: list[str]) -> list[list[float]]:
            return [[1.0, 0.0] if index == 0 else [1.0, 0.0] for index, _ in enumerate(texts)]

    failure = StructuredFailure(
        event=FailureEvent("r", 1.0, "collision", 8.0, "cut-in vehicle", ["merge"], "high"),
        actors=["ego", "vehicle"],
        interaction_pattern="adversarial_merge",
        constraints={"ego_speed_min": 0.0, "ego_speed_max": 10.0},
    )
    anchors = retrieve_anchor_scenarios(
        failure,
        [{"scenario_id": "a", "tags": ["merge"], "pattern": "adversarial_merge", "config": {}}],
        top_k=1,
        retrieval_config={"backend": "clip", "score_mode": "embedding"},
        encoder=FakeEncoder(),
    )
    assert anchors[0].score == 1.0
    assert anchors[0].score_breakdown["embedding_similarity"] == 1.0


def test_held_out_routes_are_not_inserted_into_adaptation_bank(tmp_path: Path) -> None:
    held_out = tmp_path / "bench2drive220.xml"
    held_out.write_text(
        "<routes><route id='220' town='Town12'><waypoints><position x='0' y='0' z='0'/></waypoints>"
        "<scenarios><scenario name='ControlLoss_1' type='ControlLoss'><trigger_point x='0' y='0' z='0' yaw='0'/></scenario>"
        "</scenarios></route></routes>",
        encoding="utf-8",
    )
    evolved = {
        "scenario_id": "evolved_1",
        "tags": ["control"],
        "pattern": "control_instability",
        "config": {
            "scenario_role": "evolved",
            "bench2drive_profile": {"dataset_split": "training"},
            "bench2drive_spec": _route_spec(),
        },
    }
    xml, trace = build_bench2drive_evaluation_routes_artifacts(held_out, [evolved], include_evolved=True)
    assert 'route id="220"' in xml
    assert 'route id="fdset_evolved_evolved_1"' in xml
    assert trace[0]["scenario_count"] == 1
    manifest = build_adaptation_manifest(
        [evolved],
        held_out_routes_xml=held_out,
        output_dir=tmp_path / "adaptation",
        config={"models": {"tcp": {"train_command": ""}}},
    )
    assert manifest["scenario_ids"] == ["evolved_1"]
    assert select_adaptation_scenarios([evolved], seed=1)[0]["scenario_id"] == "evolved_1"


def test_evolution_ids_include_generation_tag_for_multi_round_growth() -> None:
    failure = StructuredFailure(
        event=FailureEvent("r", 1.0, "collision", 8.0, "cut-in vehicle", ["merge"], "high"),
        actors=["ego", "vehicle"],
        interaction_pattern="adversarial_merge",
        constraints={"ego_speed_min": 0.0, "ego_speed_max": 10.0},
    )
    anchor = retrieve_anchor_scenarios(
        failure,
        [{"scenario_id": "a", "tags": ["merge"], "pattern": "adversarial_merge", "config": {}}],
        top_k=1,
    )
    first = evolve_from_anchors(failure, anchor, generation_tag="round0")
    second = evolve_from_anchors(failure, anchor, generation_tag="round1")
    assert first[0].evolved_id != second[0].evolved_id


def test_evolved_native_anchor_is_marked_for_scenic_adaptation() -> None:
    from core.scenario_bank import append_evolved_to_bank

    bank = [
        {
            "scenario_id": "scenic_anchor",
            "tags": ["scenic"],
            "pattern": "late_response",
            "config": {"scenario_role": "scenario_bank", "scenic_program_format": "scenic_native_v1"},
        }
    ]
    evolved = [{"evolved_id": "scenic_anchor_evo_round0_0", "modifications": dict(bank[0]["config"])}]
    updated = append_evolved_to_bank(bank, evolved)
    assert updated[-1]["config"]["scenario_role"] == "evolved"
    assert select_adaptation_scenarios(updated, seed=1)[0]["scenario_id"] == "scenic_anchor_evo_round0_0"


def test_materialize_feedback_benchmark_never_appends_evolved_routes(tmp_path: Path) -> None:
    held_out = tmp_path / "bench2drive220.xml"
    held_out.write_text(
        "<routes><route id='220' town='Town12'><waypoints><position x='0' y='0' z='0'/></waypoints>"
        "<scenarios><scenario name='ControlLoss_1' type='ControlLoss'><trigger_point x='0' y='0' z='0' yaw='0'/></scenario>"
        "</scenarios></route></routes>",
        encoding="utf-8",
    )
    evolved = {
        "scenario_id": "evolved_1",
        "tags": ["control"],
        "pattern": "control_instability",
        "config": {
            "scenario_role": "evolved",
            "bench2drive_spec": _route_spec(),
        },
    }
    outputs = materialize_round_inputs(
        [evolved],
        tmp_path / "round0",
        {
            "materialization": {"bench2drive": {"enabled": True}},
            "evaluation": {
                "use_held_out_routes": True,
                "held_out_routes_xml": str(held_out),
                "include_evolved_scenarios": False,
            },
        },
    )
    xml = outputs["evaluation_xml"].read_text(encoding="utf-8")
    assert 'id="220"' in xml
    assert "evolved_1" not in xml


def test_dry_adaptation_materializes_scenic_expert_data_and_checkpoint(tmp_path: Path) -> None:
    held_out = tmp_path / "bench2drive220.xml"
    held_out.write_text(
        "<routes><route id='220' town='Town12'><waypoints><position x='0' y='0' z='0'/></waypoints>"
        "<scenarios><scenario name='ControlLoss_1' type='ControlLoss'><trigger_point x='0' y='0' z='0' yaw='0'/></scenario>"
        "</scenarios></route></routes>",
        encoding="utf-8",
    )
    evolved = {
        "scenario_id": "evolved_1",
        "tags": ["control"],
        "pattern": "control_instability",
        "config": {"scenario_role": "evolved", "bench2drive_spec": _route_spec()},
    }
    manifest = prepare_adaptation_artifacts(
        tmp_path / "run",
        [evolved],
        {
            "evaluation": {"held_out_routes_xml": str(held_out)},
            "materialization": {"bench2drive": {"enabled": True}},
            "adaptation": {
                "enabled": True,
                "models": {"tcp": {"train_command": ""}},
                "expert": {"backend": "external"},
            },
        },
        round_id=0,
        dry_run=True,
        adaptation_rows=[evolved],
    )
    assert manifest is not None
    assert manifest["num_scenarios"] == 1
    assert manifest["expert_generation"]["status"] == "dry_run_proxy"
    assert manifest["fine_tuning"]["models"][0]["status"] == "dry_run_placeholder"


def test_live_adaptation_rejects_native_scenic_as_route_xml(tmp_path: Path) -> None:
    held_out = tmp_path / "bench2drive220.xml"
    held_out.write_text("<routes/>", encoding="utf-8")
    native = {
        "scenario_id": "evolved_native",
        "tags": ["control"],
        "pattern": "control_instability",
        "config": {
            "scenario_role": "evolved",
            "scenic_program_format": "scenic_native_v1",
            "scenic_program": "model scenic.simulators.carla.model\n",
            "scenic_native_execution_ready": True,
        },
    }
    with pytest.raises(ValueError, match="route-XML collection cannot collect native Scenic"):
        prepare_adaptation_artifacts(
            tmp_path / "run",
            [native],
            {
                "evaluation": {"held_out_routes_xml": str(held_out)},
                "materialization": {
                    "scenic": {"enabled": True},
                    "bench2drive": {"enabled": True},
                },
                "adaptation": {
                    "enabled": True,
                    "expert": {
                        "backend": "external",
                        "command_template": "invalid-route-collector --routes {routes_xml}",
                    },
                },
            },
            dry_run=False,
            adaptation_rows=[native],
        )


def test_live_adaptation_passes_native_manifest_to_scenic_collector(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    held_out = tmp_path / "bench2drive220.xml"
    held_out.write_text("<routes/>", encoding="utf-8")
    manifest_path = tmp_path / "scenic_execution_manifest.json"
    manifest_path.write_text('{"programs": []}', encoding="utf-8")
    native = {
        "scenario_id": "evolved_native",
        "tags": ["control"],
        "pattern": "control_instability",
        "config": {
            "scenario_role": "evolved",
            "scenic_program_format": "scenic_native_v1",
            "scenic_program": "model scenic.simulators.carla.model\n",
            "scenic_native_execution_ready": True,
        },
    }
    monkeypatch.setattr(
        closed_loop_runner,
        "materialize_adaptation_inputs",
        lambda *_args, **_kwargs: {
            "scenic_execution_manifest": manifest_path,
            "bench2drive_xml": held_out,
        },
    )
    output_dir = tmp_path / "run" / "adaptation"
    demos_path = output_dir / "expert_demos.jsonl"
    commands: list[str] = []

    def collect(command: str, *, cwd: Path) -> None:
        commands.append(command)
        demos_path.parent.mkdir(parents=True, exist_ok=True)
        demos_path.write_text('{"scenario_id":"evolved_native"}\n', encoding="utf-8")

    monkeypatch.setattr(closed_loop_runner, "run_expert_generation", collect)
    result = prepare_adaptation_artifacts(
        tmp_path / "run",
        [native],
        {
            "evaluation": {"held_out_routes_xml": str(held_out)},
            "adaptation": {
                "enabled": True,
                "expert": {
                    "backend": "external",
                    "command_template": (
                        '"{python}" pdm_lite/scenic_collector.py --manifest '
                        '"{scenic_execution_manifest}" --output-dir "{output_dir}/pdm_lite" '
                        '--demos-file "{demos_file}" --carla-host "{carla_host}" '
                        '--carla-port {carla_port} --traffic-manager-port {carla_tm_port} '
                        '--seed {seed}'
                    ),
                },
                "models": {},
            },
            "carla": {"host": "127.0.0.1", "port": 3000, "tm_port": 3050},
        },
        dry_run=False,
        adaptation_rows=[native],
    )

    assert result is not None
    assert len(commands) == 1
    assert "--manifest" in commands[0]
    assert str(manifest_path.resolve()) in commands[0]
    assert str(held_out.resolve()) not in commands[0]
    assert "--routes" not in commands[0]
