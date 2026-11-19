from __future__ import annotations

import json
import os
from pathlib import Path

from core.scenic_library import build_scenic_source_bank, index_scenic_programs
from core.models import FailureEvent, ScenarioCandidate, StructuredFailure
from core.scenario_evolve import evolve_from_anchors
from core.scenario_materialize import materialize_round_inputs
from scripts.validate_scenic_programs import _validate_one


def test_native_validation_rejects_scenic_syntax_errors(tmp_path: Path) -> None:
    source = tmp_path / "invalid.scenic"
    source.write_text("behavior Broken:\n    wait\n", encoding="utf-8")
    result = _validate_one(source, timeout_s=20.0, env=os.environ.copy())
    assert result["status"] == "syntax_error"
    assert result["return_code"] != 0


def test_index_scenic_programs_audits_sources_and_maps(tmp_path: Path) -> None:
    source_dir = tmp_path / "scenic"
    source_dir.mkdir()
    (source_dir / "carlaChallenge10.scenic").write_text(
        '"""intersection test"""\n'
        "param map = localPath('../maps/Town_Test.xodr')\n"
        "param carla_map = 'Town_Test'\n"
        "model scenic.simulators.carla.model\n"
        "behavior EgoBehavior(speed):\n    do FollowLaneBehavior(speed)\n"
        "ego = Car at spawn\n"
        "require 1 < 2\n",
        encoding="utf-8",
    )
    maps_dir = tmp_path / "maps"
    maps_dir.mkdir()
    (maps_dir / "Town_Test.xodr").write_text("<OpenDRIVE/>", encoding="utf-8")

    copy_dir = tmp_path / "copied"
    payload = index_scenic_programs(source_dir, copy_to=copy_dir)

    assert payload["program_count"] == 1
    assert payload["summary"]["native_execution_ready"] == 1
    program = payload["programs"][0]
    assert program["interaction_family"] == "unsignalized_intersection"
    assert program["carla_map"] == "'Town_Test'"
    assert program["map_references"][0]["exists"] is True
    assert Path(program["artifact_source_path"]).exists()
    assert program["sha256"]


def test_materialization_archives_user_scenic_library(tmp_path: Path) -> None:
    source_dir = tmp_path / "scenic"
    source_dir.mkdir()
    (source_dir / "example.scenic").write_text(
        "param map = localPath('../maps/Town.xodr')\nmodel scenic.domains.driving.model\n",
        encoding="utf-8",
    )

    outputs = materialize_round_inputs(
        [],
        tmp_path / "round0",
        {
            "materialization": {
                "scenic": {
                    "enabled": True,
                    "source_dir": str(source_dir),
                    "source_copy_dir": "scenic_sources",
                },
                "bench2drive": {"enabled": False},
            }
        },
    )

    index = json.loads(outputs["scenic_library_index"].read_text(encoding="utf-8"))
    assert index["program_count"] == 1
    assert outputs["scenic_source_dir"].joinpath("example.scenic").exists()


def test_native_scenic_sources_are_scenario_bank_rows_and_materialized_as_execution_units(tmp_path: Path) -> None:
    source_dir = tmp_path / "scenic"
    source_dir.mkdir()
    source = source_dir / "carlaChallenge2.scenic"
    source.write_text(
        '"""leading brake"""\n'
        "param map = localPath('../maps/Town.xodr')\n"
        "model scenic.domains.driving.model\n"
        "EGO_SPEED = 10\n"
        "ego = Car at spawn\n",
        encoding="utf-8",
    )
    (tmp_path / "maps").mkdir()
    (tmp_path / "maps" / "Town.xodr").write_text("<OpenDRIVE/>", encoding="utf-8")

    rows = build_scenic_source_bank(source_dir)
    assert len(rows) == 1
    assert rows[0]["scenario_id"] == "scenic_carlaChallenge2"
    assert rows[0]["pattern"] == "late_response"
    assert rows[0]["config"]["scenic_program_format"] == "scenic_native_v1"

    outputs = materialize_round_inputs(
        rows,
        tmp_path / "round0",
        {
            "materialization": {
                "scenic": {
                    "enabled": True,
                    "source_dir": str(source_dir),
                },
                "bench2drive": {"enabled": False},
            }
        },
    )

    native_path = outputs["scenic_program_dir"] / "native" / "scenic_carlaChallenge2.scenic"
    assert native_path.exists()
    assert "EGO_SPEED = 10" in native_path.read_text(encoding="utf-8")
    assert "localPath('../maps/Town.xodr')" in native_path.read_text(encoding="utf-8")
    assert (outputs["scenic_program_dir"] / "maps" / "Town.xodr").exists()
    manifest = json.loads(outputs["scenic_execution_manifest"].read_text(encoding="utf-8"))
    assert manifest["program_count"] == 1
    assert manifest["programs"][0]["native_execution_ready"] is True


def test_native_scenic_anchor_evolves_without_downgrading_to_generated_ir(tmp_path: Path) -> None:
    source_dir = tmp_path / "scenic"
    source_dir.mkdir()
    source = source_dir / "carlaChallenge1.scenic"
    source.write_text(
        "param map = localPath('../maps/Town.xodr')\n"
        "model scenic.domains.driving.model\n"
        "EGO_SPEED = 10\n"
        "ego = Car at spawn\n",
        encoding="utf-8",
    )
    (tmp_path / "maps").mkdir()
    (tmp_path / "maps" / "Town.xodr").write_text("<OpenDRIVE/>", encoding="utf-8")
    source_row = build_scenic_source_bank(source_dir)[0]
    source_row["config"]["ego_speed"] = 10.0
    failure = StructuredFailure(
        event=FailureEvent("r", 1.0, "collision", 8.0, "loss of control", ["control"], "high"),
        actors=["ego"],
        interaction_pattern="control_instability",
        constraints={"ego_speed_min": 0.0, "ego_speed_max": 20.0},
    )
    anchor = ScenarioCandidate(
        scenario_id=source_row["scenario_id"],
        score=1.0,
        tags=source_row["tags"],
        base_config=source_row["config"],
    )
    evolved = evolve_from_anchors(failure, [anchor], escalation_strength=0.2)[0]
    assert evolved.modifications["scenic_program_format"] == "scenic_native_v1"
    assert evolved.modifications["scenic_source_path"] == source_row["config"]["scenic_source_path"]

    outputs = materialize_round_inputs(
        [{"scenario_id": evolved.evolved_id, "tags": [], "pattern": "control_instability", "config": evolved.modifications}],
        tmp_path / "round1",
        {"materialization": {"scenic": {"enabled": True}, "bench2drive": {"enabled": False}}},
    )
    native_path = outputs["scenic_program_dir"] / "native" / f"{evolved.evolved_id}.scenic"
    assert "EGO_SPEED = 12" in native_path.read_text(encoding="utf-8")

    capped_failure = StructuredFailure(
        event=FailureEvent("r", 2.0, "collision", 0.0, "loss of control", ["control"], "high"),
        actors=["ego"],
        interaction_pattern="control_instability",
        constraints={"ego_speed_min": 0.0, "ego_speed_max": 3.0},
    )
    no_op = evolve_from_anchors(capped_failure, [anchor], escalation_strength=0.2)[0]
    assert no_op.modifications["ego_speed"] == 10.0
    assert no_op.modifications["scenic_validation_errors"] == [
        "native Scenic evolution changed no executable controls"
    ]


def test_native_merge_evolution_changes_real_response_threshold_when_speed_is_capped(tmp_path: Path) -> None:
    source_dir = tmp_path / "scenic"
    source_dir.mkdir()
    (source_dir / "carlaChallenge5.scenic").write_text(
        "param map = localPath('../maps/Town.xodr')\n"
        "model scenic.domains.driving.model\n"
        "EGO_SPEED = 10\n"
        "DIST_THRESHOLD = 15\n"
        "ego = Car at spawn\n",
        encoding="utf-8",
    )
    (tmp_path / "maps").mkdir()
    (tmp_path / "maps" / "Town.xodr").write_text("<OpenDRIVE/>", encoding="utf-8")
    source_row = build_scenic_source_bank(source_dir)[0]
    failure = StructuredFailure(
        event=FailureEvent("route_1711", 200.0, "low_speed", 0.0, "cut-in", ["merge"], "medium"),
        actors=["ego", "vehicle"],
        interaction_pattern="adversarial_merge",
        constraints={"ego_speed_min": 0.0, "ego_speed_max": 3.0},
    )
    anchor = ScenarioCandidate(
        scenario_id=source_row["scenario_id"],
        score=1.0,
        tags=source_row["tags"],
        base_config=source_row["config"],
    )
    evolved = evolve_from_anchors(failure, [anchor], escalation_strength=0.2)[0]
    assert evolved.modifications["ego_speed"] == 10.0
    assert evolved.modifications["response_distance"] == 12.0
    assert evolved.modifications["scenic_validation_errors"] == []

    outputs = materialize_round_inputs(
        [{"scenario_id": evolved.evolved_id, "tags": [], "pattern": "adversarial_merge", "config": evolved.modifications}],
        tmp_path / "round1",
        {"materialization": {"scenic": {"enabled": True}, "bench2drive": {"enabled": False}}},
    )
    native_path = outputs["scenic_program_dir"] / "native" / f"{evolved.evolved_id}.scenic"
    content = native_path.read_text(encoding="utf-8")
    assert "DIST_THRESHOLD = 12\n" in content
    assert "EGO_SPEED = 10\n" in content


def test_scenic_readiness_does_not_depend_on_install_time(tmp_path: Path, monkeypatch) -> None:
    from core import scenic_library

    source_dir = tmp_path / "scenic"
    source_dir.mkdir()
    (tmp_path / "maps").mkdir()
    (tmp_path / "maps" / "Town.xodr").write_text("<OpenDRIVE/>", encoding="utf-8")
    (source_dir / "example.scenic").write_text(
        "param map = localPath('../maps/Town.xodr')\n"
        "model scenic.domains.driving.model\n"
        "ego = Car at spawn\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(scenic_library, "_scenic_runtime_available", lambda: False)
    index = index_scenic_programs(source_dir)
    assert index["runtime"]["available"] is False
    assert index["programs"][0]["native_execution_ready"] is True
