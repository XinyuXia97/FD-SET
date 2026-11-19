from __future__ import annotations

import json
import math
import random
import subprocess
import sys
from pathlib import Path
from typing import Any


DEFAULT_MODEL_PROFILES = {
    "uniad": {
        "agent": "uniad",
        "train_command": "",
    },
    "vad": {
        "agent": "vad",
        "train_command": "",
    },
    "admlp": {
        "agent": "admlp",
        "train_command": "",
    },
    "tcp": {
        "agent": "tcp",
        "train_command": "",
    },
}


def select_adaptation_scenarios(
    scenario_bank: list[dict[str, Any]],
    *,
    max_scenarios: int | None = None,
    seed: int = 2026,
) -> list[dict[str, Any]]:
    """Select only scenario-bank rows for downstream adaptation.

    Official evaluation routes are intentionally rejected by role/split so the
    adaptation set cannot silently leak the held-out benchmark.
    """
    candidates: list[dict[str, Any]] = []
    for row in scenario_bank:
        if not isinstance(row, dict):
            continue
        config = row.get("config", {})
        if not isinstance(config, dict):
            continue
        role = str(config.get("scenario_role", "scenario_bank")).strip().lower()
        profile = config.get("bench2drive_profile", {})
        split = str(profile.get("dataset_split", "")) if isinstance(profile, dict) else ""
        if role == "evaluation" or split == "evaluation":
            raise ValueError(f"Evaluation scenario leaked into adaptation bank: {row.get('scenario_id', '')}")
        if role == "evolved":
            candidates.append(row)

    rng = random.Random(seed)
    rng.shuffle(candidates)
    if max_scenarios is not None:
        return candidates[: max(0, int(max_scenarios))]
    return candidates


def build_kinematic_expert_demo(row: dict[str, Any], *, expert_name: str = "pdm_lite_proxy") -> dict[str, Any]:
    """Create a deterministic route-level expert record when PDM-Lite is absent.

    This record is a contract fixture, not a claim of PDM-Lite equivalence.  A
    real reproduction should replace it with the external PDM-Lite command.
    """
    config = row.get("config", {})
    if not isinstance(config, dict):
        config = {}
    spec = config.get("bench2drive_spec", {})
    if not isinstance(spec, dict):
        spec = {}
    route = spec.get("route", {})
    if not isinstance(route, dict):
        route = {}
    waypoints = route.get("waypoints", [])
    if not isinstance(waypoints, list):
        waypoints = []
    target_speed = max(0.0, float(config.get("ego_speed", 8.0)))
    actions: list[dict[str, float]] = []
    previous_yaw: float | None = None
    for index in range(max(0, len(waypoints) - 1)):
        current = waypoints[index]
        following = waypoints[index + 1]
        if not isinstance(current, dict) or not isinstance(following, dict):
            continue
        dx = float(following.get("x", 0.0)) - float(current.get("x", 0.0))
        dy = float(following.get("y", 0.0)) - float(current.get("y", 0.0))
        distance = math.hypot(dx, dy)
        yaw = math.atan2(dy, dx)
        steer = 0.0 if previous_yaw is None else max(-1.0, min(1.0, (yaw - previous_yaw) / math.pi))
        actions.append(
            {
                "target_speed": target_speed,
                "steer": steer,
                "throttle": 0.5 if target_speed > 0 else 0.0,
                "brake": 0.0,
                "distance_m": distance,
            }
        )
        previous_yaw = yaw

    return {
        "scenario_id": row.get("scenario_id"),
        "route_id": route.get("id"),
        "town": route.get("town"),
        "expert": expert_name,
        "expert_backend": "kinematic_proxy",
        "actions": actions,
        "source": "scenario_bank_only",
    }


def write_proxy_expert_demos(rows: list[dict[str, Any]], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(build_kinematic_expert_demo(row), ensure_ascii=False) + "\n")
    return output_path


def build_expert_generation_command(
    template: str,
    *,
    scenario_bank_path: Path,
    output_dir: Path,
    seed: int,
    scenic_program_dir: Path | None = None,
    scenic_execution_manifest: Path | None = None,
    routes_xml: Path | None = None,
    demos_file: Path | None = None,
    round_id: int | None = None,
    carla_host: str | None = None,
    carla_port: int | None = None,
    carla_tm_port: int | None = None,
    carla_pythonapi_root: Path | str | None = None,
    carla_root: Path | str | None = None,
    expert_camera_profile: str | None = None,
) -> str:
    values: dict[str, Any] = {
        "scenario_bank": str(scenario_bank_path.resolve()),
        "output_dir": str(output_dir.resolve()),
        "seed": seed,
        "python": sys.executable,
        "expert_camera_profile": str(expert_camera_profile or "admlp"),
    }
    optional_paths = {
        "scenic_program_dir": scenic_program_dir,
        "scenic_execution_manifest": scenic_execution_manifest,
        "routes_xml": routes_xml,
        "demos_file": demos_file,
    }
    for key, path in optional_paths.items():
        values[key] = str(path.resolve()) if path is not None else ""
    if round_id is not None:
        values["round_id"] = int(round_id)
    if carla_host is not None:
        values["carla_host"] = str(carla_host)
    if carla_port is not None:
        values["carla_port"] = int(carla_port)
    if carla_tm_port is not None:
        values["carla_tm_port"] = int(carla_tm_port)
    if carla_pythonapi_root is not None:
        values["carla_pythonapi_root"] = str(carla_pythonapi_root)
    if carla_root is not None:
        values["carla_root"] = str(carla_root)
    return template.format(
        **values,
    )


def run_expert_generation(command: str, *, cwd: Path | None = None, env: dict[str, str] | None = None) -> None:
    import os

    child_env = os.environ.copy()
    if env:
        child_env.update(env)
    subprocess.run(command, shell=True, check=True, cwd=str(cwd) if cwd else None, env=child_env)


def build_adaptation_manifest(
    scenario_bank: list[dict[str, Any]],
    *,
    held_out_routes_xml: Path,
    output_dir: Path,
    config: dict[str, Any] | None = None,
    round_id: int | None = None,
) -> dict[str, Any]:
    config = config or {}
    seed = int(config.get("seed", 2026))
    selected = select_adaptation_scenarios(
        scenario_bank,
        max_scenarios=config.get("max_scenarios"),
        seed=seed,
    )
    model_profiles = config.get("models", DEFAULT_MODEL_PROFILES)
    if isinstance(model_profiles, dict):
        models = []
        for name, profile in model_profiles.items():
            row = dict(DEFAULT_MODEL_PROFILES.get(str(name).lower(), {}))
            if isinstance(profile, dict):
                row.update(profile)
            row["name"] = str(name)
            models.append(row)
    elif isinstance(model_profiles, list):
        models = [dict(row) for row in model_profiles if isinstance(row, dict)]
    else:
        models = []

    return {
        "protocol": "benchmark_feedback_scenic_pdm_finetune_v1",
        "round": round_id,
        "seed": seed,
        "held_out_routes_xml": str(held_out_routes_xml),
        "benchmark_role": "feedback_and_re_evaluation_only",
        "adaptation_source": "scenic_evolved_scenarios_only",
        "expert": {
            "name": str(config.get("expert", {}).get("name", "PDM-Lite"))
            if isinstance(config.get("expert", {}), dict)
            else "PDM-Lite",
            "backend": str(config.get("expert", {}).get("backend", "external"))
            if isinstance(config.get("expert", {}), dict)
            else "external",
        },
        "scenario_ids": [str(row.get("scenario_id", "")) for row in selected],
        "num_scenarios": len(selected),
        "models": models,
        "output_dir": str(output_dir),
        "notes": [
            "The benchmark XML is used only to evaluate the current model and provide failure feedback.",
            "Only Scenic-generated evolved scenarios are passed to the PDM-Lite expert.",
            "The benchmark XML never generates expert demonstrations or training labels.",
        ],
    }


def write_adaptation_manifest(manifest: dict[str, Any], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return output_path
