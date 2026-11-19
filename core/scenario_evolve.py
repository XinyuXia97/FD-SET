from __future__ import annotations

import random
from typing import Any

from .bench2drive_catalog import build_parameter_overrides, infer_scenario_family
from .models import EvolvedScenario, ScenarioCandidate, StructuredFailure
from .scenario_program import render_native_scenic_variant, render_scenic_program, validate_scenario_row


def evolve_from_anchors(
    failure: StructuredFailure,
    anchors: list[ScenarioCandidate],
    escalation_strength: float = 0.2,
    *,
    use_escalation: bool = True,
    reconstruction: bool = True,
    generation_tag: str | None = None,
) -> list[EvolvedScenario]:
    evolved: list[EvolvedScenario] = []
    for idx, anchor in enumerate(anchors):
        applied_strength = escalation_strength if use_escalation else 0.0
        mods = _build_modifications(
            anchor.base_config,
            failure,
            escalation_strength,
            use_escalation=use_escalation,
            reconstruction=reconstruction,
        )
        parameter_overrides = _build_bench2drive_parameter_overrides(anchor.base_config, mods)
        if parameter_overrides:
            mods["bench2drive_parameter_overrides"] = parameter_overrides
        evolved_id = _build_evolved_id(anchor.scenario_id, "evo", idx, generation_tag)
        _attach_program_artifacts(mods, evolved_id)
        evolved.append(
            EvolvedScenario(
                source_scenario_id=anchor.scenario_id,
                evolved_id=evolved_id,
                modifications=mods,
                rationale=f"Escalate {failure.interaction_pattern} with bounded perturbation",
                metadata={
                    "anchor_score": anchor.score,
                    "anchor_score_breakdown": anchor.score_breakdown or {},
                    "anchor_tags": anchor.tags,
                    "use_escalation": use_escalation,
                    "applied_escalation_strength": applied_strength,
                    "failure_interaction_pattern": failure.interaction_pattern,
                    "bench2drive_parameter_overrides": parameter_overrides,
                },
            )
        )
    return evolved


def random_perturb_from_anchors(
    failure: StructuredFailure,
    anchors: list[ScenarioCandidate],
    *,
    strength: float = 0.2,
    seed: int = 2026,
    generation_tag: str | None = None,
) -> list[EvolvedScenario]:
    """Generate the paper's unguided perturbation ablation deterministically."""
    evolved: list[EvolvedScenario] = []
    for idx, anchor in enumerate(anchors):
        rng = random.Random(seed + idx)
        mods = dict(anchor.base_config)
        for key in (
            "ego_speed",
            "merge_offset",
            "spawn_gap",
            "obstacle_dist",
            "ped_speed",
            "ped_travel_dist",
            "traffic_frequency",
            "lane_offset",
            "front_vehicle_distance",
            "behind_vehicle_distance",
        ):
            if key not in mods:
                continue
            value = float(mods[key])
            factor = 1.0 + rng.uniform(-abs(strength), abs(strength))
            mods[key] = max(0.1, round(value * factor, 3))

        mods["target_failure_type"] = failure.event.failure_type
        mods["risk_level"] = failure.event.risk_level
        mods["interaction_pattern"] = failure.interaction_pattern
        mods["tags"] = list(dict.fromkeys(failure.event.scene_tags + ["random_perturbation"]))
        parameter_overrides = _build_bench2drive_parameter_overrides(anchor.base_config, mods)
        if parameter_overrides:
            mods["bench2drive_parameter_overrides"] = parameter_overrides
        evolved_id = _build_evolved_id(anchor.scenario_id, "random", idx, generation_tag)
        _attach_program_artifacts(mods, evolved_id)
        evolved.append(
            EvolvedScenario(
                source_scenario_id=anchor.scenario_id,
                evolved_id=evolved_id,
                modifications=mods,
                rationale="Unguided bounded random perturbation for ablation",
                metadata={
                    "anchor_score": anchor.score,
                    "anchor_score_breakdown": anchor.score_breakdown or {},
                    "anchor_tags": anchor.tags,
                    "random_seed": seed + idx,
                    "failure_interaction_pattern": failure.interaction_pattern,
                    "bench2drive_parameter_overrides": parameter_overrides,
                },
            )
        )
    return evolved


def _build_modifications(
    base_cfg: dict,
    failure: StructuredFailure,
    escalation_strength: float,
    *,
    use_escalation: bool,
    reconstruction: bool,
) -> dict:
    cfg = dict(base_cfg)
    if cfg.get("scenic_program_format") == "scenic_native_v1":
        constants = cfg.get("scenic_constants", {})
        if isinstance(constants, dict):
            if "ego_speed" not in cfg and "EGO_SPEED" in constants:
                try:
                    cfg["ego_speed"] = float(constants["EGO_SPEED"])
                except (TypeError, ValueError):
                    pass
            # Native response thresholds control when ego reacts to nearby actors.
            if "response_distance" not in cfg:
                for name in ("DIST_THRESHOLD", "EGO_BRAKING_THRESHOLD"):
                    try:
                        cfg["response_distance"] = float(constants[name])
                    except (KeyError, TypeError, ValueError):
                        continue
                    break
    strength = escalation_strength if use_escalation else 0.0
    low = float(failure.constraints.get("ego_speed_min", 0.0))
    high = float(failure.constraints.get("ego_speed_max", 10.0))

    if not reconstruction:
        cfg["target_failure_type"] = failure.event.failure_type
        cfg["risk_level"] = failure.event.risk_level
        cfg["interaction_pattern"] = failure.interaction_pattern
        cfg["tags"] = list(dict.fromkeys(failure.event.scene_tags + [failure.interaction_pattern]))
        return cfg

    if "ego_speed" in cfg:
        base_speed = float(cfg["ego_speed"])
        if cfg.get("scenic_program_format") != "scenic_native_v1" or high >= base_speed:
            cfg["ego_speed"] = min(high, base_speed * (1.0 + strength))
    else:
        cfg["ego_speed"] = min(high, max(low, (low + high) / 2.0))

    if failure.interaction_pattern in {"adversarial_merge", "late_response", "junction_conflict"}:
        if "response_distance" in cfg:
            cfg["response_distance"] = max(3.0, float(cfg["response_distance"]) * (1.0 - strength))
    if failure.interaction_pattern == "adversarial_merge":
        cfg["merge_offset"] = max(5.0, float(cfg.get("merge_offset", 15.0)) * (1.0 - strength))
        if "front_vehicle_distance" in cfg:
            cfg["front_vehicle_distance"] = max(3.0, float(cfg["front_vehicle_distance"]) * (1.0 - strength))
        if "behind_vehicle_distance" in cfg:
            cfg["behind_vehicle_distance"] = max(3.0, float(cfg["behind_vehicle_distance"]) * (1.0 - strength))
    if failure.interaction_pattern == "late_response":
        cfg["obstacle_dist"] = max(8.0, float(cfg.get("obstacle_dist", 20.0)) * (1.0 - strength))
        if "traffic_frequency" in cfg:
            cfg["traffic_frequency"] = float(cfg["traffic_frequency"]) * (1.0 + strength)
        if "front_vehicle_distance" in cfg:
            cfg["front_vehicle_distance"] = max(3.0, float(cfg["front_vehicle_distance"]) * (1.0 - strength))
        if "behind_vehicle_distance" in cfg:
            cfg["behind_vehicle_distance"] = max(3.0, float(cfg["behind_vehicle_distance"]) * (1.0 - strength))
    if failure.interaction_pattern == "junction_conflict":
        if "merge_offset" in cfg:
            cfg["merge_offset"] = max(5.0, float(cfg["merge_offset"]) * (1.0 - strength))
        if "obstacle_dist" in cfg:
            cfg["obstacle_dist"] = max(8.0, float(cfg["obstacle_dist"]) * (1.0 - strength))
        if "traffic_frequency" in cfg:
            cfg["traffic_frequency"] = float(cfg["traffic_frequency"]) * (1.0 + strength)
        if "lane_offset" in cfg:
            cfg["lane_offset"] = min(1.0, float(cfg["lane_offset"]) * (1.0 + strength))
        if "front_vehicle_distance" in cfg:
            cfg["front_vehicle_distance"] = max(3.0, float(cfg["front_vehicle_distance"]) * (1.0 - strength))
        if "behind_vehicle_distance" in cfg:
            cfg["behind_vehicle_distance"] = max(3.0, float(cfg["behind_vehicle_distance"]) * (1.0 - strength))
    if failure.interaction_pattern == "control_instability":
        if "lane_offset" in cfg:
            cfg["lane_offset"] = min(1.5, float(cfg["lane_offset"]) * (1.0 + strength))
        else:
            cfg["lane_offset"] = round(0.15 * (1.0 + strength), 3)
    if failure.interaction_pattern == "pedestrian_intrusion":
        cfg["spawn_gap"] = max(5.0, float(cfg.get("spawn_gap", 14.0)) * (1.0 - strength))
        if "ped_speed" in cfg:
            cfg["ped_speed"] = max(0.5, float(cfg["ped_speed"]) * (1.0 + strength * 0.5))
        if "ped_travel_dist" in cfg:
            cfg["ped_travel_dist"] = max(5.0, float(cfg["ped_travel_dist"]) * (1.0 + strength))
        if "crossing_angle" in cfg:
            cfg["crossing_angle"] = min(90.0, float(cfg["crossing_angle"]) + (10.0 * strength))
        if "traffic_frequency" in cfg:
            cfg["traffic_frequency"] = float(cfg["traffic_frequency"]) * (1.0 + strength)

    cfg["target_failure_type"] = failure.event.failure_type
    cfg["risk_level"] = failure.event.risk_level
    cfg["interaction_pattern"] = failure.interaction_pattern
    cfg["tags"] = list(dict.fromkeys(failure.event.scene_tags + [failure.interaction_pattern]))
    return cfg


def _attach_program_artifacts(config: dict[str, Any], scenario_id: str) -> None:
    row = {"scenario_id": scenario_id, "config": config}
    if str(config.get("scenic_program_format", "")).strip() == "scenic_native_v1":
        source = config.get("scenic_program")
        if isinstance(source, str) and source.strip():
            rendered, _ = render_native_scenic_variant(row)
            config["scenic_validation_errors"] = (
                ["native Scenic evolution changed no executable controls"]
                if rendered.endswith(source.lstrip("\n"))
                else []
            )
        else:
            config.setdefault("scenic_validation_errors", list(config.get("scenic_diagnostics", [])))
        config["scenic_variant_of"] = config.get("scenic_program_id", config.get("scenic_relative_path", ""))
        config["scenic_variant_scenario_id"] = scenario_id
        return
    errors = validate_scenario_row(row, require_executable=False)
    config["scenario_program_format"] = "scenic_compatible_v1"
    config["scenic_validation_errors"] = errors
    config["scenic_program"] = render_scenic_program(row)


def _build_evolved_id(anchor_id: str, strategy: str, index: int, generation_tag: str | None) -> str:
    tag = str(generation_tag or "").strip()
    suffix = f"_{tag}" if tag else ""
    return f"{anchor_id}_{strategy}{suffix}_{index}"


def _build_bench2drive_parameter_overrides(base_cfg: dict[str, Any], evolved_cfg: dict[str, Any]) -> dict[str, dict[str, str]]:
    spec = base_cfg.get("bench2drive_spec", {})
    if not isinstance(spec, dict):
        return {}

    scenario_spec = spec.get("scenario", {})
    if not isinstance(scenario_spec, dict):
        return {}

    raw_parameters = scenario_spec.get("parameters", [])
    if not isinstance(raw_parameters, list):
        return {}
    profile = base_cfg.get("bench2drive_profile", {})
    scenario_type = str(scenario_spec.get("type", ""))
    scenario_family = str(profile.get("scenario_family", "")).strip().lower() or infer_scenario_family(
        scenario_type,
        str(evolved_cfg.get("interaction_pattern", "")),
    )
    return build_parameter_overrides(
        raw_parameters,
        scenario_type=scenario_type,
        config=evolved_cfg,
        scenario_family=scenario_family,
    )
