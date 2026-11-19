from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any

from .bench2drive_catalog import known_scenario_types, supported_parameter_tags, IGNORED_PARAMETER_TAGS, infer_scenario_family


def validate_scenario_row(row: dict[str, Any], *, require_executable: bool = False) -> list[str]:
    """Validate the executable contract shared by XML and Scenic artifacts.

    This is a simulator-independent preflight.  A later CARLA execution is
    still required to validate collisions, spawn feasibility, and behavior.
    """
    errors: list[str] = []
    scenario_id = str(row.get("scenario_id", ""))
    config = row.get("config", {})
    if not isinstance(config, dict):
        return [f"{scenario_id}: config must be an object"]

    spec = config.get("bench2drive_spec")
    if not isinstance(spec, dict):
        if require_executable:
            errors.append(f"{scenario_id}: missing bench2drive_spec")
        return errors

    route = spec.get("route")
    scenario = spec.get("scenario")
    if not isinstance(route, dict):
        errors.append(f"{scenario_id}: missing route spec")
    if not isinstance(scenario, dict):
        errors.append(f"{scenario_id}: missing scenario spec")
        return errors

    if not str(route.get("id", "")).strip():
        errors.append(f"{scenario_id}: route id is empty")
    if not str(route.get("town", "")).strip():
        errors.append(f"{scenario_id}: route town is empty")
    waypoints = route.get("waypoints", [])
    if not isinstance(waypoints, list) or not waypoints:
        errors.append(f"{scenario_id}: route has no waypoints")

    scenario_type = str(scenario.get("type", ""))
    if scenario_type not in known_scenario_types():
        errors.append(f"{scenario_id}: unsupported scenario type {scenario_type}")
    trigger = scenario.get("trigger_point")
    if not isinstance(trigger, dict):
        errors.append(f"{scenario_id}: missing trigger point")
    else:
        for key in ("x", "y", "z", "yaw"):
            try:
                if not math.isfinite(float(trigger.get(key, 0.0))):
                    errors.append(f"{scenario_id}: trigger {key} is not finite")
            except (TypeError, ValueError):
                errors.append(f"{scenario_id}: trigger {key} is not numeric")

    parameters = scenario.get("parameters", [])
    allowed_tags = supported_parameter_tags() | set(IGNORED_PARAMETER_TAGS)
    if not isinstance(parameters, list):
        errors.append(f"{scenario_id}: scenario parameters must be a list")
        parameters = []
    for parameter in parameters:
        if not isinstance(parameter, dict):
            errors.append(f"{scenario_id}: invalid parameter entry")
            continue
        tag = str(parameter.get("tag", ""))
        if tag not in allowed_tags:
            errors.append(f"{scenario_id}: unsupported parameter tag {tag}")
        attrs = parameter.get("attrs", {})
        if not isinstance(attrs, dict):
            errors.append(f"{scenario_id}: parameter {tag} attrs must be an object")

    for key, value in config.items():
        if key in {"tags", "bench2drive_spec", "bench2drive_profile", "evolution_provenance", "scenic_program"}:
            continue
        if isinstance(value, (int, float)) and not math.isfinite(float(value)):
            errors.append(f"{scenario_id}: config {key} is not finite")
        if key.endswith("speed") and isinstance(value, (int, float)) and float(value) < 0:
            errors.append(f"{scenario_id}: config {key} is negative")

    return errors


def render_scenic_program(row: dict[str, Any]) -> str:
    """Render a deterministic Scenic-style program artifact for a scenario.

    The program records route geometry, actor roles, semantic constraints, and
    failure-sensitive parameters in a human-auditable form.  Projects with the
    Scenic runtime can use this artifact as the source for their CARLA model;
    FD-SET also keeps the Bench2Drive XML compiler as a no-Scenic fallback.
    """
    scenario_id = str(row.get("scenario_id", "generated_scenario"))
    config = row.get("config", {})
    if not isinstance(config, dict):
        config = {}
    spec = config.get("bench2drive_spec", {})
    if not isinstance(spec, dict):
        spec = {}
    route = spec.get("route", {})
    scenario = spec.get("scenario", {})
    if not isinstance(route, dict):
        route = {}
    if not isinstance(scenario, dict):
        scenario = {}
    scenario_type = str(scenario.get("type", "ControlLoss"))
    profile = config.get("bench2drive_profile", {})
    if not isinstance(profile, dict):
        profile = {}
    family = str(profile.get("scenario_family", ""))
    family = family or infer_scenario_family(scenario_type, str(config.get("interaction_pattern", "")))
    trigger = scenario.get("trigger_point", {})
    if not isinstance(trigger, dict):
        trigger = {}
    waypoints = route.get("waypoints", [])
    if not isinstance(waypoints, list):
        waypoints = []

    lines = [
        f"# FD-SET scenario program: {scenario_id}",
        "# This source is generated from the structured scenario contract.",
        f"param scenario_id = {scenario_id!r}",
        f"param scenario_type = {scenario_type!r}",
        f"param scenario_family = {family!r}",
        f"param town = {str(route.get('town', 'Town01'))!r}",
        f"param ego_speed = {float(config.get('ego_speed', 0.0)):.3f}",
        f"param trigger_point = ({float(trigger.get('x', 0.0)):.3f}, {float(trigger.get('y', 0.0)):.3f}, {float(trigger.get('z', 0.0)):.3f}, {float(trigger.get('yaw', 0.0)):.3f})",
        "",
        "# Failure-relevant controls",
    ]
    for key in sorted(config):
        value = config[key]
        if key in {"bench2drive_spec", "bench2drive_profile", "tags", "scenic_program", "evolution_provenance"}:
            continue
        if isinstance(value, (int, float, str)):
            lines.append(f"param {key} = {value!r}")

    lines.extend(["", "# Route waypoints"]) 
    for index, waypoint in enumerate(waypoints):
        if not isinstance(waypoint, dict):
            continue
        lines.append(
            f"waypoint_{index} = ({float(waypoint.get('x', 0.0)):.3f}, "
            f"{float(waypoint.get('y', 0.0)):.3f}, {float(waypoint.get('z', 0.0)):.3f})"
        )
    lines.extend(
        [
            "",
            "# The CARLA adapter binds these symbolic actors to the selected model.",
            "ego = Actor('ego')",
            "traffic = Actor('traffic')",
            "require ego_speed >= 0",
            "require scenario_type != ''",
        ]
    )
    return "\n".join(lines) + "\n"



def render_native_scenic_variant(row: dict[str, Any]) -> tuple[str, list[str]]:
    """Render a source-backed Scenic variant and report controls not applied.

    The supplied programs use Scenic 2 constants for most safety controls.
    Numeric constants are safely rewritten when their names are known; all
    other reconstructed values are retained in a machine-readable comment and
    manifest so no silent parameter loss occurs.
    """
    config = row.get("config", {})
    if not isinstance(config, dict):
        config = {}
    source = _read_native_source(config)
    applied: set[str] = set()
    aliases = {
        "ego_speed": ("EGO_SPEED",),
        "ped_speed": ("PED_SPEED", "BICYCLE_MIN_SPEED"),
        "traffic_frequency": ("TRAFFIC_FREQUENCY", "FREQUENCY"),
        "spawn_gap": ("SPAWN_GAP",),
        "obstacle_dist": ("OBSTACLE_DIST",),
        "merge_offset": ("MERGE_OFFSET",),
        "front_vehicle_distance": ("FRONT_VEHICLE_DISTANCE",),
        "behind_vehicle_distance": ("BEHIND_VEHICLE_DISTANCE",),
        "lane_offset": ("LANE_OFFSET",),
        "crossing_angle": ("CROSSING_ANGLE",),
        "response_distance": ("DIST_THRESHOLD", "EGO_BRAKING_THRESHOLD"),
    }
    variant = source
    for config_key, constant_names in aliases.items():
        value = config.get(config_key)
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            continue
        for constant_name in constant_names:
            pattern = re.compile(
                rf"(?m)^(\s*{re.escape(constant_name)}\s*=\s*)"
                rf"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)(\s*(?:#.*)?$)"
            )
            updated, count = pattern.subn(rf"\g<1>{_format_scenic_number(value)}\g<3>", variant, count=1)
            if count:
                variant = updated
                applied.add(config_key)
                break

    requested = {
        key
        for key, value in config.items()
        if key in aliases and isinstance(value, (int, float)) and not isinstance(value, bool)
    }
    unapplied = sorted(requested - applied)
    scenario_id = str(row.get("scenario_id", "native_variant"))
    header = (
        f"# FD-SET native Scenic variant: {scenario_id}\n"
        f"# applied_controls: {sorted(applied)!r}\n"
        f"# unapplied_controls: {unapplied!r}\n"
    )
    return header + variant.lstrip("\n"), unapplied


def _read_native_source(config: dict[str, Any]) -> str:
    source = config.get("scenic_program")
    if isinstance(source, str) and source.strip():
        return source
    source_path = str(config.get("scenic_source_path", "")).strip()
    if source_path:
        path = Path(source_path).expanduser()
        if path.exists():
            return path.read_text(encoding="utf-8")
    return ""


def _format_scenic_number(value: int | float) -> str:
    number = float(value)
    if number.is_integer():
        return str(int(number))
    return f"{number:.6g}"
