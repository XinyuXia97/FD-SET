from __future__ import annotations

from typing import Any


PATTERN_BY_SCENARIO_TYPE: dict[str, str] = {
    "Accident": "late_response",
    "AccidentTwoWays": "late_response",
    "BlockedIntersection": "junction_conflict",
    "ConstructionObstacle": "late_response",
    "ConstructionObstacleTwoWays": "late_response",
    "ControlLoss": "control_instability",
    "CrossingBicycleFlow": "pedestrian_intrusion",
    "DynamicObjectCrossing": "pedestrian_intrusion",
    "EnterActorFlow": "adversarial_merge",
    "EnterActorFlowV2": "adversarial_merge",
    "HardBreakRoute": "control_instability",
    "HazardAtSideLane": "pedestrian_intrusion",
    "HazardAtSideLaneTwoWays": "pedestrian_intrusion",
    "HighwayCutIn": "adversarial_merge",
    "HighwayExit": "adversarial_merge",
    "InterurbanActorFlow": "adversarial_merge",
    "InterurbanAdvancedActorFlow": "adversarial_merge",
    "InvadingTurn": "junction_conflict",
    "MergerIntoSlowTraffic": "adversarial_merge",
    "MergerIntoSlowTrafficV2": "adversarial_merge",
    "NonSignalizedJunctionLeftTurn": "junction_conflict",
    "NonSignalizedJunctionLeftTurnEnterFlow": "junction_conflict",
    "NonSignalizedJunctionRightTurn": "junction_conflict",
    "OppositeVehicleRunningRedLight": "signal_violation",
    "OppositeVehicleTakingPriority": "junction_conflict",
    "ParkedObstacle": "late_response",
    "ParkedObstacleTwoWays": "late_response",
    "ParkingCrossingPedestrian": "pedestrian_intrusion",
    "ParkingCutIn": "adversarial_merge",
    "ParkingExit": "adversarial_merge",
    "PedestrianCrossing": "pedestrian_intrusion",
    "PriorityAtJunction": "junction_conflict",
    "SequentialLaneChange": "adversarial_merge",
    "SignalizedJunctionLeftTurn": "junction_conflict",
    "SignalizedJunctionLeftTurnEnterFlow": "junction_conflict",
    "SignalizedJunctionRightTurn": "junction_conflict",
    "StaticCutIn": "adversarial_merge",
    "T_Junction": "junction_conflict",
    "VanillaNonSignalizedTurn": "junction_conflict",
    "VanillaNonSignalizedTurnEncounterStopsign": "junction_conflict",
    "VanillaSignalizedTurnEncounterGreenLight": "junction_conflict",
    "VanillaSignalizedTurnEncounterRedLight": "signal_violation",
    "VehicleOpensDoorTwoWays": "late_response",
    "VehicleTurningRoute": "junction_conflict",
    "VehicleTurningRoutePedestrian": "pedestrian_intrusion",
    "YieldToEmergencyVehicle": "late_response",
}

FAMILY_BY_PATTERN = {
    "adversarial_merge": "merge",
    "pedestrian_intrusion": "pedestrian",
    "late_response": "obstacle",
    "signal_violation": "signal",
    "junction_conflict": "junction",
    "control_instability": "control",
}

SCALAR_CONTROL_BY_TAG = {
    "speed": "ego_speed",
    "flow_speed": "ego_speed",
    "bicycle_speed": "ped_speed",
    "bicycle_drive_distance": "ped_travel_dist",
    "crossing_angle": "crossing_angle",
    "direction": "turn_direction",
    "offset": "lane_offset",
    "front_vehicle_distance": "front_vehicle_distance",
    "behind_vehicle_distance": "behind_vehicle_distance",
    "blocker_model": "blocker_model",
}

IGNORED_PARAMETER_TAGS = frozenset(
    {
        "other_actor_location",
        "end_location",
        "start_actor_flow",
        "end_actor_flow",
    }
)


def infer_scenario_pattern(scenario_type: str) -> str:
    return PATTERN_BY_SCENARIO_TYPE.get(scenario_type, "generic_failure")


def infer_scenario_family(scenario_type: str, pattern: str | None = None) -> str:
    resolved_pattern = pattern or infer_scenario_pattern(scenario_type)
    return FAMILY_BY_PATTERN.get(resolved_pattern, "generic")


def infer_config_from_parameters(
    parameters: list[dict[str, Any]],
    *,
    scenario_type: str,
    pattern: str | None = None,
) -> dict[str, float | str]:
    inferred: dict[str, float | str] = {}
    scenario_family = infer_scenario_family(scenario_type, pattern)
    distance_key = _distance_control_key(scenario_family)

    for parameter in parameters:
        tag = str(parameter.get("tag", ""))
        attrs = parameter.get("attrs", {})
        if not isinstance(attrs, dict):
            continue

        if tag in SCALAR_CONTROL_BY_TAG and "value" in attrs:
            control_key = SCALAR_CONTROL_BY_TAG[tag]
            raw_value = attrs["value"]
            if control_key in {"turn_direction", "blocker_model"}:
                inferred[control_key] = str(raw_value)
            elif control_key == "ego_speed" and control_key in inferred:
                continue
            else:
                inferred[control_key] = float(raw_value)
            continue

        if tag == "distance" and "value" in attrs:
            inferred[distance_key] = float(attrs["value"])
            continue

        if tag == "source_dist_interval" and "to" in attrs:
            inferred["merge_offset"] = float(attrs["to"])
            continue

        if tag == "frequency":
            if "value" in attrs:
                inferred["traffic_frequency"] = float(attrs["value"])
            elif "to" in attrs:
                inferred["traffic_frequency"] = float(attrs["to"])

    return inferred


def build_parameter_overrides(
    parameters: list[dict[str, Any]],
    *,
    scenario_type: str,
    config: dict[str, Any],
    scenario_family: str | None = None,
) -> dict[str, dict[str, str]]:
    family = scenario_family or infer_scenario_family(scenario_type)
    distance_key = _distance_control_key(family)
    overrides: dict[str, dict[str, str]] = {}

    for parameter in parameters:
        tag = str(parameter.get("tag", ""))
        attrs = parameter.get("attrs", {})
        if not isinstance(attrs, dict):
            continue

        if tag in {"flow_speed", "speed"} and "ego_speed" in config:
            overrides[tag] = {"value": _format_scalar(config["ego_speed"])}
        elif tag == "bicycle_speed" and "ped_speed" in config:
            overrides[tag] = {"value": _format_scalar(config["ped_speed"])}
        elif tag == "bicycle_drive_distance" and "ped_travel_dist" in config:
            overrides[tag] = {"value": _format_scalar(config["ped_travel_dist"])}
        elif tag == "crossing_angle" and "crossing_angle" in config:
            overrides[tag] = {"value": _format_scalar(config["crossing_angle"])}
        elif tag == "source_dist_interval" and "merge_offset" in config:
            overrides[tag] = _build_interval_attrs(float(config["merge_offset"]))
        elif tag == "frequency" and "traffic_frequency" in config:
            overrides[tag] = _build_frequency_attrs(float(config["traffic_frequency"]), attrs)
        elif tag == "direction" and "turn_direction" in config:
            overrides[tag] = {"value": str(config["turn_direction"])}
        elif tag == "offset" and "lane_offset" in config:
            overrides[tag] = {"value": _format_scalar(config["lane_offset"])}
        elif tag == "front_vehicle_distance" and "front_vehicle_distance" in config:
            overrides[tag] = {"value": _format_scalar(config["front_vehicle_distance"])}
        elif tag == "behind_vehicle_distance" and "behind_vehicle_distance" in config:
            overrides[tag] = {"value": _format_scalar(config["behind_vehicle_distance"])}
        elif tag == "blocker_model" and "blocker_model" in config:
            overrides[tag] = {"value": str(config["blocker_model"])}
        elif tag == "distance" and distance_key in config:
            overrides[tag] = {"value": _format_scalar(config[distance_key])}

    return overrides


def known_scenario_types() -> set[str]:
    return set(PATTERN_BY_SCENARIO_TYPE)


def supported_parameter_tags() -> set[str]:
    return {
        "distance",
        "direction",
        "speed",
        "flow_speed",
        "bicycle_speed",
        "bicycle_drive_distance",
        "crossing_angle",
        "source_dist_interval",
        "frequency",
        "offset",
        "front_vehicle_distance",
        "behind_vehicle_distance",
        "blocker_model",
    }


def _distance_control_key(scenario_family: str) -> str:
    if scenario_family == "pedestrian":
        return "spawn_gap"
    return "obstacle_dist"


def _build_interval_attrs(value: float) -> dict[str, str]:
    lower = max(5.0, value * 0.6)
    upper = max(lower + 5.0, value)
    return {
        "from": _format_scalar(lower),
        "to": _format_scalar(upper),
    }


def _build_frequency_attrs(value: float, existing_attrs: dict[str, Any]) -> dict[str, str]:
    if "value" in existing_attrs and "from" not in existing_attrs and "to" not in existing_attrs:
        return {"value": _format_scalar(value)}

    lower = max(5.0, value * 0.4)
    upper = max(lower + 5.0, value)
    return {
        "from": _format_scalar(lower),
        "to": _format_scalar(upper),
    }


def _format_scalar(value: Any) -> str:
    if isinstance(value, int):
        return str(value)
    return f"{float(value):.3f}".rstrip("0").rstrip(".")
