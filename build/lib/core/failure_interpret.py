from __future__ import annotations

from pathlib import Path
from typing import Any

from .interpretation_backend import call_model_interpreter
from .models import FailureEvent


DEFAULT_PROMPT_PATH = Path("prompts/vlm_prompt.txt")
ALLOWED_INTERACTION_PATTERNS = {
    "adversarial_merge",
    "pedestrian_intrusion",
    "late_response",
    "signal_violation",
    "junction_conflict",
    "control_instability",
}


def load_interpretation_prompt(prompt_path: Path | None = None) -> str:
    resolved_path = prompt_path or DEFAULT_PROMPT_PATH
    if not resolved_path.exists():
        raise FileNotFoundError(f"Interpretation prompt not found: {resolved_path}")
    return resolved_path.read_text(encoding="utf-8").strip()


def interpret_failure(
    event: FailureEvent,
    *,
    prompt_template: str | None = None,
    prompt_path: Path | None = None,
    config: dict[str, Any] | None = None,
) -> dict[str, object]:
    """Interpret a failure event with either a model backend or deterministic rules."""
    config = config or {}
    backend = str(config.get("backend", "rule_based")).strip().lower()
    resolved_prompt_path = prompt_path or DEFAULT_PROMPT_PATH
    loaded_prompt = prompt_template or load_interpretation_prompt(resolved_prompt_path)

    if backend == "rule_based":
        semantics = _interpret_failure_rule_based(
            event,
            prompt_path=resolved_prompt_path,
            prompt_loaded=bool(loaded_prompt),
        )
        semantics["interpreter"] = {
            "backend": "rule_based_v1",
            "prompt_source": str(resolved_prompt_path),
            "prompt_loaded": bool(loaded_prompt),
        }
        return _normalize_semantics(event, semantics)

    try:
        semantics = call_model_interpreter(
            event,
            prompt_template=loaded_prompt,
            config=config,
        )
        if not isinstance(semantics, dict):
            raise ValueError("Model interpreter returned a non-object payload")
        semantics.setdefault("failure_type", event.failure_type)
        semantics.setdefault("key_entities", ["ego"])
        semantics.setdefault("interaction_pattern", "generic_failure")
        semantics.setdefault("risk_cues", [])
        semantics["interpreter"] = {
            "backend": backend,
            "prompt_source": str(resolved_prompt_path),
            "prompt_loaded": bool(loaded_prompt),
        }
        return _normalize_semantics(event, semantics)
    except Exception as exc:
        if not bool(config.get("fallback_to_rule_based", True)):
            raise
        semantics = _interpret_failure_rule_based(
            event,
            prompt_path=resolved_prompt_path,
            prompt_loaded=bool(loaded_prompt),
        )
        semantics["interpreter"] = {
            "backend": "rule_based_v1",
            "prompt_source": str(resolved_prompt_path),
            "prompt_loaded": bool(loaded_prompt),
            "fallback_reason": str(exc),
            "requested_backend": backend,
        }
        return _normalize_semantics(event, semantics)


def interpret_failures(
    events: list[FailureEvent],
    *,
    prompt_path: Path | None = None,
    config: dict[str, Any] | None = None,
) -> dict[str, dict[str, object]]:
    resolved_prompt_path = prompt_path or DEFAULT_PROMPT_PATH
    prompt_template = load_interpretation_prompt(resolved_prompt_path)
    return {
        f"{event.route_id}:{event.timestamp_s:.2f}": interpret_failure(
            event,
            prompt_template=prompt_template,
            prompt_path=resolved_prompt_path,
            config=config,
        )
        for event in events
    }


def _interpret_failure_rule_based(
    event: FailureEvent,
    *,
    prompt_path: Path,
    prompt_loaded: bool,
) -> dict[str, object]:
    desc = event.description.lower()
    metadata = event.metadata or {}
    infraction_key = str(metadata.get("infraction_key", "")).lower()
    semantics: dict[str, object] = {
        "failure_type": event.failure_type,
        "key_entities": ["ego"],
        "interaction_pattern": "generic_failure",
        "risk_cues": [],
        "capability_weakness": "failure response under mixed traffic",
        "spatial_pattern": ", ".join(event.scene_tags),
        "temporal_pattern": "late or unsafe response",
        "risk_pattern": event.risk_level,
    }

    if "merge" in desc or "cut-in" in desc:
        semantics["interaction_pattern"] = "adversarial_merge"
        semantics["key_entities"] = ["ego", "vehicle"]
        semantics["risk_cues"] = ["lateral_intrusion", "short_ttc"]
    elif "pedestrian" in desc:
        semantics["interaction_pattern"] = "pedestrian_intrusion"
        semantics["key_entities"] = ["ego", "pedestrian"]
        semantics["risk_cues"] = ["occlusion", "late_detection"]
    elif "intersection" in desc or "junction" in desc or "turning" in desc:
        semantics["interaction_pattern"] = "junction_conflict"
        semantics["key_entities"] = ["ego", "vehicle"]
        semantics["risk_cues"] = ["priority_conflict", "crossing_trajectory"]
    elif "late braking" in desc:
        semantics["interaction_pattern"] = "late_response"
        semantics["key_entities"] = ["ego", "static_vehicle"]
        semantics["risk_cues"] = ["delayed_brake"]
    elif "control loss" in desc or "hard brake" in desc or "skid" in desc:
        semantics["interaction_pattern"] = "control_instability"
        semantics["key_entities"] = ["ego"]
        semantics["risk_cues"] = ["unstable_dynamics", "low_friction"]
    elif "red light" in desc:
        semantics["interaction_pattern"] = "signal_violation"
        semantics["risk_cues"] = ["traffic_signal_noncompliance"]
    elif "blocked" in desc:
        semantics["interaction_pattern"] = "blocked_progress"
        semantics["risk_cues"] = ["stalled_agent"]
    elif event.failure_type in {"route_deviation", "offroad"} or infraction_key in {"outside_route_lanes", "route_dev"}:
        semantics["interaction_pattern"] = "route_departure"
        semantics["risk_cues"] = ["lane_departure", "route_loss"]
    elif event.failure_type in {"timeout", "blocked"} or infraction_key in {"route_timeout", "vehicle_blocked", "scenario_timeouts"}:
        semantics["interaction_pattern"] = "blocked_progress"
        semantics["risk_cues"] = ["stalled_agent", "goal_not_reached"]
    elif event.failure_type in {"red_light", "stop_infraction"} or infraction_key in {"red_light", "stop_infraction"}:
        semantics["interaction_pattern"] = "signal_violation"
        semantics["risk_cues"] = ["traffic_rule_noncompliance"]

    capability_by_pattern = {
        "adversarial_merge": "lateral conflict handling",
        "pedestrian_intrusion": "vulnerable-road-user anticipation",
        "late_response": "hazard detection and braking timing",
        "signal_violation": "traffic-rule compliance",
        "junction_conflict": "right-of-way reasoning",
        "control_instability": "ego motion control stability",
        "blocked_progress": "progress under obstruction",
        "route_departure": "route keeping",
    }
    semantics["capability_weakness"] = capability_by_pattern.get(
        str(semantics["interaction_pattern"]),
        "failure response under mixed traffic",
    )

    return semantics


def _normalize_semantics(event: FailureEvent, semantics: dict[str, object]) -> dict[str, object]:
    """Enforce the fixed paper schema and keep downstream patterns canonical."""
    normalized = dict(semantics)
    pattern = str(normalized.get("interaction_pattern", "")).strip()
    pattern_aliases = {
        "blocked_progress": "late_response",
        "route_departure": "control_instability",
        "generic_failure": "control_instability",
    }
    pattern = pattern_aliases.get(pattern, pattern)
    if pattern not in ALLOWED_INTERACTION_PATTERNS:
        pattern = _pattern_from_event(event)
    normalized["failure_type"] = str(normalized.get("failure_type", event.failure_type))
    normalized["interaction_pattern"] = pattern
    normalized["key_entities"] = _normalize_list(normalized.get("key_entities", ["ego"]))
    normalized["risk_cues"] = _normalize_list(normalized.get("risk_cues", []))
    normalized.setdefault("capability_weakness", "failure response under mixed traffic")
    normalized.setdefault("spatial_pattern", ", ".join(event.scene_tags) or "unknown")
    normalized.setdefault("temporal_pattern", "failure-time interaction")
    normalized.setdefault("risk_pattern", event.risk_level)
    normalized.setdefault("road_topology", "unknown")
    normalized.setdefault("traffic_state", "unknown")
    normalized["relative_relations"] = _normalize_list(normalized.get("relative_relations", []))
    normalized["context_conditions"] = _normalize_list(normalized.get("context_conditions", []))
    return normalized


def _normalize_list(value: object) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if value is None:
        return []
    return [str(value).strip()]


def _pattern_from_event(event: FailureEvent) -> str:
    lowered = f"{event.failure_type} {event.description}".lower()
    if "merge" in lowered or "cut-in" in lowered:
        return "adversarial_merge"
    if "pedestrian" in lowered or "bicycle" in lowered or "cyclist" in lowered:
        return "pedestrian_intrusion"
    if "red light" in lowered or "stop" in lowered:
        return "signal_violation"
    if "junction" in lowered or "intersection" in lowered or "turn" in lowered:
        return "junction_conflict"
    if "control" in lowered or "offroad" in lowered or "deviation" in lowered:
        return "control_instability"
    return "late_response"
