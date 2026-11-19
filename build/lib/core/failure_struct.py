from __future__ import annotations

from .models import FailureEvent, StructuredFailure


def structure_failures(
    events: list[FailureEvent],
    semantics: dict[str, dict[str, object]] | None = None,
    *,
    use_structure: bool = True,
) -> list[StructuredFailure]:
    """Map failure events into structured representation for retrieval/evolution."""
    if not use_structure:
        return [
            StructuredFailure(
                event=e,
                actors=["ego"],
                interaction_pattern="generic_failure",
                constraints={"target_failure_type": e.failure_type},
                semantic_evidence=None,
            )
            for e in events
        ]

    out: list[StructuredFailure] = []
    for e in events:
        key = f"{e.route_id}:{e.timestamp_s:.2f}"
        sem = (semantics or {}).get(key)
        out.append(
            StructuredFailure(
                event=e,
                actors=_infer_actors(e, sem),
                interaction_pattern=_infer_pattern(e, sem),
                constraints=_infer_constraints(e, sem),
                semantic_evidence=sem,
            )
        )
    return out


def _infer_actors(event: FailureEvent, semantic: dict[str, object] | None) -> list[str]:
    if semantic and isinstance(semantic.get("key_entities"), list):
        return [str(x) for x in semantic["key_entities"]]
    desc = event.description.lower()
    actors = ["ego"]
    if "vehicle" in desc or "merge" in desc or "cut-in" in desc:
        actors.append("vehicle")
    if "pedestrian" in desc:
        actors.append("pedestrian")
    if "parked" in desc:
        actors.append("static_vehicle")
    return actors


def _infer_pattern(event: FailureEvent, semantic: dict[str, object] | None) -> str:
    if semantic and isinstance(semantic.get("interaction_pattern"), str):
        return str(semantic["interaction_pattern"])
    desc = event.description.lower()
    if "cut-in" in desc or "merge" in desc:
        return "adversarial_merge"
    if "junction" in desc or "intersection" in desc or "turning" in desc:
        return "junction_conflict"
    if "late braking" in desc:
        return "late_response"
    if "control loss" in desc or "hard brake" in desc or "skid" in desc:
        return "control_instability"
    if event.failure_type == "red_light":
        return "signal_violation"
    return "generic_failure"


def _infer_constraints(event: FailureEvent, semantic: dict[str, object] | None) -> dict[str, float | str | list[str]]:
    risk_cues: list[str] = []
    if semantic and isinstance(semantic.get("risk_cues"), list):
        risk_cues = [str(x) for x in semantic["risk_cues"]]
    constraints: dict[str, float | str | list[str]] = {
        "target_failure_type": event.failure_type,
        "ego_speed_min": max(0.0, event.ego_speed - 1.5),
        "ego_speed_max": event.ego_speed + 3.0,
        "risk_level": event.risk_level,
        "risk_cues": risk_cues,
    }
    if semantic:
        for key in (
            "capability_weakness",
            "spatial_pattern",
            "temporal_pattern",
            "risk_pattern",
            "road_topology",
            "traffic_state",
        ):
            value = semantic.get(key)
            if isinstance(value, str) and value.strip():
                constraints[key] = value.strip()
        for key in ("relative_relations", "context_conditions"):
            value = semantic.get(key)
            if isinstance(value, list):
                constraints[key] = [str(item) for item in value]
    return constraints
