from __future__ import annotations

from core.failure_struct import structure_failures
from core.models import FailureEvent


def test_structure_failures_infers_junction_and_control_patterns() -> None:
    events = [
        FailureEvent(
            route_id="route_junction",
            timestamp_s=10.0,
            failure_type="blocked",
            ego_speed=6.0,
            description="intersection turning conflict caused blockage",
            scene_tags=["urban", "junction"],
            risk_level="high",
        ),
        FailureEvent(
            route_id="route_control",
            timestamp_s=20.0,
            failure_type="control_loss",
            ego_speed=7.0,
            description="control loss after hard brake and skid",
            scene_tags=["urban", "control"],
            risk_level="high",
        ),
    ]

    structured = structure_failures(events)

    assert structured[0].interaction_pattern == "junction_conflict"
    assert structured[1].interaction_pattern == "control_instability"
