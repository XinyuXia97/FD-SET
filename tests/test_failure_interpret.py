from __future__ import annotations

from pathlib import Path

from core.failure_interpret import interpret_failures
from core.models import FailureEvent


def test_interpret_failures_loads_prompt_metadata(tmp_path: Path) -> None:
    prompt_path = tmp_path / "prompt.txt"
    prompt_path.write_text("You are a safety analyst.", encoding="utf-8")

    events = [
        FailureEvent(
            route_id="route_001",
            timestamp_s=10.0,
            failure_type="collision",
            ego_speed=8.0,
            description="cut-in vehicle merged aggressively",
            scene_tags=["urban", "merge"],
            risk_level="high",
        )
    ]

    interpreted = interpret_failures(events, prompt_path=prompt_path)
    payload = interpreted["route_001:10.00"]

    assert payload["interaction_pattern"] == "adversarial_merge"
    assert payload["interpreter"]["backend"] == "rule_based_v1"
    assert payload["interpreter"]["prompt_source"] == str(prompt_path)
    assert payload["interpreter"]["prompt_loaded"] is True


def test_rule_based_interpreter_handles_junction_and_control_patterns(tmp_path: Path) -> None:
    prompt_path = tmp_path / "prompt.txt"
    prompt_path.write_text("You are a safety analyst.", encoding="utf-8")

    events = [
        FailureEvent(
            route_id="route_010",
            timestamp_s=10.0,
            failure_type="blocked",
            ego_speed=5.0,
            description="intersection turning conflict with blocked vehicle",
            scene_tags=["urban", "junction"],
            risk_level="high",
        ),
        FailureEvent(
            route_id="route_011",
            timestamp_s=11.0,
            failure_type="control_loss",
            ego_speed=6.0,
            description="control loss after hard brake and skid",
            scene_tags=["urban", "control"],
            risk_level="high",
        ),
    ]

    interpreted = interpret_failures(events, prompt_path=prompt_path)

    assert interpreted["route_010:10.00"]["interaction_pattern"] == "junction_conflict"
    assert interpreted["route_011:11.00"]["interaction_pattern"] == "control_instability"
