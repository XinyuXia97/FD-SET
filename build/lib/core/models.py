from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any


@dataclass
class FailureEvent:
    route_id: str
    timestamp_s: float
    failure_type: str
    ego_speed: float
    description: str
    scene_tags: list[str]
    risk_level: str
    metadata: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class StructuredFailure:
    event: FailureEvent
    actors: list[str]
    interaction_pattern: str
    constraints: dict[str, Any]
    semantic_evidence: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["event"] = self.event.to_dict()
        return payload


@dataclass
class ScenarioCandidate:
    scenario_id: str
    score: float
    tags: list[str]
    base_config: dict[str, Any]
    score_breakdown: dict[str, float] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EvolvedScenario:
    source_scenario_id: str
    evolved_id: str
    modifications: dict[str, Any]
    rationale: str
    metadata: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
