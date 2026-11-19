from __future__ import annotations

from typing import Any, Iterable

from .models import ScenarioCandidate, StructuredFailure
from .retrieval_encoder import EmbeddingUnavailable, TextEncoder, build_text_encoder, cosine_similarity


def retrieve_anchor_scenarios(
    failure: StructuredFailure,
    scenario_bank: Iterable[dict],
    top_k: int = 3,
    *,
    retrieval_config: dict[str, Any] | None = None,
    encoder: TextEncoder | None = None,
) -> list[ScenarioCandidate]:
    """Retrieve anchors using the configured CLIP or deterministic scorer."""
    retrieval_config = retrieval_config or {}
    backend = str(retrieval_config.get("backend", "lexical")).strip().lower()
    score_mode = str(retrieval_config.get("score_mode", "hybrid")).strip().lower()
    embedding_weight = float(retrieval_config.get("embedding_weight", 1.0))
    rule_weight = float(retrieval_config.get("rule_weight", 1.0))
    embedding_rows: list[list[float]] | None = None
    embedding_failure: Exception | None = None
    if backend in {"clip", "embedding", "hybrid"}:
        try:
            encoder = encoder or build_text_encoder(retrieval_config)
            texts = [serialize_structured_failure(failure)]
            rows = list(scenario_bank)
            embedding_rows = encoder.encode(texts + [serialize_scenario(row) for row in rows])
            failure_embedding = embedding_rows[0]
            scenario_embeddings = embedding_rows[1:]
        except EmbeddingUnavailable as exc:
            embedding_failure = exc
            if not bool(retrieval_config.get("fallback_to_lexical", False)):
                raise
            rows = list(scenario_bank)
            failure_embedding = []
            scenario_embeddings = [[] for _ in rows]
    else:
        rows = list(scenario_bank)
        failure_embedding = []
        scenario_embeddings = [[] for _ in rows]

    scored: list[ScenarioCandidate] = []
    f_tags = set(failure.event.scene_tags)

    for row, scenario_embedding in zip(rows, scenario_embeddings):
        tags = list(row.get("tags", []))
        tag_overlap = len(f_tags.intersection(tags))
        pattern_bonus = 2.0 if row.get("pattern") == failure.interaction_pattern else 0.0

        cfg = dict(row.get("config", {}))
        cfg_speed = float(cfg.get("ego_speed", 0.0))
        min_speed = float(failure.constraints.get("ego_speed_min", 0.0))
        max_speed = float(failure.constraints.get("ego_speed_max", 999.0))
        speed_bonus = 1.0 if min_speed <= cfg_speed <= max_speed else 0.0

        text_bonus = _compute_text_bonus(failure, row)
        control_bonus = _compute_control_bonus(failure, cfg)

        rule_score = float(tag_overlap) + pattern_bonus + speed_bonus + text_bonus + control_bonus
        embedding_similarity = cosine_similarity(failure_embedding, scenario_embedding)
        if embedding_rows is not None and score_mode in {"embedding", "cosine"}:
            score = embedding_weight * embedding_similarity
        elif embedding_rows is not None:
            score = embedding_weight * embedding_similarity + rule_weight * rule_score
        else:
            score = rule_score
        score_breakdown = {
            "tag_overlap": float(tag_overlap),
            "pattern_bonus": float(pattern_bonus),
            "speed_bonus": float(speed_bonus),
            "text_bonus": float(text_bonus),
            "control_bonus": float(control_bonus),
            "embedding_similarity": float(embedding_similarity),
        }
        if embedding_failure is not None:
            score_breakdown["embedding_fallback"] = 1.0
        scored.append(
            ScenarioCandidate(
                scenario_id=str(row.get("scenario_id", "unknown")),
                score=score,
                tags=tags,
                base_config=cfg,
                score_breakdown=score_breakdown,
            )
        )

    scored.sort(key=lambda x: x.score, reverse=True)
    return scored[:top_k]


def serialize_structured_failure(failure: StructuredFailure) -> str:
    """Serialize the structured failure fields as CLIP retrieval text."""
    return " ".join(
        [
            f"failure type {failure.event.failure_type}",
            f"interaction pattern {failure.interaction_pattern}",
            f"entities {' '.join(failure.actors)}",
            f"scene tags {' '.join(failure.event.scene_tags)}",
            f"description {failure.event.description}",
            f"constraints {failure.constraints}",
        ]
    )


def serialize_scenario(row: dict) -> str:
    config = row.get("config", {})
    if not isinstance(config, dict):
        config = {}
    profile = config.get("bench2drive_profile", {})
    return " ".join(
        [
            f"scenario id {row.get('scenario_id', '')}",
            f"pattern {row.get('pattern', '')}",
            f"description {config.get('scenic_description', '')}",
            f"tags {' '.join(str(tag) for tag in row.get('tags', []))}",
            f"profile {profile}",
            f"configuration {config}",
        ]
    )


def _compute_text_bonus(failure: StructuredFailure, row: dict) -> float:
    failure_tokens = _tokenize(
        " ".join(
            [
                failure.event.description,
                failure.interaction_pattern,
                failure.event.failure_type,
                " ".join(failure.actors),
            ]
        )
    )
    if not failure_tokens:
        return 0.0

    row_tokens = _tokenize(
        " ".join(
            [
                str(row.get("scenario_id", "")),
                str(row.get("pattern", "")),
                " ".join(str(tag) for tag in row.get("tags", [])),
                str(dict(row.get("config", {})).get("scenic_description", "")),
                str(dict(row.get("config", {})).get("bench2drive_profile", {})),
            ]
        )
    )
    if not row_tokens:
        return 0.0

    overlap = len(failure_tokens.intersection(row_tokens))
    return min(2.0, overlap * 0.25)


def _compute_control_bonus(failure: StructuredFailure, cfg: dict) -> float:
    expected_controls = {
        "adversarial_merge": {"merge_offset", "front_vehicle_distance", "behind_vehicle_distance"},
        "pedestrian_intrusion": {"spawn_gap", "ped_speed", "ped_travel_dist", "crossing_angle"},
        "late_response": {"obstacle_dist", "traffic_frequency"},
        "signal_violation": {"turn_direction"},
        "junction_conflict": {"obstacle_dist", "lane_offset", "turn_direction", "merge_offset"},
        "control_instability": {"lane_offset", "ego_speed"},
    }.get(failure.interaction_pattern, set())
    if not expected_controls:
        return 0.0

    present = len([key for key in expected_controls if key in cfg])
    return min(1.5, present * 0.3)


def _tokenize(text: str) -> set[str]:
    normalized = []
    for char in text.lower():
        normalized.append(char if char.isalnum() else " ")
    return {token for token in "".join(normalized).split() if len(token) >= 3}
