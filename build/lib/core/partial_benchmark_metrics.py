from __future__ import annotations

import json
import math
from pathlib import Path
from statistics import fmean
from typing import Any

from .bench2drive_adapter import parse_metrics


_SCORE_FIELDS = ("score_composed", "score_route", "score_penalty")


def build_partial_checkpoint(
    records: list[dict[str, Any]],
    skipped: list[dict[str, Any]],
    expected: int,
    destination: Path,
) -> dict[str, Any]:
    """Write an official-shape checkpoint containing only actually scored routes.

    Route indices are zero-based positions in the original routes XML. The audit
    section records those positions and the route IDs so skipped routes cannot
    be mistaken for scored entries.
    """
    if isinstance(expected, bool) or not isinstance(expected, int) or expected <= 0:
        raise ValueError("expected must be a positive route count")

    measured_by_index: dict[int, dict[str, Any]] = {}
    skipped_by_index: dict[int, dict[str, Any]] = {}

    for item in records:
        index = _route_index(item, "records")
        if index in measured_by_index:
            raise ValueError(f"Duplicate measured route index: {index}")
        record = item.get("record")
        if not isinstance(record, dict):
            raise ValueError(f"Measured route {index} must contain a checkpoint record")
        route_id = record.get("route_id")
        if route_id is None or not str(route_id).strip():
            raise ValueError(f"Measured route {index} is missing route_id")
        status = str(record.get("status", ""))
        if not status or status == "Started" or any(
            marker in status
            for marker in ("Simulation crashed", "Agent crashed", "sensors were invalid")
        ):
            raise ValueError(f"Route {index} has no valid scored result: {status!r}")
        scores = record.get("scores")
        if not isinstance(scores, dict):
            raise ValueError(f"Measured route {index} is missing scored fields")
        for field in _SCORE_FIELDS:
            value = scores.get(field)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"Measured route {index} is missing valid {field}")
        measured_by_index[index] = record

    for item in skipped:
        index = _route_index(item, "skipped")
        if index in skipped_by_index:
            raise ValueError(f"Duplicate skipped route index: {index}")
        route_id = item.get("route_id")
        reason = item.get("reason")
        if route_id is None or not str(route_id).strip():
            raise ValueError(f"Skipped route {index} is missing route_id")
        if reason is None or not str(reason).strip():
            raise ValueError(f"Skipped route {index} is missing reason")
        skipped_by_index[index] = item

    overlap = measured_by_index.keys() & skipped_by_index.keys()
    if overlap:
        raise ValueError(f"Routes cannot be both measured and skipped: {sorted(overlap)}")
    all_indices = measured_by_index.keys() | skipped_by_index.keys()
    expected_indices = set(range(expected))
    unexpected = all_indices - expected_indices
    if unexpected:
        raise ValueError(f"Route indices outside expected range: {sorted(unexpected)}")
    missing = expected_indices - all_indices
    if missing:
        raise ValueError(f"Routes are neither measured nor skipped: {sorted(missing)}")
    if not measured_by_index:
        raise ValueError("Cannot build a checkpoint with zero measured routes")

    ordered_indices = sorted(measured_by_index)
    measured_records = [measured_by_index[index] for index in ordered_indices]
    scores_mean = {
        field: fmean(float(measured_by_index[index]["scores"][field]) for index in ordered_indices)
        for field in _SCORE_FIELDS
    }
    complete = len(measured_records) == expected
    skipped_audit = [
        {
            "index": index,
            "route_id": str(skipped_by_index[index]["route_id"]),
            "reason": str(skipped_by_index[index]["reason"]),
        }
        for index in sorted(skipped_by_index)
    ]
    audit = {
        "source": "partial_benchmark_metrics.build_partial_checkpoint",
        "expected": expected,
        "measured_count": len(measured_records),
        "measured_routes": [
            {"index": index, "route_id": str(measured_by_index[index]["route_id"])}
            for index in ordered_indices
        ],
        "skipped_count": len(skipped_audit),
        "skipped_routes": skipped_audit,
    }
    payload: dict[str, Any] = {
        "entry_status": "Finished" if complete else "Partial",
        "_checkpoint": {
            "progress": [len(measured_records), expected],
            "global_record": {"scores_mean": scores_mean},
            "records": measured_records,
        },
        "audit": audit,
    }

    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    # Use the same parser used by benchmark consumers, including checkpoint
    # record-derived SR/CR, and verify the written artifact remains readable.
    parse_metrics(destination, keys=["DS", "RC", "IP", "SR", "CR"], strict=True)
    return payload


def _route_index(item: Any, category: str) -> int:
    if not isinstance(item, dict):
        raise ValueError(f"Each {category} entry must be an object")
    index = item.get("index")
    if isinstance(index, bool) or not isinstance(index, int) or index < 0:
        raise ValueError(f"Each {category} entry must have a non-negative integer index")
    return index
