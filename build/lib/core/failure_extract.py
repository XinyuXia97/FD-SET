from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

from .models import FailureEvent


DEFAULT_FAILURE_TYPES = {"collision", "offroad", "red_light", "timeout"}
BENCH2DRIVE_INFRACTION_TYPE_MAP = {
    "collisions_layout": "collision",
    "collisions_pedestrian": "collision",
    "collisions_vehicle": "collision",
    "red_light": "red_light",
    "stop_infraction": "stop_infraction",
    "outside_route_lanes": "offroad",
    "route_dev": "route_deviation",
    "route_timeout": "timeout",
    "vehicle_blocked": "blocked",
    "scenario_timeouts": "timeout",
    "yield_emergency_vehicle_infractions": "yield_violation",
    "min_speed_infractions": "low_speed",
}
DEFAULT_FIELD_MAP = {
    "route_id": "route_id",
    "timestamp_s": "timestamp_s",
    "event_type": "event_type",
    "ego_speed": "ego_speed",
    "description": "note",
    "scene_tags": "scene_tags",
    "risk_level": "risk_level",
}


def extract_failures_from_log(log_file: Path, config: dict[str, Any] | None = None) -> list[FailureEvent]:
    """Parse JSONL or JSON logs into normalized failure events.

    Supported config keys:
    - format: `jsonl` or `json` (defaults to suffix inference)
    - records_path: dotted path to the record list for JSON payloads
    - accepted_failure_types: explicit allowlist of failure types
    - field_map: map normalized names to dotted source paths
    - scene_tags_separator: split tags when the source field is a string
    """
    if not log_file.exists():
        return []

    parser_cfg = config or {}
    input_format = str(parser_cfg.get("format", "")).strip().lower()
    if not input_format:
        input_format = "jsonl" if log_file.suffix.lower() == ".jsonl" else "json"

    if input_format == "bench2drive_checkpoint":
        return _extract_failures_from_bench2drive_checkpoint(log_file, parser_cfg)

    records = _load_records(log_file, parser_cfg, input_format=input_format)
    field_map = _build_field_map(parser_cfg.get("field_map"))
    accepted_failure_types_cfg = parser_cfg.get("accepted_failure_types")
    if accepted_failure_types_cfg is None:
        accepted_failure_types = {str(value) for value in DEFAULT_FAILURE_TYPES}
    else:
        accepted_failure_types = {str(value) for value in accepted_failure_types_cfg}

    failures: list[FailureEvent] = []
    for record in records:
        if not isinstance(record, dict):
            continue

        event_type = str(_get_mapped_value(record, field_map, "event_type", "")).strip()
        if event_type not in accepted_failure_types:
            continue

        risk_level = str(_get_mapped_value(record, field_map, "risk_level", "")).strip()
        if not risk_level:
            risk_level = _risk_from_type(event_type)

        failures.append(
            FailureEvent(
                route_id=str(_get_mapped_value(record, field_map, "route_id", "unknown")),
                timestamp_s=float(_get_mapped_value(record, field_map, "timestamp_s", 0.0)),
                failure_type=event_type,
                ego_speed=float(_get_mapped_value(record, field_map, "ego_speed", 0.0)),
                description=str(_get_mapped_value(record, field_map, "description", "")),
                scene_tags=_normalize_scene_tags(
                    _get_mapped_value(record, field_map, "scene_tags", []),
                    separator=str(parser_cfg.get("scene_tags_separator", ",")),
                ),
                risk_level=risk_level,
                metadata=None,
            )
        )
    return failures


def generate_mock_failures() -> list[FailureEvent]:
    return [
        FailureEvent(
            route_id="route_001",
            timestamp_s=23.2,
            failure_type="collision",
            ego_speed=8.5,
            description="cut-in vehicle merged aggressively from right lane",
            scene_tags=["urban", "merge", "dense_traffic"],
            risk_level="high",
            metadata=None,
        ),
        FailureEvent(
            route_id="route_008",
            timestamp_s=41.9,
            failure_type="offroad",
            ego_speed=6.1,
            description="late braking near parked vehicle causing lane departure",
            scene_tags=["urban", "parked_vehicle"],
            risk_level="medium",
            metadata=None,
        ),
    ]


def _load_records(log_file: Path, config: dict[str, Any], *, input_format: str) -> list[dict[str, Any]]:
    if input_format == "jsonl":
        return [
            json.loads(line)
            for line in log_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    if input_format == "json":
        payload = json.loads(log_file.read_text(encoding="utf-8"))
        records_path = str(config.get("records_path", "")).strip()
        if records_path:
            payload = _resolve_dotted_path(payload, records_path, [])
        if isinstance(payload, list):
            return [record for record in payload if isinstance(record, dict)]
        raise ValueError(f"JSON payload does not resolve to a record list: {log_file}")

    raise ValueError(f"Unsupported failure_extract.format: {input_format}")


def _extract_failures_from_bench2drive_checkpoint(log_file: Path, config: dict[str, Any]) -> list[FailureEvent]:
    payload = json.loads(log_file.read_text(encoding="utf-8"))
    records_path = str(config.get("records_path", "_checkpoint.records"))
    records = _resolve_dotted_path(payload, records_path, [])
    if not isinstance(records, list):
        raise ValueError(f"Bench2Drive checkpoint records path does not resolve to a list: {records_path}")

    accepted_failure_types_cfg = config.get("accepted_failure_types")
    accepted_failure_types = (
        {str(value) for value in accepted_failure_types_cfg}
        if accepted_failure_types_cfg is not None
        else None
    )

    failures: list[FailureEvent] = []
    for record in records:
        if not isinstance(record, dict):
            continue
        failures.extend(_extract_failures_from_bench2drive_record(record, accepted_failure_types, config=config))
    return failures


def _extract_failures_from_bench2drive_record(
    record: dict[str, Any],
    accepted_failure_types: set[str] | None,
    *,
    config: dict[str, Any],
) -> list[FailureEvent]:
    route_id = str(record.get("route_id", "unknown"))
    scenario_name = str(record.get("scenario_name", "")).strip()
    town_name = str(record.get("town_name", "")).strip()
    status = str(record.get("status", "")).strip()
    save_name = str(record.get("save_name", "")).strip()
    duration_game = float(_resolve_dotted_path(record, "meta.duration_game", 0.0))
    failure_frame = _first_present(
        record,
        (
            "meta.failure_frame",
            "meta.frame",
            "failure_frame",
            "frame",
        ),
    )
    scores = dict(record.get("scores", {})) if isinstance(record.get("scores"), dict) else {}
    infractions = record.get("infractions", {})
    if not isinstance(infractions, dict):
        infractions = {}

    scene_tags = _normalize_bench2drive_scene_tags(scenario_name, town_name, status)
    context_summary = _load_bench2drive_context_summary(
        save_name=save_name,
        log_file=Path(config.get("__log_file__", "")) if config.get("__log_file__") else None,
        config=config,
        failure_timestamp_s=duration_game,
        failure_frame=failure_frame,
    )
    failures: list[FailureEvent] = []
    event_counter = 0

    for infraction_key, raw_messages in infractions.items():
        failure_type = BENCH2DRIVE_INFRACTION_TYPE_MAP.get(str(infraction_key), str(infraction_key))
        if accepted_failure_types is not None and failure_type not in accepted_failure_types:
            continue

        if not isinstance(raw_messages, list):
            continue

        for raw_message in raw_messages:
            event_counter += 1
            message = str(raw_message)
            failures.append(
                FailureEvent(
                    route_id=route_id,
                    timestamp_s=duration_game + event_counter / 1000.0,
                    failure_type=failure_type,
                    ego_speed=0.0,
                    description=message,
                    scene_tags=scene_tags,
                    risk_level=_risk_from_type(failure_type),
                    metadata=_build_bench2drive_metadata(
                        scenario_name=scenario_name,
                        town_name=town_name,
                        status=status,
                        save_name=save_name,
                        scores=scores,
                        infraction_key=str(infraction_key),
                        infraction_index=event_counter - 1,
                        context_summary=context_summary,
                    ),
                )
            )

    if failures:
        return failures

    derived_failure_type = _failure_type_from_status(status)
    if not derived_failure_type:
        return []
    if accepted_failure_types is not None and derived_failure_type not in accepted_failure_types:
        return []

    return [
        FailureEvent(
            route_id=route_id,
            timestamp_s=duration_game,
            failure_type=derived_failure_type,
            ego_speed=0.0,
            description=status,
            scene_tags=scene_tags,
            risk_level=_risk_from_type(derived_failure_type),
            metadata=_build_bench2drive_metadata(
                scenario_name=scenario_name,
                town_name=town_name,
                status=status,
                save_name=save_name,
                scores=scores,
                infraction_key="status",
                infraction_index=0,
                context_summary=context_summary,
            ),
        )
    ]


def _build_field_map(override_field_map: Any) -> dict[str, str]:
    field_map = dict(DEFAULT_FIELD_MAP)
    if isinstance(override_field_map, dict):
        for key, value in override_field_map.items():
            field_map[str(key)] = str(value)
    return field_map


def _get_mapped_value(record: dict[str, Any], field_map: dict[str, str], field_name: str, default: Any) -> Any:
    source_path = field_map.get(field_name)
    if not source_path:
        return default
    return _resolve_dotted_path(record, source_path, default)


def _resolve_dotted_path(payload: Any, dotted_path: str, default: Any) -> Any:
    current = payload
    for raw_part in dotted_path.split("."):
        part = raw_part.strip()
        if not part:
            return default

        if isinstance(current, dict):
            if part not in current:
                return default
            current = current[part]
            continue

        if isinstance(current, list) and part.isdigit():
            index = int(part)
            if index >= len(current):
                return default
            current = current[index]
            continue

        return default
    return current


def _normalize_scene_tags(raw_tags: Any, *, separator: str) -> list[str]:
    if isinstance(raw_tags, list):
        return [str(tag) for tag in raw_tags]
    if isinstance(raw_tags, str):
        return [part.strip() for part in raw_tags.split(separator) if part.strip()]
    return []


def _normalize_bench2drive_scene_tags(scenario_name: str, town_name: str, status: str) -> list[str]:
    tags = ["bench2drive"]
    if scenario_name:
        tags.append(scenario_name)
    if town_name:
        tags.append(town_name)
    if status:
        tags.append(status.lower().replace(" ", "_"))
    return list(dict.fromkeys(tags))


def _build_bench2drive_metadata(
    *,
    scenario_name: str,
    town_name: str,
    status: str,
    save_name: str,
    scores: dict[str, Any],
    infraction_key: str,
    infraction_index: int,
    context_summary: dict[str, Any] | None,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "source": "bench2drive_checkpoint",
        "scenario_name": scenario_name,
        "town_name": town_name,
        "status": status,
        "save_name": save_name,
        "infraction_key": infraction_key,
        "infraction_index": infraction_index,
        "scores": scores,
    }
    if context_summary is not None:
        metadata["context_summary"] = context_summary
    return metadata


def _load_bench2drive_context_summary(
    *,
    save_name: str,
    log_file: Path | None,
    config: dict[str, Any],
    failure_timestamp_s: float | None = None,
    failure_frame: Any = None,
) -> dict[str, Any] | None:
    if not save_name:
        return None

    metric_info_path = _resolve_metric_info_path(save_name=save_name, log_file=log_file, config=config)
    payload: dict[str, Any] = {}
    if metric_info_path is not None and metric_info_path.exists():
        raw_payload = json.loads(metric_info_path.read_text(encoding="utf-8"))
        if isinstance(raw_payload, dict):
            payload = raw_payload

    ordered_frames = [
        payload[key]
        for key in sorted(payload.keys(), key=_frame_sort_key)
        if isinstance(payload[key], dict)
    ]

    locations = [_extract_vector(frame.get("location")) for frame in ordered_frames]
    accelerations = [_extract_vector(frame.get("acceleration")) for frame in ordered_frames]
    angular_velocities = [_extract_vector(frame.get("angular_velocity")) for frame in ordered_frames]

    path_length_m = 0.0
    step_distances: list[float] = []
    prev_location: tuple[float, float, float] | None = None
    for location in locations:
        if location is None:
            continue
        if prev_location is not None:
            step_distance = math.dist(prev_location, location)
            step_distances.append(step_distance)
            path_length_m += step_distance
        prev_location = location

    max_accel = max((_vector_magnitude(vec) for vec in accelerations if vec is not None), default=0.0)
    max_yaw_rate = max((abs(vec[2]) for vec in angular_velocities if vec is not None), default=0.0)

    summary: dict[str, Any] = {
        "num_frames": len(ordered_frames),
        "path_length_m": round(path_length_m, 3),
        "mean_step_distance_m": round(sum(step_distances) / len(step_distances), 3) if step_distances else 0.0,
        "max_accel_magnitude": round(max_accel, 3),
        "max_yaw_rate": round(max_yaw_rate, 3),
    }
    if metric_info_path is not None:
        summary["metric_info_path"] = str(metric_info_path)
    evidence_frames = _resolve_evidence_frames(
        save_name=save_name,
        log_file=log_file,
        config=config,
        failure_timestamp_s=failure_timestamp_s,
        failure_frame=failure_frame,
    )
    if evidence_frames:
        summary["evidence_frames"] = evidence_frames
    if locations and locations[0] is not None:
        summary["start_location"] = list(locations[0])
    if locations and locations[-1] is not None:
        summary["end_location"] = list(locations[-1])
    if not ordered_frames and not evidence_frames:
        return None
    return summary


def _resolve_metric_info_path(save_name: str, log_file: Path | None, config: dict[str, Any]) -> Path | None:
    path_template = str(config.get("metric_info_path_template", "")).strip()
    if path_template:
        return Path(
            path_template.format(
                save_name=save_name,
                log_dir=str(log_file.parent) if log_file is not None else "",
                log_file=str(log_file) if log_file is not None else "",
            )
        )

    metric_info_root = str(config.get("metric_info_root", "")).strip()
    metric_info_filename = str(config.get("metric_info_filename", "metric_info.json")).strip()
    if metric_info_root:
        return Path(metric_info_root) / save_name / metric_info_filename
    if log_file is not None:
        return log_file.parent / save_name / metric_info_filename
    return None


def _resolve_evidence_frames(
    save_name: str,
    log_file: Path | None,
    config: dict[str, Any],
    *,
    failure_timestamp_s: float | None = None,
    failure_frame: Any = None,
) -> list[str]:
    frame_limit = int(config.get("frame_limit", 3))
    if frame_limit <= 0:
        return []

    frame_paths: list[Path] = []
    frame_paths_template = str(config.get("frame_paths_template", "")).strip()
    if frame_paths_template:
        resolved = frame_paths_template.format(
            save_name=save_name,
            log_dir=str(log_file.parent) if log_file is not None else "",
            log_file=str(log_file) if log_file is not None else "",
        )
        for item in resolved.split(":"):
            candidate = Path(item.strip())
            if candidate.exists() and candidate.is_file():
                frame_paths.append(candidate)
        return [str(path) for path in frame_paths[:frame_limit]]

    frame_glob = str(config.get("frame_glob", "*.png")).strip()
    frame_dir_template = str(config.get("frame_dir_template", "")).strip()
    if frame_dir_template:
        frame_dir = Path(
            frame_dir_template.format(
                save_name=save_name,
                log_dir=str(log_file.parent) if log_file is not None else "",
                log_file=str(log_file) if log_file is not None else "",
            )
        )
    elif str(config.get("frame_root", "")).strip():
        frame_dir = Path(str(config.get("frame_root", "")).strip()) / save_name
    elif str(config.get("metric_info_root", "")).strip():
        frame_dir = Path(str(config.get("metric_info_root", "")).strip()) / save_name
    elif log_file is not None:
        frame_dir = log_file.parent / save_name
    else:
        return []

    if frame_dir.exists() and frame_dir.is_dir():
        frame_paths = sorted(
            (path for path in frame_dir.glob(frame_glob) if path.is_file()),
            key=lambda path: _frame_sort_key(path.stem),
        )
    else:
        frame_paths = []
    selected = _select_temporal_frames(
        frame_paths,
        frame_limit=frame_limit,
        failure_frame=failure_frame,
        failure_timestamp_s=failure_timestamp_s,
        frame_rate=float(config.get("frame_rate", 10.0)),
        window_before=int(config.get("evidence_window_before", 2)),
        window_after=int(config.get("evidence_window_after", 2)),
        frame_stride=max(1, int(config.get("evidence_frame_stride", 1))),
    )
    if selected:
        return [str(path) for path in selected]

    video_path = _resolve_evidence_video(save_name=save_name, log_file=log_file, config=config)
    if video_path is not None:
        return [
            str(path)
            for path in _extract_video_evidence_frames(
                video_path,
                save_name=save_name,
                log_file=log_file,
                config=config,
                failure_timestamp_s=failure_timestamp_s,
                frame_limit=frame_limit,
            )
        ]
    return []


def _select_temporal_frames(
    frame_paths: list[Path],
    *,
    frame_limit: int,
    failure_frame: Any,
    failure_timestamp_s: float | None,
    frame_rate: float,
    window_before: int,
    window_after: int,
    frame_stride: int,
) -> list[Path]:
    if not frame_paths:
        return []

    center: int | None = None
    try:
        if failure_frame is not None:
            center = int(float(failure_frame))
    except (TypeError, ValueError):
        center = None
    if center is None and failure_timestamp_s is not None and frame_rate > 0:
        center = int(round(failure_timestamp_s * frame_rate))

    if center is None:
        center = len(frame_paths) // 2
    numeric_indices = [_numeric_frame_index(path) for path in frame_paths]
    if any(value is not None for value in numeric_indices):
        closest = min(
            range(len(frame_paths)),
            key=lambda index: abs((numeric_indices[index] if numeric_indices[index] is not None else index) - center),
        )
    else:
        closest = max(0, min(len(frame_paths) - 1, center))

    candidate_indices = list(
        range(
            max(0, closest - max(0, window_before)),
            min(len(frame_paths), closest + max(0, window_after) + 1),
            max(1, frame_stride),
        )
    )
    if len(candidate_indices) > frame_limit:
        candidate_indices = _evenly_spaced_indices(candidate_indices, frame_limit)
    return [frame_paths[index] for index in candidate_indices]


def _evenly_spaced_indices(values: list[int], limit: int) -> list[int]:
    if limit <= 0:
        return []
    if len(values) <= limit:
        return values
    if limit == 1:
        return [values[len(values) // 2]]
    positions = [round(index * (len(values) - 1) / (limit - 1)) for index in range(limit)]
    return [values[position] for position in positions]


def _numeric_frame_index(path: Path) -> int | None:
    match = re.search(r"(\d+)(?:\D*)$", path.stem)
    return int(match.group(1)) if match else None


def _resolve_evidence_video(save_name: str, log_file: Path | None, config: dict[str, Any]) -> Path | None:
    template = str(config.get("video_path_template", "")).strip()
    if template:
        candidate = Path(
            template.format(
                save_name=save_name,
                log_dir=str(log_file.parent) if log_file is not None else "",
                log_file=str(log_file) if log_file is not None else "",
            )
        )
        return candidate if candidate.exists() and candidate.is_file() else None

    video_root = str(config.get("video_root", "")).strip()
    if video_root:
        candidate_dir = Path(video_root) / save_name
    elif log_file is not None:
        candidate_dir = log_file.parent / save_name
    else:
        return None
    if not candidate_dir.exists() or not candidate_dir.is_dir():
        return None
    video_glob = str(config.get("video_glob", "*.mp4")).strip()
    candidates = sorted(path for path in candidate_dir.glob(video_glob) if path.is_file())
    return candidates[0] if candidates else None


def _extract_video_evidence_frames(
    video_path: Path,
    *,
    save_name: str,
    log_file: Path | None,
    config: dict[str, Any],
    failure_timestamp_s: float | None,
    frame_limit: int,
) -> list[Path]:
    try:
        import cv2  # type: ignore
    except ModuleNotFoundError:
        return []

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        return []
    fps = float(capture.get(cv2.CAP_PROP_FPS) or config.get("frame_rate", 10.0) or 10.0)
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    center_time = float(failure_timestamp_s or 0.0)
    before_s = float(config.get("video_window_before_s", 2.0))
    after_s = float(config.get("video_window_after_s", 2.0))
    start = max(0, int((center_time - before_s) * fps))
    end = min(total_frames - 1, int((center_time + after_s) * fps)) if total_frames else int((center_time + after_s) * fps)
    requested = _evenly_spaced_indices(list(range(start, max(start + 1, end + 1))), frame_limit)

    output_root = str(config.get("evidence_output_root", "")).strip()
    if output_root:
        output_dir = Path(output_root) / save_name
    elif log_file is not None:
        output_dir = log_file.parent / "evidence_frames" / save_name
    else:
        output_dir = video_path.parent / "evidence_frames" / save_name
    output_dir.mkdir(parents=True, exist_ok=True)

    outputs: list[Path] = []
    for frame_index in requested:
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = capture.read()
        if not ok:
            continue
        output_path = output_dir / f"fdset_evidence_{frame_index:06d}.jpg"
        if cv2.imwrite(str(output_path), frame):
            outputs.append(output_path)
    capture.release()
    return outputs


def _first_present(payload: dict[str, Any], paths: tuple[str, ...]) -> Any:
    for path in paths:
        value = _resolve_dotted_path(payload, path, None)
        if value is not None:
            return value
    return None


def _frame_sort_key(value: str) -> tuple[int, str]:
    return (0, value) if value.isdigit() else (1, value)


def _extract_vector(value: Any) -> tuple[float, float, float] | None:
    if isinstance(value, list) and len(value) >= 3:
        return float(value[0]), float(value[1]), float(value[2])
    if isinstance(value, dict):
        if {"x", "y", "z"}.issubset(value.keys()):
            return float(value["x"]), float(value["y"]), float(value["z"])
    return None


def _vector_magnitude(vector: tuple[float, float, float]) -> float:
    return math.sqrt(vector[0] ** 2 + vector[1] ** 2 + vector[2] ** 2)


def _failure_type_from_status(status: str) -> str | None:
    lowered = status.lower()
    if "timed out" in lowered or "timeout" in lowered:
        return "timeout"
    if "deviated from the route" in lowered:
        return "route_deviation"
    if "blocked" in lowered:
        return "blocked"
    if "failed" in lowered:
        return "failure"
    return None


def _risk_from_type(failure_type: str) -> str:
    if failure_type == "collision":
        return "high"
    if failure_type in {"red_light", "offroad", "stop_infraction", "route_deviation"}:
        return "medium"
    if failure_type in {"blocked", "timeout"}:
        return "medium"
    return "low"
