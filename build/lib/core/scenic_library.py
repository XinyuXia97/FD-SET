from __future__ import annotations

import hashlib
import importlib.util
import importlib.metadata
import re
import shutil
from pathlib import Path
from typing import Any


SCENIC_LIBRARY_SCHEMA = "fdset.scenic_library.v1"

_FAMILY_HINTS = (
    ("carlaChallenge1", "road_debris"),
    ("carlaChallenge2", "longitudinal_braking"),
    ("carlaChallenge3", "static_obstacle"),
    ("carlaChallenge4", "intersection_bicycle"),
    ("carlaChallenge5", "slow_lead_overtake"),
    ("carlaChallenge6", "opposite_lane_blockage"),
    ("carlaChallenge7", "cross_traffic"),
    ("carlaChallenge8", "unprotected_left_turn"),
    ("carlaChallenge9", "right_turn_crossing"),
    ("carlaChallenge10", "unsignalized_intersection"),
    ("fdset_intersection_left", "left_turn_navigation"),
    ("fdset_intersection_right", "right_turn_navigation"),
    ("fdset_intersection_straight", "straight_junction_navigation"),
    ("fdset_green_light_start", "green_light_start"),
    ("fdset_signal_stop_restart", "signal_stop_restart"),
    ("fdset_lead_vehicle_hard_brake", "lead_vehicle_braking"),
    ("fdset_merge_cut_in", "cut_in"),
    ("fdset_pedestrian_crossing", "pedestrian_crossing"),
    ("fdset_bicycle_crossing", "bicycle_crossing"),
    ("fdset_roadside_narrow_passage", "narrow_passage"),
)

_PATTERN_BY_FAMILY = {
    "road_debris": "late_response",
    "longitudinal_braking": "late_response",
    "static_obstacle": "late_response",
    "intersection_bicycle": "late_response",
    "slow_lead_overtake": "adversarial_merge",
    "opposite_lane_blockage": "junction_conflict",
    "red_light_crossing": "signal_violation",
    "cross_traffic": "junction_conflict",
    "unprotected_left_turn": "junction_conflict",
    "right_turn_crossing": "junction_conflict",
    "unsignalized_intersection": "junction_conflict",
    "obstacle_avoidance": "late_response",
    "intersection_conflict": "junction_conflict",
    "left_turn_navigation": "junction_conflict",
    "right_turn_navigation": "junction_conflict",
    "straight_junction_navigation": "junction_conflict",
    "green_light_start": "junction_conflict",
    "signal_stop_restart": "signal_violation",
    "lead_vehicle_braking": "late_response",
    "cut_in": "adversarial_merge",
    "pedestrian_crossing": "pedestrian_intrusion",
    "bicycle_crossing": "late_response",
    "narrow_passage": "late_response",
}


def index_scenic_programs(
    source_dir: Path,
    *,
    copy_to: Path | None = None,
) -> dict[str, Any]:
    """Index Scenic source programs without importing the Scenic runtime.

    A directory containing ``.scenic`` files is source data, not the Scenic
    Python package. The index therefore performs a conservative source-level
    audit and records exactly why native execution is or is not ready.
    """
    source_dir = source_dir.expanduser().resolve()
    if not source_dir.exists() or not source_dir.is_dir():
        raise FileNotFoundError(f"Scenic source directory not found: {source_dir}")

    files = sorted(path for path in source_dir.rglob("*.scenic") if path.is_file())
    runtime_available = _scenic_runtime_available()
    runtime_version = _scenic_runtime_version()
    programs: list[dict[str, Any]] = []
    for source_path in files:
        relative_path = source_path.relative_to(source_dir)
        entry = _inspect_program(source_path, relative_path)
        if copy_to is not None:
            copied_path = copy_to / relative_path
            copied_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_path, copied_path)
            entry["artifact_source_path"] = str(copied_path.resolve())
        programs.append(entry)

    native_ready = sum(1 for program in programs if program["native_execution_ready"])
    missing_maps = sorted(
        {
            str(reference["resolved_path"])
            for program in programs
            for reference in program["map_references"]
            if not reference["exists"]
        }
    )
    family_counts: dict[str, int] = {}
    for program in programs:
        family = program["interaction_family"]
        family_counts[family] = family_counts.get(family, 0) + 1
    return {
        "schema_version": SCENIC_LIBRARY_SCHEMA,
        "source_dir": str(source_dir),
        "program_count": len(programs),
        "runtime": {
            "available": runtime_available,
            "version": runtime_version,
            "import_target": "scenic.core/scenic.syntax",
            "note": "The indexed directory contains Scenic source programs, not the Scenic Python runtime.",
        },
        "summary": {
            "native_execution_ready": native_ready,
            "missing_map_count": len(missing_maps),
            "missing_maps": missing_maps,
            "interaction_family_counts": family_counts,
        },
        "programs": programs,
    }


def build_scenic_source_bank(
    source_dir: Path,
    *,
    source_index: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Build scenario-bank rows from the supplied native Scenic programs.

    These rows are deliberately separate from Bench2Drive route rows.  A
    Scenic source program is the executable template used for retrieval and
    reconstruction; it does not have a fake route XML representation.  The
    native program and its audit metadata stay inside ``config`` so the normal
    scenario-bank persistence and retrieval paths can use them unchanged.
    """
    source_dir = source_dir.expanduser().resolve()
    index = source_index or index_scenic_programs(source_dir)
    rows: list[dict[str, Any]] = []
    for program in index.get("programs", []):
        if not isinstance(program, dict):
            continue
        relative_path = str(program.get("relative_path", "")).strip()
        program_id = str(program.get("program_id", Path(relative_path).stem)).strip()
        if not relative_path or not program_id:
            continue
        source_path = Path(str(program.get("source_path", source_dir / relative_path))).expanduser()
        if not source_path.exists():
            continue
        family = str(program.get("interaction_family", "unknown")).strip() or "unknown"
        pattern = _PATTERN_BY_FAMILY.get(family, "generic_failure")
        carla_map = _strip_inline_comment(str(program.get("carla_map", ""))).strip().strip("'\"")
        tags = ["scenic", "native", family, pattern]
        if carla_map:
            tags.append(carla_map)
        source_text = source_path.read_text(encoding="utf-8")
        config = {
            "scenario_role": "scenario_bank",
            "scenario_origin": "scenic_source",
            "interaction_pattern": pattern,
            "scenic_program_format": "scenic_native_v1",
            "scenic_program_id": program_id,
            "scenic_relative_path": relative_path,
            "scenic_source_path": str(source_path.resolve()),
            "scenic_source_sha256": str(program.get("sha256", "")),
            "scenic_program": source_text,
            "scenic_interaction_family": family,
            "scenic_description": str(program.get("description", "")),
            "scenic_model": str(program.get("model", "")),
            "scenic_parameters": dict(program.get("parameters", {})),
            "scenic_constants": dict(program.get("constants", {})),
            "scenic_behaviors": list(program.get("behaviors", [])),
            "scenic_monitors": list(program.get("monitors", [])),
            "scenic_actors": list(program.get("actors", [])),
            "scenic_requirements": list(program.get("requirements", [])),
            "scenic_map_references": list(program.get("map_references", [])),
            "scenic_native_execution_ready": bool(program.get("native_execution_ready", False)),
            "scenic_diagnostics": list(program.get("diagnostics", [])),
            "carla_map": carla_map,
        }
        rows.append(
            {
                "scenario_id": f"scenic_{_safe_scenario_id(relative_path)}",
                "tags": list(dict.fromkeys(tags)),
                "pattern": pattern,
                "config": config,
            }
        )
    return rows


def merge_scenic_source_bank(
    scenario_bank: list[dict[str, Any]],
    source_dir: Path,
    *,
    source_index: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Idempotently add native Scenic source rows to an existing bank."""
    existing_ids = {str(row.get("scenario_id", "")) for row in scenario_bank}
    for row in build_scenic_source_bank(source_dir, source_index=source_index):
        if str(row["scenario_id"]) not in existing_ids:
            scenario_bank.append(row)
            existing_ids.add(str(row["scenario_id"]))
    return scenario_bank


def _inspect_program(source_path: Path, relative_path: Path) -> dict[str, Any]:
    text = source_path.read_text(encoding="utf-8")
    description_match = re.match(r"\s*\"\"\"(.*?)\"\"\"", text, flags=re.DOTALL)
    description = description_match.group(1).strip() if description_match else ""

    model = _first_line_value(text, r"^\s*model\s+(.+?)\s*$")
    params = _parse_assignments(text, r"^\s*param\s+([A-Za-z_]\w*)\s*=\s*(.+?)\s*$")
    constants = _parse_assignments(text, r"^\s*([A-Z][A-Z0-9_]*)\s*=\s*(.+?)\s*$")
    behaviors = sorted(set(re.findall(r"^\s*behavior\s+([A-Za-z_]\w*)\s*\(", text, flags=re.MULTILINE)))
    monitors = sorted(set(re.findall(r"^\s*monitor\s+([A-Za-z_]\w*)\s*:", text, flags=re.MULTILINE)))
    requirements = [
        line.strip()[len("require") :].strip()
        for line in text.splitlines()
        if line.strip().startswith("require ")
    ]
    actors = [
        {"name": name, "type": actor_type}
        for name, actor_type in re.findall(
            r"^\s*([A-Za-z_]\w*)\s*=\s*([A-Z][A-Za-z0-9_]*)\s+(?:at|following|on)\b",
            text,
            flags=re.MULTILINE,
        )
    ]
    map_references = _parse_map_references(text, source_path)
    family = _infer_family(source_path.stem, text)

    diagnostics: list[str] = []
    if not model:
        diagnostics.append("missing Scenic model declaration")
    if not map_references:
        diagnostics.append("missing param map declaration")
    for reference in map_references:
        if not reference["exists"]:
            diagnostics.append(f"referenced map is missing: {reference['resolved_path']}")
    if not actors:
        diagnostics.append("no actor declarations were detected")
    source_ready = not diagnostics

    return {
        "program_id": source_path.stem,
        "relative_path": relative_path.as_posix(),
        "source_path": str(source_path.resolve()),
        "sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
        "description": description,
        "interaction_family": family,
        "model": _strip_inline_comment(model),
        "parameters": {key: _strip_inline_comment(value).strip() for key, value in params.items()},
        "constants": {key: _strip_inline_comment(value).strip() for key, value in constants.items()},
        "behaviors": behaviors,
        "monitors": monitors,
        "actors": actors,
        "requirements": requirements,
        "map_references": map_references,
        "carla_map": _strip_inline_comment(params.get("carla_map", "")).strip(),
        "native_execution_ready": source_ready,
        "diagnostics": diagnostics,
    }


def _parse_assignments(text: str, pattern: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for match in re.finditer(pattern, text, flags=re.MULTILINE):
        values[match.group(1)] = match.group(2).strip()
    return values


def _first_line_value(text: str, pattern: str) -> str:
    match = re.search(pattern, text, flags=re.MULTILINE)
    return match.group(1).strip() if match else ""


def _parse_map_references(text: str, source_path: Path) -> list[dict[str, Any]]:
    references: list[dict[str, Any]] = []
    for quote, relative_map in re.findall(
        r"param\s+map\s*=\s*localPath\(\s*(['\"])(.*?)\1\s*\)", text
    ):
        resolved_path = (source_path.parent / relative_map).resolve()
        references.append(
            {
                "expression": f"localPath({relative_map!r})",
                "relative_path": relative_map,
                "resolved_path": str(resolved_path),
                "exists": resolved_path.exists() and resolved_path.is_file(),
            }
        )
    return references


def _infer_family(stem: str, text: str) -> str:
    for prefix, family in _FAMILY_HINTS:
        if stem == prefix or stem.startswith(f"{prefix}_"):
            return family
    lowered = text.lower()
    keyword_families = (
        (("red light", "traffic light", "runs a red"), "red_light_crossing"),
        (("left turn", "left_turn"), "unprotected_left_turn"),
        (("right turn", "right_turn"), "right_turn_crossing"),
        (("obstacle", "blocking object", "debris", "trash"), "obstacle_avoidance"),
        (("intersection", "maneuver"), "intersection_conflict"),
    )
    for keywords, family in keyword_families:
        if any(keyword in lowered for keyword in keywords):
            return family
    return "unknown"


def _strip_inline_comment(value: str) -> str:
    return value.split("#", 1)[0].rstrip()


def _safe_scenario_id(relative_path: str) -> str:
    stem = Path(relative_path).with_suffix("").as_posix()
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "__", stem).strip("_")
    return safe or "program"


def _scenic_runtime_available() -> bool:
    return importlib.util.find_spec("scenic.core") is not None and importlib.util.find_spec("scenic.syntax") is not None


def _scenic_runtime_version() -> str | None:
    try:
        return importlib.metadata.version("scenic")
    except importlib.metadata.PackageNotFoundError:
        return None
