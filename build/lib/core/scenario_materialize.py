from __future__ import annotations

import copy
import json
import re
import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

from .bench2drive_catalog import build_parameter_overrides, infer_scenario_family
from .scenic_library import index_scenic_programs
from .scenario_program import render_native_scenic_variant, render_scenic_program, validate_scenario_row


def materialize_round_inputs(
    scenario_bank: list[dict[str, Any]],
    round_dir: Path,
    cfg: dict[str, Any],
) -> dict[str, Path]:
    round_dir.mkdir(parents=True, exist_ok=True)
    outputs = {"json": round_dir / "scenarios.json"}
    outputs["json"].write_text(json.dumps(scenario_bank, indent=2), encoding="utf-8")

    scenic_cfg = cfg.get("materialization", {}).get("scenic", {})
    if bool(scenic_cfg.get("enabled", False)):
        scenic_dir = round_dir / str(scenic_cfg.get("program_dir", "scenic_programs"))
        generated_count = 0
        native_entries: list[dict[str, Any]] = []
        source_index: dict[str, Any] | None = None
        source_copy_dir: Path | None = None
        source_dir_raw = str(scenic_cfg.get("source_dir", "")).strip()
        if source_dir_raw:
            source_dir = Path(source_dir_raw).expanduser()
            source_copy_dir_raw = str(scenic_cfg.get("source_copy_dir", "")).strip()
            source_copy_dir = round_dir / source_copy_dir_raw if source_copy_dir_raw else None
            source_index = index_scenic_programs(source_dir, copy_to=source_copy_dir)

        for row_index, row in enumerate(scenario_bank):
            if not isinstance(row, dict):
                continue
            scenario_id = str(row.get("scenario_id", f"scenario_{row_index}"))
            safe_id = "".join(char if char.isalnum() or char in {"-", "_", "."} else "_" for char in scenario_id)
            config = row.get("config", {})
            if not isinstance(config, dict):
                config = {}
            if str(config.get("scenic_program_format", "")).strip() == "scenic_native_v1":
                native_dir = scenic_dir / "native"
                native_dir.mkdir(parents=True, exist_ok=True)
                native_program, unapplied = render_native_scenic_variant(row)
                map_artifacts, native_program = _materialize_native_map_references(
                    native_program,
                    config,
                    scenic_dir,
                )
                native_path = native_dir / f"{safe_id}.scenic"
                native_path.write_text(native_program, encoding="utf-8")
                native_entries.append(
                    {
                        "scenario_id": scenario_id,
                        "source_scenario_id": config.get("scenic_program_id", ""),
                        "relative_source_path": config.get("scenic_relative_path", ""),
                        "program_path": str(native_path.resolve()),
                        "map": config.get("carla_map", ""),
                        "native_execution_ready": bool(config.get("scenic_native_execution_ready", False)),
                        "unapplied_controls": unapplied,
                        "map_artifacts": map_artifacts,
                        "command": [
                            sys.executable,
                            "-m",
                            "scenic",
                            str(native_path.resolve()),
                            "--simulate",
                            "--count",
                            "1",
                        ],
                    }
                )
                continue
            if not isinstance(config.get("bench2drive_spec"), dict):
                continue
            scenic_dir.mkdir(parents=True, exist_ok=True)
            (scenic_dir / f"{safe_id}.scenic").write_text(render_scenic_program(row), encoding="utf-8")
            generated_count += 1
        if generated_count or native_entries:
            outputs["scenic_program_dir"] = scenic_dir
        if native_entries:
            execution_manifest_path = round_dir / str(
                scenic_cfg.get("execution_manifest_file_name", "scenic_execution_manifest.json")
            )
            execution_manifest_path.write_text(
                json.dumps(
                    {
                        "schema_version": "fdset.scenic_execution_manifest.v1",
                        "execution_mode": "native_scenic",
                        "program_count": len(native_entries),
                        "programs": native_entries,
                        "note": "These programs are native Scenic execution units; they are not converted to Bench2Drive route XML.",
                    },
                    indent=2,
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            outputs["scenic_execution_manifest"] = execution_manifest_path

        if source_index is not None:
            index_name = str(scenic_cfg.get("index_file_name", "scenic_library_index.json"))
            index_path = round_dir / index_name
            index_path.write_text(json.dumps(source_index, indent=2, ensure_ascii=False), encoding="utf-8")
            outputs["scenic_library_index"] = index_path
            if source_copy_dir is not None:
                outputs["scenic_source_dir"] = source_copy_dir

    bench_cfg = cfg.get("materialization", {}).get("bench2drive", {})
    enabled = bool(bench_cfg.get("enabled", True))
    if not enabled:
        return outputs

    xml_payload, trace_payload = build_bench2drive_routes_artifacts(scenario_bank)
    if xml_payload is not None:
        route_file_name = str(bench_cfg.get("route_file_name", "scenarios_bench2drive.xml"))
        xml_path = round_dir / route_file_name
        xml_path.write_text(xml_payload, encoding="utf-8")
        outputs["bench2drive_xml"] = xml_path
        trace_file_name = str(bench_cfg.get("trace_file_name", "materialization_trace.json"))
        trace_path = round_dir / trace_file_name
        trace_path.write_text(json.dumps(trace_payload, indent=2), encoding="utf-8")
        outputs["bench2drive_trace"] = trace_path

    evaluation_cfg = cfg.get("evaluation", {})
    held_out_routes_raw = str(evaluation_cfg.get("held_out_routes_xml", "")).strip()
    if bool(evaluation_cfg.get("use_held_out_routes", False)) and held_out_routes_raw:
        if bool(evaluation_cfg.get("include_evolved_scenarios", False)):
            raise ValueError(
                "evaluation.include_evolved_scenarios must be false: evolved Scenic scenarios "
                "are adaptation data, not part of the fixed benchmark feedback set"
            )
        held_out_routes_xml = Path(held_out_routes_raw).expanduser()
        evaluation_xml, evaluation_trace = build_bench2drive_evaluation_routes_artifacts(
            held_out_routes_xml,
            scenario_bank,
            # The official benchmark is a fixed feedback/re-evaluation set.
            # Evolved Scenic scenarios are materialized separately for the
            # PDM-Lite data-generation stage and must never be appended here.
            include_evolved=False,
            route_id_prefix=str(evaluation_cfg.get("evolved_route_id_prefix", "fdset_evolved")),
        )
        evaluation_name = str(evaluation_cfg.get("routes_file_name", "evaluation_routes.xml"))
        evaluation_path = round_dir / evaluation_name
        evaluation_path.write_text(evaluation_xml, encoding="utf-8")
        outputs["evaluation_xml"] = evaluation_path
        evaluation_trace_name = str(evaluation_cfg.get("trace_file_name", "evaluation_materialization_trace.json"))
        evaluation_trace_path = round_dir / evaluation_trace_name
        evaluation_trace_path.write_text(json.dumps(evaluation_trace, indent=2), encoding="utf-8")
        outputs["evaluation_trace"] = evaluation_trace_path
    return outputs


def materialize_adaptation_inputs(
    scenario_bank: list[dict[str, Any]],
    output_dir: Path,
    cfg: dict[str, Any],
) -> dict[str, Path]:
    """Materialize only Scenic/PDM-Lite inputs for one adaptation round.

    This intentionally disables the evaluation stage.  The returned route
    XML, when possible, describes the generated adaptation scenarios only;
    it is never the official ``bench2drive220.xml`` evaluation input.
    """
    adaptation_cfg = copy.deepcopy(cfg)
    adaptation_cfg["evaluation"] = {"use_held_out_routes": False}
    materialization_cfg = adaptation_cfg.setdefault("materialization", {})
    if not isinstance(materialization_cfg, dict):
        materialization_cfg = {}
        adaptation_cfg["materialization"] = materialization_cfg
    scenic_cfg = materialization_cfg.setdefault("scenic", {})
    if not isinstance(scenic_cfg, dict):
        scenic_cfg = {}
        materialization_cfg["scenic"] = scenic_cfg
    # Adaptation data must always have a Scenic artifact, even when the caller
    # disabled Scenic materialization for the benchmark-evaluation path.
    scenic_cfg["enabled"] = True
    scenic_cfg["program_dir"] = str(scenic_cfg.get("program_dir", "scenic_programs"))
    bench_cfg = materialization_cfg.setdefault("bench2drive", {})
    if not isinstance(bench_cfg, dict):
        bench_cfg = {}
        materialization_cfg["bench2drive"] = bench_cfg
    bench_cfg["route_file_name"] = str(bench_cfg.get("route_file_name", "adaptation_routes.xml"))
    return materialize_round_inputs(scenario_bank, output_dir, adaptation_cfg)


def _materialize_native_map_references(
    program: str,
    config: dict[str, Any],
    scenic_dir: Path,
) -> tuple[list[str], str]:
    """Copy maps next to a round's native programs and rewrite localPath refs."""
    references = config.get("scenic_map_references", [])
    if not isinstance(references, list):
        return [], program

    map_dir = scenic_dir / "maps"
    copied: list[str] = []
    rewritten = program
    for reference in references:
        if not isinstance(reference, dict):
            continue
        source_raw = str(reference.get("resolved_path", "")).strip()
        relative_raw = str(reference.get("relative_path", "")).strip()
        if not source_raw or not relative_raw:
            continue
        source_path = Path(source_raw).expanduser()
        if not source_path.exists() or not source_path.is_file():
            continue
        target = map_dir / source_path.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, target)
        copied.append(str(target.resolve()))
        rewritten = re.sub(
            rf"(param\s+map\s*=\s*localPath\(\s*)(['\"]){re.escape(relative_raw)}\2",
            rf"\g<1>\g<2>../maps/{source_path.name}\g<2>",
            rewritten,
            count=1,
        )
    return list(dict.fromkeys(copied)), rewritten


def build_bench2drive_routes_xml(scenario_bank: list[dict[str, Any]]) -> str | None:
    xml_payload, _ = build_bench2drive_routes_artifacts(scenario_bank)
    return xml_payload


def build_bench2drive_routes_artifacts(scenario_bank: list[dict[str, Any]]) -> tuple[str | None, list[dict[str, Any]]]:
    routes_root = ET.Element("routes")
    route_count = 0
    trace_rows: list[dict[str, Any]] = []

    for scenario_row in scenario_bank:
        materialized, trace_row = _materialize_bench2drive_route(scenario_row)
        if materialized is None:
            continue
        routes_root.append(materialized)
        route_count += 1
        if trace_row is not None:
            trace_rows.append(trace_row)

    if route_count == 0:
        return None, trace_rows

    _indent_xml(routes_root)
    return ET.tostring(routes_root, encoding="unicode"), trace_rows


def build_bench2drive_evaluation_routes_artifacts(
    held_out_routes_xml: Path,
    scenario_bank: list[dict[str, Any]],
    *,
    include_evolved: bool = False,
    route_id_prefix: str = "fdset_evolved",
) -> tuple[str, list[dict[str, Any]]]:
    """Materialize the fixed benchmark, optionally with an explicit stress set.

    The held-out XML is copied verbatim and is never inserted into the scenario
    bank.  ``include_evolved`` exists only for callers which explicitly need a
    separate stress-test XML; the paper feedback loop leaves it disabled so
    the benchmark file remains exactly the fixed ``bench2drive220.xml`` set.
    """
    if not held_out_routes_xml.exists():
        raise FileNotFoundError(f"Held-out Bench2Drive routes XML not found: {held_out_routes_xml}")

    root = ET.parse(held_out_routes_xml).getroot()
    if root.tag != "routes":
        raise ValueError(f"Held-out routes XML must have a <routes> root: {held_out_routes_xml}")

    trace_rows: list[dict[str, Any]] = [
        {
            "source": "held_out_routes",
            "route_count": sum(1 for _ in root.iter("route")),
            "scenario_count": sum(1 for _ in root.iter("scenario")),
            "routes_xml": str(held_out_routes_xml),
        }
    ]
    if include_evolved:
        for scenario_row in scenario_bank:
            config = scenario_row.get("config", {})
            if not isinstance(config, dict):
                continue
            if str(config.get("scenario_role", "")).strip().lower() != "evolved":
                continue
            scenario_id = str(scenario_row.get("scenario_id", "generated"))
            materialized, trace_row = _materialize_bench2drive_route(
                scenario_row,
                route_id_override=f"{route_id_prefix}_{scenario_id}",
            )
            if materialized is None:
                continue
            root.append(materialized)
            if trace_row is not None:
                trace_row["source"] = "evolved_scenario"
                trace_rows.append(trace_row)

    _indent_xml(root)
    payload = ET.tostring(root, encoding="unicode")
    validate_bench2drive_routes_xml(payload)
    return payload, trace_rows


def validate_bench2drive_routes_xml(xml_payload: str) -> dict[str, int]:
    """Perform structural validation before an XML route file reaches CARLA."""
    from .bench2drive_catalog import known_scenario_types, supported_parameter_tags, IGNORED_PARAMETER_TAGS

    root = ET.fromstring(xml_payload)
    if root.tag != "routes":
        raise ValueError("Bench2Drive route XML must have a <routes> root")

    route_count = 0
    scenario_count = 0
    known_types = known_scenario_types()
    known_tags = supported_parameter_tags() | set(IGNORED_PARAMETER_TAGS)
    for route in root.iter("route"):
        route_count += 1
        if not route.attrib.get("id") or not route.attrib.get("town"):
            raise ValueError("Every Bench2Drive route must define id and town")
        if route.find("waypoints") is None:
            raise ValueError(f"Route {route.attrib.get('id')} is missing waypoints")
        for scenario in route.iter("scenario"):
            scenario_count += 1
            scenario_type = str(scenario.attrib.get("type", ""))
            if scenario_type not in known_types:
                raise ValueError(f"Unsupported Bench2Drive scenario type: {scenario_type}")
            if scenario.find("trigger_point") is None:
                raise ValueError(f"Scenario {scenario.attrib.get('name', '')} is missing trigger_point")
            for child in scenario:
                if child.tag not in {"trigger_point", *known_tags}:
                    raise ValueError(f"Unsupported Bench2Drive parameter tag: {child.tag}")

    if route_count == 0 or scenario_count == 0:
        raise ValueError("Bench2Drive route XML must contain at least one route and scenario")
    return {"route_count": route_count, "scenario_count": scenario_count}


def _materialize_bench2drive_route(
    scenario_row: dict[str, Any],
    *,
    route_id_override: str | None = None,
) -> tuple[ET.Element | None, dict[str, Any] | None]:
    validation_errors = validate_scenario_row(scenario_row, require_executable=False)
    if validation_errors:
        raise ValueError("Invalid scenario row: " + "; ".join(validation_errors))
    config = dict(scenario_row.get("config", {}))
    spec = config.get("bench2drive_spec")
    if not isinstance(spec, dict):
        return None, None

    route_spec = spec.get("route")
    scenario_spec = spec.get("scenario")
    if not isinstance(route_spec, dict) or not isinstance(scenario_spec, dict):
        return None, None

    scenario_type = str(scenario_spec.get("type", "ControlLoss"))
    scenario_profile = config.get("bench2drive_profile", {})
    scenario_family = _resolve_scenario_family(config, scenario_type)

    route_elem = ET.Element(
        "route",
        {
            "id": str(route_id_override or route_spec.get("id", scenario_row.get("scenario_id", "generated_route"))),
            "town": str(route_spec.get("town", "Town01")),
        },
    )

    waypoints_elem = ET.SubElement(route_elem, "waypoints")
    for waypoint in route_spec.get("waypoints", []):
        if not isinstance(waypoint, dict):
            continue
        ET.SubElement(
            waypoints_elem,
            "position",
            {
                "x": str(waypoint.get("x", 0.0)),
                "y": str(waypoint.get("y", 0.0)),
                "z": str(waypoint.get("z", 0.0)),
            },
        )

    weather_entries = route_spec.get("weathers", [])
    if isinstance(weather_entries, list) and weather_entries:
        weathers_elem = ET.SubElement(route_elem, "weathers")
        for weather in weather_entries:
            if not isinstance(weather, dict):
                continue
            ET.SubElement(weathers_elem, "weather", {str(k): str(v) for k, v in weather.items()})

    scenarios_elem = ET.SubElement(route_elem, "scenarios")
    scenario_elem = ET.SubElement(
        scenarios_elem,
        "scenario",
        {
            "name": str(scenario_row.get("scenario_id", scenario_spec.get("name", "GeneratedScenario"))),
            "type": str(scenario_spec.get("type", "ControlLoss")),
        },
    )

    trigger = scenario_spec.get("trigger_point", {})
    if isinstance(trigger, dict):
        ET.SubElement(
            scenario_elem,
            "trigger_point",
            {
                "x": str(trigger.get("x", 0.0)),
                "y": str(trigger.get("y", 0.0)),
                "z": str(trigger.get("z", 0.0)),
                "yaw": str(trigger.get("yaw", 0.0)),
            },
        )

    parameters = copy.deepcopy(scenario_spec.get("parameters", []))
    if not isinstance(parameters, list):
        parameters = []
    parameters, applied_updates = _apply_modifications_to_parameters(parameters, config, scenario_type=scenario_type)

    for parameter in parameters:
        if not isinstance(parameter, dict):
            continue
        tag = str(parameter.get("tag", "")).strip()
        attrs = parameter.get("attrs", {})
        if not tag or not isinstance(attrs, dict):
            continue
        ET.SubElement(scenario_elem, tag, {str(k): str(v) for k, v in attrs.items()})

    trace_row = {
        "scenario_id": str(scenario_row.get("scenario_id", "")),
        "route_id": str(route_spec.get("id", "")),
        "town": str(route_spec.get("town", "")),
        "scenario_type": scenario_type,
        "scenario_family": scenario_family,
        "parameter_tags": [str(parameter.get("tag", "")) for parameter in parameters if isinstance(parameter, dict)],
        "applied_parameter_updates": applied_updates,
        "profile": scenario_profile if isinstance(scenario_profile, dict) else {},
    }
    return route_elem, trace_row


def _apply_modifications_to_parameters(
    parameters: list[dict[str, Any]],
    config: dict[str, Any],
    *,
    scenario_type: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    updated = copy.deepcopy(parameters)
    applied_updates: list[dict[str, Any]] = []
    scenario_family = _resolve_scenario_family(config, scenario_type)

    catalog_overrides = build_parameter_overrides(
        updated,
        scenario_type=scenario_type,
        config=config,
        scenario_family=scenario_family,
    )
    _apply_parameter_override_map(updated, catalog_overrides, "catalog_mapping", applied_updates)
    _apply_direct_parameter_overrides(updated, config, applied_updates=applied_updates)
    return updated, applied_updates


def _apply_direct_parameter_overrides(
    parameters: list[dict[str, Any]],
    config: dict[str, Any],
    *,
    applied_updates: list[dict[str, Any]],
) -> None:
    raw_overrides = config.get("bench2drive_parameter_overrides", {})
    if not isinstance(raw_overrides, dict):
        return

    for tag, attrs in raw_overrides.items():
        if not isinstance(attrs, dict):
            continue
        normalized = {str(k): str(v) for k, v in attrs.items()}
        _update_parameter_attrs(parameters, str(tag), normalized, "direct_override", applied_updates)


def _apply_parameter_override_map(
    parameters: list[dict[str, Any]],
    overrides: dict[str, dict[str, str]],
    reason: str,
    applied_updates: list[dict[str, Any]],
) -> None:
    for tag, attrs in overrides.items():
        _update_parameter_attrs(parameters, tag, attrs, reason, applied_updates)


def _update_parameter_attrs(
    parameters: list[dict[str, Any]],
    tag: str,
    attrs: dict[str, str],
    reason: str,
    applied_updates: list[dict[str, Any]],
) -> None:
    for parameter in parameters:
        if not isinstance(parameter, dict):
            continue
        if str(parameter.get("tag", "")) != tag:
            continue
        parameter_attrs = parameter.get("attrs", {})
        if not isinstance(parameter_attrs, dict):
            continue
        parameter_attrs.update(attrs)
        parameter["attrs"] = parameter_attrs
        applied_updates.append(
            {
                "tag": tag,
                "attrs": dict(attrs),
                "reason": reason,
            }
        )


def _resolve_scenario_family(config: dict[str, Any], scenario_type: str) -> str:
    profile = config.get("bench2drive_profile", {})
    if isinstance(profile, dict):
        family = str(profile.get("scenario_family", "")).strip().lower()
        if family:
            return family

    return infer_scenario_family(scenario_type, str(config.get("interaction_pattern", "")).strip().lower())


def _indent_xml(elem: ET.Element, level: int = 0) -> None:
    indent = "\n" + level * "   "
    child_indent = "\n" + (level + 1) * "   "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = child_indent
        for child in elem:
            _indent_xml(child, level + 1)
        if not elem[-1].tail or not elem[-1].tail.strip():
            elem[-1].tail = indent
    if level and (not elem.tail or not elem.tail.strip()):
        elem.tail = indent
