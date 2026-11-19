from __future__ import annotations

"""Run a real CARLA audit for each native Scenic source and explicit seed."""

import argparse
import hashlib
import gzip
import json
import math
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.scenic_library import index_scenic_programs  # noqa: E402
from pdm_lite.camera_profiles import CAMERA_PROFILES  # noqa: E402


def _safe(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "program"


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def _read_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                value = json.loads(line)
                if isinstance(value, dict):
                    rows.append(value)
    return rows


def _read_gzip_json(path: Path) -> Any:
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return json.load(stream)


def _finite(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _valid_windows(samples: list[dict[str, Any]]) -> int:
    """Count real 11-point windows sampled contiguously at 0.5-second intervals."""
    rows = []
    for row in samples:
        position = row.get("pos_global")
        acceleration = row.get("acceleration")
        stamp = _finite(row.get("timestamp_s"))
        command = _finite(row.get("command"))
        valid = (
            isinstance(position, list) and len(position) >= 2
            and _finite(position[0]) is not None and _finite(position[1]) is not None
            and _finite(row.get("theta")) is not None and _finite(row.get("speed")) is not None
            and isinstance(acceleration, list) and len(acceleration) == 3
            and all(_finite(value) is not None for value in acceleration)
            and stamp is not None and command is not None and command == int(command)
            and int(command) in {-1, 1, 2, 3, 4, 5, 6}
        )
        if valid:
            rows.append((stamp, row))
    rows.sort(key=lambda pair: pair[0])
    runs: list[int] = []
    run = 0
    previous = None
    for stamp, _ in rows:
        if previous is not None and not math.isclose(stamp - previous, 0.5, rel_tol=0.0, abs_tol=0.02):
            runs.append(run)
            run = 0
        run += 1
        previous = stamp
    if run:
        runs.append(run)
    return sum(max(0, length - 10) for length in runs)


def _audit_output(output_dir: Path, source: dict[str, Any], seed: int, command: list[str],
                  process_status: int, log_path: Path, manifest_path: Path) -> dict[str, Any]:
    result_path = output_dir / "native_scenic_policy_results.json"
    result: dict[str, Any] = {}
    if result_path.is_file():
        try:
            result = _load_json(result_path)
        except (OSError, ValueError):
            pass
    program_rows = result.get("programs", []) if isinstance(result, dict) else []
    run = next((r for r in program_rows if isinstance(r, dict)), {})
    route = output_dir / "pdm_lite_dataset" / "FDSET_SCENIC_Rep0_0"
    measurement_files = sorted((route / "measurements").glob("*.json.gz"))
    boxes_files = sorted((route / "boxes").glob("*.json.gz"))
    measurements: list[dict[str, Any]] = []
    measurement_times: dict[str, float] = {}
    measurement_rows_by_frame: dict[str, dict[str, Any]] = {}
    for path in measurement_files:
        try:
            row = _read_gzip_json(path)
            if isinstance(row, dict):
                frame = path.name[:-8]
                measurements.append(row)
                measurement_rows_by_frame[frame] = row
                stamp = _finite(row.get("timestamp_s"))
                if stamp is not None:
                    measurement_times[frame] = stamp
        except (OSError, ValueError):
            continue
    boxes_by_frame: dict[str, list[dict[str, Any]]] = {}
    for path in boxes_files:
        try:
            values = _read_gzip_json(path)
            if isinstance(values, list):
                boxes_by_frame[path.name[:-8]] = [v for v in values if isinstance(v, dict)]
        except (OSError, ValueError):
            continue
    hazards = {
        "vehicle": sum(bool(r.get("vehicle_hazard")) for r in measurements),
        "walker": sum(bool(r.get("walker_hazard")) for r in measurements),
        "traffic_light": sum(bool(r.get("light_hazard")) for r in measurements),
        "stop_sign": sum(bool(r.get("stop_sign_hazard")) for r in measurements),
    }
    reductions: dict[str, int] = {}
    for row in measurements:
        obj = row.get("speed_reduced_by_obj_type")
        if obj is not None and str(obj):
            key = str(obj)
            reductions[key] = reductions.get(key, 0) + 1
    reductions_by_target: dict[str, int] = {}
    for row in measurements:
        obj_type = row.get("speed_reduced_by_obj_type")
        obj_id = row.get("speed_reduced_by_obj_id")
        if obj_type is not None or obj_id is not None:
            key = f"{obj_type or 'unknown'}#{obj_id if obj_id is not None else 'unknown'}"
            reductions_by_target[key] = reductions_by_target.get(key, 0) + 1
    commands: dict[str, int] = {}
    for row in measurements:
        value = row.get("command")
        if isinstance(value, (int, float)):
            key = str(int(value))
            commands[key] = commands.get(key, 0) + 1
    navigation_commands = {"left": 0, "right": 0, "straight": 0, "lane_follow": 0, "other": 0}
    command_names = {"1": "left", "2": "right", "3": "straight", "4": "lane_follow"}
    for key, count in commands.items():
        navigation_commands[command_names.get(key, "other")] += count

    ego_samples = [(r, _finite(r.get("speed"))) for r in measurements]
    ego_samples = [(r, s) for r, s in ego_samples if s is not None]
    displacement_speeds: list[float] = []
    near_counts = {"vehicle": 0, "walker": 0, "static": 0, "bicycle": 0}
    type_counts: dict[str, int] = {}
    type_near_counts: dict[str, int] = {}
    stop_light_boxes = 0
    light_affected_frames: set[str] = set()
    actor_motion: dict[str, list[tuple[float, list[float], float | None]]] = {}
    actor_class_by_id: dict[str, str] = {}
    actor_type_by_id: dict[str, str] = {}
    for index, path in enumerate(boxes_files):
        records = boxes_by_frame.get(path.name[:-8], [])
        stamp = measurement_times.get(path.name[:-8])
        for item in records:
            klass = str(item.get("class", "")).lower()
            type_id = str(item.get("type_id", ""))
            dist = _finite(item.get("distance"))
            if type_id:
                type_counts[type_id] = type_counts.get(type_id, 0) + 1
            if klass == "car" and dist is not None and dist <= 20:
                near_counts["vehicle"] += 1
                if type_id:
                    type_near_counts[type_id] = type_near_counts.get(type_id, 0) + 1
                if "bicycle" in type_id.lower() or "bike" in type_id.lower() or "diamondback" in type_id.lower():
                    near_counts["bicycle"] += 1
            elif klass == "walker" and dist is not None and dist <= 25:
                near_counts["walker"] += 1
                if type_id:
                    type_near_counts[type_id] = type_near_counts.get(type_id, 0) + 1
            elif klass == "static" and dist is not None and dist <= 25:
                near_counts["static"] += 1
                if type_id:
                    type_near_counts[type_id] = type_near_counts.get(type_id, 0) + 1
            if (
                ("traffic_light" in klass or "trafficlight" in klass or "traffic_light" in type_id)
                and bool(item.get("affects_ego", False))
            ):
                stop_light_boxes += 1
                light_affected_frames.add(path.name[:-8])
            ident = item.get("id")
            matrix = item.get("matrix")
            xyz = None
            if isinstance(matrix, list) and len(matrix) >= 3:
                xyz = [_finite(matrix[0][3]), _finite(matrix[1][3]), _finite(matrix[2][3])]
            if klass != "ego_car" and ident is not None and xyz is not None and None not in xyz:
                speed = _finite(item.get("speed"))
                actor_type_by_id[str(ident)] = type_id
                actor_class_by_id[str(ident)] = klass
                actor_motion.setdefault(str(ident), []).append(
                    (stamp if stamp is not None else float(index), xyz, speed)
                )
    ego_displacement = []
    for left, right in zip(measurements, measurements[1:]):
        p0, p1 = left.get("pos_global"), right.get("pos_global")
        t0, t1 = _finite(left.get("timestamp_s")), _finite(right.get("timestamp_s"))
        if isinstance(p0, list) and isinstance(p1, list) and len(p0) >= 2 and len(p1) >= 2 and t0 is not None and t1 is not None and t1 > t0:
            x0, y0, x1, y1 = map(_finite, (p0[0], p0[1], p1[0], p1[1]))
            if None not in (x0, y0, x1, y1):
                ego_displacement.append(math.hypot(x1 - x0, y1 - y0) / (t1 - t0))
    ego_speed_values = [s for _, s in ego_samples]
    valid_pose = [
        (r["pos_global"], _finite(r.get("theta")))
        for r in measurements
        if isinstance(r.get("pos_global"), list) and len(r["pos_global"]) >= 2
        and all(_finite(value) is not None for value in r["pos_global"][:2])
        and _finite(r.get("theta")) is not None
    ]
    ego_start_to_end_displacement = (
        math.dist(valid_pose[0][0][:2], valid_pose[-1][0][:2])
        if len(valid_pose) >= 2 else 0.0
    )
    heading_delta = (
        abs(math.atan2(math.sin(valid_pose[-1][1] - valid_pose[0][1]),
                       math.cos(valid_pose[-1][1] - valid_pose[0][1])))
        if len(valid_pose) >= 2 else None
    )
    speed_errors = [abs(a - b) for a, b in zip(ego_speed_values[1:], ego_displacement)]
    actor_speed_checks = []
    actor_displacement: dict[str, float] = {}
    actor_speed_errors_by_id: dict[str, list[float]] = {}
    for ident, rows in actor_motion.items():
        distances = []
        for a, b in zip(rows, rows[1:]):
            dt = b[0] - a[0]
            if dt > 0:
                distances.append(math.dist(a[1], b[1]))
                displacement = distances[-1] / dt
                if b[2] is not None:
                    error = abs(displacement - abs(b[2]))
                    actor_speed_checks.append(error)
                    actor_speed_errors_by_id.setdefault(ident, []).append(error)
        if distances:
            actor_displacement[ident] = sum(distances)
    effective_light = sum(
        bool(measurement_rows_by_frame.get(frame, {}).get("light_hazard"))
        for frame in light_affected_frames
    )
    green_frames = [
        frame for frame in sorted(light_affected_frames)
        if not bool(measurement_rows_by_frame.get(frame, {}).get("light_hazard"))
        and (_finite(measurement_rows_by_frame.get(frame, {}).get("speed")) or 0) > 0.5
    ]
    red_stop_frames = [
        frame for frame in sorted(light_affected_frames)
        if bool(measurement_rows_by_frame.get(frame, {}).get("light_hazard"))
        and (_finite(measurement_rows_by_frame.get(frame, {}).get("speed")) or 0) < 0.5
    ]
    effective_hazards = {
        "vehicle": hazards["vehicle"] if near_counts["vehicle"] else 0,
        "walker": hazards["walker"] if near_counts["walker"] else 0,
        "traffic_light": effective_light,
    }
    family = str(source.get("interaction_family", ""))
    bicycle_family = family in {"bicycle_crossing", "intersection_bicycle"}
    pedestrian_family = family == "pedestrian_crossing"
    static_family = family in {"road_debris", "static_obstacle", "narrow_passage"}
    navigation_target = {
        "left_turn_navigation": "left", "right_turn_navigation": "right",
        "straight_junction_navigation": "straight",
        "unprotected_left_turn": "left", "right_turn_crossing": "right",
    }.get(family)
    navigation_observable = bool(navigation_target and navigation_commands[navigation_target])
    bicycle_type_ids = {
        type_id for type_id in set(type_counts) | set(reductions)
        if any(name in type_id.lower() for name in ("diamondback", "gazelle", "omafiets", "bicycle", "bike"))
    }
    bicycle_actor_ids = {ident for ident, type_id in actor_type_by_id.items() if type_id in bicycle_type_ids}
    bicycle_moved = any(actor_displacement.get(ident, 0) > 1 for ident in bicycle_actor_ids)
    bicycle_reduction = any(reductions.get(type_id, 0) for type_id in bicycle_type_ids)
    walker_moved = any(
        actor_displacement.get(ident, 0) > 1 for ident, klass in actor_class_by_id.items() if klass == "walker"
    )
    if navigation_target and family in {"unprotected_left_turn", "right_turn_crossing"}:
        interaction_observable = bool(
            navigation_observable and near_counts["vehicle"] and (
                effective_hazards["vehicle"] or
                any(obj.startswith("vehicle.") for obj in reductions)
            )
        )
        interaction_basis = "near_turn_command_and_conflicting_vehicle_response" if interaction_observable else "none"
    elif family == "straight_junction_navigation":
        # CARLA labels some straight connectors LANEFOLLOW throughout.
        # A sampled through-junction goal plus a measured straight crossing
        # proves the maneuver even when no RoadOption.STRAIGHT is emitted.
        straight_crossing = (
            len(valid_pose) >= 8 and ego_start_to_end_displacement > 30
            and heading_delta is not None and heading_delta < 0.2
            and bool(navigation_commands["lane_follow"])
        )
        interaction_observable = navigation_observable or straight_crossing
        interaction_basis = (
            "near_straight_command" if navigation_observable
            else "sampled_straight_goal_and_measured_through_junction_trajectory"
            if straight_crossing else "none"
        )
    elif navigation_target:
        interaction_observable = navigation_observable
        interaction_basis = f"near_{navigation_target}_command" if interaction_observable else "none"
    elif bicycle_family:
        interaction_observable = (
            bicycle_moved and bicycle_reduction
            and any(type_near_counts.get(type_id, 0) for type_id in bicycle_type_ids)
        )
        interaction_basis = "moving_bicycle_and_expert_speed_reduction" if interaction_observable else "none"
    elif pedestrian_family:
        interaction_observable = near_counts["walker"] > 0 and walker_moved and hazards["walker"] > 0
        interaction_basis = "moving_walker_and_walker_hazard" if interaction_observable else "none"
    elif family == "green_light_start":
        interaction_observable = bool(green_frames)
        interaction_basis = "affected_green_traffic_light_and_moving_ego" if interaction_observable else "none"
    elif family == "signal_stop_restart":
        interaction_observable = bool(red_stop_frames and any(frame > red_stop_frames[0] for frame in green_frames))
        interaction_basis = "affected_red_stop_then_green_departure" if interaction_observable else "none"
    elif family == "narrow_passage":
        interaction_observable = (
            near_counts["static"] > 0
            and ego_start_to_end_displacement > 30
            and not run.get("collisions", 0)
        )
        interaction_basis = "nearby_static_obstacle_and_collision_free_passage" if interaction_observable else "none"
    elif static_family:
        interaction_observable = near_counts["static"] > 0
        interaction_basis = "static_box_observed" if interaction_observable else "none"
    else:
        interaction_observable = bool(
            near_counts["vehicle"] and (
                effective_hazards["vehicle"] or
                any(obj.startswith("vehicle.") for obj in reductions)
            )
        )
        interaction_basis = "nearby_vehicle_and_expert_response" if interaction_observable else "none"
    status = "completed" if process_status == 0 and run.get("status") == "completed" else (
        "failed" if process_status != 0 or run.get("status") in {"error", "skipped"} else "incomplete"
    )
    measured_windows = _valid_windows(measurements)
    eligible_measurements = (
        [row for row in measurements if not row.get("stop_sign_hazard") and not row.get("light_hazard")]
        if source.get("program_id") == "fdset_green_light_start" else measurements
    )
    safe_windows = _valid_windows(eligible_measurements)
    return {
        "scenario_id": source.get("program_id"), "source_path": source.get("source_path"), "seed": seed,
        "status": status, "execution_status": run.get("status", "missing"),
        "termination_reason": run.get("termination_reason"), "steps": run.get("steps", 0),
        "collision_count": run.get("collisions", 0),
        "measurement_frames": len(measurements), "box_frames": len(boxes_files), "command_counts": commands,
        "hazard_frame_counts": hazards, "effective_hazard_frame_counts": effective_hazards,
        "speed_reduced_by_object_counts": reductions,
        "speed_reduced_by_target_counts": reductions_by_target,
        "nearby_actor_frame_counts": near_counts,
        "affecting_traffic_light_box_count": stop_light_boxes,
        "ego_speed_mps": {"samples": len(ego_speed_values), "min": min(ego_speed_values) if ego_speed_values else None,
                          "max": max(ego_speed_values) if ego_speed_values else None,
                          "mean": sum(ego_speed_values) / len(ego_speed_values) if ego_speed_values else None},
        "ego_displacement_speed_mps": {"samples": len(ego_displacement), "mean": sum(ego_displacement) / len(ego_displacement) if ego_displacement else None,
                                       "speed_displacement_mae_mps": sum(speed_errors) / len(speed_errors) if speed_errors else None},
        "ego_start_to_end_displacement_m": ego_start_to_end_displacement,
        "ego_start_to_end_heading_delta_rad": heading_delta,
        "actor_box_total_displacement_m": sum(actor_displacement.values()),
        "actor_type_id_by_id": actor_type_by_id,
        "actor_box_speed_displacement_mae_mps": sum(actor_speed_checks) / len(actor_speed_checks) if actor_speed_checks else None,
        "actor_box_speed_displacement_mae_mps_by_id": {
            ident: sum(errors) / len(errors) for ident, errors in actor_speed_errors_by_id.items() if errors
        },
        "observed_type_id_counts": type_counts,
        "nearby_type_id_counts": type_near_counts,
        "trace_file": run.get("trace_file"),
        "navigation_command_counts": navigation_commands,
        "navigation_command_observable": navigation_observable,
        "interaction_observable": interaction_observable,
        "interaction_basis": interaction_basis,
        "interaction_status": (
            "unknown_due_to_run_failure" if status != "completed"
            else "navigation_command_observed" if navigation_observable and interaction_observable
            else "observed" if interaction_observable else "unobserved"
        ),
        "measurement_contiguous_0_5s_windows": measured_windows,
        "trainable_contiguous_0_5s_windows": (
            safe_windows if status == "completed" and not run.get("collisions", 0) else 0
        ),
        "artifacts": {"output_dir": str(output_dir.resolve()), "manifest": str(manifest_path.resolve()),
                      "result": str(result_path.resolve()), "log": str(log_path.resolve())},
        "command": command, "return_code": process_status,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=ROOT / "scenic")
    parser.add_argument("--seeds", default="2026,2027,2028", help="Comma-separated explicit integer seeds")
    parser.add_argument("--max-steps", type=int, default=600)
    parser.add_argument("--camera-profile", choices=tuple(CAMERA_PROFILES), default="admlp")
    parser.add_argument("--carla-port", type=int, default=3000)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts" / "scenic_bank_audit")
    parser.add_argument("--pythonapi-root", type=Path, default=ROOT.parent / "carla")
    parser.add_argument("--carla-host", default="127.0.0.1")
    parser.add_argument("--timeout", type=float, default=10.0)
    args = parser.parse_args()
    seeds = [int(part.strip()) for part in args.seeds.split(",") if part.strip()]
    if not seeds or len(set(seeds)) != len(seeds) or args.max_steps < 1 or args.carla_port < 1:
        parser.error("seeds must be unique/non-empty, max-steps and carla-port positive")
    source_dir = args.source_dir.expanduser().resolve()
    output_root = args.output.expanduser().resolve()
    index = index_scenic_programs(source_dir)
    runtime_hasher = hashlib.sha256()
    for relative_path in (
        "core/scenic_policy_adapter.py",
        "pdm_lite/agent/autopilot.py",
        "pdm_lite/agent/data_agent.py",
        "pdm_lite/scenic_collector.py",
    ):
        runtime_hasher.update(relative_path.encode("utf-8"))
        runtime_hasher.update((ROOT / relative_path).read_bytes())
    runtime_sha256 = runtime_hasher.hexdigest()
    output_root.mkdir(parents=True, exist_ok=True)
    audit_path = output_root / "audit.json"
    summaries: dict[tuple[str, int], dict[str, Any]] = {}
    if audit_path.is_file():
        try:
            prior = _load_json(audit_path)
            for row in prior.get("runs", []):
                if isinstance(row, dict) and row.get("status") == "completed":
                    summaries[(str(row.get("scenario_id")), int(row.get("seed")))] = row
        except (OSError, ValueError, TypeError):
            pass

    for source in index["programs"]:
        for seed in seeds:
            key = (str(source["program_id"]), seed)
            run_dir = output_root / _safe(key[0]) / f"seed_{seed}"
            summary_file = run_dir / "audit.json"
            cached = summaries.get(key)
            if cached is None and summary_file.is_file():
                try:
                    cached = _load_json(summary_file)
                except (OSError, ValueError):
                    pass
            cache_valid = False
            if isinstance(cached, dict) and cached.get("return_code") == 0:
                try:
                    artifacts = cached["artifacts"]
                    collector_dir = Path(artifacts["output_dir"])
                    cached_result = _load_json(collector_dir / "native_scenic_policy_results.json")
                    cached_run = next(
                        item for item in cached_result.get("programs", [])
                        if isinstance(item, dict) and item.get("status") == "completed"
                    )
                    cache_valid = (
                        cached.get("source_sha256") == source.get("sha256")
                        and cached.get("runtime_sha256") == runtime_sha256
                        and cached.get("max_steps") == args.max_steps
                        and cached.get("camera_profile") == args.camera_profile
                        and cached.get("carla_port") == args.carla_port
                        and summary_file.is_file()
                        and Path(artifacts["manifest"]).is_file()
                        and (collector_dir / "pdm_lite_manifest.json").is_file()
                        and _load_json(collector_dir / "pdm_lite_manifest.json").get("camera_profile") == args.camera_profile
                        and Path(artifacts["log"]).is_file()
                        and cached.get("carla_host") == args.carla_host
                        and cached.get("timeout") == args.timeout
                        and cached.get("pythonapi_root") == str(args.pythonapi_root.expanduser().resolve())
                        and Path(str(cached_run.get("trace_file", ""))).is_file()
                        and cached.get("scenario_id") == key[0]
                        and cached.get("seed") == seed
                    )
                except (KeyError, OSError, StopIteration, TypeError, ValueError):
                    cache_valid = False
            if cache_valid:
                fresh = _audit_output(
                    collector_dir, source, seed, cached["command"], 0,
                    Path(artifacts["log"]), Path(artifacts["manifest"]),
                )
                if fresh["status"] == "completed" and fresh["measurement_frames"]:
                    cached.update(fresh)
                    summaries[key] = cached
                    summary_file.write_text(json.dumps(cached, indent=2, ensure_ascii=False), encoding="utf-8")
                    print(json.dumps({"scenario_id": key[0], "seed": seed, "status": cached["status"],
                                      "interaction_status": cached["interaction_status"], "cached": True}), flush=True)
                    continue
            run_dir.mkdir(parents=True, exist_ok=True)
            attempt_dir = run_dir / f"attempt_{len(list(run_dir.glob('attempt_*'))) + 1:03d}"
            attempt_dir.mkdir()
            manifest_path = attempt_dir / "manifest.json"
            source_path = Path(source["source_path"]).resolve()
            manifest = {
                "schema_version": "fdset.scenic_execution_manifest.v1", "execution_mode": "native_scenic",
                "program_count": 1, "programs": [{
                    "scenario_id": key[0], "source_scenario_id": key[0],
                    "relative_source_path": source.get("relative_path", ""),
                    "program_path": str(source_path), "map": source.get("carla_map", ""),
                    "native_execution_ready": bool(source.get("native_execution_ready")),
                    "map_artifacts": [str(x.get("resolved_path")) for x in source.get("map_references", []) if x.get("exists")],
                }],
            }
            manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
            output_dir = attempt_dir / "pdm_lite"
            demos_file = output_dir / "expert_demos.jsonl"
            command = [sys.executable, str(ROOT / "pdm_lite" / "scenic_collector.py"), "--manifest", str(manifest_path),
                       "--output-dir", str(output_dir), "--demos-file", str(demos_file), "--carla-host", args.carla_host,
                       "--carla-port", str(args.carla_port), "--traffic-manager-port", str(args.carla_port + 50),
                       "--pythonapi-root", str(args.pythonapi_root.expanduser().resolve()), "--seed", str(seed),
                       "--max-steps", str(args.max_steps), "--camera-profile", args.camera_profile, "--timeout", str(args.timeout)]
            log_path = attempt_dir / "collector.log"
            started = time.time()
            with log_path.open("wb") as log:
                try:
                    completed = subprocess.run(
                        command, cwd=str(ROOT), stdout=log, stderr=subprocess.STDOUT, check=False,
                        timeout=max(args.timeout * 3, args.max_steps * 3),
                    )
                    return_code = completed.returncode
                except subprocess.TimeoutExpired:
                    log.write(b"CARLA collector exceeded the per-run wall timeout\n")
                    return_code = 124
                except OSError as exc:
                    log.write((f"Could not start collector: {exc}\n").encode("utf-8", errors="replace"))
                    return_code = 127
            row = _audit_output(output_dir, source, seed, command, return_code, log_path, manifest_path)
            row["duration_seconds"] = time.time() - started
            row["max_steps"] = args.max_steps
            row["carla_port"] = args.carla_port
            row["camera_profile"] = args.camera_profile
            row["carla_host"] = args.carla_host
            row["timeout"] = args.timeout
            row["pythonapi_root"] = str(args.pythonapi_root.expanduser().resolve())
            row["source_sha256"] = source.get("sha256")
            row["runtime_sha256"] = runtime_sha256
            row["native_execution_ready"] = bool(source.get("native_execution_ready"))
            row["diagnostics"] = source.get("diagnostics", [])
            row["index_family"] = source.get("interaction_family")
            if not source.get("native_execution_ready") and row["status"] == "incomplete":
                row["status"] = "failed"
                row["execution_status"] = "not_ready"
                row["interaction_status"] = "unknown_due_to_run_failure"
            row["completed_at_epoch"] = time.time()
            summary_file.write_text(json.dumps(row, indent=2, ensure_ascii=False), encoding="utf-8")
            summaries[key] = row
            rows = [summaries[(str(item["program_id"]), seed_value)] for item in index["programs"] for seed_value in seeds if (str(item["program_id"]), seed_value) in summaries]
            aggregate = {"schema_version": "fdset.scenic_bank_audit.v1", "source_dir": str(source_dir),
                         "camera_profile": args.camera_profile, "requested_programs": len(index["programs"]),
                         "requested_seeds": seeds, "completed_runs": sum(r.get("status") == "completed" for r in rows),
                         "runs": rows}
            temp = audit_path.with_suffix(".json.tmp")
            temp.write_text(json.dumps(aggregate, indent=2, ensure_ascii=False), encoding="utf-8")
            os.replace(temp, audit_path)
            print(json.dumps({"scenario_id": key[0], "seed": seed, "status": row["status"],
                              "interaction_status": row["interaction_status"], "steps": row["steps"],
                              "windows": row["trainable_contiguous_0_5s_windows"], "summary": str(audit_path)}), flush=True)

    rows = [summaries[(str(item["program_id"]), seed_value)] for item in index["programs"]
            for seed_value in seeds if (str(item["program_id"]), seed_value) in summaries]
    aggregate = {"schema_version": "fdset.scenic_bank_audit.v1", "source_dir": str(source_dir),
                 "camera_profile": args.camera_profile, "requested_programs": len(index["programs"]),
                 "requested_seeds": seeds,
                 "completed_runs": sum(row.get("status") == "completed" for row in rows), "runs": rows}
    temp = audit_path.with_suffix(".json.tmp")
    temp.write_text(json.dumps(aggregate, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(temp, audit_path)
    print(json.dumps({"audit": str(audit_path), "runs": len(summaries), "completed": sum(r.get("status") == "completed" for r in summaries.values())}, ensure_ascii=False))


if __name__ == "__main__":
    main()
