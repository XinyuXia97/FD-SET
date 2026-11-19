#!/usr/bin/env python3
"""Collect PDM-Lite demonstrations by running native Scenic programs in CARLA.

Scenic owns scene generation and non-ego behaviors; the extracted PDM-Lite
DataAgent owns ego controls, measurements, and the full sensor suite. No route
XML or held-out Bench2Drive evaluator is involved.
"""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
from typing import Any

from core.scenic_policy_adapter import load_native_execution_manifest, run_native_policy_manifest
from pdm_lite.camera_profiles import CAMERA_PROFILES, cameras_for_profile


SENSOR_FILES = {
    "rgb": ".jpg",
    "rgb_augmented": ".jpg",
    "depth": ".png",
    "depth_augmented": ".png",
    "semantics": ".png",
    "semantics_augmented": ".png",
    "lidar": ".laz",
    "bev_semantics": ".png",
    "bev_semantics_augmented": ".png",
    "boxes": ".json.gz",
}


def build_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "seed": args.seed,
        "carla": {
            "host": args.carla_host,
            "port": args.carla_port,
            "tm_port": args.traffic_manager_port,
            "connection_timeout_s": args.timeout,
            "pythonapi_root": str(args.pythonapi_root.expanduser().resolve()),
            "require_server": True,
        },
        "materialization": {
            "scenic": {
                "policy_execution": {
                    "enabled": True,
                    "expert_mode": True,
                    "camera_profile": args.camera_profile,
                    "max_steps": args.max_steps,
                    "max_programs": args.max_programs,
                    "seed": args.seed,
                },
            },
        },
    }


def index_native_demos(
    execution: dict[str, Any],
    dataset_dir: Path,
    demos_file: Path,
    *,
    camera_profile: str = "admlp",
) -> dict[str, Any]:
    """Index collision-free, uncontaminated contiguous expert windows."""
    camera_ids = cameras_for_profile(camera_profile)
    if execution.get("status") != "completed":
        raise RuntimeError(f"Scenic expert execution did not complete: {execution.get('status')}")
    programs = execution.get("programs", [])
    if not programs or any(row.get("status") != "completed" for row in programs):
        raise RuntimeError("Scenic expert collection requires every program to complete; inspect native results")

    records: list[dict[str, Any]] = []
    dropped_collision: list[str] = []
    retained_scenarios: list[str] = []
    for index, program in enumerate(programs):
        scenario_id = program["scenario_id"]
        if program.get("collisions", 0):
            dropped_collision.append(scenario_id)
            continue
        route_dir = dataset_dir / f"FDSET_SCENIC_Rep0_{index}"
        measurement_files = sorted((route_dir / "measurements").glob("*.json.gz"))
        if not measurement_files:
            raise RuntimeError(f"No PDM-Lite measurements for Scenic scenario {scenario_id}")
        frames: list[tuple[float, dict[str, Any], Path, str]] = []
        for measurement_file in measurement_files:
            frame = measurement_file.name[:-len(".json.gz")]
            with gzip.open(measurement_file, "rt", encoding="utf-8") as handle:
                measurement = json.load(handle)
            if any(key not in measurement for key in ("steer", "throttle", "brake")):
                raise RuntimeError(f"PDM-Lite control missing from {measurement_file}")
            if scenario_id == "fdset_green_light_start" and (
                measurement.get("stop_sign_hazard") or measurement.get("light_hazard")
            ):
                continue
            timestamp = measurement.get("timestamp_s")
            time_s = float(timestamp) if timestamp is not None else float(frame) * 0.5
            frames.append((time_s, measurement, measurement_file, frame))

        windows: list[list[tuple[float, dict[str, Any], Path, str]]] = []
        current: list[tuple[float, dict[str, Any], Path, str]] = []
        previous: float | None = None
        for item in frames:
            time_s = item[0]
            if previous is not None and abs(time_s - previous - 0.5) > 1e-6:
                if len(current) >= 11:
                    windows.append(current)
                current = []
            current.append(item)
            previous = time_s
        if len(current) >= 11:
            windows.append(current)
        if not windows:
            continue
        retained_scenarios.append(scenario_id)
        for window in windows:
            for _, measurement, measurement_file, frame in window:
                sensor_files = {
                    name: route_dir / name / f"{frame}{suffix}"
                    for name, suffix in SENSOR_FILES.items()
                }
                sensor_files.update({name: route_dir / name / f"{frame}.jpg" for name in camera_ids})
                missing = [str(path) for path in sensor_files.values() if not path.is_file()]
                if missing:
                    raise RuntimeError(
                        f"Incomplete PDM-Lite sensor frame {scenario_id}:{frame}: {missing}"
                    )
                records.append({
                    "expert": "PDM-Lite",
                    "expert_backend": "pdm_lite_native_scenic",
                    "scenario_id": scenario_id,
                    "camera_profile": camera_profile,
                    "scenic_program": program["program_path"],
                    "frame": frame,
                    "dataset_dir": str(dataset_dir.resolve()),
                    "route_data_dir": str(route_dir.resolve()),
                    "measurement_file": str(measurement_file.resolve()),
                    "sensor_files": {key: str(value.resolve()) for key, value in sensor_files.items()},
                    "control": {key: measurement[key] for key in ("steer", "throttle", "brake")},
                    "measurement": measurement,
                })
    demos_file.parent.mkdir(parents=True, exist_ok=True)
    with demos_file.open("w", encoding="utf-8") as output:
        for record in records:
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
    return {
        "protocol": "fdset_native_scenic_pdm_lite_v1",
        "expert": "PDM-Lite",
        "camera_profile": camera_profile,
        "synthetic": False,
        "measured": True,
        "dataset_dir": str(dataset_dir.resolve()),
        "demos_file": str(demos_file.resolve()),
        "num_scenarios": len(programs),
        "num_measurements": len(records),
        "scenarios": [row["scenario_id"] for row in programs],
        "retained_scenarios": retained_scenarios,
        "dropped_collision_scenarios": dropped_collision,
        "num_programs_dropped_collision": len(dropped_collision),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True, help="Materialized native Scenic manifest")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--demos-file", type=Path, required=True)
    parser.add_argument("--carla-host", default="127.0.0.1")
    parser.add_argument("--carla-port", type=int, default=3000)
    parser.add_argument("--traffic-manager-port", type=int, default=3050)
    parser.add_argument("--pythonapi-root", type=Path, default=Path(__file__).resolve().parents[2] / "carla")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--max-steps", type=int, default=600)
    parser.add_argument("--max-programs", type=int, default=None, help="Limit a single-scene smoke run")
    parser.add_argument("--camera-profile", choices=tuple(CAMERA_PROFILES), default="admlp")
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.max_steps < 1 or args.timeout <= 0 or (args.max_programs is not None and args.max_programs < 1):
        parser.error("--max-steps, --timeout and --max-programs must be positive")
    manifest = args.manifest.expanduser().resolve()
    load_native_execution_manifest(manifest)
    output_dir = args.output_dir.expanduser().resolve()
    demos_file = args.demos_file.expanduser().resolve()
    cfg = build_config(args)
    execution = run_native_policy_manifest(manifest, cfg, output_dir, dry_run=args.dry_run)
    if args.dry_run:
        print(json.dumps({"synthetic": True, "measured": False, "plan": execution}, ensure_ascii=False, indent=2))
        return
    dataset_dir = output_dir / "pdm_lite_dataset"
    result = index_native_demos(execution, dataset_dir, demos_file, camera_profile=args.camera_profile)
    (output_dir / "pdm_lite_manifest.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
