#!/usr/bin/env python3
"""Validate PDM-Lite demonstrations and fine-tune official Zoo planning heads.

Only measured ego trajectories supervise planning: the stock detection,
occupancy, tracking and mapping heads remain frozen and require no fake labels.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np

SAMPLE_PERIOD_S = 0.5
FUTURE_STEPS = 6
CAMERAS = (
    "CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT",
    "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT",
)
REQUIRED_EGO = ("pos_global", "theta", "speed", "timestamp_s", "command")


def _read_demos(path: Path) -> dict[str, list[dict[str, Any]]]:
    scenarios: dict[str, list[dict[str, Any]]] = {}
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON in {path}:{line_number}: {exc}") from exc
            if row.get("expert_backend") != "pdm_lite_native_scenic":
                raise ValueError(f"unverified PDM-Lite expert backend in {path}:{line_number}")
            scenario = row.get("scenario_id")
            if not isinstance(scenario, str) or not scenario:
                raise ValueError(f"missing scenario_id in {path}:{line_number}")
            measurement = row.get("measurement")
            if not isinstance(measurement, dict):
                raise ValueError(f"missing measurement object in {path}:{line_number}")
            missing = [key for key in REQUIRED_EGO if key not in measurement]
            if missing:
                raise ValueError(f"measurement in {path}:{line_number} lacks {', '.join(missing)}")
            position = measurement["pos_global"]
            if not isinstance(position, (list, tuple)) or len(position) != 2:
                raise ValueError(f"measurement.pos_global in {path}:{line_number} must be [x, y]")
            for key in ("theta", "speed", "timestamp_s"):
                try:
                    value = float(measurement[key])
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"measurement.{key} in {path}:{line_number} must be numeric") from exc
                if not math.isfinite(value):
                    raise ValueError(f"measurement.{key} in {path}:{line_number} must be finite")
            for index, value in enumerate(position):
                if not math.isfinite(float(value)):
                    raise ValueError(f"measurement.pos_global[{index}] in {path}:{line_number} must be finite")
            episode = row.get("training_episode_id") or row.get("dataset_dir") or scenario
            if not isinstance(episode, str):
                raise ValueError(f"invalid training episode in {path}:{line_number}")
            scenarios.setdefault(episode, []).append(row)
    if not scenarios:
        raise ValueError(f"no PDM-Lite measurements found in {path}")
    return scenarios


def _camera_paths(row: dict[str, Any]) -> dict[str, Path]:
    sensors = row.get("sensor_files")
    if not isinstance(sensors, dict):
        raise ValueError(f"scenario {row['scenario_id']!r} frame {row.get('frame')!r}: missing sensor_files")
    missing = [name for name in CAMERAS if name not in sensors]
    if missing:
        raise ValueError(
            f"scenario {row['scenario_id']!r} frame {row.get('frame')!r}: missing synchronized Zoo camera keys: "
            + ", ".join(missing)
        )
    paths = {name: Path(sensors[name]).expanduser() for name in CAMERAS}
    absent = [f"{name}={path}" for name, path in paths.items() if not path.is_file()]
    if absent:
        raise ValueError(f"missing real camera image files: {', '.join(absent)}")
    if len({path.resolve() for path in paths.values()}) != len(CAMERAS):
        raise ValueError("UniAD/VAD camera sensor_files must reference six distinct real camera frames")
    return paths




def export_trajectories(demos: Path, output: Path, manifest_path: Path) -> dict[str, Any]:
    """Export contiguous measured world-frame positions without relabeling them."""
    if output.suffix != ".npz":
        raise ValueError(f"trajectory intermediate output must use .npz: {output}")
    grouped = _read_demos(demos)
    samples: list[dict[str, Any]] = []
    scenarios_summary = []
    for scenario_id, rows in sorted(grouped.items()):
        rows.sort(key=lambda row: float(row["measurement"]["timestamp_s"]))
        times = [float(row["measurement"]["timestamp_s"]) for row in rows]
        if len(set(times)) != len(times):
            raise ValueError(f"scenario {scenario_id!r} has duplicate measurement timestamps")
        for previous, current in zip(times, times[1:]):
            delta = current - previous
            if not math.isclose(delta, SAMPLE_PERIOD_S, rel_tol=0.0, abs_tol=0.02):
                raise ValueError(
                    f"scenario {scenario_id!r} measurements are not contiguous: {previous:.3f}s -> {current:.3f}s "
                    f"(expected {SAMPLE_PERIOD_S}s)"
                )
        count = 0
        for start in range(len(rows) - FUTURE_STEPS):
            source = rows[start]
            future = rows[start + 1:start + FUTURE_STEPS + 1]
            current_state = source["measurement"]
            future_xy = np.asarray([row["measurement"]["pos_global"] for row in future], dtype=np.float64)
            if future_xy.shape != (FUTURE_STEPS, 2) or not np.isfinite(future_xy).all():
                raise ValueError(f"scenario {scenario_id!r} has invalid measured future positions")
            cameras = _camera_paths(source)
            samples.append({
                "scenario_id": source["scenario_id"],
                "frame": source.get("frame"),
                "timestamp_s": float(current_state["timestamp_s"]),
                "position_world_xy": np.asarray(current_state["pos_global"], dtype=np.float64),
                "theta": float(current_state["theta"]),
                "speed": float(current_state["speed"]),
                "command": current_state["command"],
                "future_position_world_xy": future_xy,
                "camera_files": np.asarray([str(cameras[name].resolve()) for name in CAMERAS]),
            })
            count += 1
        if count == 0:
            raise ValueError(f"scenario {scenario_id!r} requires at least {FUTURE_STEPS + 1} contiguous measurements")
        scenarios_summary.append({"scenario_id": scenario_id, "measurement_count": len(rows), "window_count": count})
    output.parent.mkdir(parents=True, exist_ok=True)
    # Object arrays retain scenario/frame provenance and are never presented as a Zoo dataset.
    np.savez_compressed(
        output,
        scenario_id=np.asarray([sample["scenario_id"] for sample in samples]),
        frame=np.asarray([str(sample["frame"]) for sample in samples]),
        timestamp_s=np.asarray([sample["timestamp_s"] for sample in samples], dtype=np.float64),
        position_world_xy=np.stack([sample["position_world_xy"] for sample in samples]),
        theta=np.asarray([sample["theta"] for sample in samples], dtype=np.float64),
        speed=np.asarray([sample["speed"] for sample in samples], dtype=np.float32),
        command=np.asarray([sample["command"] for sample in samples]),
        future_position_world_xy=np.stack([sample["future_position_world_xy"] for sample in samples]),
        camera_files=np.stack([sample["camera_files"] for sample in samples]),
    )
    manifest = {
        "format": "measured_pdm_lite_bev_world_trajectory_intermediate_v1",
        "not_zoo_training_dataset": True,
        "source_demos": str(demos.resolve()),
        "output_npz": str(output.resolve()),
        "camera_order": list(CAMERAS),
        "sample_period_s": SAMPLE_PERIOD_S,
        "future_steps": FUTURE_STEPS,
        "sample_count": len(samples),
        "scenarios": scenarios_summary,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest


def train(family: str, demos: Path, zoo_root: Path, baseline: Path, output: Path, *, epochs: int,
          batch_size: int, learning_rate: float, device: str = "cpu") -> dict[str, Any]:
    """Adapt the original Zoo UniAD/VAD checkpoint's differentiable planner."""
    if family not in ("uniad", "vad"):
        raise ValueError("family must be 'uniad' or 'vad'")
    if epochs < 1 or batch_size < 1 or not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("epochs, batch-size, and learning-rate must be positive")
    if not zoo_root.is_dir():
        raise ValueError(f"Zoo checkout does not exist: {zoo_root}")
    if not baseline.is_file():
        raise ValueError(f"baseline checkpoint does not exist: {baseline}")
    if output.resolve() == baseline.resolve():
        raise ValueError("output checkpoint must not overwrite the baseline")
    from pdm_lite.zoo_planner_training import train as train_zoo_planner
    return train_zoo_planner(family, demos, zoo_root, baseline, output, epochs=epochs,
                             batch_size=batch_size, learning_rate=learning_rate, device=device)


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)
    convert = subparsers.add_parser("convert", help="export provenance-preserving measured trajectories")
    convert.add_argument("--demos", type=Path, required=True)
    convert.add_argument("--output", type=Path, required=True)
    convert.add_argument("--manifest", type=Path, required=True)
    adapt = subparsers.add_parser("adapt", help="fine-tune official Zoo planning model")
    adapt.add_argument("--family", choices=("uniad", "vad"), required=True)
    adapt.add_argument("--demos", type=Path, required=True)
    adapt.add_argument("--zoo-root", type=Path, required=True)
    adapt.add_argument("--baseline", type=Path, required=True)
    adapt.add_argument("--output", type=Path, required=True)
    adapt.add_argument("--epochs", type=int, default=1)
    adapt.add_argument("--batch-size", type=int, default=1)
    adapt.add_argument("--learning-rate", type=float, default=1e-5)
    adapt.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    if args.action == "convert":
        result = export_trajectories(args.demos, args.output, args.manifest)
    else:
        result = train(args.family, args.demos, args.zoo_root, args.baseline, args.output,
                       epochs=args.epochs, batch_size=args.batch_size,
                       learning_rate=args.learning_rate, device=args.device)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
