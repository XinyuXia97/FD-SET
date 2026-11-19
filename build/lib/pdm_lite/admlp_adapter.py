#!/usr/bin/env python3
"""Convert measured PDM-Lite demonstrations and fine-tune Zoo ADMLP."""
from __future__ import annotations

import argparse
import importlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np

SAMPLE_PERIOD_S = 0.5
HISTORY_POINTS = 5
FUTURE_POINTS = 6
WINDOW_POINTS = HISTORY_POINTS + FUTURE_POINTS
REQUIRED_MEASUREMENT = ("pos_global", "theta", "command", "speed", "acceleration", "timestamp_s")


def _number(value: Any, field: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"measurement field {field!r} must be numeric") from exc
    if not math.isfinite(result):
        raise ValueError(f"measurement field {field!r} must be finite")
    return result


def _measurement(record: dict[str, Any]) -> dict[str, Any]:
    payload = record.get("measurement")
    if not isinstance(payload, dict):
        raise ValueError("expert demo record has no measurement object")
    missing = [key for key in REQUIRED_MEASUREMENT if key not in payload]
    if missing:
        raise ValueError(f"measurement is missing required fields: {', '.join(missing)}")
    position = payload["pos_global"]
    acceleration = payload["acceleration"]
    if not isinstance(position, (list, tuple)) or len(position) < 2:
        raise ValueError("measurement field 'pos_global' must contain x and y")
    if not isinstance(acceleration, (list, tuple)) or len(acceleration) != 3:
        raise ValueError("measurement field 'acceleration' must be the measured [x, y, z] vector")
    command_value = _number(payload["command"], "command")
    command = int(command_value)
    if command_value != command:
        raise ValueError("measurement field 'command' must be an integer")
    if command not in (-1, 1, 2, 3, 4, 5, 6):
        raise ValueError(f"unsupported Zoo ADMLP command value {command}")
    return {
        "x": _number(position[0], "pos_global[0]"),
        "y": _number(position[1], "pos_global[1]"),
        "theta": _number(payload["theta"], "theta"),
        "command": command,
        "speed": _number(payload["speed"], "speed"),
        "speed_acc": [_number(value, "acceleration") for value in acceleration],
        "timestamp_s": _number(payload["timestamp_s"], "timestamp_s"),
    }


def _load_records(demos_file: Path) -> dict[str, list[dict[str, Any]]]:
    scenarios: dict[str, list[dict[str, Any]]] = {}
    with demos_file.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {demos_file}:{line_number}: {exc}") from exc
            if record.get("expert_backend") != "pdm_lite_native_scenic":
                raise ValueError(
                    f"unverified native Scenic PDM-Lite expert backend at {demos_file}:{line_number}"
                )
            scenario = record.get("scenario_id")
            if not isinstance(scenario, str) or not scenario:
                raise ValueError(f"missing scenario_id at {demos_file}:{line_number}")
            item = _measurement(record)
            item["frame"] = record.get("frame")
            item["scenario_id"] = scenario
            episode = record.get("training_episode_id") or record.get("dataset_dir") or scenario
            if not isinstance(episode, str):
                raise ValueError(f"invalid training episode at {demos_file}:{line_number}")
            scenarios.setdefault(episode, []).append(item)
    if not scenarios:
        raise ValueError(f"no expert demo measurements in {demos_file}")
    return scenarios


def convert(demos_file: Path, output_file: Path, manifest_file: Path) -> dict[str, Any]:
    """Convert only contiguous, fully measured 5-history/6-future windows."""
    if output_file.suffix != ".npy":
        raise ValueError(f"Zoo ADMLP dataset output must end in .npy: {output_file}")
    grouped = _load_records(demos_file)
    samples: dict[str, list[Any]] = {key: [] for key in (
        "input_x", "input_y", "input_theta", "input_command", "input_speed", "input_speed_acc",
        "future_x", "future_y", "future_theta",
    )}
    provenance: list[dict[str, Any]] = []
    for scenario_id, rows in sorted(grouped.items()):
        rows.sort(key=lambda row: row["timestamp_s"])
        times = [row["timestamp_s"] for row in rows]
        if len(set(times)) != len(times):
            raise ValueError(f"scenario {scenario_id!r} contains duplicate timestamps")
        for previous, current in zip(times, times[1:]):
            delta = current - previous
            if not math.isclose(delta, SAMPLE_PERIOD_S, rel_tol=0.0, abs_tol=0.02):
                raise ValueError(
                    f"scenario {scenario_id!r} has non-contiguous measurements: "
                    f"{previous:.6f}s -> {current:.6f}s ({delta:.6f}s; expected 0.5s)"
                )
        scenario_count = 0
        for start in range(max(0, len(rows) - WINDOW_POINTS + 1)):
            window = rows[start:start + WINDOW_POINTS]
            hist, future = window[:HISTORY_POINTS], window[HISTORY_POINTS:]
            for key, field in (("input_x", "x"), ("input_y", "y"), ("input_theta", "theta"),
                               ("input_command", "command"), ("input_speed", "speed"),
                               ("input_speed_acc", "speed_acc"), ("future_x", "x"),
                               ("future_y", "y"), ("future_theta", "theta")):
                source = hist if key.startswith("input_") else future
                samples[key].append([row[field] for row in source])
            scenario_count += 1
        if scenario_count == 0:
            raise ValueError(
                f"scenario {scenario_id!r} has {len(rows)} valid measurements; "
                f"at least {WINDOW_POINTS} contiguous 0.5s points are required"
            )
        provenance.append({
            "scenario_id": rows[0]["scenario_id"],
            "training_episode_id": scenario_id,
            "measurement_count": len(rows),
            "window_count": scenario_count,
            "first_timestamp_s": times[0],
            "last_timestamp_s": times[-1],
        })
    if not samples["input_x"]:
        raise ValueError("no valid training windows were produced")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_file, samples, allow_pickle=True)
    manifest_file.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "format": "zoo_admlp_measured_pdm_lite_v1",
        "source_demos": str(demos_file.resolve()),
        "output_npy": str(output_file.resolve()),
        "sample_period_s": SAMPLE_PERIOD_S,
        "history_points": HISTORY_POINTS,
        "future_points": FUTURE_POINTS,
        "sample_count": len(samples["input_x"]),
        "scenarios": provenance,
    }
    manifest_file.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest


def _import_zoo(zoo_root: Path):
    root = zoo_root.expanduser().resolve()
    admlp_dir = root / "ADMLP"
    if not (admlp_dir / "model.py").is_file() or not (admlp_dir / "data.py").is_file() or not (admlp_dir / "config.py").is_file():
        raise RuntimeError(f"Zoo root {root} must contain ADMLP/model.py, data.py, and config.py")
    sys.path.insert(0, str(root))
    try:
        data_module = importlib.import_module("ADMLP.data")
        model_module = importlib.import_module("ADMLP.model")
        config_module = importlib.import_module("ADMLP.config")
    except Exception as exc:
        raise RuntimeError(f"could not import official Zoo ADMLP modules from {root}: {exc}") from exc
    if not hasattr(data_module, "CARLA_Data") or not hasattr(model_module, "ADMLP") or not hasattr(config_module, "GlobalConfig"):
        raise RuntimeError("Zoo ADMLP lacks required official CARLA_Data/ADMLP/GlobalConfig classes")
    return data_module.CARLA_Data, model_module.ADMLP, config_module.GlobalConfig


def _state_dict(checkpoint: Any) -> dict[str, Any]:
    if isinstance(checkpoint, dict) and isinstance(checkpoint.get("state_dict"), dict):
        state = checkpoint["state_dict"]
    elif isinstance(checkpoint, dict):
        state = checkpoint
    else:
        raise ValueError("baseline checkpoint must be a state_dict or contain a state_dict mapping")
    normalized = {key[6:]: value for key, value in state.items() if key.startswith("model.")}
    if not normalized:
        normalized = state
    return normalized


def train(zoo_root: Path, data_file: Path, baseline: Path, output: Path, *, epochs: int, batch_size: int,
          learning_rate: float, device: str = "cpu") -> dict[str, Any]:
    if epochs < 1 or batch_size < 1 or not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("epochs, batch-size, and learning-rate must be positive")
    try:
        import torch
        from torch.utils.data import DataLoader
        import torch.nn.functional as F
    except ImportError as exc:
        raise RuntimeError("ADMLP training requires PyTorch") from exc
    try:
        carla_data, model_class, config_class = _import_zoo(zoo_root)
        dataset = carla_data(str(data_file))
    except Exception as exc:
        if isinstance(exc, RuntimeError):
            raise
        raise RuntimeError(f"could not load official Zoo ADMLP data/model dependencies: {exc}") from exc
    count = len(dataset)
    if count < 1:
        raise ValueError("ADMLP training data contains no complete examples")
    try:
        target_device = torch.device(device)
        if target_device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(f"requested device {device!r}, but CUDA is unavailable")
        model = model_class(config_class()).to(target_device)
    except Exception as exc:
        if isinstance(exc, RuntimeError) and "CUDA is unavailable" in str(exc):
            raise
        raise RuntimeError(f"could not construct official Zoo ADMLP: {exc}") from exc
    try:
        raw_checkpoint = torch.load(baseline, map_location=target_device, weights_only=False)
        model.load_state_dict(_state_dict(raw_checkpoint), strict=True)
    except Exception as exc:
        raise RuntimeError(f"baseline checkpoint is incompatible with official Zoo ADMLP: {exc}") from exc
    initial = {key: value.detach().clone() for key, value in model.state_dict().items()}
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-2)
    model.train()
    updates = 0
    for _ in range(epochs):
        for batch in loader:
            batch = {key: value.to(target_device) if hasattr(value, "to") else value for key, value in batch.items()}
            prediction = model({"input": batch["input"]})
            target = batch["waypoints"].float()
            theta = batch["thetas"].float()
            loss = F.l1_loss(prediction[:, :, 0], target[:, :, 0]) + F.l1_loss(
                prediction[:, :, 1], target[:, :, 1]
            ) + F.l1_loss(prediction[:, :, 2], theta)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            updates += 1
    if updates == 0 or all(torch.equal(initial[key], value) for key, value in model.state_dict().items()):
        raise RuntimeError("ADMLP training performed no weight update")
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": {f"model.{key}": value.detach().cpu() for key, value in model.state_dict().items()}}, output)
    return {"checkpoint": str(output.resolve()), "examples": count, "epochs": epochs, "updates": updates,
            "model": "official Zoo ADMLP", "device": str(target_device)}


def _require_artifacts_output(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    artifacts = (Path(__file__).resolve().parents[1] / "artifacts").resolve()
    try:
        resolved.relative_to(artifacts)
    except ValueError as exc:
        raise ValueError(f"output must be inside project artifacts/: {resolved}") from exc
    return resolved


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    convert_parser = commands.add_parser("convert", help="convert measured PDM-Lite expert demos")
    convert_parser.add_argument("--demos", type=Path, required=True)
    convert_parser.add_argument("--output", type=Path, required=True, help="official Zoo .npy dataset")
    convert_parser.add_argument("--manifest", type=Path, required=True, help="separate scenario provenance JSON")
    train_parser = commands.add_parser("train", help="fine-tune official Zoo ADMLP from a baseline")
    adapt_parser = commands.add_parser("adapt", help="convert measured demos then fine-tune official Zoo ADMLP")
    train_parser.add_argument("--data", type=Path, required=True)
    adapt_parser.add_argument("--demos", type=Path, required=True)
    for model_parser in (train_parser, adapt_parser):
        model_parser.add_argument("--zoo-root", type=Path, required=True)
        model_parser.add_argument("--baseline", type=Path, required=True)
        model_parser.add_argument("--output", type=Path, required=True, help="artifacts checkpoint output")
        model_parser.add_argument("--epochs", type=int, required=True)
        model_parser.add_argument("--batch-size", type=int, required=True)
        model_parser.add_argument("--learning-rate", type=float, required=True)
        model_parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    try:
        if args.action == "convert":
            result = convert(args.demos, _require_artifacts_output(args.output),
                             _require_artifacts_output(args.manifest))
        else:
            output = _require_artifacts_output(args.output)
            if args.action == "adapt":
                dataset = output.parent / "admlp_measured.npy"
                convert(args.demos, dataset, output.parent / "admlp_scenario_manifest.json")
            else:
                dataset = args.data
            result = train(args.zoo_root, dataset, args.baseline, output,
                           epochs=args.epochs, batch_size=args.batch_size,
                           learning_rate=args.learning_rate, device=args.device)
    except (ValueError, RuntimeError, OSError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
