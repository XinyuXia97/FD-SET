#!/usr/bin/env python3
"""Adapt the official Bench2DriveZoo TCP model from measured PDM-Lite demos."""
from __future__ import annotations

import argparse
import importlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable

SAMPLE_PERIOD_S = 0.5
FUTURE_POINTS = 4
CAMERA_KEYS = ("CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT")
CONTROL_KEYS = ("steer", "throttle", "brake")
REQUIRED_MEASUREMENT = (
    "pos_global", "theta", "speed", "command", "timestamp_s", "target_point",
    *CONTROL_KEYS,
)


def _number(value: Any, field: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"measurement field {field!r} must be numeric") from exc
    if not math.isfinite(result):
        raise ValueError(f"measurement field {field!r} must be finite")
    return result


def _point(value: Any, field: str) -> tuple[float, float]:
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        raise ValueError(f"measurement field {field!r} must contain x and y")
    return (_number(value[0], f"{field}[0]"), _number(value[1], f"{field}[1]"))


def _record(record: dict[str, Any], line_number: int) -> dict[str, Any]:
    if record.get("expert_backend") != "pdm_lite_native_scenic":
        raise ValueError(f"unverified native Scenic PDM-Lite expert backend at line {line_number}")
    scenario = record.get("scenario_id")
    if not isinstance(scenario, str) or not scenario:
        raise ValueError(f"missing scenario_id at line {line_number}")
    measurement = record.get("measurement")
    if not isinstance(measurement, dict):
        raise ValueError(f"expert demo line {line_number} has no measurement object")
    missing = [field for field in REQUIRED_MEASUREMENT if field not in measurement]
    if missing:
        raise ValueError(f"measurement at line {line_number} is missing required fields: {', '.join(missing)}")
    files = record.get("sensor_files")
    if not isinstance(files, dict):
        raise ValueError(f"expert demo line {line_number} has no sensor_files object")
    missing_cameras = [key for key in CAMERA_KEYS if key not in files]
    if missing_cameras:
        raise ValueError(f"TCP requires genuine synchronized camera sensor_files: {', '.join(missing_cameras)}")
    camera_paths = {}
    for key in CAMERA_KEYS:
        path = Path(files[key]).expanduser()
        if not path.is_file():
            raise ValueError(f"missing TCP camera image {key}: {path}")
        camera_paths[key] = path
    if len(set(camera_paths.values())) != len(CAMERA_KEYS):
        raise ValueError(f"TCP camera sensor_files at line {line_number} must reference three distinct views")
    pos = _point(measurement["pos_global"], "pos_global")
    target = _point(measurement["target_point"], "target_point")
    command_value = _number(measurement["command"], "command")
    command = int(command_value)
    if command != command_value or command not in (-1, 1, 2, 3, 4, 5, 6):
        raise ValueError("measurement field 'command' must be Zoo command -1 or an integer in 1..6")
    controls = tuple(_number(measurement[key], key) for key in CONTROL_KEYS)
    if not -1 <= controls[0] <= 1 or not 0 <= controls[1] <= 1 or not 0 <= controls[2] <= 1:
        raise ValueError(f"measured controls at line {line_number} are outside CARLA steer/throttle/brake ranges")
    return {
        "scenario_id": scenario,
        "training_episode_id": record.get("training_episode_id") or record.get("dataset_dir") or scenario,
        "timestamp": _number(measurement["timestamp_s"], "timestamp_s"),
        "x": pos[0], "y": pos[1], "theta": _number(measurement["theta"], "theta"),
        "speed": _number(measurement["speed"], "speed"),
        "command": 4 if command < 0 else command,
        "target_point": target,
        "controls": controls,
        "cameras": camera_paths,
    }


def _load_samples(demos_file: Path) -> list[dict[str, Any]]:
    scenarios: dict[str, list[dict[str, Any]]] = {}
    with demos_file.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {demos_file}:{line_number}: {exc}") from exc
            item = _record(record, line_number)
            scenarios.setdefault(item["training_episode_id"], []).append(item)
    if not scenarios:
        raise ValueError(f"no expert demo measurements in {demos_file}")

    samples = []
    for scenario_id, rows in sorted(scenarios.items()):
        rows.sort(key=lambda item: item["timestamp"])
        times = [row["timestamp"] for row in rows]
        if len(set(times)) != len(times):
            raise ValueError(f"scenario {scenario_id!r} contains duplicate timestamps")
        for previous, current in zip(times, times[1:]):
            delta = current - previous
            if not math.isclose(delta, SAMPLE_PERIOD_S, rel_tol=0.0, abs_tol=0.02):
                raise ValueError(f"scenario {scenario_id!r} has non-contiguous measurements: {delta:.6f}s; expected 0.5s")
        if len(rows) <= FUTURE_POINTS:
            raise ValueError(f"scenario {scenario_id!r} requires at least {FUTURE_POINTS + 1} contiguous measured frames")
        for index in range(len(rows) - FUTURE_POINTS):
            row = rows[index]
            # PDM preprocess_compass already subtracts pi/2 from raw IMU heading.
            theta = row["theta"]
            c, s = math.cos(theta), math.sin(theta)
            waypoints = []
            for future in rows[index + 1:index + FUTURE_POINTS + 1]:
                dx, dy = future["x"] - row["x"], future["y"] - row["y"]
                waypoints.append((c * dx + s * dy, -s * dx + c * dy))
            samples.append({**row, "waypoints": waypoints})
    if not samples:
        raise ValueError("no complete four-step TCP trajectory examples were produced")
    return samples


def _import_zoo(zoo_root: Path):
    root = zoo_root.expanduser().resolve()
    tcp_dir = root / "TCP"
    if not all((tcp_dir / filename).is_file() for filename in ("model.py", "data.py", "config.py")):
        raise RuntimeError(f"Zoo root {root} must contain TCP/model.py, data.py, and config.py")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    try:
        model_module = importlib.import_module("TCP.model")
        config_module = importlib.import_module("TCP.config")
    except Exception as exc:
        raise RuntimeError(f"could not import official Zoo TCP model from {root}: {exc}") from exc
    if not hasattr(model_module, "TCP") or not hasattr(config_module, "GlobalConfig"):
        raise RuntimeError("Zoo TCP lacks required official TCP/GlobalConfig classes")
    return model_module.TCP, model_module.Discrete_Actions_DICT, config_module.GlobalConfig


def _state_dict(checkpoint: Any) -> dict[str, Any]:
    if isinstance(checkpoint, dict) and isinstance(checkpoint.get("state_dict"), dict):
        state = checkpoint["state_dict"]
    elif isinstance(checkpoint, dict):
        state = checkpoint
    else:
        raise ValueError("baseline checkpoint must be a state_dict or contain a state_dict mapping")
    normalized = {key[6:]: value for key, value in state.items() if key.startswith("model.")}
    return normalized or state


def _image_array(path: Path):
    import numpy as np
    from PIL import Image
    with Image.open(path) as image:
        return np.array(image.convert("RGB"))


def _prepare_image(sample: dict[str, Any], *, torch):
    import numpy as np
    images = [_image_array(sample["cameras"][key]) for key in CAMERA_KEYS]
    front, left, right = images
    if front.shape[:2] != (900, 1600) or left.shape[:2] != front.shape[:2] or right.shape[:2] != front.shape[:2]:
        raise ValueError("TCP camera frames must all be synchronized 1600x900 RGB images")
    composite = np.concatenate((left[:, :1400], front[:, 200:1400], right[:, 200:]), axis=1)
    rgb = torch.from_numpy(composite).permute(2, 0, 1).unsqueeze(0).float()
    rgb = torch.nn.functional.interpolate(rgb, size=(256, 900), mode="bilinear", align_corners=False)
    rgb = rgb.squeeze(0).permute(1, 2, 0).byte().numpy()
    tensor = torch.from_numpy(rgb.copy()).permute(2, 0, 1).float() / 255.0
    mean = tensor.new_tensor((0.485, 0.456, 0.406))[:, None, None]
    std = tensor.new_tensor((0.229, 0.224, 0.225))[:, None, None]
    return (tensor - mean) / std


def train(zoo_root: Path, demos_file: Path, baseline: Path, output: Path, *, epochs: int,
          batch_size: int, learning_rate: float, device: str = "cpu") -> dict[str, Any]:
    if epochs < 1 or batch_size < 1 or not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("epochs, batch-size, and learning-rate must be positive")
    try:
        import torch
        import torch.nn.functional as F
        from torch.utils.data import DataLoader, Dataset
    except ImportError as exc:
        raise RuntimeError("TCP adaptation requires PyTorch and Pillow") from exc
    samples = _load_samples(demos_file)
    model_class, action_dict, config_class = _import_zoo(zoo_root)
    try:
        target_device = torch.device(device)
        if target_device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(f"requested device {device!r}, but CUDA is unavailable")
        # The Zoo constructor asks Torch Hub for ImageNet weights even though
        # the complete baseline is loaded immediately below. Avoid that
        # unnecessary network access; restore the original constructor.
        constructor_globals = model_class.__init__.__globals__
        original_resnet34 = constructor_globals.get("resnet34")
        if original_resnet34 is not None:
            def local_resnet34(*args, **kwargs):
                kwargs["pretrained"] = False
                return original_resnet34(*args, **kwargs)
            constructor_globals["resnet34"] = local_resnet34
        try:
            model = model_class(config_class()).to(target_device)
        finally:
            if original_resnet34 is not None:
                constructor_globals["resnet34"] = original_resnet34
        raw = torch.load(baseline, map_location=target_device, weights_only=False)
        model.load_state_dict(_state_dict(raw), strict=True)
    except Exception as exc:
        if isinstance(exc, RuntimeError) and "CUDA is unavailable" in str(exc):
            raise
        raise RuntimeError(f"could not construct official Zoo TCP or load baseline checkpoint: {exc}") from exc

    action_values = torch.tensor([action_dict[index][:3] for index in range(len(action_dict))], dtype=torch.float32)
    action_reverse = torch.tensor([bool(action_dict[index][3]) for index in range(len(action_dict))])
    class DemoDataset(Dataset):
        def __len__(self):
            return len(samples)

        def __getitem__(self, index):
            sample = samples[index]
            command = sample["command"] - 1
            one_hot = torch.nn.functional.one_hot(torch.tensor(command), num_classes=6).float()
            target = torch.tensor(sample["target_point"], dtype=torch.float32)
            state = torch.cat((torch.tensor([sample["speed"] / 12.0]), target, one_hot))
            control = torch.tensor((sample["controls"][1], sample["controls"][0], sample["controls"][2]),
                                   dtype=torch.float32)
            # TCP's genuine 39-way action head cannot represent continuous PDM controls exactly.
            # A temperature-scaled soft target preserves the real measured controls without
            # rejecting them or pretending they are exact Zoo action IDs.
            distances = (action_values - control).abs().mean(dim=1)
            distances = distances + action_reverse.to(dtype=distances.dtype) * 10.0
            action_target = torch.softmax(-distances / 0.08, dim=0)
            return (_prepare_image(sample, torch=torch), state, target,
                    torch.tensor(sample["speed"] / 12.0), torch.tensor(sample["waypoints"], dtype=torch.float32),
                    action_target)

    dataset = DemoDataset()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    trainable_prefixes = ("decoder_traj.", "output_traj.", "speed_branch.", "policy_head.", "action_head.")
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.startswith(trainable_prefixes))
    trainable = {name: parameter for name, parameter in model.named_parameters() if parameter.requires_grad}
    optimizer = torch.optim.AdamW(trainable.values(), lr=learning_rate, weight_decay=1e-4)
    initial = {name: parameter.detach().clone() for name, parameter in trainable.items()}
    model.train()
    model.perception.eval()
    updates = 0
    total_loss = 0.0
    for _ in range(epochs):
        for image, state, target_point, speed_target, waypoints, action_target in loader:
            image = image.to(target_device)
            state = state.to(target_device)
            target_point = target_point.to(target_device)
            speed_target = speed_target.to(target_device).view(-1, 1)
            waypoints = waypoints.to(target_device)
            action_target = action_target.to(target_device)
            prediction = model(image, state, target_point)
            waypoint_loss = F.l1_loss(prediction["pred_wp"], waypoints)
            speed_loss = F.l1_loss(prediction["pred_speed"], speed_target)
            action_loss = -(action_target * F.log_softmax(prediction["action_index"], dim=1)).sum(dim=1).mean()
            loss = waypoint_loss + 0.05 * speed_loss + action_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            updates += 1
            total_loss += float(loss.detach())
    if updates == 0 or all(torch.equal(initial[name], parameter) for name, parameter in trainable.items()):
        raise RuntimeError("TCP adaptation performed no weight update")
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": {f"model.{key}": value.detach().cpu() for key, value in model.state_dict().items()}}, output)
    return {"checkpoint": str(output.resolve()), "examples": len(dataset), "epochs": epochs,
            "updates": updates, "mean_loss": total_loss / updates, "model": "official Zoo TCP",
            "supervision": ["measured_future_waypoints", "measured_speed", "measured_steer_throttle_brake_soft_action_target"],
            "device": str(target_device)}



def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    adapt = subparsers.add_parser("adapt", help="adapt a Zoo TCP checkpoint from measured PDM-Lite demos")
    adapt.add_argument("--demos", required=True, type=Path)
    adapt.add_argument("--zoo-root", required=True, type=Path)
    adapt.add_argument("--baseline", required=True, type=Path)
    adapt.add_argument("--output", required=True, type=Path)
    adapt.add_argument("--epochs", required=True, type=int)
    adapt.add_argument("--batch-size", required=True, type=int)
    adapt.add_argument("--learning-rate", required=True, type=float)
    adapt.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    result = train(args.zoo_root, args.demos, args.baseline, args.output.expanduser().resolve(), epochs=args.epochs,
                   batch_size=args.batch_size, learning_rate=args.learning_rate, device=args.device)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
