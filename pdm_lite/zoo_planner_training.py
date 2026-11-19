"""Planning-only imitation on the original Bench2DriveZoo UniAD/VAD models.

The frozen perception, tracking, map and occupancy heads run on measured six-view
CARLA imagery. Only the original planning head is updated; no object/map labels
are inferred from PDM's ego trajectory.
"""
from __future__ import annotations

import importlib
import math
from pathlib import Path
from typing import Any

import numpy as np

from pdm_lite.camera_profiles import ZOO_CAMERA_SPECS
from pdm_lite.zoo_runtime import prepare_zoo_runtime

CAMERAS = tuple(ZOO_CAMERA_SPECS)
LIDAR2EGO = np.array([[0., 1., 0., -.39], [-1., 0., 0., 0.],
                      [0., 0., 1., 1.84], [0., 0., 0., 1.]])


def camera_calibration() -> tuple[np.ndarray, np.ndarray]:
    """Compute the Zoo agent's lidar2cam/lidar2img from its actual CARLA sensors."""
    lidar2cam, lidar2img = [], []
    for name in CAMERAS:
        spec = ZOO_CAMERA_SPECS[name]
        yaw = math.radians(float(spec["yaw"]))
        c, s = math.cos(yaw), math.sin(yaw)
        cam = np.eye(4)
        cam[:3, :3] = [[c, -s, 0], [0, 0, -1], [s, c, 0]]
        delta = np.array([-.39 - float(spec["x"]), -float(spec["y"]), 1.84 - float(spec["z"])])
        cam[:3, 3] = cam[:3, :3] @ np.array([delta[1], delta[0], delta[2]])
        # The rotation above acts on lidar coordinates (ego y, ego x, ego z).
        # The camera offset must therefore be expressed in the same lidar axes.
        fx = float(spec["width"]) / (2 * math.tan(math.radians(float(spec["fov"])) / 2))
        intrinsics = np.eye(4)
        intrinsics[0, 0] = intrinsics[1, 1] = fx
        intrinsics[0, 2] = int(spec["width"]) / 2
        intrinsics[1, 2] = int(spec["height"]) / 2
        lidar2cam.append(cam)
        lidar2img.append(intrinsics @ cam)
    return np.stack(lidar2cam), np.stack(lidar2img)


def lidar2global(measurement: dict[str, Any]) -> np.ndarray:
    # PDM has already converted raw IMU compass into CARLA yaw.
    theta = -float(measurement["theta"])
    c, s = math.cos(theta), math.sin(theta)
    pose = np.eye(4)
    pose[:2, :2] = [[c, -s], [s, c]]
    pose[:2, 3] = [float(measurement["pos_global"][0]), -float(measurement["pos_global"][1])]
    return pose @ LIDAR2EGO

def local_future(source: dict[str, Any], future: list[dict[str, Any]]) -> np.ndarray:
    """World poses to the exact B2D dataset world2lidar_current @ lidar2world_future target."""
    world2current = np.linalg.inv(lidar2global(source))
    return np.stack([(world2current @ lidar2global(row))[:2, 3] for row in future]).astype(np.float32)


def _near_command(measurement: dict[str, Any]) -> int:
    # The original Zoo agents condition on the current, not the next, route command.
    command = int(measurement["command"])
    if command < 0:
        command = 4
    if not 1 <= command <= 6:
        raise ValueError(f"invalid measured Zoo command: {command}")
    return command


def _model_input(row: dict[str, Any], family: str, pipeline: Any, device: Any, calibration: tuple[np.ndarray, np.ndarray]) -> dict[str, Any]:
    import cv2
    import torch
    from mmcv.core.bbox import get_box_type
    from mmcv.parallel.collate import collate
    from pdm_lite.bev_planning_adapter import _camera_paths

    measurement = row["measurement"]
    for name in ("imu_acceleration", "imu_angular_velocity"):
        values = measurement.get(name)
        if not isinstance(values, list) or len(values) != 3 or not all(math.isfinite(float(v)) for v in values):
            raise ValueError(f"{family} requires measured three-axis {name} (recollect PDM-Lite demonstrations)")
    images = []
    for name, path in _camera_paths(row).items():
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None or image.shape != (900, 1600, 3):
            raise ValueError(f"{name} must be a real 1600x900 RGB camera image: {path}")
        # The unchanged Zoo agent first flips CARLA BGR to RGB, then JPEG-decodes as BGR.
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        ok, encoded = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 20])
        if not ok:
            raise ValueError(f"could not decode measured camera: {path}")
        images.append(cv2.imdecode(encoded, cv2.IMREAD_COLOR))
    raw_theta = float(measurement["theta"]) + np.pi / 2
    ego_theta = -raw_theta + np.pi / 2
    quaternion = np.array([math.cos(ego_theta / 2), 0, 0, math.sin(ego_theta / 2)])
    can_bus = np.zeros(18)
    can_bus[:2] = [measurement["pos_global"][0], -measurement["pos_global"][1]]
    can_bus[3:7] = quaternion
    can_bus[7] = float(measurement["speed"])
    can_bus[10:13] = measurement["imu_acceleration"]
    can_bus[11] *= -1
    can_bus[13:16] = -np.asarray(measurement["imu_angular_velocity"])
    can_bus[16:18] = [ego_theta, math.degrees(ego_theta)]
    command = _near_command(measurement) - 1
    lid2world = lidar2global(measurement)
    shape = np.stack(images, axis=-1).shape
    data = {
        "img": images, "lidar2cam": calibration[0], "lidar2img": calibration[1],
        "folder": " ", "scene_token": f"{row['scenario_id']}:{row['frame']}",
        "frame_idx": 0, "timestamp": float(measurement["timestamp_s"]),
        "box_type_3d": get_box_type("LiDAR")[0], "can_bus": can_bus,
        "command": command, "l2g_r_mat": lid2world[:3, :3], "l2g_t": lid2world[:3, 3],
        "img_shape": shape, "ori_shape": shape, "pad_shape": shape,
    }
    if family == "vad":
        onehot = np.zeros(6)
        onehot[command] = 1
        data["ego_fut_cmd"] = onehot
    batch = collate([pipeline(data)], samples_per_gpu=1)
    for key, value in batch.items():
        if key != "img_metas" and torch.is_tensor(value[0]):
            value[0] = value[0].to(device)
    return batch


def train(family: str, demos: Path, zoo_root: Path, baseline: Path, output: Path, *,
          epochs: int, batch_size: int, learning_rate: float, device: str) -> dict[str, Any]:
    from pdm_lite.bev_planning_adapter import FUTURE_STEPS, SAMPLE_PERIOD_S, _camera_paths, _read_demos

    if batch_size != 1:
        raise ValueError("official Zoo inference/planning models support batch-size 1 only")
    grouped = _read_demos(demos)
    samples = []
    for scenario, records in grouped.items():
        records.sort(key=lambda row: float(row["measurement"]["timestamp_s"]))
        for first, second in zip(records, records[1:]):
            delta = float(second["measurement"]["timestamp_s"]) - float(first["measurement"]["timestamp_s"])
            if not math.isclose(delta, SAMPLE_PERIOD_S, rel_tol=0., abs_tol=.02):
                raise ValueError(f"scenario {scenario!r}: discontinuous measured frames ({delta:.3f}s)")
        for index in range(len(records) - FUTURE_STEPS):
            row = records[index]
            _camera_paths(row)
            values = row["measurement"]
            for name in ("imu_acceleration", "imu_angular_velocity"):
                if name not in values:
                    raise ValueError(f"missing real {name} in PDM-Lite measurements; recollect expert demonstrations")
            future = [record["measurement"] for record in records[index + 1:index + FUTURE_STEPS + 1]]
            samples.append((row, local_future(values, future)))
    if not samples:
        raise ValueError("at least seven contiguous 0.5s PDM-Lite frames per scenario required")

    import torch
    prepare_zoo_runtime(zoo_root)
    from mmcv import Config
    # The CUDA extensions must be registered before importing the Zoo mmcv package.
    from mmcv.models import build_model
    from mmcv.utils import load_checkpoint
    from mmcv.datasets.pipelines import Compose

    config = zoo_root / ("adzoo/uniad/configs/stage2_e2e/base_e2e_b2d.py" if family == "uniad"
                         else "adzoo/vad/configs/VAD/VAD_base_e2e_b2d.py")
    cfg = Config.fromfile(str(config))
    if family == "uniad":
        cfg.model["motion_head"]["anchor_info_path"] = str(zoo_root / cfg.model["motion_head"]["anchor_info_path"])
    if cfg.get("plugin", False):
        plugin_dir = str(cfg.plugin_dir)
        importlib.import_module("Bench2DriveZoo." + plugin_dir.strip("/").replace("/", "."))
    model = build_model(cfg.model, train_cfg=cfg.get("train_cfg"), test_cfg=cfg.get("test_cfg"))
    load_checkpoint(model, str(baseline), map_location="cpu", strict=True)
    target_device = torch.device(device)
    model.to(target_device)
    model.eval()
    # Both original agent forward_test methods support one frame per sample;
    # temporal history cannot leak across independent Scenic rollouts.
    model.video_test_mode = False
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    planner = model.planning_head if family == "uniad" else model.pts_bbox_head.ego_fut_decoder
    planner.train()
    for parameter in planner.parameters():
        parameter.requires_grad_(True)
    initial = [parameter.detach().clone() for parameter in planner.parameters()]
    optimizer = torch.optim.AdamW(planner.parameters(), lr=learning_rate)
    pipeline = Compose([item for item in cfg.inference_only_pipeline
                        if item["type"] not in ("LoadMultiViewImageFromFilesInCeph", "LoadMultiViewImageFromFiles")])
    calibration = camera_calibration()
    updates = 0
    total_loss = 0.
    for _ in range(epochs):
        for row, trajectory in samples:
            # Remove temporal caches after each update: next expert frame can belong
            # to a different run, and a cached tensor would retain a stale graph.
            model.prev_frame_infos = []
            model.prev_frame_info = {"prev_bev": None, "scene_token": None, "prev_pos": 0, "prev_angle": 0}
            if family == "uniad":
                model.test_track_instances = None
                model.scene_token = None
            batch = _model_input(row, family, pipeline, target_device, calibration)
            result = model(batch, return_loss=False, rescale=True)[0]
            command = _near_command(row["measurement"])
            if family == "uniad":
                prediction = result["planning"]["result_planning"]["sdc_traj"]
                if prediction.ndim == 3:
                    prediction = prediction[0]
            else:
                prediction = result["pts_bbox"]["ego_fut_preds"][command - 1].cumsum(dim=0)
            target = torch.as_tensor(trajectory, dtype=prediction.dtype, device=prediction.device)
            if prediction.shape != target.shape:
                raise ValueError(f"Zoo {family} planner output {tuple(prediction.shape)} != measured {tuple(target.shape)}")
            loss = torch.nn.functional.smooth_l1_loss(prediction, target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            updates += 1
            total_loss += float(loss.detach())
    if updates == 0 or all(torch.equal(before, after) for before, after in zip(initial, planner.parameters())):
        raise RuntimeError(f"official {family} planner did not update; refusing to write unchanged checkpoint")
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "meta": {"source": "PDM-Lite measured Scenic planner fine-tune",
                                                          "model": family, "updates": updates}}, output)
    return {"checkpoint": str(output.resolve()), "model": f"official Zoo {family}", "examples": len(samples),
            "epochs": epochs, "updates": updates, "mean_loss": total_loss / updates,
            "supervision": "measured 0.5s lidar-frame future trajectory; planning head only"}
