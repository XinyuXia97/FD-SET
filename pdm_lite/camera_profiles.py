"""Bench2DriveZoo camera geometry for native Scenic expert collection.

These are the actual agent sensors, not alternate views synthesized from PDM's
front camera. Image IDs are also the sensor_files keys in expert_demos.jsonl.
"""

from __future__ import annotations


ZOO_CAMERA_SPECS: dict[str, dict[str, float | int | str]] = {
    "CAM_FRONT": {
        "type": "sensor.camera.rgb", "x": 0.80, "y": 0.0, "z": 1.60,
        "roll": 0.0, "pitch": 0.0, "yaw": 0.0,
        "width": 1600, "height": 900, "fov": 70,
    },
    "CAM_FRONT_LEFT": {
        "type": "sensor.camera.rgb", "x": 0.27, "y": -0.55, "z": 1.60,
        "roll": 0.0, "pitch": 0.0, "yaw": -55.0,
        "width": 1600, "height": 900, "fov": 70,
    },
    "CAM_FRONT_RIGHT": {
        "type": "sensor.camera.rgb", "x": 0.27, "y": 0.55, "z": 1.60,
        "roll": 0.0, "pitch": 0.0, "yaw": 55.0,
        "width": 1600, "height": 900, "fov": 70,
    },
    "CAM_BACK": {
        "type": "sensor.camera.rgb", "x": -2.0, "y": 0.0, "z": 1.60,
        "roll": 0.0, "pitch": 0.0, "yaw": 180.0,
        "width": 1600, "height": 900, "fov": 110,
    },
    "CAM_BACK_LEFT": {
        "type": "sensor.camera.rgb", "x": -0.32, "y": -0.55, "z": 1.60,
        "roll": 0.0, "pitch": 0.0, "yaw": -110.0,
        "width": 1600, "height": 900, "fov": 70,
    },
    "CAM_BACK_RIGHT": {
        "type": "sensor.camera.rgb", "x": -0.32, "y": 0.55, "z": 1.60,
        "roll": 0.0, "pitch": 0.0, "yaw": 110.0,
        "width": 1600, "height": 900, "fov": 70,
    },
}

_CAMERA_IDS = tuple(ZOO_CAMERA_SPECS)
CAMERA_PROFILES = {
    "admlp": (),
    "tcp": _CAMERA_IDS[:3],
    "uniad": _CAMERA_IDS,
    "vad": _CAMERA_IDS,
}


def cameras_for_profile(profile: str) -> tuple[str, ...]:
    """Select genuine RGB sensors for the model consuming this expert run."""
    try:
        return CAMERA_PROFILES[profile]
    except KeyError as exc:
        raise ValueError(f"Unsupported PDM-Lite camera profile: {profile}") from exc
