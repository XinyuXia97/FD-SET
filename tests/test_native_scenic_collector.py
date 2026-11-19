from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest

from pdm_lite.scenic_collector import SENSOR_FILES, index_native_demos


def _dataset(tmp_path: Path) -> tuple[Path, dict]:
    dataset_dir = tmp_path / "pdm_lite_dataset"
    route_dir = dataset_dir / "FDSET_SCENIC_Rep0_0"
    measurements = route_dir / "measurements"
    measurements.mkdir(parents=True)
    for frame_index in range(11):
        frame = f"{frame_index:04d}"
        with gzip.open(measurements / f"{frame}.json.gz", "wt", encoding="utf-8") as output:
            json.dump({
                "steer": 0.25, "throttle": 0.5, "brake": False,
                "timestamp_s": frame_index * 0.5,
            }, output)
        for modality, suffix in SENSOR_FILES.items():
            folder = route_dir / modality
            folder.mkdir(exist_ok=True)
            (folder / f"{frame}{suffix}").write_bytes(b"sensor frame")
    execution = {
        "status": "completed",
        "programs": [{"scenario_id": "scenic_cut_in", "program_path": str(tmp_path / "cut_in.scenic"), "status": "completed"}],
    }
    return dataset_dir, execution


def test_native_expert_indexes_scenic_provenance_and_real_modalities(tmp_path: Path) -> None:
    dataset_dir, execution = _dataset(tmp_path)
    index_file = tmp_path / "expert_demos.jsonl"
    result = index_native_demos(execution, dataset_dir, index_file)
    rows = [json.loads(line) for line in index_file.read_text(encoding="utf-8").splitlines()]
    row = rows[0]
    assert result["measured"] is True
    assert result["num_measurements"] == 11
    assert row["scenario_id"] == "scenic_cut_in"
    assert row["control"] == {"steer": 0.25, "throttle": 0.5, "brake": False}
    assert all(Path(path).is_file() for path in row["sensor_files"].values())
    assert row["expert_backend"] == "pdm_lite_native_scenic"


def test_native_expert_rejects_partial_sensor_frames_and_failed_scenes(tmp_path: Path) -> None:
    dataset_dir, execution = _dataset(tmp_path)
    index_file = tmp_path / "expert_demos.jsonl"
    (dataset_dir / "FDSET_SCENIC_Rep0_0" / "lidar" / "0000.laz").unlink()
    with pytest.raises(RuntimeError, match="Incomplete PDM-Lite sensor frame"):
        index_native_demos(execution, dataset_dir, index_file)
    assert not index_file.exists()

    execution["programs"][0]["status"] = "error"
    with pytest.raises(RuntimeError, match="every program to complete"):
        index_native_demos(execution, dataset_dir, index_file)
    assert not index_file.exists()


def test_collision_program_is_excluded_but_reported(tmp_path: Path) -> None:
    dataset_dir, execution = _dataset(tmp_path)
    execution["programs"][0]["collisions"] = 1
    demos = tmp_path / "expert_demos.jsonl"
    result = index_native_demos(execution, dataset_dir, demos)
    assert demos.read_text() == ""
    assert result["num_measurements"] == 0
    assert result["dropped_collision_scenarios"] == ["scenic_cut_in"]


@pytest.mark.parametrize("hazard", ["stop_sign_hazard", "light_hazard"])
def test_green_light_hazards_break_windows_without_stitching(
    tmp_path: Path, hazard: str,
) -> None:
    dataset_dir, execution = _dataset(tmp_path)
    execution["programs"][0]["scenario_id"] = "fdset_green_light_start"
    route = dataset_dir / "FDSET_SCENIC_Rep0_0"
    measurements = route / "measurements"
    # A rejected middle frame leaves two five-frame runs, neither trainable.
    middle = measurements / "0005.json.gz"
    with gzip.open(middle, "wt", encoding="utf-8") as output:
        json.dump({
            "steer": 0.25, "throttle": 0.5, "brake": False,
            "timestamp_s": 2.5, hazard: True,
        }, output)
    result = index_native_demos(execution, dataset_dir, tmp_path / "demos.jsonl")
    assert result["num_measurements"] == 0


def test_timestamp_gap_does_not_form_training_window(tmp_path: Path) -> None:
    dataset_dir, execution = _dataset(tmp_path)
    route = dataset_dir / "FDSET_SCENIC_Rep0_0"
    for frame_index in range(6, 11):
        path = route / "measurements" / f"{frame_index:04d}.json.gz"
        with gzip.open(path, "rt", encoding="utf-8") as source:
            measurement = json.load(source)
        measurement["timestamp_s"] += 1.0
        with gzip.open(path, "wt", encoding="utf-8") as output:
            json.dump(measurement, output)
    result = index_native_demos(execution, dataset_dir, tmp_path / "demos.jsonl")
    assert result["num_measurements"] == 0


@pytest.mark.parametrize("profile,expected_cameras", [("tcp", 3), ("uniad", 6), ("vad", 6)])
def test_multiview_expert_requires_distinct_synchronized_camera_frames(
    tmp_path: Path, profile: str, expected_cameras: int,
) -> None:
    from pdm_lite.camera_profiles import cameras_for_profile

    dataset_dir, execution = _dataset(tmp_path)
    demos_file = tmp_path / "expert_demos.jsonl"
    cameras = cameras_for_profile(profile)
    assert len(cameras) == expected_cameras
    with pytest.raises(RuntimeError, match="CAM_FRONT"):
        index_native_demos(execution, dataset_dir, demos_file, camera_profile=profile)
    assert not demos_file.exists()

    route_dir = dataset_dir / "FDSET_SCENIC_Rep0_0"
    for camera in cameras:
        (route_dir / camera).mkdir()
        for frame_index in range(11):
            (route_dir / camera / f"{frame_index:04d}.jpg").write_bytes(camera.encode("ascii"))
    result = index_native_demos(execution, dataset_dir, demos_file, camera_profile=profile)
    indexed = json.loads(demos_file.read_text(encoding="utf-8").splitlines()[0])
    assert result["camera_profile"] == profile
    assert {name: Path(indexed["sensor_files"][name]).read_bytes() for name in cameras} == {
        name: name.encode("ascii") for name in cameras
    }

def test_scenic_speed_limit_resource_samples_map_waypoints(tmp_path: Path) -> None:
    import numpy as np
    from pdm_lite.scenic_resources import ensure_speed_limit_map

    class Location:
        def __init__(self, x: float) -> None:
            self.x, self.y, self.z = x, x + 1, 0.0

    class Waypoint:
        def __init__(self, x: float) -> None:
            self.transform = type("Transform", (), {"location": Location(x)})()

    class Map:
        name = "/Game/Carla/Maps/OpenDriveMap"

        def generate_waypoints(self, spacing: float) -> list:
            assert spacing == 1.0
            return [Waypoint(2.0), Waypoint(8.0)]

        def get_all_landmarks_of_type(self, kind: object) -> list:
            return []

    path = ensure_speed_limit_map(Map(), tmp_path / "speed_limits")
    payload = np.load(path, allow_pickle=True).item()
    assert payload["locations"].tolist() == [[2.0, 3.0, 0.0], [8.0, 9.0, 0.0]]
    assert payload["speed_limits"].tolist() == [50.0, 50.0]


def test_scenic_speed_limit_cache_never_crosses_to_different_road_geometry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import numpy as np
    from pdm_lite import scenic_resources

    references = tmp_path / "agent" / "speed_limits"
    references.mkdir(parents=True)
    np.save(references / "Town07_speed_limits.npy", {
        "locations": np.asarray([[2.0, 3.0, 0.0], [8.0, 9.0, 0.0]]),
        "speed_limits": np.asarray([30.0, 40.0]),
    })
    monkeypatch.setattr(scenic_resources, "__file__", str(tmp_path / "scenic_resources.py"))

    class Map:
        name = "/Game/Carla/Maps/OpenDriveMap"

        def __init__(self, changed: bool) -> None:
            self.changed = changed

        def generate_waypoints(self, spacing: float) -> list:
            x_values = (2.0, 9.0 if self.changed else 8.0)
            return [
                type("Waypoint", (), {"transform": type("Transform", (), {
                    "location": type("Location", (), {"x": x, "y": x + 1, "z": 0.0})(),
                })()})()
                for x in x_values
            ]

        def get_all_landmarks_of_type(self, kind: object) -> list:
            if not self.changed:
                raise AssertionError("identical map must reuse validated speed limits")
            return []

    matched = scenic_resources.ensure_speed_limit_map(Map(False), tmp_path / "matched")
    different = scenic_resources.ensure_speed_limit_map(Map(True), tmp_path / "different")
    assert np.load(matched, allow_pickle=True).item()["speed_limits"].tolist() == [30.0, 40.0]
    assert np.load(different, allow_pickle=True).item()["speed_limits"].tolist() == [50.0, 50.0]
