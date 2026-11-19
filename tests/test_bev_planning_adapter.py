from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from pdm_lite import bev_planning_adapter as adapter
from pdm_lite.zoo_planner_training import _near_command


def _demo_rows(directory: Path, *, camera_names: tuple[str, ...] = adapter.CAMERAS, count: int = 7) -> list[dict[str, object]]:
    directory.mkdir(parents=True, exist_ok=True)
    rows = []
    for frame in range(count):
        sensors = {}
        for camera in camera_names:
            path = directory / f"{camera}-{frame}.jpg"
            path.write_bytes(b"real image fixture")
            sensors[camera] = str(path)
        rows.append({
            "expert": "PDM-Lite",
            "expert_backend": "pdm_lite_native_scenic",
            "scenario_id": "scenario-1",
            "frame": str(frame),
            "sensor_files": sensors,
            "measurement": {
                "pos_global": [float(frame), float(frame * 2)],
                "theta": 0.1 * frame,
                "speed": 3.0 + frame,
                "timestamp_s": 10.0 + adapter.SAMPLE_PERIOD_S * frame,
                "command": 1,
                "next_command": 2,
            },
        })
    return rows


def _write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_export_preserves_only_measured_world_trajectory_and_camera_provenance(tmp_path: Path) -> None:
    rows = _demo_rows(tmp_path / "frames")
    demos = tmp_path / "demos.jsonl"
    _write_rows(demos, rows)
    archive_path = tmp_path / "world-traj.npz"
    manifest_path = tmp_path / "manifest.json"

    manifest = adapter.export_trajectories(demos, archive_path, manifest_path)

    assert manifest["not_zoo_training_dataset"] is True
    assert manifest["sample_count"] == 1
    with np.load(archive_path) as archive:
        assert archive["position_world_xy"].tolist() == [[0.0, 0.0]]
        assert archive["future_position_world_xy"].tolist() == [[[1.0, 2.0], [2.0, 4.0], [3.0, 6.0], [4.0, 8.0], [5.0, 10.0], [6.0, 12.0]]]
        assert archive["camera_files"].shape == (1, 6)
        assert archive["command"].tolist() == [1]


def test_export_does_not_join_repeated_scenic_template_across_seeds(tmp_path: Path) -> None:
    first = _demo_rows(tmp_path / "seed2026")
    second = _demo_rows(tmp_path / "seed2027")
    for row in first:
        row["training_episode_id"] = "scenario-1:2026:0-6"
    for row in second:
        row["training_episode_id"] = "scenario-1:2027:0-6"
        row["measurement"]["timestamp_s"] += 2.13
        row["measurement"]["pos_global"][0] += 100
    demos = tmp_path / "demos.jsonl"
    _write_rows(demos, first + second)
    manifest = adapter.export_trajectories(demos, tmp_path / "trajectories.npz", tmp_path / "manifest.json")
    assert manifest["sample_count"] == 2
    with np.load(tmp_path / "trajectories.npz") as archive:
        assert sorted(archive["position_world_xy"][:, 0].tolist()) == [0., 100.]
        assert sorted(archive["future_position_world_xy"][:, -1, 0].tolist()) == [6., 106.]


def test_zoo_planner_uses_turn_command_from_measured_near_route() -> None:
    assert _near_command({"command": 1, "next_command": 4}) == 1
    assert _near_command({"command": -1, "next_command": 4}) == 4


def test_export_rejects_missing_synchronized_camera_key(tmp_path: Path) -> None:
    rows = _demo_rows(tmp_path / "frames", camera_names=adapter.CAMERAS[:-1])
    demos = tmp_path / "demos.jsonl"
    _write_rows(demos, rows)

    with pytest.raises(ValueError, match="CAM_BACK_RIGHT"):
        adapter.export_trajectories(demos, tmp_path / "out.npz", tmp_path / "manifest.json")



def test_export_rejects_one_camera_reused_as_six_views(tmp_path: Path) -> None:
    rows = _demo_rows(tmp_path / "frames")
    for row in rows:
        row["sensor_files"]["CAM_BACK"] = row["sensor_files"]["CAM_FRONT"]
    demos = tmp_path / "demos.jsonl"
    _write_rows(demos, rows)
    with pytest.raises(ValueError, match="six distinct"):
        adapter.export_trajectories(demos, tmp_path / "out.npz", tmp_path / "manifest.json")


def test_train_rejects_absent_measured_imu_without_writing_checkpoint(tmp_path: Path) -> None:
    rows = _demo_rows(tmp_path / "frames")
    demos = tmp_path / "demos.jsonl"
    _write_rows(demos, rows)
    zoo_root = tmp_path / "zoo"
    zoo_root.mkdir()
    baseline = tmp_path / "baseline.pth"
    baseline.write_bytes(b"baseline")
    output = tmp_path / "changed.pth"

    with pytest.raises(ValueError, match="missing real imu_acceleration"):
        adapter.train("vad", demos, zoo_root, baseline, output, epochs=1, batch_size=1, learning_rate=1e-5)
    assert not output.exists()


def test_train_rejects_missing_camera_modality_before_annotations(tmp_path: Path) -> None:
    rows = _demo_rows(tmp_path / "frames", camera_names=("CAM_FRONT",))
    demos = tmp_path / "demos.jsonl"
    _write_rows(demos, rows)
    zoo_root = tmp_path / "zoo"
    zoo_root.mkdir()
    baseline = tmp_path / "baseline.pth"
    baseline.write_bytes(b"baseline")

    with pytest.raises(ValueError, match="missing synchronized Zoo camera keys"):
        adapter.train("uniad", demos, zoo_root, baseline, tmp_path / "out.pth",
                      epochs=1, batch_size=1, learning_rate=1e-5)


def test_lidar_frame_target_and_sensor_calibration_match_original_zoo_geometry() -> None:
    from pdm_lite.zoo_planner_training import camera_calibration, local_future

    lidar2cam, lidar2img = camera_calibration()
    assert lidar2cam.shape == (6, 4, 4)
    np.testing.assert_allclose(lidar2cam[0][:3, 3], [0, -0.24, -1.19], atol=1e-6)
    np.testing.assert_allclose(lidar2img[1][0, :2], [0, 1394.75744], atol=1e-3)
    np.testing.assert_allclose(lidar2img[3][2, :2], [0, -1], atol=1e-6)
    source = {"pos_global": [10., 20.], "theta": 0.}
    future = [{"pos_global": [11., 20.], "theta": 0.},
              {"pos_global": [12., 20.], "theta": 0.}]
    np.testing.assert_allclose(local_future(source, future), [[0., 1.], [0., 2.]], atol=1e-5)
