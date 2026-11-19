from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest

from scripts.prepare_scenic_training_demos import _read_run, prepare


def _audit_tree(root: Path, *, bad_green_frame: bool = False, gap: bool = False, missing: bool = False,
                six_cameras: bool = False) -> Path:
    runs = []
    for scenario_index in range(20):
        scenario = "fdset_green_light_start" if scenario_index == 2 else f"scene-{scenario_index:02d}"
        for seed in (2026, 2027, 2028):
            output = root / scenario / str(seed) / "pdm_lite"
            output.mkdir(parents=True)
            records = []
            for frame in range(11):
                timestamp = frame * 0.5 + (0.5 if gap and scenario_index == 1 and seed == 2026 and frame >= 6 else 0)
                measure_file = output / f"{frame:04}.json.gz"
                measurement = {"steer": 0.1, "throttle": 0.2, "brake": 0.0, "timestamp_s": timestamp,
                               "pos_global": [frame, 0, 0], "theta": 0.0, "command": 4, "speed": 1.0,
                               "acceleration": [0.0, 0.0, 0.0]}
                if bad_green_frame and scenario_index == 2 and seed == 2026 and frame == 5:
                    measurement["stop_sign_hazard"] = True
                with gzip.open(measure_file, "wt", encoding="utf-8") as handle:
                    json.dump(measurement, handle)
                sensor = output / f"{frame:04}.jpg"
                if not (missing and scenario_index == 3 and seed == 2026 and frame == 0):
                    sensor.write_bytes(b"real-file-placeholder")
                cameras = ("CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT",
                           "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT") if six_cameras else ("CAM_FRONT",)
                sensor_files = {}
                for camera in cameras:
                    camera_path = sensor if camera == "CAM_FRONT" else output / f"{frame:04}-{camera}.jpg"
                    if camera != "CAM_FRONT":
                        camera_path.write_bytes(b"real-file-placeholder")
                    sensor_files[camera] = str(camera_path)
                records.append({"expert": "PDM-Lite", "expert_backend": "pdm_lite_native_scenic",
                                "scenario_id": scenario, "camera_profile": "uniad" if six_cameras else "admlp",
                                "frame": f"{frame:04}", "measurement_file": str(measure_file),
                                "sensor_files": sensor_files, "measurement": measurement})
            (output / "expert_demos.jsonl").write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
            runs.append({"scenario_id": scenario, "seed": seed, "status": "completed", "execution_status": "completed",
                         "return_code": 0, "collision_count": 1 if scenario_index == 0 else 0,
                         "source_sha256": f"{scenario_index + 1:064x}", "runtime_sha256": "a" * 64,
                         "artifacts": {"output_dir": str(output)}})
    audit = root / "audit.json"
    audit.write_text(json.dumps({"schema_version": "fdset.scenic_bank_audit.v1", "requested_programs": 20,
                                 "requested_seeds": [2026, 2027, 2028], "completed_runs": 60, "runs": runs}), encoding="utf-8")
    return audit


def test_excludes_collision_runs_and_green_hazard_frame_without_stitching(tmp_path: Path) -> None:
    audit = _audit_tree(tmp_path / "inputs", bad_green_frame=True)
    report = prepare(audit, tmp_path / "out")
    rows = [json.loads(line) for line in (tmp_path / "out/expert_demos.jsonl").read_text().splitlines()]
    assert not any(row["scenario_id"] == "scene-00" for row in rows)
    assert not any(row["scenario_id"] == "fdset_green_light_start" and "/2026/" in row["measurement_file"]
                   for row in rows)
    assert report["scenarios"]["scene-00"]["dropped_runs"] == 3
    assert report["scenarios"]["scene-00"]["dropped_reasons"] == ["collision_run"] * 3
    assert report["scenarios"]["fdset_green_light_start"]["admitted_frames"] == 22


def test_disjoint_timestamps_do_not_form_eleven_frame_window(tmp_path: Path) -> None:
    audit = _audit_tree(tmp_path / "inputs", gap=True)
    report = prepare(audit, tmp_path / "out")
    rows = [json.loads(line) for line in (tmp_path / "out/expert_demos.jsonl").read_text().splitlines()]
    assert not any(row["scenario_id"] == "scene-01" and "/2026/" in row["measurement_file"] for row in rows)
    assert report["scenarios"]["scene-01"]["admitted_frames"] == 22


def test_missing_real_sensor_drops_run_and_output_is_deterministic(tmp_path: Path) -> None:
    audit = _audit_tree(tmp_path / "inputs", missing=True)
    first = prepare(audit, tmp_path / "first")
    first_bytes = (tmp_path / "first/expert_demos.jsonl").read_bytes()
    second = prepare(audit, tmp_path / "second")
    assert first_bytes == (tmp_path / "second/expert_demos.jsonl").read_bytes()
    assert first["scenarios"]["scene-03"]["admitted_frames"] == 22
    assert first["scenarios"]["scene-03"]["dropped_reasons"] == ["invalid_or_incomplete_data"]
    assert json.loads((tmp_path / "first/training_report.json").read_text()) == first
    assert second["admitted_frames"] == first["admitted_frames"]


def test_six_camera_admission_serves_uniad_vad_and_tcp_without_old_sensorless_data(tmp_path: Path) -> None:
    audit = _audit_tree(tmp_path / "sixcam", six_cameras=True)
    for profile in ("uniad", "vad", "tcp"):
        report = prepare(audit, tmp_path / profile, required_camera_profile=profile)
        assert report["admitted_frames"] == 627
        rows = [json.loads(line) for line in (tmp_path / profile / "expert_demos.jsonl").read_text().splitlines()]
        assert len({row["training_episode_id"] for row in rows}) == 57
        assert all(len(row["sensor_files"]) == 6 for row in rows)
    old = _audit_tree(tmp_path / "old")
    with pytest.raises(ValueError, match="incompatible with uniad"):
        _read_run(json.loads(old.read_text())["runs"][3], "uniad")


def test_rejects_incomplete_or_malformed_bank_identity(tmp_path: Path) -> None:
    audit = _audit_tree(tmp_path / "inputs")
    payload = json.loads(audit.read_text())
    payload["runs"].pop()
    audit.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="exactly 60"):
        prepare(audit, tmp_path / "out")
