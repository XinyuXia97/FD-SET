from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from pdm_lite import tcp_adapter


def _write_demos(tmp_path: Path, *, count: int = 6) -> Path:
    rows = []
    camera_paths = {}
    for name, color in (("CAM_FRONT", (10, 20, 30)), ("CAM_FRONT_LEFT", (40, 50, 60)),
                        ("CAM_FRONT_RIGHT", (70, 80, 90))):
        path = tmp_path / f"{name}.png"
        from PIL import Image
        Image.fromarray(np.full((900, 1600, 3), color, dtype=np.uint8)).save(path)
        camera_paths[name] = str(path)
    for frame in range(count):
        rows.append({
            "expert_backend": "pdm_lite_native_scenic",
            "scenario_id": "measured-scene",
            "frame": str(frame),
            "sensor_files": camera_paths,
            "measurement": {
                "pos_global": [frame * 0.5, 0.0],
                "theta": 0.0,
                "speed": 1.0 + frame * 0.1,
                "command": 1,
                "next_command": 4,
                "timestamp_s": frame * 0.5,
                "target_point": [10.0, 0.0],
                "steer": 0.02 * frame,
                "throttle": 0.2,
                "brake": 0.0,
            },
        })
    demos = tmp_path / "demos.jsonl"
    demos.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return demos


def test_load_samples_requires_real_cameras_controls_and_four_future_positions(tmp_path: Path) -> None:
    demos = _write_demos(tmp_path)
    samples = tcp_adapter._load_samples(demos)
    assert len(samples) == 2
    assert samples[0]["waypoints"] == [(0.5, 0.0), (1.0, 0.0), (1.5, 0.0), (2.0, 0.0)]
    assert samples[0]["controls"] == (0.0, 0.2, 0.0)
    assert samples[0]["command"] == 1

    rows = [json.loads(line) for line in demos.read_text(encoding="utf-8").splitlines()]
    del rows[0]["sensor_files"]["CAM_FRONT_LEFT"]
    demos.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    with pytest.raises(ValueError, match="CAM_FRONT_LEFT"):
        tcp_adapter._load_samples(demos)

    rows = [json.loads(line) for line in _write_demos(tmp_path).read_text(encoding="utf-8").splitlines()]
    del rows[0]["measurement"]["steer"]
    demos.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    with pytest.raises(ValueError, match="steer"):
        tcp_adapter._load_samples(demos)


def test_load_samples_does_not_join_seeds_of_the_same_scenic_template(tmp_path: Path) -> None:
    demos = _write_demos(tmp_path)
    first = [json.loads(line) for line in demos.read_text(encoding="utf-8").splitlines()]
    second = json.loads(json.dumps(first))
    for row in first:
        row["training_episode_id"] = "measured-scene:2026:0-5"
    for row in second:
        row["training_episode_id"] = "measured-scene:2027:0-5"
        row["measurement"]["timestamp_s"] += 2.13
        row["measurement"]["pos_global"][0] += 100
    demos.write_text("".join(json.dumps(row) + "\n" for row in first + second), encoding="utf-8")
    samples = tcp_adapter._load_samples(demos)
    assert len(samples) == 4
    assert sorted({sample["x"] for sample in samples}) == [0., .5, 100., 100.5]
    assert all(sample["waypoints"][-1] == (2., 0.) for sample in samples)


def test_load_samples_rejects_incomplete_trajectory_and_missing_camera_file(tmp_path: Path) -> None:
    demos = _write_demos(tmp_path, count=4)
    with pytest.raises(ValueError, match="at least 5"):
        tcp_adapter._load_samples(demos)

    demos = _write_demos(tmp_path)
    rows = [json.loads(line) for line in demos.read_text(encoding="utf-8").splitlines()]
    rows[0]["sensor_files"]["CAM_FRONT"] = str(tmp_path / "absent.png")
    demos.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    with pytest.raises(ValueError, match="missing TCP camera image"):
        tcp_adapter._load_samples(demos)


def test_train_updates_compatible_tcp_checkpoint_from_real_sample_fields(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    torch = pytest.importorskip("torch")
    demos = _write_demos(tmp_path)

    class TinyTCP(torch.nn.Module):
        def __init__(self, _config):
            super().__init__()
            self.perception = torch.nn.Identity()
            self.output_traj = torch.nn.Linear(1, 8)
            self.speed_branch = torch.nn.Linear(1, 1)
            self.action_head = torch.nn.Linear(1, 39)

        def forward(self, image, state, _target_point):
            signal = image.mean(dim=(1, 2, 3), keepdim=False).unsqueeze(1)
            speed = self.speed_branch(state[:, :1])
            waypoints = self.output_traj(signal).reshape(-1, 4, 2)
            action = self.action_head(signal)
            return {"pred_wp": waypoints, "pred_speed": speed, "action_index": action}

    class Config:
        pass

    action_dict = {index: (0.0, 0.0, 0.0, False) for index in range(39)}
    monkeypatch.setattr(tcp_adapter, "_import_zoo", lambda _root: (TinyTCP, action_dict, Config))
    baseline = tmp_path / "baseline.ckpt"
    torch.save({"state_dict": {f"model.{key}": value.detach().clone()
                                for key, value in TinyTCP(Config()).state_dict().items()}}, baseline)
    output = tmp_path / "adapted.ckpt"
    result = tcp_adapter.train(Path("unused"), demos, baseline, output, epochs=1, batch_size=2,
                               learning_rate=1e-2)
    assert result["examples"] == 2
    assert result["updates"] == 1
    checkpoint = torch.load(output, map_location="cpu", weights_only=False)
    assert all(key.startswith("model.") for key in checkpoint["state_dict"])
    changed = torch.load(baseline, map_location="cpu", weights_only=False)["state_dict"]
    assert any(not torch.equal(changed[key], checkpoint["state_dict"][key]) for key in changed)
    model = TinyTCP(Config())
    model.load_state_dict(tcp_adapter._state_dict(checkpoint), strict=True)
