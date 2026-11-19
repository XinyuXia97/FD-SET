from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

from pdm_lite import admlp_adapter


def _write_demos(path: Path, *, scenarios: tuple[str, ...] = ("scene-a",), count: int = 11) -> None:
    rows = []
    for scenario_index, scenario in enumerate(scenarios):
        for index in range(count):
            rows.append({
                "scenario_id": scenario,
                "expert_backend": "pdm_lite_native_scenic",
                "frame": f"{index:04}",
                "measurement": {
                    "pos_global": [float(index + scenario_index * 100), float(index * 2), 0.0],
                    "theta": 1.25 + index * 0.01,
                    "command": 4,
                    "next_command": 2,
                    "speed": 7.5 + index,
                    "acceleration": [0.1 + index, -0.2, 0.3],
                    "timestamp_s": float(index) * 0.5,
                },
            })
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _zoo_root() -> Path:
    configured = os.environ.get("BENCH2DRIVE_ZOO_ROOT")
    candidates = [Path(configured)] if configured else []
    candidates += [Path("../Bench2DriveZoo"), Path("../Bench2DriveZoo_1")]
    for candidate in candidates:
        if (candidate / "ADMLP" / "model.py").is_file():
            return candidate.resolve()
    pytest.skip("official Zoo ADMLP checkout unavailable; set BENCH2DRIVE_ZOO_ROOT")


def test_convert_creates_contiguous_real_windows_and_scenario_manifest(tmp_path: Path) -> None:
    demos = tmp_path / "expert_demos.jsonl"
    output = tmp_path / "admlp.npy"
    manifest_path = tmp_path / "scenario_manifest.json"
    _write_demos(demos, scenarios=("scene-a", "scene-b"), count=11)

    manifest = admlp_adapter.convert(demos, output, manifest_path)

    data = np.load(output, allow_pickle=True).item()
    assert set(data) == {
        "input_x", "input_y", "input_theta", "input_command", "input_speed", "input_speed_acc",
        "future_x", "future_y", "future_theta",
    }
    assert np.asarray(data["input_x"]).shape == (2, 5)
    assert np.asarray(data["future_x"]).shape == (2, 6)
    assert data["input_x"][0] == [0., 1., 2., 3., 4.]
    assert data["future_x"][0] == [5., 6., 7., 8., 9., 10.]
    assert data["input_command"][0] == [4., 4., 4., 4., 4.]
    assert data["input_speed"][0] == [7.5, 8.5, 9.5, 10.5, 11.5]
    assert data["input_speed_acc"][0][2] == [2.1, -0.2, 0.3]
    assert data["input_theta"][0][0] == pytest.approx(1.25)
    assert [entry["scenario_id"] for entry in manifest["scenarios"]] == ["scene-a", "scene-b"]
    assert manifest["sample_count"] == 2
    assert json.loads(manifest_path.read_text(encoding="utf-8")) == manifest


def test_convert_keeps_repeated_scenario_runs_as_separate_episodes(tmp_path: Path) -> None:
    demos = tmp_path / "expert_demos.jsonl"
    _write_demos(demos)
    rows = [json.loads(line) for line in demos.read_text(encoding="utf-8").splitlines()]
    repeated = json.loads(json.dumps(rows))
    for row in rows:
        row["training_episode_id"] = "scene-a:2026:0-10"
    for row in repeated:
        row["training_episode_id"] = "scene-a:2027:0-10"
        row["measurement"]["timestamp_s"] += 2.13
        row["measurement"]["pos_global"][0] += 100
    demos.write_text("".join(json.dumps(row) + "\n" for row in rows + repeated), encoding="utf-8")
    manifest = admlp_adapter.convert(demos, tmp_path / "episodes.npy", tmp_path / "episodes.json")
    samples = np.load(tmp_path / "episodes.npy", allow_pickle=True).item()
    assert manifest["sample_count"] == 2
    assert {entry["training_episode_id"] for entry in manifest["scenarios"]} == {
        "scene-a:2026:0-10", "scene-a:2027:0-10",
    }
    assert {tuple(history) for history in samples["input_x"]} == {
        (0., 1., 2., 3., 4.), (100., 101., 102., 103., 104.),
    }


def test_convert_rejects_missing_real_acceleration_and_discontinuous_time(tmp_path: Path) -> None:
    demos = tmp_path / "expert_demos.jsonl"
    _write_demos(demos)
    rows = [json.loads(line) for line in demos.read_text(encoding="utf-8").splitlines()]
    del rows[5]["measurement"]["acceleration"]
    demos.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    with pytest.raises(ValueError, match="acceleration"):
        admlp_adapter.convert(demos, tmp_path / "bad.npy", tmp_path / "bad.json")

    _write_demos(demos)
    rows = [json.loads(line) for line in demos.read_text(encoding="utf-8").splitlines()]
    rows[6]["measurement"]["timestamp_s"] = 3.25
    demos.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    with pytest.raises(ValueError, match="non-contiguous"):
        admlp_adapter.convert(demos, tmp_path / "gap.npy", tmp_path / "gap.json")


def test_convert_rejects_proxy_expert_even_with_complete_measurements(tmp_path: Path) -> None:
    demos = tmp_path / "proxy.jsonl"
    _write_demos(demos)
    rows = [json.loads(line) for line in demos.read_text(encoding="utf-8").splitlines()]
    rows[0]["expert_backend"] = "kinematic_proxy"
    demos.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    with pytest.raises(ValueError, match="unverified native Scenic"):
        admlp_adapter.convert(demos, tmp_path / "proxy.npy", tmp_path / "proxy.json")


def test_convert_rejects_too_few_measurements_instead_of_padding(tmp_path: Path) -> None:
    demos = tmp_path / "short.jsonl"
    _write_demos(demos, count=4)
    with pytest.raises(ValueError, match="at least 11 contiguous"):
        admlp_adapter.convert(demos, tmp_path / "short.npy", tmp_path / "short.json")


def test_train_updates_official_model_and_writes_agent_compatible_checkpoint(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    zoo_root = _zoo_root()
    # Use the actual Zoo classes to create a shape-authentic baseline fixture.
    _, model_class, config_class = admlp_adapter._import_zoo(zoo_root)
    model = model_class(config_class())
    baseline = tmp_path / "baseline.ckpt"
    torch.save({"state_dict": {f"model.{key}": value.detach().clone()
                                for key, value in model.state_dict().items()}}, baseline)
    demos = tmp_path / "expert_demos.jsonl"
    _write_demos(demos)
    dataset_file = tmp_path / "admlp.npy"
    admlp_adapter.convert(demos, dataset_file, tmp_path / "manifest.json")

    output = tmp_path / "artifacts" / "adapted.ckpt"
    result = admlp_adapter.train(
        zoo_root, dataset_file, baseline, output, epochs=1, batch_size=1,
        learning_rate=1e-3, device="cpu",
    )

    saved = torch.load(output, map_location="cpu", weights_only=False)
    assert result["updates"] == 1
    assert result["examples"] == 1
    assert saved["state_dict"]
    assert all(key.startswith("model.") for key in saved["state_dict"])
    model.load_state_dict(admlp_adapter._state_dict(saved), strict=True)
    assert any(not torch.equal(model.state_dict()[key], value)
               for key, value in admlp_adapter._state_dict(torch.load(baseline, map_location="cpu", weights_only=False)).items())


def test_train_rejects_unavailable_requested_cuda_before_checkpoint_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    zoo_root = _zoo_root()
    _, model_class, config_class = admlp_adapter._import_zoo(zoo_root)
    baseline = tmp_path / "baseline.ckpt"
    torch.save({"state_dict": {f"model.{key}": value for key, value in model_class(config_class()).state_dict().items()}}, baseline)
    demos = tmp_path / "expert_demos.jsonl"
    _write_demos(demos)
    data_file = tmp_path / "admlp.npy"
    admlp_adapter.convert(demos, data_file, tmp_path / "manifest.json")
    output = tmp_path / "must-not-exist.ckpt"
    with pytest.raises(RuntimeError, match="CUDA is unavailable"):
        admlp_adapter.train(zoo_root, data_file, baseline, output, epochs=1, batch_size=1,
                            learning_rate=1e-3, device="cuda")
    assert not output.exists()
