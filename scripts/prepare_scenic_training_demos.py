from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

WINDOW_SIZE = 11
FRAME_PERIOD_S = 0.5
HASH_RE = re.compile(r"^[0-9a-f]{64}$")


def _load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _valid_hash(value: Any) -> bool:
    return isinstance(value, str) and HASH_RE.fullmatch(value) is not None


def _check_audit(audit: dict[str, Any]) -> list[dict[str, Any]]:
    runs = audit.get("runs")
    if audit.get("schema_version") != "fdset.scenic_bank_audit.v1" or not isinstance(runs, list):
        raise ValueError("Unsupported or malformed Scenic audit")
    if audit.get("requested_programs") != 20 or audit.get("requested_seeds") != [2026, 2027, 2028]:
        raise ValueError("Audit must describe the expected 20-program, three-seed bank")
    if len(runs) != 60 or audit.get("completed_runs") != 60:
        raise ValueError("Audit must contain exactly 60 run records")
    identities: set[tuple[str, int]] = set()
    sources: dict[str, str] = {}
    runtime_versions: set[str] = set()
    for run in runs:
        if not isinstance(run, dict):
            raise ValueError("Malformed run entry")
        scenario, seed = run.get("scenario_id"), run.get("seed")
        if not isinstance(scenario, str) or seed not in (2026, 2027, 2028):
            raise ValueError("Malformed run identity")
        identity = (scenario, seed)
        if identity in identities:
            raise ValueError(f"Duplicate audit run identity: {identity}")
        identities.add(identity)
        if not _valid_hash(run.get("source_sha256")) or not _valid_hash(run.get("runtime_sha256")):
            raise ValueError(f"Missing or malformed source/runtime fingerprint for {scenario}:{seed}")
        runtime_versions.add(run["runtime_sha256"])
        if scenario in sources and sources[scenario] != run["source_sha256"]:
            raise ValueError(f"Source fingerprint changed between runs of {scenario}")
        sources[scenario] = run["source_sha256"]
    expected = {(scenario, seed) for scenario in sources for seed in (2026, 2027, 2028)}
    if len(sources) != 20 or identities != expected or len(runtime_versions) != 1:
        raise ValueError("Audit must cover 20 scenarios, all three seeds, and one expert runtime version")
    return runs


def _read_run(run: dict[str, Any], required_camera_profile: str | None) -> tuple[list[dict[str, Any]], str, int]:
    scenario = run["scenario_id"]
    artifacts = run.get("artifacts")
    if not isinstance(artifacts, dict) or not artifacts.get("output_dir"):
        raise ValueError("missing artifacts.output_dir")
    demos_path = Path(artifacts["output_dir"]) / "expert_demos.jsonl"
    if not demos_path.is_file():
        raise ValueError(f"missing expert demos: {demos_path}")
    source_rows = [json.loads(line) for line in demos_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not source_rows:
        return [], "", 0
    for record in source_rows:
        if record.get("scenario_id") != scenario or record.get("expert_backend") != "pdm_lite_native_scenic":
            raise ValueError(f"unexpected expert demo identity/backend in {demos_path}")
        measurement_path = Path(record.get("measurement_file", ""))
        if not measurement_path.is_file():
            raise ValueError(f"missing measurement file: {measurement_path}")
        sensor_files = record.get("sensor_files")
        if not isinstance(sensor_files, dict) or not sensor_files:
            raise ValueError(f"missing sensor-file references in {demos_path}")
        for name, path in sensor_files.items():
            if not Path(path).is_file():
                raise ValueError(f"missing sensor file {name}: {path}")
    profile_set = {record.get("camera_profile") for record in source_rows}
    if len(profile_set) != 1:
        raise ValueError(f"mixed camera profiles in {demos_path}")
    profile = next(iter(profile_set))
    compatible_profiles = {
        "admlp": {"admlp", "uniad", "vad", "tcp"},
        "uniad": {"uniad", "vad"},
        "vad": {"uniad", "vad"},
        "tcp": {"uniad", "vad", "tcp"},
    }
    if required_camera_profile and profile not in compatible_profiles[required_camera_profile]:
        raise ValueError(f"{scenario}:{run['seed']} uses {profile}, incompatible with {required_camera_profile}")
    rows: list[tuple[float, int, dict[str, Any], dict[str, Any]]] = []
    for record in source_rows:
        frame = str(record.get("frame", ""))
        measurement_path = Path(record["measurement_file"])
        with gzip.open(measurement_path, "rt", encoding="utf-8") as handle:
            measurement = json.load(handle)
        if not isinstance(measurement, dict) or not all(k in measurement for k in ("steer", "throttle", "brake")):
            raise ValueError(f"malformed measurement/control in {measurement_path}")
        if required_camera_profile in {"uniad", "vad", "tcp"}:
            required = (
                {"CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT", "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"}
                if required_camera_profile in {"uniad", "vad"}
                else {"CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT"}
            )
            if not required.issubset(record.get("sensor_files", {})):
                raise ValueError(f"{scenario}:{run['seed']} lacks required {required_camera_profile} camera sensors")
        if scenario == "fdset_green_light_start" and (measurement.get("stop_sign_hazard") or measurement.get("light_hazard")):
            continue
        timestamp = measurement.get("timestamp_s", int(frame) * FRAME_PERIOD_S if frame.isdigit() else None)
        try:
            timestamp = float(timestamp)
            frame_number = int(frame)
        except (TypeError, ValueError):
            raise ValueError(f"invalid frame/timestamp in {measurement_path}") from None
        if not math.isfinite(timestamp):
            raise ValueError(f"invalid timestamp in {measurement_path}")
        rows.append((timestamp, frame_number, record, measurement))
    rows.sort(key=lambda item: (item[0], item[1]))
    windows: list[list[tuple[float, int, dict[str, Any], dict[str, Any]]]] = []
    segment: list[tuple[float, int, dict[str, Any], dict[str, Any]]] = []
    for row in rows:
        if segment and (abs(row[0] - segment[-1][0] - FRAME_PERIOD_S) > 1e-6 or row[1] != segment[-1][1] + 1):
            if len(segment) >= WINDOW_SIZE:
                windows.append(segment)
            segment = []
        segment.append(row)
    if len(segment) >= WINDOW_SIZE:
        windows.append(segment)
    admitted: dict[str, dict[str, Any]] = {}
    for window in windows:
        episode_id = f"{scenario}:{run['seed']}:{window[0][1]}-{window[-1][1]}"
        for _, _, record, _ in window:
            admitted[record["frame"]] = {**record, "training_episode_id": episode_id}
    return [admitted[key] for key in sorted(admitted, key=lambda value: int(value))], profile, sum(
        len(window) - WINDOW_SIZE + 1 for window in windows
    )


def prepare(audit_path: Path, output_dir: Path, required_camera_profile: str | None = None) -> dict[str, Any]:
    audit = _load_json(audit_path)
    runs = _check_audit(audit)
    admitted_records: list[dict[str, Any]] = []
    summary: dict[str, dict[str, Any]] = defaultdict(lambda: {"admitted_frames": 0, "windows": 0, "dropped_runs": 0, "dropped_reasons": []})
    dropped: list[dict[str, Any]] = []
    for run in sorted(runs, key=lambda row: (row["scenario_id"], row["seed"])):
        scenario = run["scenario_id"]
        item = summary[scenario]
        if run.get("status") != "completed" or run.get("execution_status") != "completed" or run.get("return_code") != 0:
            reason = "execution_incomplete"
        elif not isinstance(run.get("collision_count"), (int, float)) or run["collision_count"] != 0:
            reason = "collision_run" if run.get("collision_count", 0) else "invalid_collision_count"
        else:
            reason = ""
        if reason:
            dropped.append({"scenario_id": scenario, "seed": run["seed"], "reason": reason, "collision_count": run.get("collision_count")})
            item["dropped_runs"] += 1
            item["dropped_reasons"].append(reason)
            continue
        try:
            records, _, window_count = _read_run(run, required_camera_profile)
        except (OSError, ValueError, json.JSONDecodeError) as error:
            dropped.append({"scenario_id": scenario, "seed": run["seed"], "reason": "invalid_or_incomplete_data", "detail": str(error)})
            item["dropped_runs"] += 1
            item["dropped_reasons"].append("invalid_or_incomplete_data")
            continue
        if not records:
            dropped.append({"scenario_id": scenario, "seed": run["seed"], "reason": "no_safe_windows"})
            item["dropped_runs"] += 1
            item["dropped_reasons"].append("no_safe_windows")
            continue
        frames = {record["frame"] for record in records}
        item["admitted_frames"] += len(frames)
        item["windows"] += window_count
        admitted_records.extend(records)
    output_dir.mkdir(parents=True, exist_ok=True)
    demos_path = output_dir / "expert_demos.jsonl"
    with demos_path.open("w", encoding="utf-8") as handle:
        for record in admitted_records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    report = {"schema_version": "fdset.scenic_training_index.v1", "audit": str(audit_path.resolve()),
              "required_camera_profile": required_camera_profile, "input_runs": len(runs),
              "admitted_frames": len(admitted_records), "dropped_runs": dropped,
              "scenarios": {key: summary[key] for key in sorted(summary)}, "training_index": str(demos_path.resolve())}
    (output_dir / "training_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Merge safe contiguous Scenic expert demonstrations from an existing audit")
    parser.add_argument("--audit", type=Path, required=True, help="Path to the existing audit.json")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for merged index and report")
    parser.add_argument("--required-camera-profile", choices=("admlp", "uniad", "vad", "tcp"), help="Require this recorded camera profile")
    args = parser.parse_args(argv)
    try:
        report = prepare(args.audit, args.output_dir, args.required_camera_profile)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(json.dumps({"training_index": report["training_index"], "admitted_frames": report["admitted_frames"], "dropped_runs": len(report["dropped_runs"])}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
