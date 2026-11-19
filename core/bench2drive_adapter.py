from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


DEFAULT_METRIC_KEYS = ["DS", "SR", "Eff", "Comf", "CR"]
BENCH2DRIVE_LABEL_MAP = {
    "Avg. driving score": "DS",
    "Avg. route completion": "RC",
    "Avg. infraction penalty": "IP",
    "Driving Efficiency": "Eff",
    "Driving Smoothness": "Comf",
    "L2m": "L2m",
}
COLLISION_KEYS = {"collisions_vehicle", "collisions_pedestrian", "collisions_layout"}


def build_eval_command(
    template: str,
    *,
    bench2drive_root: Path,
    round_id: int,
    scenario_file: Path,
    out_dir: Path,
    host: str | None = None,
    port: int | None = None,
    traffic_manager_port: int | None = None,
    track: str | None = None,
    agent: str | None = None,
    agent_config: str | None = None,
    gpu_rank: int | None = None,
    resume: bool | None = None,
) -> str:
    values: dict[str, Any] = {
        "bench2drive_root": str(bench2drive_root.resolve()),
        "python": sys.executable,
        "round_id": round_id,
        "scenario_file": str(scenario_file.resolve()),
        "out_dir": str(out_dir.resolve()),
    }
    if host is not None:
        values["host"] = host
    if port is not None:
        values["port"] = port
    if traffic_manager_port is not None:
        values["traffic_manager_port"] = traffic_manager_port
    if track is not None:
        values["track"] = track
    if agent is not None:
        values["agent"] = agent
    if agent_config is not None:
        values["agent_config"] = agent_config
    if gpu_rank is not None:
        values["gpu_rank"] = gpu_rank
    if resume is not None:
        values["resume"] = str(resume)
    return template.format(
        **values,
    )


def run_eval_command(
    command: str,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    log_file: Path | None = None,
) -> None:
    child_env = os.environ.copy()
    if env:
        child_env.update(env)
    if log_file is None:
        subprocess.run(command, shell=True, check=True, cwd=str(cwd) if cwd else None, env=child_env)
        return
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with log_file.open("w", encoding="utf-8") as handle:
        subprocess.run(
            command,
            shell=True,
            check=True,
            cwd=str(cwd) if cwd else None,
            env=child_env,
            stdout=handle,
            stderr=subprocess.STDOUT,
        )


def calculate_official_motion_metrics(
    checkpoint_file: Path, evidence_root: Path, bench2drive_root: Path,
) -> dict[str, float]:
    """Use Bench2Drive's own trajectory metric tool, not checkpoint defaults."""
    tool_path = bench2drive_root / "tools" / "efficiency_smoothness_benchmark.py"
    if not tool_path.is_file():
        raise FileNotFoundError(f"Official motion metric tool is missing: {tool_path}")
    if not evidence_root.is_dir():
        raise FileNotFoundError(f"Motion metric trajectories are missing: {evidence_root}")
    result = subprocess.run(
        [sys.executable, str(tool_path), "-f", str(checkpoint_file.resolve()),
         "-m", str(evidence_root.resolve())],
        cwd=str(bench2drive_root),
        capture_output=True,
        text=True,
        check=True,
    )
    labels = {"Driving Efficiency": "Eff", "Driving Smoothness": "Comf"}
    metrics: dict[str, float] = {}
    for line in result.stdout.splitlines():
        label, separator, raw_value = line.strip().partition("=")
        if separator and label in labels:
            score = float(raw_value)
            if not math.isfinite(score):
                raise ValueError(f"Official {label} is not finite: {raw_value}")
            metrics[labels[label]] = score * 100.0 if label == "Driving Smoothness" else score
    if len(metrics) != len(labels):
        raise RuntimeError(
            f"Official motion tool did not report both metrics: {result.stdout!r}"
        )
    return metrics


def parse_metrics(
    metrics_file: Path,
    keys: list[str] | None = None,
    *,
    collision_rate_scale: str = "percent",
    strict: bool = False,
) -> dict[str, float]:
    keys = keys or DEFAULT_METRIC_KEYS
    if not metrics_file.exists():
        if strict:
            raise FileNotFoundError(f"Benchmark checkpoint is missing: {metrics_file}")
        return {k: 0.0 for k in keys}

    payload: dict[str, Any] = json.loads(metrics_file.read_text(encoding="utf-8"))
    candidates = _collect_metric_candidates(payload, collision_rate_scale=collision_rate_scale)
    if strict:
        missing = [key for key in keys if key not in candidates]
        if missing:
            raise ValueError(f"Benchmark checkpoint lacks required official metrics: {', '.join(missing)}")
    return {key: float(candidates.get(key, 0.0)) for key in keys}


def _collect_metric_candidates(payload: dict[str, Any], *, collision_rate_scale: str = "percent") -> dict[str, float]:
    candidates: dict[str, float] = {}
    candidates.update(_extract_direct_metrics(payload))
    candidates.update(_extract_label_value_metrics(payload))
    candidates.update(
        _extract_bench2drive_checkpoint_metrics(
            payload,
            collision_rate_scale=collision_rate_scale,
        )
    )
    return candidates


def _extract_direct_metrics(payload: dict[str, Any]) -> dict[str, float]:
    result: dict[str, float] = {}

    for key, value in payload.items():
        if isinstance(value, (int, float)):
            result[str(key)] = float(value)

    nested_metrics = payload.get("metrics")
    if isinstance(nested_metrics, dict):
        for key, value in nested_metrics.items():
            if isinstance(value, (int, float)):
                result[str(key)] = float(value)

    summary = payload.get("summary")
    if isinstance(summary, dict):
        for key, value in summary.items():
            if isinstance(value, (int, float)):
                result[str(key)] = float(value)

    return result


def _extract_label_value_metrics(payload: dict[str, Any]) -> dict[str, float]:
    labels = payload.get("labels")
    values = payload.get("values")
    if not isinstance(labels, list) or not isinstance(values, list):
        return {}

    result: dict[str, float] = {}
    for label, value in zip(labels, values):
        metric_key = BENCH2DRIVE_LABEL_MAP.get(str(label))
        if not metric_key:
            continue
        result[metric_key] = _normalize_rate(value)
    return result


def _extract_bench2drive_checkpoint_metrics(
    payload: dict[str, Any],
    *,
    collision_rate_scale: str = "percent",
) -> dict[str, float]:
    checkpoint = payload.get("_checkpoint")
    if not isinstance(checkpoint, dict):
        return {}

    result: dict[str, float] = {}
    global_record = checkpoint.get("global_record", {})
    if isinstance(global_record, dict):
        scores_mean = global_record.get("scores_mean", {})
        if isinstance(scores_mean, dict):
            if "score_composed" in scores_mean:
                result["DS"] = float(scores_mean["score_composed"])
            if "score_route" in scores_mean:
                result["RC"] = float(scores_mean["score_route"])
            if "score_penalty" in scores_mean:
                result["IP"] = float(scores_mean["score_penalty"])

    records = checkpoint.get("records", [])
    if isinstance(records, list) and records:
        total_records = len(records)
        successful_records = 0
        collision_records = 0
        for record in records:
            if not isinstance(record, dict):
                continue
            if _is_success_record(record):
                successful_records += 1
            if _has_collision(record):
                collision_records += 1

        result["SR"] = 100.0 * successful_records / total_records
        collision_rate = collision_records / total_records
        if str(collision_rate_scale).strip().lower() in {"fraction", "ratio", "unit"}:
            result["CR"] = collision_rate
        else:
            result["CR"] = 100.0 * collision_rate

    if "driving score" in payload:
        result["DS"] = _normalize_rate(payload["driving score"])
    if "success rate" in payload:
        result["SR"] = _normalize_rate(payload["success rate"])

    return result


def _is_success_record(record: dict[str, Any]) -> bool:
    status = str(record.get("status", ""))
    if status not in {"Completed", "Perfect"}:
        return False

    infractions = record.get("infractions", {})
    if not isinstance(infractions, dict):
        return False

    for key, value in infractions.items():
        if key == "min_speed_infractions":
            continue
        if isinstance(value, list) and value:
            return False
    return True


def _has_collision(record: dict[str, Any]) -> bool:
    infractions = record.get("infractions", {})
    if not isinstance(infractions, dict):
        return False

    for key in COLLISION_KEYS:
        value = infractions.get(key, [])
        if isinstance(value, list) and value:
            return True
    return False


def _normalize_rate(value: Any) -> float:
    numeric = float(value)
    return numeric * 100.0 if 0.0 <= numeric <= 1.0 else numeric
