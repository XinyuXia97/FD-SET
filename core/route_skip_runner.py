"""Run a Bench2Drive XML one route at a time without mutating its source checkpoint."""
from __future__ import annotations

import hashlib
import json
import math
import os
import shlex
import signal
import socket
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
from .evidence_retention import append_retention_receipts, compact_new_route_evidence


_MANIFEST_NAME = "route_skip_manifest.json"
_SERVER_STARTUP_TIMEOUT_S = 180
_SERVER_CONNECT_TIMEOUT_S = 5
_SERVER_LOSS_GRACE_S = 30
_DISK_GUARD_PERCENT = 95
_EXECUTION_ERROR_MARKERS = (
    "simulation crashed",
    "agent crashed",
    "agent couldn't be set up",
    "agent could not be set up",
    "sensors were invalid",
    "sensor configuration invalid",
)
_ROUTE_FAILURE_LOG_MARKERS = (
    "failed to connect to carla",
    "unable to connect to carla",
    "connection refused",
    "connection reset",
    "connection aborted",
    "server is not responding",
    "carla server has shut down",
    "carla server is not running",
    "rpc error",
    "rpc timeout",
    "timeout while waiting for the server",
    "timed out waiting for the server",
    "time-out of 600s",
    "timeout of 600s",
    "timed out while waiting for the simulator",
    "time-out while waiting for the simulator",
)


def run_remaining_routes(
    command: str,
    *,
    bench_root: Path,
    scenario_file: Path,
    output_dir: Path,
    environment: Dict[str, str],
    host: str,
    port: int,
    server_launcher: Path,
    server_gpu_adapter: int,
    route_timeout_s: int = 1800,
    failure_extract_config: Optional[Dict[str, Any]] = None,
    evidence_max_images: int = 3,
) -> Dict[str, Any]:
    """Score remaining routes independently and durably record unscored skips.

    The checkpoint named in ``command`` is an immutable input. Each route gets a
    fresh evaluator checkpoint under ``output_dir/route_segments``; the segment
    checkpoint and attempt log are the audit evidence for that route.
    """
    if route_timeout_s < 1:
        raise ValueError("route_timeout_s must be positive")
    if port < 1 or port >= 65535:
        raise ValueError("port must allow a CARLA RPC port and its streaming port")
    if server_gpu_adapter < 0:
        raise ValueError("server_gpu_adapter must be non-negative")

    bench_root = Path(bench_root).expanduser().resolve()
    scenario_file = Path(scenario_file).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    server_launcher = Path(server_launcher).expanduser().resolve()
    if not scenario_file.is_file():
        raise FileNotFoundError("Bench2Drive routes XML is missing: {}".format(scenario_file))

    route_ids = _read_route_ids(scenario_file)
    expected = len(route_ids)
    args = shlex.split(command)
    if not args:
        raise ValueError("Bench2Drive evaluator command is empty")
    source_checkpoint = _checkpoint_argument(args).expanduser().resolve()
    if source_checkpoint.exists() and not source_checkpoint.is_file():
        raise ValueError("Original checkpoint path is not a file: {}".format(source_checkpoint))
    if _routes_argument(args).expanduser().resolve() != scenario_file:
        raise ValueError("Evaluator --routes does not match scenario_file")

    if source_checkpoint.is_file():
        source_payload = json.loads(source_checkpoint.read_text(encoding="utf-8"))
        source_digest = _sha256_file(source_checkpoint)
    else:
        # Fresh rounds have no original checkpoint; only segments hold results.
        source_payload = {"_checkpoint": {"progress": [0, expected], "records": []}}
        source_digest = None
    prefix_count, scored_records, source_skips = _read_scored_prefix(
        source_payload, route_ids, source_checkpoint
    )

    _check_disk_guard(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    segment_root = output_dir / "route_segments"
    segment_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / _MANIFEST_NAME
    manifest = _load_or_create_manifest(
        manifest_path,
        source_checkpoint,
        source_digest,
        scenario_file,
        expected,
        source_skips,
    )

    completed = {item["index"]: item for item in manifest["completed"]}
    skipped = {item["index"]: item for item in manifest["skipped"]}
    _validate_manifest_routes(completed, skipped, route_ids, prefix_count)

    # The existing checkpoint is a route-order prefix. A failed simulator record
    # is retained in that file as raw evidence, but is never surfaced as a score.
    for item in source_skips:
        skipped[item["index"]] = item
    expected_prefix_skips = {item["index"] for item in source_skips}
    recorded_prefix_skips = {index for index in skipped if index < prefix_count}
    if recorded_prefix_skips != expected_prefix_skips:
        raise ValueError("Route skip manifest conflicts with the original checkpoint's scored prefix")
    _store_outcomes(manifest, completed, skipped)
    _atomic_write_json(manifest_path, manifest)

    runtime_env = os.environ.copy()
    runtime_env.update({str(key): str(value) for key, value in environment.items()})
    evidence_root: Optional[Path] = None
    preexisting_evidence: set[str] = set()
    if failure_extract_config is not None:
        if not runtime_env.get("SAVE_PATH"):
            raise RuntimeError("Evidence retention requires a configured SAVE_PATH")
        evidence_root = Path(runtime_env["SAVE_PATH"]).resolve()
        if evidence_root.is_dir():
            preexisting_evidence = {path.name for path in evidence_root.iterdir() if path.is_dir()}
    owned_server: Optional[subprocess.Popen] = None
    server_events: List[Dict[str, Any]] = manifest.setdefault("server_events", [])
    external_failure_streak = 0

    try:
        for index in range(prefix_count, expected):
            if index in skipped:
                continue
            preferred_checkpoint = completed.get(index, {}).get("checkpoint")
            completed_path = _find_completed_segment_path(
                segment_root, index, route_ids[index], preferred_checkpoint
            )
            if completed_path is not None:
                record = _read_segment_record(completed_path, route_ids[index])
                if record is None or not _valid_scored_record(
                    record, "RouteScenario_{}_rep0".format(route_ids[index])
                ):
                    raise RuntimeError(
                        "Completed route {} has no valid segment checkpoint; refusing to rerun or fabricate it".format(
                            route_ids[index]
                        )
                    )
                _mark_recovered_attempt(manifest, index, completed_path, output_dir)
                completed[index] = {
                    "index": index,
                    "route_id": route_ids[index],
                    "checkpoint": _relative_or_absolute(completed_path, output_dir),
                    "status": str(record.get("status", "")),
                }
                _store_outcomes(manifest, completed, skipped)
                _atomic_write_json(manifest_path, manifest)
                scored_records.append({"index": index, "record": record})
                external_failure_streak = 0
                continue
            if index in completed:
                raise RuntimeError(
                    "Completed route {} has no valid segment checkpoint; refusing to rerun or fabricate it".format(
                        route_ids[index]
                    )
                )

            _check_disk_guard(output_dir)
            owned_server = _ensure_server_before_route(
                owned_server,
                host,
                port,
                server_launcher,
                server_gpu_adapter,
                runtime_env,
                output_dir,
                server_events,
                manifest,
                manifest_path,
            )
            attempt_number = _next_attempt_number(manifest["attempts"], index)
            attempt_dir = segment_root / "route_{:04d}_{}".format(index, _safe_path_component(route_ids[index])) / (
                "attempt_{:03d}".format(attempt_number)
            )
            attempt_dir.mkdir(parents=True, exist_ok=True)
            checkpoint_path = attempt_dir / "ckpt.json"
            debug_path = attempt_dir / "live_results.txt"
            log_path = attempt_dir / "evaluation.log"
            attempt = {
                "index": index,
                "route_id": route_ids[index],
                "attempt": attempt_number,
                "status": "running",
                "started_at": _utc_now(),
                "segment_dir": _relative_or_absolute(attempt_dir, output_dir),
                "checkpoint": _relative_or_absolute(checkpoint_path, output_dir),
                "log": _relative_or_absolute(log_path, output_dir),
            }
            manifest["attempts"].append(attempt)
            _atomic_write_json(manifest_path, manifest)

            route_command = _route_command(
                args, checkpoint_path, debug_path, route_ids[index]
            )
            try:
                timed_out, return_code = _run_evaluator(
                    route_command,
                    bench_root,
                    runtime_env,
                    log_path,
                    route_timeout_s,
                    output_dir,
                    host,
                    port,
                )
            except Exception as exc:
                attempt.update({
                    "status": "error",
                    "finished_at": _utc_now(),
                    "reason": str(exc),
                })
                _atomic_write_json(manifest_path, manifest)
                raise
            server_state = _probe_server(host, port, runtime_env, bench_root)
            record = _read_segment_record(checkpoint_path, route_ids[index])
            skip_reason = _execution_error_reason(record)
            if skip_reason is None and record is None:
                skip_reason = _non_scored_attempt_reason(
                    timed_out, route_timeout_s, log_path, server_state
                )

            if record is not None and skip_reason is None:
                if not _valid_scored_record(record, "RouteScenario_{}_rep0".format(route_ids[index])):
                    skip_reason = "segment checkpoint has no valid measured route score"
                else:
                    attempt.update({
                        "status": "completed",
                        "finished_at": _utc_now(),
                        "returncode": return_code,
                        "entry_status": _checkpoint_entry_status(checkpoint_path),
                    })
                    completed[index] = {
                        "index": index,
                        "route_id": route_ids[index],
                        "checkpoint": _relative_or_absolute(checkpoint_path, output_dir),
                        "status": str(record.get("status", "")),
                    }
                    scored_records.append({"index": index, "record": record})
                    external_failure_streak = 0
                    _store_outcomes(manifest, completed, skipped)
                    _atomic_write_json(manifest_path, manifest)
                    if evidence_root is not None:
                        receipts = compact_new_route_evidence(
                            evidence_root=evidence_root,
                            route_id=route_ids[index],
                            checkpoint=checkpoint_path,
                            record=record,
                            failure_config=failure_extract_config,
                            preexisting_names=preexisting_evidence,
                            max_images=evidence_max_images,
                        )
                        append_retention_receipts(output_dir / "evidence_retention.jsonl", receipts)
                    continue

            if skip_reason is None:
                attempt.update({
                    "status": "error",
                    "finished_at": _utc_now(),
                    "returncode": return_code,
                    "reason": "evaluator exited without evidence of a route-specific non-scored failure",
                })
                _atomic_write_json(manifest_path, manifest)
                raise RuntimeError(
                    "Route {} evaluator did not produce a scored record and no route-specific failure was proven; "
                    "inspect {}".format(route_ids[index], log_path)
                )

            skipped_item = {
                "index": index,
                "route_id": route_ids[index],
                "reason": skip_reason,
            }
            if record is not None:
                skipped_item["checkpoint"] = _relative_or_absolute(checkpoint_path, output_dir)
            attempt.update({
                "status": "skipped",
                "finished_at": _utc_now(),
                "returncode": return_code,
                "reason": skip_reason,
                "server_state_after_attempt": server_state,
            })
            skipped[index] = skipped_item
            _store_outcomes(manifest, completed, skipped)
            _atomic_write_json(manifest_path, manifest)
            if evidence_root is not None:
                receipts = compact_new_route_evidence(
                    evidence_root=evidence_root,
                    route_id=route_ids[index],
                    checkpoint=checkpoint_path,
                    record=None,
                    failure_config=failure_extract_config,
                    preexisting_names=preexisting_evidence,
                    max_images=evidence_max_images,
                )
                append_retention_receipts(output_dir / "evidence_retention.jsonl", receipts)

            restart_required = bool(
                timed_out
                or server_state != "healthy"
                or (record is not None and _requires_server_restart(record))
                or _is_simulator_crash_reason(skip_reason)
            )
            if restart_required:
                if owned_server is None and server_state == "healthy":
                    external_failure_streak += 1
                    if external_failure_streak > 1:
                        server_events.append({
                            "time": _utc_now(),
                            "event": "stopped_after_repeated_failure_with_external_server",
                            "route_id": route_ids[index],
                            "reason": "healthy caller-owned CARLA cannot be restarted safely",
                        })
                        _atomic_write_json(manifest_path, manifest)
                        raise RuntimeError(
                            "Route {} failed while the unowned CARLA server remained healthy after an earlier "
                            "route failure; refusing to skip more routes without a safe server restart".format(
                                route_ids[index]
                            )
                        )
                else:
                    external_failure_streak = 0
                owned_server = _restart_after_route_failure(
                    owned_server,
                    host,
                    port,
                    server_launcher,
                    server_gpu_adapter,
                    runtime_env,
                    output_dir,
                    server_events,
                    manifest,
                    manifest_path,
                    server_state,
                    route_ids[index],
                )
                if owned_server is not None:
                    external_failure_streak = 0
        # The source checkpoint is input-only. Detect an external mutation rather
        # than returning a result based on an unstable prefix.
        if (source_digest is None and source_checkpoint.exists()) or (
            source_digest is not None
            and (not source_checkpoint.is_file() or _sha256_file(source_checkpoint) != source_digest)
        ):
            raise RuntimeError("Original Bench2Drive checkpoint changed while route segments were running")
        _check_disk_guard(output_dir)
        scored_records.sort(key=lambda item: item["index"])
        result_skips = sorted(skipped.values(), key=lambda item: item["index"])
        return {"records": scored_records, "skipped": result_skips, "expected": expected}
    finally:
        if sys.exc_info()[0] is not None:
            _stop_owned_server(owned_server)


def _read_route_ids(scenario_file: Path) -> List[str]:
    root = ET.parse(str(scenario_file)).getroot()
    route_ids: List[str] = []
    for route in root.iter("route"):
        route_id = route.get("id")
        if route_id is None or not route_id.strip():
            raise ValueError("Every route in {} must have a non-empty id".format(scenario_file))
        # The official RouteParser sorts subset IDs numerically; reject IDs it
        # cannot process before creating any route attempts.
        try:
            int(route_id)
        except ValueError as exc:
            raise ValueError("Official --routes-subset requires numeric route IDs: {!r}".format(route_id)) from exc
        route_ids.append(route_id)
    if not route_ids:
        raise ValueError("No routes in Bench2Drive XML: {}".format(scenario_file))
    if len(set(route_ids)) != len(route_ids):
        raise ValueError("Bench2Drive XML contains duplicate route IDs")
    return route_ids


def _read_scored_prefix(
    payload: Dict[str, Any], route_ids: Sequence[str], source_checkpoint: Path
) -> Tuple[int, List[Dict[str, Any]], List[Dict[str, Any]]]:
    checkpoint = payload.get("_checkpoint")
    if not isinstance(checkpoint, dict):
        raise ValueError("Original checkpoint has no _checkpoint object: {}".format(source_checkpoint))
    progress = checkpoint.get("progress")
    records = checkpoint.get("records")
    if (not isinstance(progress, list) or len(progress) != 2
            or not all(isinstance(value, int) and not isinstance(value, bool) for value in progress)):
        raise ValueError("Original checkpoint has invalid _checkpoint.progress")
    completed_count, expected = progress
    if expected != len(route_ids) or not 0 <= completed_count <= expected:
        raise ValueError(
            "Original checkpoint progress {!r} does not match XML route count {}".format(progress, len(route_ids))
        )
    if not isinstance(records, list) or len(records) != completed_count:
        raise ValueError(
            "Original checkpoint progress/records disagree: progress={}, records={}".format(
                completed_count, len(records) if isinstance(records, list) else "invalid"
            )
        )

    xml_index = {route_id: index for index, route_id in enumerate(route_ids)}
    by_index: Dict[int, Dict[str, Any]] = {}
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("Original checkpoint contains a non-object route record")
        route_index = _route_record_index(record, xml_index)
        if route_index in by_index:
            raise ValueError("Original checkpoint contains duplicate route records for XML index {}".format(route_index))
        by_index[route_index] = record
    if set(by_index) != set(range(completed_count)):
        raise ValueError("Original checkpoint records are not a complete XML-indexed prefix")

    scored: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []
    for index in range(completed_count):
        record = by_index[index]
        reason = _execution_error_reason(record)
        if reason is not None:
            skipped.append({
                "index": index,
                "route_id": route_ids[index],
                "reason": "original checkpoint contains a non-scored execution failure: {}".format(reason),
                "checkpoint": str(source_checkpoint),
            })
        elif not _valid_scored_record(record, "RouteScenario_{}_rep0".format(route_ids[index])):
            raise ValueError(
                "Original checkpoint record for route {} lacks valid measured scores".format(route_ids[index])
            )
        else:
            scored.append({"index": index, "record": record})
    return completed_count, scored, skipped


def _route_record_index(record: Dict[str, Any], xml_index: Dict[str, int]) -> int:
    route_id = record.get("route_id")
    if not isinstance(route_id, str):
        raise ValueError("Checkpoint route record is missing route_id")
    prefix = "RouteScenario_"
    separator = "_rep"
    if not route_id.startswith(prefix) or separator not in route_id:
        raise ValueError("Unrecognized Bench2Drive checkpoint route_id: {!r}".format(route_id))
    route_name, repetition = route_id[len(prefix):].rsplit(separator, 1)
    if repetition != "0" or route_name not in xml_index:
        raise ValueError("Checkpoint route does not match XML repetition 0: {!r}".format(route_id))
    return xml_index[route_name]


def _valid_scored_record(record: Dict[str, Any], expected_route_id: str) -> bool:
    if record.get("route_id") != expected_route_id:
        return False
    if not record.get("status") or record.get("status") == "Started":
        return False
    if _execution_error_reason(record) is not None:
        return False
    scores = record.get("scores")
    if not isinstance(scores, dict):
        return False
    for name in ("score_route", "score_penalty", "score_composed"):
        value = scores.get(name)
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(float(value)):
            return False
    return True


def _execution_error_reason(record: Optional[Dict[str, Any]]) -> Optional[str]:
    if record is None:
        return None
    status = str(record.get("status", "")).strip()
    lowered = status.lower()
    for marker in _EXECUTION_ERROR_MARKERS:
        if marker in lowered:
            return status or marker
    return None


def _requires_server_restart(record: Dict[str, Any]) -> bool:
    status = str(record.get("status", "")).lower()
    return "simulation crashed" in status


def _checkpoint_argument(args: Sequence[str]) -> Path:
    value = _option_value(args, "--checkpoint")
    if value is None:
        raise ValueError("Evaluator command must include --checkpoint")
    return Path(value)


def _routes_argument(args: Sequence[str]) -> Path:
    value = _option_value(args, "--routes")
    if value is None:
        raise ValueError("Evaluator command must include --routes")
    return Path(value)


def _option_value(args: Sequence[str], option: str) -> Optional[str]:
    for index, arg in enumerate(args):
        if arg == option:
            if index + 1 >= len(args):
                raise ValueError("Evaluator command has no value after {}".format(option))
            return args[index + 1]
        if arg.startswith(option + "="):
            return arg.split("=", 1)[1]
    return None


def _route_command(
    original: Sequence[str], checkpoint_path: Path, debug_path: Path, route_id: str
) -> List[str]:
    args: List[str] = []
    index = 0
    replaced_checkpoint = False
    replaced_debug = False
    while index < len(original):
        arg = original[index]
        if arg == "--resume":
            # The official evaluator's type=bool parser treats the string
            # "False" as true; remove the option rather than forwarding it.
            index += 1
            if index < len(original) and not original[index].startswith("--"):
                index += 1
            continue
        if arg.startswith("--resume="):
            index += 1
            continue
        if arg in ("--checkpoint", "--debug-checkpoint", "--routes-subset"):
            option = arg
            if index + 1 >= len(original):
                raise ValueError("Evaluator command has no value after {}".format(option))
            if option == "--checkpoint":
                args.extend([option, str(checkpoint_path)])
                replaced_checkpoint = True
            elif option == "--debug-checkpoint":
                args.extend([option, str(debug_path)])
                replaced_debug = True
            # Remove any template subset; each invocation is exactly one route.
            index += 2
            continue
        if arg.startswith("--checkpoint="):
            args.append("--checkpoint={}".format(checkpoint_path))
            replaced_checkpoint = True
            index += 1
            continue
        if arg.startswith("--debug-checkpoint="):
            args.append("--debug-checkpoint={}".format(debug_path))
            replaced_debug = True
            index += 1
            continue
        if arg.startswith("--routes-subset="):
            index += 1
            continue
        args.append(arg)
        index += 1
    if not replaced_checkpoint:
        args.extend(["--checkpoint", str(checkpoint_path)])
    if not replaced_debug:
        args.extend(["--debug-checkpoint", str(debug_path)])
    args.extend(["--routes-subset", route_id])
    return args


def _read_segment_record(checkpoint_path: Path, route_id: str) -> Optional[Dict[str, Any]]:
    if not checkpoint_path.is_file():
        return None
    try:
        payload = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        checkpoint = payload.get("_checkpoint")
        if not isinstance(checkpoint, dict):
            return None
        progress = checkpoint.get("progress")
        records = checkpoint.get("records")
        if (progress != [1, 1] or not isinstance(records, list) or len(records) != 1
                or not isinstance(records[0], dict)):
            return None
        if records[0].get("route_id") != "RouteScenario_{}_rep0".format(route_id):
            return None
        return records[0]
    except (OSError, ValueError, TypeError):
        return None


def _checkpoint_entry_status(checkpoint_path: Path) -> Optional[str]:
    try:
        payload = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    status = payload.get("entry_status")
    return status if isinstance(status, str) else None


def _find_completed_segment_path(
    segment_root: Path, index: int, route_id: str, preferred: Optional[str]
) -> Optional[Path]:
    candidates: List[Path] = []
    if preferred:
        path = Path(preferred)
        if not path.is_absolute():
            path = segment_root.parent / path
        candidates.append(path)
    route_dir = segment_root / "route_{:04d}_{}".format(index, _safe_path_component(route_id))
    if route_dir.is_dir():
        candidates.extend(sorted(route_dir.glob("attempt_*/ckpt.json"), reverse=True))
    seen = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        record = _read_segment_record(resolved, route_id)
        if record is not None and _valid_scored_record(record, "RouteScenario_{}_rep0".format(route_id)):
            return resolved
    return None


def _non_scored_attempt_reason(
    timed_out: bool,
    timeout_s: int,
    log_path: Path,
    server_state: str,
) -> Optional[str]:
    if timed_out:
        return "route evaluator exceeded its {} second timeout without a scored checkpoint".format(timeout_s)
    if server_state != "healthy":
        return "CARLA server became unavailable during route evaluation ({})".format(server_state)
    try:
        log_text = log_path.read_text(encoding="utf-8", errors="replace").lower()
    except OSError:
        log_text = ""
    for marker in _ROUTE_FAILURE_LOG_MARKERS:
        if marker in log_text:
            return "route-specific simulator connection failure: {}".format(marker)
    return None


def _is_simulator_crash_reason(reason: str) -> bool:
    lowered = reason.lower()
    return any(marker in lowered for marker in (
        "simulation crashed", "carla server", "carla server became unavailable",
        "simulator connection failure", "evaluator exceeded",
    ))


def _run_evaluator(
    args: Sequence[str],
    bench_root: Path,
    environment: Dict[str, str],
    log_path: Path,
    timeout_s: int,
    guard_path: Path,
    host: str,
    port: int,
) -> Tuple[bool, Optional[int]]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    process: Optional[subprocess.Popen] = None
    timed_out = False
    try:
        with log_path.open("w", encoding="utf-8") as log:
            process = subprocess.Popen(
                list(args),
                cwd=str(bench_root),
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            deadline = time.monotonic() + timeout_s
            server_unavailable_since: Optional[float] = None
            while True:
                try:
                    _check_disk_guard(guard_path)
                except Exception:
                    _terminate_process(process)
                    raise
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    timed_out = True
                    _terminate_process(process)
                    break
                try:
                    process.wait(timeout=min(1.0, remaining))
                    break
                except subprocess.TimeoutExpired:
                    try:
                        with socket.create_connection((host, port), timeout=0.5):
                            server_unavailable_since = None
                    except OSError:
                        now = time.monotonic()
                        if server_unavailable_since is None:
                            server_unavailable_since = now
                        elif now - server_unavailable_since >= _SERVER_LOSS_GRACE_S:
                            # The caller checks the server again before recording
                            # a skipped route; never manufacture a missing score.
                            _terminate_process(process)
                            break
        _terminate_process(process)
        return timed_out, process.returncode
    except BaseException:
        if process is not None and process.poll() is None:
            _terminate_process(process)
        raise


def _terminate_process(process: subprocess.Popen) -> None:
    # The evaluator may spawn worker processes; it owns a dedicated session.
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        return
    process.wait()


def _probe_server(host: str, port: int, environment: Dict[str, str], cwd: Path) -> str:
    """Return healthy, unbound, or bound_unhealthy without taking server ownership."""
    try:
        with socket.create_connection((host, port), timeout=1.0):
            pass
    except OSError:
        return "unbound"

    probe = (
        "import carla, sys; "
        "client = carla.Client(sys.argv[1], int(sys.argv[2])); "
        "client.set_timeout({}); "
        "print(client.get_server_version())"
    ).format(_SERVER_CONNECT_TIMEOUT_S)
    try:
        result = subprocess.run(
            [sys.executable, "-c", probe, host, str(port)],
            cwd=str(cwd),
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=_SERVER_CONNECT_TIMEOUT_S + 3,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return "bound_unhealthy"
    return "healthy" if result.returncode == 0 and result.stdout.strip() else "bound_unhealthy"



def _ensure_server_before_route(
    owned_server: Optional[subprocess.Popen],
    host: str,
    port: int,
    launcher: Path,
    gpu_adapter: int,
    environment: Dict[str, str],
    output_dir: Path,
    events: List[Dict[str, Any]],
    manifest: Dict[str, Any],
    manifest_path: Path,
) -> Optional[subprocess.Popen]:
    state = _probe_server(host, port, environment, launcher.parent)
    if state == "healthy":
        if owned_server is not None and owned_server.poll() is not None:
            owned_server = None
        return owned_server
    if state == "bound_unhealthy":
        if owned_server is None:
            raise RuntimeError(
                "CARLA port {}:{} remains bound but unhealthy; refusing to kill an unowned server".format(host, port)
            )
        _stop_owned_server(owned_server)
        owned_server = None
    elif owned_server is not None:
        _stop_owned_server(owned_server)
        owned_server = None
    return _launch_owned_server(
        host, port, launcher, gpu_adapter, environment, output_dir,
        events, manifest, manifest_path, "CARLA unavailable before route",
    )


def _restart_after_route_failure(
    owned_server: Optional[subprocess.Popen],
    host: str,
    port: int,
    launcher: Path,
    gpu_adapter: int,
    environment: Dict[str, str],
    output_dir: Path,
    events: List[Dict[str, Any]],
    manifest: Dict[str, Any],
    manifest_path: Path,
    server_state: str,
    failed_route_id: str,
) -> Optional[subprocess.Popen]:
    if owned_server is None and server_state == "healthy":
        # Preserve a healthy caller-owned server. If it has actually died or
        # stopped answering RPC, the unbound/unhealthy path below launches the
        # isolated server without ever terminating an external process.
        events.append({
            "time": _utc_now(),
            "event": "retained_healthy_external_server_after_route_failure",
            "route_id": failed_route_id,
            "reason": "server remained RPC healthy and is not owned by this runner",
        })
        _atomic_write_json(manifest_path, manifest)
        return None
    if owned_server is not None:
        _stop_owned_server(owned_server)
        owned_server = None
    state = _probe_server(host, port, environment, launcher.parent)
    if state == "healthy":
        events.append({
            "time": _utc_now(),
            "event": "using_recovered_healthy_server",
            "route_id": failed_route_id,
            "reason": "a healthy CARLA server was already available after the owned server stopped",
        })
        _atomic_write_json(manifest_path, manifest)
        return None
    if state == "bound_unhealthy":
        raise RuntimeError(
            "CARLA port {}:{} is bound but unhealthy after route {}; refusing to kill an unowned server".format(
                host, port, failed_route_id
            )
        )
    return _launch_owned_server(
        host, port, launcher, gpu_adapter, environment, output_dir,
        events, manifest, manifest_path,
        "route {} failed and CARLA was {}".format(failed_route_id, server_state),
    )


def _launch_owned_server(
    host: str,
    port: int,
    launcher: Path,
    gpu_adapter: int,
    environment: Dict[str, str],
    output_dir: Path,
    events: List[Dict[str, Any]],
    manifest: Dict[str, Any],
    manifest_path: Path,
    reason: str,
) -> subprocess.Popen:
    if not launcher.is_file():
        raise FileNotFoundError("Patched CARLA launcher is missing: {}".format(launcher))
    _check_disk_guard(output_dir)
    server_logs = output_dir / "server_restarts"
    server_logs.mkdir(parents=True, exist_ok=True)
    launch_number = 1 + sum(1 for event in events if event.get("event") == "launched_owned_server")
    log_path = server_logs / "carla_{:03d}.log".format(launch_number)
    args = [
        str(launcher),
        "-RenderOffScreen",
        "-nosound",
        "-carla-rpc-port={}".format(port),
        "-carla-streaming-port={}".format(port + 1),
        "-quality-level=Low",
        "-graphicsadapter={}".format(gpu_adapter),
    ]
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            args,
            cwd=str(launcher.parent),
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    try:
        event = {
            "time": _utc_now(),
            "event": "launched_owned_server",
            "pid": process.pid,
            "process_group": process.pid,
            "host": host,
            "port": port,
            "streaming_port": port + 1,
            "graphicsadapter": gpu_adapter,
            "launcher": str(launcher),
            "log": _relative_or_absolute(log_path, output_dir),
            "reason": reason,
        }
        events.append(event)
        _atomic_write_json(manifest_path, manifest)

        deadline = time.monotonic() + _SERVER_STARTUP_TIMEOUT_S
        last_state = "unbound"
        while time.monotonic() < deadline:
            _check_disk_guard(output_dir)
            if process.poll() is not None:
                raise RuntimeError(
                    "Patched CARLA exited during startup with code {}; inspect {}".format(process.returncode, log_path)
                )
            last_state = _probe_server(host, port, environment, launcher.parent)
            if last_state == "healthy":
                event["ready_at"] = _utc_now()
                _atomic_write_json(manifest_path, manifest)
                return process
            time.sleep(1.0)
        raise RuntimeError(
            "Patched CARLA did not become RPC healthy within {} seconds (state={}); inspect {}".format(
                _SERVER_STARTUP_TIMEOUT_S, last_state, log_path
            )
        )
    except BaseException:
        _stop_owned_server(process)
        raise


def _stop_owned_server(process: Optional[subprocess.Popen]) -> None:
    if process is None:
        return
    pgid = process.pid
    try:
        os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        return
    except PermissionError as exc:
        raise RuntimeError("Cannot stop runner-owned CARLA process group {}: {}".format(pgid, exc)) from exc
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return
    try:
        os.killpg(pgid, signal.SIGKILL)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=2)
    except subprocess.TimeoutExpired:
        pass


def _check_disk_guard(path: Path) -> None:
    existing = path
    while not existing.exists() and existing != existing.parent:
        existing = existing.parent
    try:
        stats = os.statvfs(str(existing))
    except OSError as exc:
        raise RuntimeError("Cannot read filesystem usage for the disk guard: {}".format(exc)) from exc
    used = stats.f_blocks - stats.f_bfree
    available = stats.f_bavail
    percent = 100 if used + available == 0 else int(math.ceil(100.0 * used / (used + available)))
    if percent >= _DISK_GUARD_PERCENT:
        raise RuntimeError(
            "Global disk guard stopped route evaluation at {}% usage (threshold {}%)".format(
                percent, _DISK_GUARD_PERCENT
            )
        )
    for ancestor in (path,) + tuple(path.parents):
        alert_path = ancestor / "disk_alert.json"
        if not alert_path.is_file():
            continue
        try:
            alert = json.loads(alert_path.read_text(encoding="utf-8"))
            used_percent = int(alert.get("used_percent", 0))
            threshold = int(alert.get("threshold_percent", _DISK_GUARD_PERCENT))
        except (OSError, ValueError, TypeError, AttributeError):
            raise RuntimeError("Global disk guard alert is present but unreadable: {}".format(alert_path))
        if used_percent >= threshold and threshold >= _DISK_GUARD_PERCENT:
            raise RuntimeError(
                "Global disk guard alert {} records {}% usage (threshold {}%)".format(
                    alert_path, used_percent, threshold
                )
            )


def _load_or_create_manifest(
    path: Path,
    source_checkpoint: Path,
    source_digest: str,
    scenario_file: Path,
    expected: int,
    source_skips: List[Dict[str, Any]],
) -> Dict[str, Any]:
    if path.exists():
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
            raise ValueError("Unsupported route skip manifest: {}".format(path))
        if manifest.get("source_checkpoint") != str(source_checkpoint):
            raise ValueError("Route skip manifest references a different source checkpoint")
        if manifest.get("source_sha256") != source_digest:
            raise ValueError("Original checkpoint changed since route skip manifest was written")
        if manifest.get("scenario_file") != str(scenario_file) or manifest.get("expected") != expected:
            raise ValueError("Route skip manifest does not match this XML evaluation")
        for key in ("attempts", "completed", "skipped", "server_events"):
            if not isinstance(manifest.get(key), list):
                raise ValueError("Route skip manifest field {!r} is invalid".format(key))
        for key in ("completed", "skipped"):
            seen_indices = set()
            for item in manifest[key]:
                if (not isinstance(item, dict)
                        or not isinstance(item.get("index"), int)
                        or isinstance(item.get("index"), bool)):
                    raise ValueError("Route skip manifest field {!r} has an invalid outcome".format(key))
                if item["index"] in seen_indices:
                    raise ValueError("Route skip manifest field {!r} has duplicate indices".format(key))
                seen_indices.add(item["index"])
        for attempt in manifest["attempts"]:
            if (not isinstance(attempt, dict)
                    or not isinstance(attempt.get("index"), int)
                    or isinstance(attempt.get("index"), bool)
                    or not isinstance(attempt.get("attempt"), int)
                    or isinstance(attempt.get("attempt"), bool)):
                raise ValueError("Route skip manifest has an invalid attempt record")
        return manifest
    return {
        "schema_version": 1,
        "created_at": _utc_now(),
        "source_checkpoint": str(source_checkpoint),
        "source_sha256": source_digest,
        "scenario_file": str(scenario_file),
        "expected": expected,
        "attempts": [],
        "completed": [],
        "skipped": list(source_skips),
        "server_events": [],
    }


def _validate_manifest_routes(
    completed: Dict[int, Dict[str, Any]],
    skipped: Dict[int, Dict[str, Any]],
    route_ids: Sequence[str],
    prefix_count: int,
) -> None:
    if set(completed).intersection(skipped):
        raise ValueError("Route skip manifest marks a route both completed and skipped")
    for index, item in list(completed.items()) + list(skipped.items()):
        if index < 0 or index >= len(route_ids) or item.get("route_id") != route_ids[index]:
            raise ValueError("Route skip manifest outcome does not match XML index {}".format(index))
        if index < prefix_count and index in completed:
            # A source-prefix score is authoritative; duplicated segment outcomes
            # indicate the resume state was built against a different run.
            raise ValueError("Route skip manifest duplicates a route already present in the source prefix")


def _store_outcomes(
    manifest: Dict[str, Any],
    completed: Dict[int, Dict[str, Any]],
    skipped: Dict[int, Dict[str, Any]],
) -> None:
    if set(completed).intersection(skipped):
        raise ValueError("A route cannot be both completed and skipped")
    manifest["completed"] = [completed[index] for index in sorted(completed)]
    manifest["skipped"] = [skipped[index] for index in sorted(skipped)]


def _mark_recovered_attempt(
    manifest: Dict[str, Any], index: int, checkpoint_path: Path, output_dir: Path
) -> None:
    checkpoint_ref = _relative_or_absolute(checkpoint_path, output_dir)
    for attempt in reversed(manifest["attempts"]):
        if attempt.get("index") == index and attempt.get("checkpoint") == checkpoint_ref:
            if attempt.get("status") != "completed":
                attempt["status"] = "completed"
                attempt["recovered_at"] = _utc_now()
                attempt["recovered_from_checkpoint"] = True
            return


def _next_attempt_number(attempts: Sequence[Dict[str, Any]], index: int) -> int:
    return 1 + max(
        [item.get("attempt", 0) for item in attempts if item.get("index") == index and isinstance(item.get("attempt"), int)]
        or [0]
    )


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=".{}.".format(path.name), suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, str(path))
        directory_fd = os.open(str(path.parent), os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except OSError:
            pass
        raise


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_path_component(value: str) -> str:
    return "".join(char if char.isalnum() or char in "-_" else "_" for char in value)


def _relative_or_absolute(path: Path, base: Path) -> str:
    try:
        return str(path.resolve().relative_to(base.resolve()))
    except ValueError:
        return str(path.resolve())


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
