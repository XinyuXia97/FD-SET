#!/usr/bin/env python3
"""Evaluate baseline and Scenic-adapted Zoo agents on the same official 220 routes."""
from __future__ import annotations

import argparse
import json
import math
import os
import signal
import shutil
import socket
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from core.bench2drive_adapter import calculate_official_motion_metrics, parse_metrics  # noqa: E402
from core.closed_loop_runner import build_bench2drive_runtime_env  # noqa: E402

BASELINES = {
    "uniad": "uniad_base_b2d.pth", "vad": "vad_b2d_base.pth",
    "admlp": "admlp_b2d.ckpt", "tcp": "tcp_b2d.ckpt",
}
ADAPTED = {
    "uniad": "uniad_adapted.pth", "vad": "vad_adapted.pth",
    "admlp": "admlp_adapted.ckpt", "tcp": "tcp_adapted.ckpt",
}
CRASH_TRUNCATED_STATUS = "Failed - Crash-truncated (non-official)"
UNSTARTED_STATUS = "Failed - Skipped before first tick (non-official)"
TIMEOUT_STATUS = "Failed - Wall-timeout skipped (non-official)"
UNSCORED_CRASH_STATUS = "Failed - Unscored CARLA crash skipped (non-official)"
TIMEOUT_SCORED_STATUS = "Failed - Wall-timeout partial score (non-official)"


def skip_unstarted_route(checkpoint: Path, snapshot: Path, evidence: Path,
                         ordered_routes: list[ET.Element], route_id: str, reason: str) -> dict:
    """Apply an explicit zero-driving policy only when no route tick exists."""
    payload = json.loads(checkpoint.read_text(encoding="utf-8"))
    state = payload["_checkpoint"]
    index, total = state["progress"]
    expected = f"RouteScenario_{route_id}_rep0"
    if (total != len(ordered_routes) or index >= total
            or ordered_routes[index].get("id") != route_id or len(state["records"]) != index):
        raise ValueError(f"Unstarted skip does not match next route in {checkpoint}")
    if snapshot.is_file() or any(evidence.glob(expected + "_*")):
        raise ValueError(f"Route {expected} has evidence; do not assign unstarted zero")
    route = ordered_routes[index]
    scenario = route.find("scenarios/scenario")
    if scenario is None or not state["records"]:
        raise ValueError(f"Route {expected} lacks route metadata for an auditable skip")
    record = {
        "index": index, "route_id": expected, "scenario_name": scenario.get("name"),
        "weather_id": "not-run",
        "save_name": f"{expected}_{route.get('town')}_{scenario.get('name')}_not-run",
        "status": UNSTARTED_STATUS, "num_infractions": 0,
        "infractions": {key: [] for key in state["records"][-1]["infractions"]},
        "scores": {"score_route": 0, "score_penalty": 1, "score_composed": 0},
        "meta": {"route_length": 0, "duration_game": 0, "duration_system": 0},
    }
    state["records"].append(record)
    state["progress"] = [index + 1, total]
    payload["eligible"] = False
    temporary = checkpoint.with_name(checkpoint.name + ".skip.tmp")
    temporary.write_text(json.dumps(payload, indent=4) + "\n", encoding="utf-8")
    os.replace(temporary, checkpoint)
    audit = {"route_id": expected, "checkpoint": str(checkpoint), "reason": reason,
             "score_source": "explicit policy zero: no CARLA tick or motion evidence",
             "official_eligible": False, "motion_metrics_available": False, "record": record}
    (checkpoint.parent / f"skipped_unstarted_{route_id}.json").write_text(
        json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    return record

def skip_unscored_crash(checkpoint: Path, snapshot: Path, evidence: Path,
                        ordered_routes: list[ET.Element], route_id: str, reason: str) -> dict:
    """Policy-zero an interrupted route only when no genuine live score exists."""
    payload = json.loads(checkpoint.read_text(encoding="utf-8"))
    state = payload["_checkpoint"]
    index, total = state["progress"]
    expected = f"RouteScenario_{route_id}_rep0"
    if (total != len(ordered_routes) or index >= total
            or ordered_routes[index].get("id") != route_id or len(state["records"]) != index):
        raise ValueError(f"Unscored crash does not match next route in {checkpoint}")
    paths = list(evidence.glob(expected + "_*"))
    if snapshot.is_file() or not paths or any((path / "metric_info.json").is_file() for path in paths):
        raise ValueError(f"Route {expected} has genuine score evidence or no partial evidence")
    route = ordered_routes[index]
    scenario = route.find("scenarios/scenario")
    if scenario is None or not state["records"]:
        raise ValueError(f"Route {expected} lacks route metadata for an auditable skip")
    record = {
        "index": index, "route_id": expected, "scenario_name": scenario.get("name"),
        "weather_id": "not-scored",
        "save_name": f"{expected}_{route.get('town')}_{scenario.get('name')}_crash-not-scored",
        "status": UNSCORED_CRASH_STATUS, "num_infractions": 0,
        "infractions": {key: [] for key in state["records"][-1]["infractions"]},
        "scores": {"score_route": 0, "score_penalty": 1, "score_composed": 0},
        "meta": {"route_length": 0, "duration_game": 0, "duration_system": 0},
    }
    audit = {"route_id": expected, "checkpoint": str(checkpoint), "reason": reason,
             "score_source": "explicit policy zero: CARLA crashed without usable live criteria",
             "official_eligible": False, "motion_metrics_available": False,
             "preserved_evidence": [str(path) for path in paths], "record": record}
    (checkpoint.parent / f"skipped_unscored_crash_{route_id}.json").write_text(
        json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    state["records"].append(record)
    state["progress"] = [index + 1, total]
    payload["eligible"] = False
    temporary = checkpoint.with_name(checkpoint.name + ".unscored.tmp")
    temporary.write_text(json.dumps(payload, indent=4) + "\n", encoding="utf-8")
    os.replace(temporary, checkpoint)
    return record


def _accept_live_snapshot(checkpoint: Path, snapshot: Path, evidence: Path,
                          ordered_route_ids: list[str], route_id: str, *,
                          status: str, evidence_prefix: str, audit_name: str,
                          audit_details: dict) -> dict:
    """Preserve Bench2Drive's actual live criteria, never infer scores from camera frames."""
    payload = json.loads(checkpoint.read_text(encoding="utf-8"))
    state = payload["_checkpoint"]
    index, total = state["progress"]
    expected = f"RouteScenario_{route_id}_rep0"
    if (total != len(ordered_route_ids) or index >= total
            or ordered_route_ids[index] != route_id or len(state["records"]) != index):
        raise ValueError(f"Live snapshot does not match the next route in {checkpoint}")
    record = json.loads(snapshot.read_text(encoding="utf-8"))
    name = record.get("save_name")
    scores = record.get("scores", {})
    infractions = record.get("infractions", {})
    meta = record.get("meta", {})
    if (record.get("route_id") != expected or record.get("index") != index
            or not isinstance(name, str) or not name.startswith(expected + "_")
            or Path(name).name != name
            or not (evidence / (evidence_prefix + name) / "metric_info.json").is_file()
            or not isinstance(infractions, dict)
            or not all(isinstance(events, list) for events in infractions.values())
            or not infractions or sum(map(len, infractions.values())) != record.get("num_infractions")
            or not all(isinstance(scores.get(key), (int, float)) and math.isfinite(scores[key])
                       for key in ("score_route", "score_penalty", "score_composed"))
            or not 0 <= scores["score_route"] <= 100
            or not 0 <= scores["score_penalty"] <= 1
            # Bench2Drive rounds route %, penalty, and composed score separately to 6 digits.
            or not math.isclose(scores["score_composed"],
                                scores["score_route"] * scores["score_penalty"],
                                rel_tol=0, abs_tol=5.2e-5)
            or not isinstance(meta.get("duration_game"), (int, float))
            or not math.isfinite(meta["duration_game"]) or meta["duration_game"] <= 0):
        raise ValueError(f"Live snapshot lacks genuine live criteria: {snapshot}")
    if scores["score_route"] <= 0:
        raise ValueError(f"Live snapshot has no measured route completion; retry instead of assigning zero: {snapshot}")
    record["status"] = status
    state["records"].append(record)
    state["progress"] = [index + 1, total]
    payload["eligible"] = False
    temporary = checkpoint.with_name(checkpoint.name + ".live.tmp")
    temporary.write_text(json.dumps(payload, indent=4) + "\n", encoding="utf-8")
    os.replace(temporary, checkpoint)
    audit = {"route_id": expected, "checkpoint": str(checkpoint), "source": str(snapshot),
             "score_source": "Bench2Drive live criteria at last completed tick",
             "official_eligible": False, "record": record, **audit_details}
    (checkpoint.parent / audit_name).write_text(json.dumps(audit, indent=2) + "\n",
                                                 encoding="utf-8")
    return record


def accept_crash_snapshot(checkpoint: Path, snapshot: Path, evidence: Path,
                          ordered_route_ids: list[str], route_id: str, carla_exit: int, *,
                          family: str = "tcp") -> dict:
    if family not in BASELINES:
        raise ValueError(f"Unknown model family: {family}")
    return _accept_live_snapshot(
        checkpoint, snapshot, evidence, ordered_route_ids, route_id,
        status=CRASH_TRUNCATED_STATUS,
        evidence_prefix="bench2drive220_" if family == "vad" else "",
        audit_name=f"crash_truncation_{route_id}.json",
        audit_details={"carla_exit": carla_exit})


def accept_timeout_snapshot(checkpoint: Path, snapshot: Path, evidence: Path,
                            ordered_route_ids: list[str], route_id: str,
                            elapsed_seconds: float, limit_seconds: float, *,
                            family: str = "vad") -> dict:
    """Score a timed-out route only from its matching, measured live criteria."""
    if not math.isfinite(elapsed_seconds) or not math.isfinite(limit_seconds) or (
            limit_seconds <= 0 or elapsed_seconds < limit_seconds):
        raise ValueError("No elapsed wall-clock timeout to score")
    if family not in BASELINES:
        raise ValueError(f"Unknown model family: {family}")
    return _accept_live_snapshot(
        checkpoint, snapshot, evidence, ordered_route_ids, route_id,
        status=TIMEOUT_SCORED_STATUS,
        evidence_prefix="bench2drive220_" if family == "vad" else "",
        audit_name=f"timeout_partial_{route_id}.json",
        audit_details={"elapsed_wall_seconds": elapsed_seconds, "limit_wall_seconds": limit_seconds})


def mark_checkpoint_ineligible(checkpoint: Path) -> None:
    payload = json.loads(checkpoint.read_text(encoding="utf-8"))
    payload["eligible"] = False
    temporary = checkpoint.with_name(checkpoint.name + ".eligibility.tmp")
    temporary.write_text(json.dumps(payload, indent=4) + "\n", encoding="utf-8")
    os.replace(temporary, checkpoint)




def checkpoint_progress(checkpoint: Path) -> int:
    """Read a checkpoint that Bench2Drive may be rewriting in place."""
    for _ in range(19):
        try:
            return json.loads(checkpoint.read_text(encoding="utf-8"))["_checkpoint"]["progress"][0]
        except (OSError, json.JSONDecodeError):
            time.sleep(0.1)
    return json.loads(checkpoint.read_text(encoding="utf-8"))["_checkpoint"]["progress"][0]


def coverage(checkpoint: Path, route_ids: set[str], *,
             allow_non_official: bool = False) -> tuple[int, bool]:
    if not checkpoint.is_file():
        return 0, False
    payload = json.loads(checkpoint.read_text(encoding="utf-8"))
    state = payload.get("_checkpoint", {})
    records = state.get("records", [])
    observed = {str(row.get("route_id", "")) for row in records if isinstance(row, dict)}
    expected = {f"RouteScenario_{route_id}_rep0" for route_id in route_ids}
    progress = state.get("progress", [])
    execution_errors = ("Simulation crashed", "Agent crashed", "sensors were invalid",
                        "Crash-truncated", "Skipped before first tick", "Wall-timeout skipped",
                        "Wall-timeout partial score", "Unscored CARLA crash skipped")
    complete = (
        payload.get("entry_status") == "Finished"
        and observed == expected
        and len(records) == len(expected)
        and progress == [len(expected), len(expected)]
        and not any(
            marker in str(row.get("status", ""))
            and not (allow_non_official and row.get("status")
                     in (CRASH_TRUNCATED_STATUS, UNSTARTED_STATUS, TIMEOUT_STATUS,
                         TIMEOUT_SCORED_STATUS, UNSCORED_CRASH_STATUS)
                     and marker in row["status"])
            for row in records
            for marker in execution_errors
        )
    )
    return len(observed & expected), complete


def prune_completed_evidence(checkpoint: Path, evidence: Path, family: str, route_ids: set[str]) -> None:
    """Keep only scored routes' genuine motion metrics, not gigabytes of debug images."""
    if not checkpoint.is_file():
        return
    try:
        records = json.loads(checkpoint.read_text(encoding="utf-8"))["_checkpoint"]["records"]
    except (OSError, ValueError, KeyError):
        return  # The evaluator may be replacing its checkpoint concurrently.
    valid_routes = {f"RouteScenario_{route_id}_rep0" for route_id in route_ids}
    image_dirs = {"rgb", "rgb_front", "rgb_front_right", "rgb_front_left",
                  "rgb_back", "rgb_back_right", "rgb_back_left", "bev", "meta"}
    for record in records:
        if not isinstance(record, dict) or record.get("route_id") not in valid_routes:
            continue
        name = record.get("save_name")
        if not isinstance(name, str) or not name.startswith(record["route_id"] + "_"):
            continue
        if Path(name).name != name or name in {".", ".."}:
            continue
        actual = evidence / ("bench2drive220_" + name if family == "vad" else name)
        metric = actual / "metric_info.json"
        if not metric.is_file():
            continue
        if family == "vad":
            # The stock VAD agent prefixes route names, but Bench2Drive's stock
            # motion tool looks them up using the unprefixed checkpoint save_name.
            alias = evidence / name
            alias.mkdir(exist_ok=True)
            link = alias / "metric_info.json"
            if not link.exists():
                link.symlink_to(metric)
        for child in actual.iterdir():
            if child.name in image_dirs and child.is_dir() and not child.is_symlink():
                shutil.rmtree(child)


def available(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=2):
            return True
    except OSError:
        return False


def start_carla(launcher: Path, workdir: Path, gpu: int, graphics_adapter: int,
                port: int) -> tuple[subprocess.Popen, object]:
    log = (workdir / "carla.log").open("a", encoding="utf-8")
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
    process = subprocess.Popen(
        [str(launcher), "-RenderOffScreen", "-nosound", f"-carla-rpc-port={port}",
         f"-carla-streaming-port={port + 1}", "-quality-level=Low",
         f"-graphicsadapter={graphics_adapter}"],
        cwd=str(workdir), env=environment, stdout=log, stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    try:
        for _ in range(300):
            if available("127.0.0.1", port):
                return process, log
            if process.poll() is not None:
                raise RuntimeError(f"CARLA exited during startup: {process.returncode}; see {workdir / 'carla.log'}")
            time.sleep(1)
        raise TimeoutError("CARLA did not open its RPC port within 300 seconds")
    except BaseException:
        stop_process(process)
        log.close()
        raise


def stop_process(process: subprocess.Popen) -> None:
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        process.wait()
        return
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        try:
            os.killpg(process.pid, 0)
        except ProcessLookupError:
            break
        time.sleep(0.5)
    else:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    process.wait()


def archive_vad_adapted_for_live_timeouts(output: Path, route_ids: set[str]) -> Path:
    """Preserve policy-zero results before remeasuring only the adapted VAD routes."""
    baseline = output / "vad/baseline/ckpt.json"
    if coverage(baseline, route_ids, allow_non_official=True) != (len(route_ids), True):
        raise ValueError(f"VAD baseline must already be complete: {baseline}")
    adapted = output / "vad/adapted"
    archive = output / "vad/adapted_before_live_timeouts"
    if archive.exists() or not (adapted / "ckpt.json").is_file():
        raise ValueError(f"Cannot safely archive VAD adapted checkpoint: {adapted}")
    records = json.loads((adapted / "ckpt.json").read_text(encoding="utf-8"))["_checkpoint"]["records"]
    if not any(row["status"] == TIMEOUT_STATUS for row in records):
        raise ValueError(f"No previous policy-zero VAD timeouts to remeasure: {adapted}")
    adapted.rename(archive)
    return archive


def evaluate(args: argparse.Namespace) -> dict:
    bench = args.bench_root.resolve()
    routes = bench / "leaderboard/data/bench2drive220.xml"
    ordered_routes = ET.parse(routes).getroot().findall(".//route")
    ordered_route_ids = [item.get("id") for item in ordered_routes]
    route_ids = set(ordered_route_ids)
    if len(route_ids) != 220:
        raise ValueError(f"expected 220 unique official routes in {routes}; found {len(route_ids)}")
    if args.truncate_crashed_routes or args.route_wall_timeout_seconds:
        scorer = bench / "leaderboard/leaderboard/utils/statistics_manager.py"
        if not scorer.is_file() or "FDSET_CRASH_ROUTE_SNAPSHOT" not in scorer.read_text(encoding="utf-8"):
            raise RuntimeError(f"Apply {ROOT / 'scripts/bench2drive_crash_snapshot.patch'} to {bench}")
    config = json.loads((ROOT / "configs/paper_reproduction.json").read_text(encoding="utf-8"))
    args.output.mkdir(parents=True, exist_ok=True)
    if args.restart_vad_adapted_for_live_timeouts:
        archive_vad_adapted_for_live_timeouts(args.output, route_ids)
    if args.skip_unstarted_route_id:
        run_dir = args.output / args.models[0] / "baseline"
        skip_unstarted_route(run_dir / "ckpt.json", run_dir / "crash_route_snapshot.json",
                             run_dir / "evidence", ordered_routes, args.skip_unstarted_route_id,
                             "Previous evaluation exhausted CARLA restart limit before world loaded")
    server, server_log = start_carla(args.carla_launcher.resolve(), args.output,
                                     args.gpu, args.graphics_adapter, args.port)
    report = {"route_xml": str(routes), "expected_routes": 220, "models": {}}
    try:
        for family in args.models:
            report["models"][family] = {}
            for variant in ((args.variant,) if args.variant else ("baseline", "adapted")):
                run_dir = args.output / family / variant
                evidence = run_dir / "evidence"
                evidence.mkdir(parents=True, exist_ok=True)
                checkpoint = run_dir / "ckpt.json"
                baseline = ROOT.parent / "Bench2DriveZoo/ckpts" / BASELINES[family]
                weights = baseline if variant == "baseline" else args.checkpoints / ADAPTED[family]
                if not weights.is_file():
                    raise FileNotFoundError(weights)
                count, complete = coverage(checkpoint, route_ids,
                                           allow_non_official=bool(args.truncate_crashed_routes
                                                                   or args.route_wall_timeout_seconds
                                                                   or args.skip_unstarted_routes))
                timeout_enabled = bool(args.route_wall_timeout_seconds
                                       and (family == "vad" or variant == "adapted"))
                profile = config["model_profiles"][family]
                run_config = dict(config)
                run_config["bench2drive"] = {**config["bench2drive"], **profile}
                environment = {**os.environ, **build_bench2drive_runtime_env(run_config, bench)}
                environment.update({"CUDA_VISIBLE_DEVICES": str(args.gpu), "IS_BENCH2DRIVE": "1",
                                    "SAVE_PATH": str(evidence), "ROUTES": str(routes),
                                    "OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2",
                                    "TMPDIR": str(args.output / "temp")})
                snapshot = run_dir / "crash_route_snapshot.json"
                if args.truncate_crashed_routes or timeout_enabled:
                    environment["FDSET_CRASH_ROUTE_SNAPSHOT"] = str(snapshot)
                if family == "tcp":
                    environment["PLANNER_TYPE"] = "merge_ctrl_traj"
                agent_path = (bench / profile["agent_path"]).resolve()
                agent_config = str(weights.resolve())
                if family in ("uniad", "vad"):
                    environment["FDSET_ZOO_AGENT_NAME"] = family
                    environment["FDSET_METRICS_ONLY"] = "1"
                    config_path = profile["agent_config"].split("+", 1)[0]
                    agent_config = str((bench / config_path).resolve()) + "+" + agent_config
                command = [str(ROOT / ".venv/bin/python"),
                           "leaderboard/leaderboard/leaderboard_evaluator.py",
                           "--host", "127.0.0.1", "--port", str(args.port),
                           "--traffic-manager-port", str(args.port + 50),
                           "--routes", str(routes), "--repetitions", "1", "--track", "SENSORS",
                           "--checkpoint", str(checkpoint), "--agent", str(agent_path),
                           "--agent-config", agent_config, "--resume", "True", "--gpu-rank", "0"]
                if args.truncate_crashed_routes or timeout_enabled:
                    command.extend(["--debug", "2", "--debug-checkpoint",
                                    str(run_dir / "live_results.txt")])
                retries = 0
                no_tick_retries = 0
                route_watch_index = count
                route_watch_start = time.monotonic()
                prune_completed_evidence(checkpoint, evidence, family, route_ids)
                while not complete:
                    previous = count
                    if args.truncate_crashed_routes or timeout_enabled:
                        snapshot.unlink(missing_ok=True)
                    with (run_dir / "evaluator.log").open("a", encoding="utf-8") as log:
                        log.write("\nCOMMAND: " + " ".join(command) + "\n")
                        log.flush()
                        evaluator = subprocess.Popen(command, cwd=str(bench), env=environment,
                                                     stdout=log, stderr=subprocess.STDOUT,
                                                     start_new_session=True)
                        timed_out_route = None
                        last_checkpoint_mtime = None
                        while evaluator.poll() is None:
                            if server.poll() is not None:
                                stop_process(evaluator)
                                break
                            checkpoint_mtime = checkpoint.stat().st_mtime_ns if checkpoint.exists() else None
                            if checkpoint_mtime is not None and checkpoint_mtime != last_checkpoint_mtime:
                                prune_completed_evidence(checkpoint, evidence, family, route_ids)
                                last_checkpoint_mtime = checkpoint_mtime
                                if timeout_enabled:
                                    index = checkpoint_progress(checkpoint)
                                    if index != route_watch_index:
                                        route_watch_index = index
                                        route_watch_start = time.monotonic()
                            if (timeout_enabled
                                    and route_watch_index < len(ordered_routes)
                                    and time.monotonic() - route_watch_start >= args.route_wall_timeout_seconds):
                                timed_out_route = ordered_route_ids[route_watch_index]
                                stop_process(evaluator)
                                break
                            time.sleep(5)
                        code = evaluator.wait()
                    if timed_out_route is not None:
                        current_index = checkpoint_progress(checkpoint)
                        if current_index == route_watch_index:
                            elapsed = time.monotonic() - route_watch_start
                            accept_timeout_snapshot(checkpoint, snapshot, evidence,
                                                    ordered_route_ids, timed_out_route,
                                                    elapsed, args.route_wall_timeout_seconds,
                                                    family=family)
                            stop_process(server)
                    carla_exit = server.poll()
                    if (carla_exit is not None and snapshot.is_file()
                            and (args.truncate_crashed_routes or timeout_enabled)):
                        current = json.loads(checkpoint.read_text(encoding="utf-8"))["_checkpoint"]
                        next_index = current["progress"][0]
                        if next_index < len(ordered_route_ids):
                            route_id = ordered_route_ids[next_index]
                            if json.loads(snapshot.read_text(encoding="utf-8")).get("route_id") == (
                                    f"RouteScenario_{route_id}_rep0"):
                                accept_crash_snapshot(checkpoint, snapshot, evidence,
                                                      ordered_route_ids, route_id, carla_exit,
                                                      family=family)
                    if (carla_exit is not None and args.skip_unstarted_routes
                            and variant == "baseline" and not snapshot.is_file()):
                        no_tick_retries += 1
                        if no_tick_retries >= args.max_unstarted_restarts:
                            current = json.loads(checkpoint.read_text(encoding="utf-8"))["_checkpoint"]
                            next_index = current["progress"][0]
                            if next_index < len(ordered_routes):
                                route_id = ordered_route_ids[next_index]
                                reason = f"CARLA exited {no_tick_retries} times without usable live criteria"
                                if any(evidence.glob(f"RouteScenario_{route_id}_rep0_*")):
                                    skip_unscored_crash(checkpoint, snapshot, evidence, ordered_routes,
                                                        route_id, reason)
                                else:
                                    skip_unstarted_route(checkpoint, snapshot, evidence, ordered_routes,
                                                         route_id, reason)
                    prune_completed_evidence(checkpoint, evidence, family, route_ids)
                    count, complete = coverage(
                        checkpoint, route_ids, allow_non_official=bool(
                            args.truncate_crashed_routes or args.route_wall_timeout_seconds
                            or args.skip_unstarted_routes))
                    if timeout_enabled and count != route_watch_index:
                        route_watch_index = count
                        route_watch_start = time.monotonic()
                    print(json.dumps({"model": family, "variant": variant,
                                      "routes": count, "complete": complete, "exit_code": code}), flush=True)
                    if complete:
                        break
                    if server.poll() is None and code == 0:
                        raise RuntimeError(f"evaluator exited without all 220 routes: {checkpoint}")
                    if server.poll() is None and count == previous:
                        raise RuntimeError(f"evaluator made no progress; inspect {run_dir / 'evaluator.log'}")
                    retries = 0 if count > previous else retries + 1
                    if count > previous:
                        no_tick_retries = 0
                    if retries > args.max_restarts:
                        raise RuntimeError(f"too many failed evaluator/CARLA restarts: {run_dir}")
                    if server.poll() is not None:
                        server_log.close()
                        server, server_log = start_carla(args.carla_launcher.resolve(), args.output,
                                                         args.gpu, args.graphics_adapter, args.port)
                records = json.loads(checkpoint.read_text(encoding="utf-8"))["_checkpoint"]["records"]
                truncated = [row["route_id"] for row in records if row["status"] == CRASH_TRUNCATED_STATUS]
                skipped = [row["route_id"] for row in records if row["status"] == UNSTARTED_STATUS]
                timed_out = [row["route_id"] for row in records if row["status"] == TIMEOUT_STATUS]
                timeout_scored = [row["route_id"] for row in records
                                  if row["status"] == TIMEOUT_SCORED_STATUS]
                unscored = [row["route_id"] for row in records if row["status"] == UNSCORED_CRASH_STATUS]
                if truncated or skipped or timed_out or timeout_scored or unscored:
                    mark_checkpoint_ineligible(checkpoint)
                metrics = parse_metrics(checkpoint, ["DS", "SR", "CR"], strict=True)
                if skipped or timed_out or timeout_scored or unscored:
                    metrics.update({"Eff": None, "Comf": None})
                else:
                    try:
                        metrics.update(calculate_official_motion_metrics(checkpoint, evidence, bench))
                    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as exc:
                        raise RuntimeError(f"official Eff/Comf unavailable for {family}/{variant}: {exc}") from exc
                report["models"][family][variant] = {
                    "route_count": count, "checkpoint": str(checkpoint), "metrics": metrics,
                    "official_eligible": not (truncated or skipped or timed_out or timeout_scored or unscored),
                    "crash_truncated_routes": truncated, "unstarted_skipped_routes": skipped,
                    "wall_timeout_skipped_routes": timed_out,
                    "wall_timeout_partial_score_routes": timeout_scored,
                    "unscored_crash_skipped_routes": unscored,
                }
                (args.output / "comparison.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    finally:
        stop_process(server)
        server_log.close()
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bench-root", type=Path, default=ROOT.parent / "Bench2Drive")
    parser.add_argument("--carla-launcher", type=Path, default=ROOT.parent / "carla/CarlaUE4.sh")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/four_model_evaluation/official220")
    parser.add_argument("--checkpoints", type=Path, default=ROOT / "artifacts/four_model_training/checkpoints")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--graphics-adapter", type=int, default=0,
                        help="CARLA/UE graphics adapter index; override for multi-GPU hosts")
    parser.add_argument("--port", type=int, default=32100, help="Dedicated CARLA RPC port")
    parser.add_argument("--max-restarts", type=int, default=40)
    parser.add_argument("--truncate-crashed-routes", action="store_true",
                        help="Non-official: score crashed routes from live criteria and continue")
    parser.add_argument("--skip-unstarted-route-id", type=str,
                        help="Non-official: skip a previously failed, never-started ADMLP/UniAD baseline route")
    parser.add_argument("--skip-unstarted-routes", action="store_true",
                        help="Non-official: skip future routes that repeatedly fail before their first tick")
    parser.add_argument("--max-unstarted-restarts", type=int, default=3)
    parser.add_argument("--route-wall-timeout-seconds", type=float, default=0,
                        help="Non-official: score the last genuine CARLA tick at the wall-clock limit")
    parser.add_argument("--restart-vad-adapted-for-live-timeouts", action="store_true",
                        help="One-time: archive policy-zero VAD adapted run, remeasure adapted only")
    parser.add_argument("--model", action="append", dest="models", choices=tuple(BASELINES))
    parser.add_argument("--variant", choices=("baseline", "adapted"),
                        help="Evaluate only this variant; leave the other checkpoint untouched")
    args = parser.parse_args()
    args.output = args.output.expanduser().resolve()
    args.checkpoints = args.checkpoints.expanduser().resolve()
    args.models = args.models or list(BASELINES)
    if args.truncate_crashed_routes and args.models not in (["admlp"], ["tcp"]):
        parser.error("--truncate-crashed-routes requires --model admlp or --model tcp")
    if args.skip_unstarted_route_id or args.skip_unstarted_routes:
        if args.models not in (["admlp"], ["uniad"]):
            parser.error("--skip-unstarted-routes requires --model admlp or --model uniad")
        if args.models == ["admlp"] and not args.truncate_crashed_routes:
            parser.error("ADMLP unstarted-route skips require --truncate-crashed-routes")
    if args.skip_unstarted_route_id and not args.skip_unstarted_routes:
        parser.error("--skip-unstarted-route-id requires --skip-unstarted-routes")
    if args.skip_unstarted_route_id and args.variant == "adapted":
        parser.error("--skip-unstarted-route-id applies only to baseline")
    if args.route_wall_timeout_seconds and (
            args.models not in (["vad"], ["uniad"], ["tcp"], ["admlp"])
            or args.route_wall_timeout_seconds <= 0
            or not math.isfinite(args.route_wall_timeout_seconds)):
        parser.error("--route-wall-timeout-seconds requires one model and a finite positive limit")
    if args.restart_vad_adapted_for_live_timeouts and (
            args.models != ["vad"] or not args.route_wall_timeout_seconds):
        parser.error("--restart-vad-adapted-for-live-timeouts requires timed --model vad")
    (args.output / "temp").mkdir(parents=True, exist_ok=True)
    print(json.dumps(evaluate(args), indent=2))


if __name__ == "__main__":
    main()
