#!/usr/bin/env python3
"""Evaluate baseline and Scenic-adapted Zoo agents on the same official 220 routes."""
from __future__ import annotations

import argparse
import json
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


def coverage(checkpoint: Path, route_ids: set[str]) -> tuple[int, bool]:
    if not checkpoint.is_file():
        return 0, False
    payload = json.loads(checkpoint.read_text(encoding="utf-8"))
    state = payload.get("_checkpoint", {})
    records = state.get("records", [])
    observed = {str(row.get("route_id", "")) for row in records if isinstance(row, dict)}
    expected = {f"RouteScenario_{route_id}_rep0" for route_id in route_ids}
    progress = state.get("progress", [])
    execution_errors = ("Simulation crashed", "Agent crashed", "sensors were invalid")
    complete = (
        payload.get("entry_status") == "Finished"
        and observed == expected
        and len(records) == len(expected)
        and progress == [len(expected), len(expected)]
        and not any(
            marker in str(row.get("status", ""))
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
    for _ in range(120):
        if process.poll() is not None:
            raise RuntimeError(f"CARLA exited during startup: {process.returncode}; see {workdir / 'carla.log'}")
        if available("127.0.0.1", port):
            return process, log
        time.sleep(1)
    process.terminate()
    raise TimeoutError("CARLA did not open its RPC port within 120 seconds")


def stop_process(process: subprocess.Popen) -> None:
    if process.poll() is None:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()


def evaluate(args: argparse.Namespace) -> dict:
    bench = args.bench_root.resolve()
    routes = bench / "leaderboard/data/bench2drive220.xml"
    route_ids = {item.get("id") for item in ET.parse(routes).getroot().findall(".//route")}
    if len(route_ids) != 220:
        raise ValueError(f"expected 220 unique official routes in {routes}; found {len(route_ids)}")
    config = json.loads((ROOT / "configs/paper_reproduction.json").read_text(encoding="utf-8"))
    args.output.mkdir(parents=True, exist_ok=True)
    server, server_log = start_carla(args.carla_launcher.resolve(), args.output,
                                     args.gpu, args.graphics_adapter, args.port)
    report = {"route_xml": str(routes), "expected_routes": 220, "models": {}}
    try:
        for family in args.models:
            report["models"][family] = {}
            for variant in ("baseline", "adapted"):
                run_dir = args.output / family / variant
                evidence = run_dir / "evidence"
                evidence.mkdir(parents=True, exist_ok=True)
                checkpoint = run_dir / "ckpt.json"
                baseline = ROOT.parent / "Bench2DriveZoo/ckpts" / BASELINES[family]
                weights = baseline if variant == "baseline" else args.checkpoints / ADAPTED[family]
                if not weights.is_file():
                    raise FileNotFoundError(weights)
                count, complete = coverage(checkpoint, route_ids)
                profile = config["model_profiles"][family]
                run_config = dict(config)
                run_config["bench2drive"] = {**config["bench2drive"], **profile}
                environment = {**os.environ, **build_bench2drive_runtime_env(run_config, bench)}
                environment.update({"CUDA_VISIBLE_DEVICES": str(args.gpu), "IS_BENCH2DRIVE": "1",
                                    "SAVE_PATH": str(evidence), "ROUTES": str(routes),
                                    "OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2",
                                    "TMPDIR": str(args.output / "temp")})
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
                retries = 0
                prune_completed_evidence(checkpoint, evidence, family, route_ids)
                while not complete:
                    previous = count
                    with (run_dir / "evaluator.log").open("a", encoding="utf-8") as log:
                        log.write("\nCOMMAND: " + " ".join(command) + "\n")
                        log.flush()
                        evaluator = subprocess.Popen(command, cwd=str(bench), env=environment,
                                                     stdout=log, stderr=subprocess.STDOUT,
                                                     start_new_session=True)
                        last_checkpoint_mtime = None
                        while evaluator.poll() is None:
                            if server.poll() is not None:
                                stop_process(evaluator)
                                break
                            checkpoint_mtime = checkpoint.stat().st_mtime_ns if checkpoint.exists() else None
                            if checkpoint_mtime is not None and checkpoint_mtime != last_checkpoint_mtime:
                                prune_completed_evidence(checkpoint, evidence, family, route_ids)
                                last_checkpoint_mtime = checkpoint_mtime
                            time.sleep(5)
                        code = evaluator.wait()
                    prune_completed_evidence(checkpoint, evidence, family, route_ids)
                    count, complete = coverage(checkpoint, route_ids)
                    print(json.dumps({"model": family, "variant": variant,
                                      "routes": count, "complete": complete, "exit_code": code}), flush=True)
                    if complete:
                        break
                    if server.poll() is None and code == 0:
                        raise RuntimeError(f"evaluator exited without all 220 routes: {checkpoint}")
                    if server.poll() is None and count == previous:
                        raise RuntimeError(f"evaluator made no progress; inspect {run_dir / 'evaluator.log'}")
                    retries += 1
                    if retries > args.max_restarts:
                        raise RuntimeError(f"too many failed evaluator/CARLA restarts: {run_dir}")
                    if server.poll() is not None:
                        server_log.close()
                        server, server_log = start_carla(args.carla_launcher.resolve(), args.output,
                                                         args.gpu, args.graphics_adapter, args.port)
                metrics = parse_metrics(checkpoint, ["DS", "SR", "CR"], strict=True)
                try:
                    metrics.update(calculate_official_motion_metrics(checkpoint, evidence, bench))
                except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as exc:
                    raise RuntimeError(f"official Eff/Comf unavailable for {family}/{variant}: {exc}") from exc
                report["models"][family][variant] = {"route_count": count, "checkpoint": str(checkpoint),
                                                       "metrics": metrics}
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
    parser.add_argument("--max-restarts", type=int, default=40)
    parser.add_argument("--model", action="append", dest="models", choices=tuple(BASELINES))
    args = parser.parse_args()
    args.models = args.models or list(BASELINES)
    (args.output / "temp").mkdir(parents=True, exist_ok=True)
    print(json.dumps(evaluate(args), indent=2))


if __name__ == "__main__":
    main()
