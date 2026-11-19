#!/usr/bin/env python3
"""Alert on stopped or stalled four-model evaluations without altering scores or processes."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = {
    "vad": "official220_vad/vad",
    "uniad": "official220_uniad/uniad",
    "tcp": "official220_tcp/tcp",
}


def active_runners() -> dict[tuple[str, Path, str | None], int]:
    """Identify only this repository's evaluation commands, not other users' CARLA jobs."""
    result = {}
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        try:
            args = (proc / "cmdline").read_bytes().decode(errors="replace").split("\0")
            if not any(arg.endswith("scripts/evaluate_four_models_official220.py") for arg in args):
                continue
            if "--model" not in args or "--output" not in args:
                continue
            model = args[args.index("--model") + 1]
            variant = args[args.index("--variant") + 1] if "--variant" in args else None
            output = Path(args[args.index("--output") + 1])
            if not output.is_absolute():
                output = Path(os.readlink(proc / "cwd")) / output
            result[model, output.resolve(), variant] = int(proc.name)
        except (OSError, IndexError):
            continue
    return result

def runner_started(pid: int) -> float:
    """Grant a new CARLA runner startup time even if it inherits old route logs."""
    return (Path("/proc") / str(pid)).stat().st_ctime


def read_checkpoint(path: Path) -> dict | None:
    if not path.is_file():
        return None
    for attempt in range(20):
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            if attempt == 19:
                raise
            time.sleep(0.1)
    raise AssertionError("unreachable")


def stalled_reason(now: float, last_heartbeat: float, last_route: float,
                   idle_limit: float, route_limit: float) -> str | None:
    if now - last_heartbeat > idle_limit:
        return f"no evaluator activity for {round(now - last_heartbeat)}s"
    if now - last_route > route_limit:
        return f"no route checkpoint advancement for {round(now - last_route)}s"
    return None


def inspect(output_root: Path, models: list[str], idle_limit: float,
            route_limit: float, progress_seen: dict[str, tuple[str, int, float]] | None = None,
            variant_only: str | None = None) -> list[dict]:
    now = time.time()
    runners = active_runners()
    statuses = []
    for model in models:
        output = (output_root / OUTPUTS[model]).resolve()
        for variant in ((variant_only,) if variant_only else ("baseline", "adapted")):
            run = output / variant
            checkpoint = run / "ckpt.json"
            data = read_checkpoint(checkpoint)
            progress = data["_checkpoint"]["progress"] if data else [0, 220]
            if data and data.get("entry_status") == "Finished" and progress == [220, 220]:
                continue
            pid = (runners.get((model, output.parent, variant))
                   or runners.get((model, output.parent, None)))
            if pid is None:
                raise RuntimeError(f"{model}/{variant}: evaluator runner absent at {progress}")
            marker = (variant, progress[0])
            if progress_seen is not None:
                seen = progress_seen.get(model)
                if seen is None or seen[:2] != marker:
                    progress_seen[model] = (variant, progress[0], now)
                last_route = progress_seen[model][2]
            else:
                last_route = now  # A single check cannot establish when progress last changed.
            log = run / "evaluator.log"
            snapshot = run / "crash_route_snapshot.json"
            heartbeats = [path.stat().st_mtime for path in (log, snapshot) if path.is_file()]
            last_heartbeat = max([runner_started(pid), *heartbeats])
            problem = stalled_reason(now, last_heartbeat, last_route, idle_limit, route_limit)
            if problem:
                raise RuntimeError(f"{model}/{variant}: {problem}; checkpoint={progress}, pid={pid}")
            statuses.append({"model": model, "variant": variant, "progress": progress,
                             "pid": pid, "heartbeat_age_s": round(now - last_heartbeat),
                             "route_no_progress_s": round(now - last_route)})
            break
    return statuses


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path,
                        default=ROOT / "artifacts/four_model_evaluation")
    parser.add_argument("--model", action="append", choices=tuple(OUTPUTS))
    parser.add_argument("--variant", choices=("baseline", "adapted"),
                        help="Monitor only this variant, including when its baseline is incomplete")
    parser.add_argument("--interval-seconds", type=float, default=60)
    parser.add_argument("--idle-seconds", type=float, default=1200)
    parser.add_argument("--route-stall-seconds", type=float, default=43200)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    if args.interval_seconds <= 0 or args.idle_seconds <= 0 or args.route_stall_seconds <= args.idle_seconds:
        parser.error("monitor intervals must be positive and route-stall must exceed idle")
    models = args.model or list(OUTPUTS)
    progress_seen: dict[str, tuple[str, int, float]] = {}
    while True:
        try:
            statuses = inspect(args.output_root.resolve(), models, args.idle_seconds,
                               args.route_stall_seconds, progress_seen, args.variant)
        except (OSError, KeyError, ValueError, RuntimeError) as exc:
            print(f"EVALUATION STALLED: {exc}", file=sys.stderr, flush=True)
            raise SystemExit(1) from exc
        print(json.dumps({"checked_at": dt.datetime.now().astimezone().isoformat(),
                          "active": statuses}), flush=True)
        if args.once or not statuses:
            return
        time.sleep(args.interval_seconds)


if __name__ == "__main__":
    main()
