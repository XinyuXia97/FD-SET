#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the paper protocol model matrix.")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "paper_reproduction.json")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--carla-port", type=int, help="Override the CARLA RPC port for every model")
    parser.add_argument("--traffic-manager-port", type=int, help="Override the traffic manager port")
    parser.add_argument("--model", action="append", dest="models")
    parser.add_argument("--protocol", choices=("fixed_policy_evolution", "downstream_adaptation"),
                        help="Override the protocol selected by the paper config")
    parser.add_argument("--strategy", action="append", dest="strategies")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--stop-on-error", action="store_true")
    parser.add_argument("--resume-existing", action="store_true", help="Resume existing per-model run directories")
    parser.add_argument(
        "--skip-failed-routes",
        action="store_true",
        help="Continue past unscored route crashes; report measured-subset coverage, not a full official score",
    )
    return parser.parse_args()


def _summary_metrics(summary: dict[str, Any]) -> dict[str, Any]:
    """Keep auditable aggregate and per-round metrics from the runner summary."""
    return {
        "metrics_keys": summary.get("metrics_keys", []),
        "num_rounds": summary.get("num_rounds", 0),
        "improvement": summary.get("improvement", {}),
        "rounds": [
            {
                key: row.get(key)
                for key in (
                    "round",
                    "baseline_metrics",
                    "post_adaptation_metrics",
                    "metrics",
                    "metric_delta_after_adaptation",
                    "benchmark_coverage",
                    "post_adaptation_coverage",
                    "official_benchmark_complete",
                )
                if key in row
            }
            for row in summary.get("rounds", [])
            if isinstance(row, dict)
        ],
    }


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("Paper reproduction config must be an object")
    if args.carla_port is not None or args.traffic_manager_port is not None:
        carla_cfg = dict(config.get("carla", {}))
        if args.carla_port is not None:
            if args.carla_port < 1:
                raise ValueError("--carla-port must be positive")
            carla_cfg["port"] = args.carla_port
        if args.traffic_manager_port is not None:
            if args.traffic_manager_port < 1:
                raise ValueError("--traffic-manager-port must be positive")
            carla_cfg["tm_port"] = args.traffic_manager_port
        config["carla"] = carla_cfg
    protocol = args.protocol or str(config.get("protocol", "downstream_adaptation"))
    if protocol not in {"fixed_policy_evolution", "downstream_adaptation"}:
        raise ValueError(f"Unsupported paper protocol: {protocol}")
    experiment = config.get("experiment", {})
    if not isinstance(experiment, dict):
        experiment = {}
    models = args.models or [str(item) for item in experiment.get("models", ["uniad"])]
    strategies = [str(item) for item in args.strategies] if args.strategies else ["full"]
    profiles = config.get("model_profiles", {})
    if not isinstance(profiles, dict):
        profiles = {}
    if protocol == "downstream_adaptation" and not args.dry_run:
        adaptation_models = config.get("adaptation", {}).get("models", {})
        unsupported = [
            model_name for model_name in models
            if not isinstance(adaptation_models.get(model_name), dict)
            or not str(adaptation_models[model_name].get("train_command", "")).strip()
        ]
        if unsupported:
            raise ValueError(
                "Live downstream adaptation requires a real train_command for each selected model: "
                + ", ".join(unsupported)
            )
    output_root = (args.output_root or Path(str(config.get("output_root", "artifacts/paper_suite")))).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "config": str(args.config),
        "protocol": protocol,
        "synthetic": bool(args.dry_run),
        "models": models,
        "strategies": strategies,
        "runs": [],
    }

    for model_name in models:
        model_cfg = profiles.get(model_name, {})
        if not isinstance(model_cfg, dict):
            model_cfg = {}
        for strategy in strategies:
            run_config = copy.deepcopy(config)
            bench_cfg = dict(run_config.get("bench2drive", {}))
            bench_cfg.update(model_cfg)
            run_config["bench2drive"] = bench_cfg
            if model_name in {"uniad", "vad"}:
                # These Zoo agents name front-camera files by 2 Hz indices;
                # AD-MLP/TCP store RGB frames using the 20 Hz tick number.
                failure_cfg = dict(run_config.get("failure_extract", {}))
                failure_cfg["frame_dir_template"] = "{log_dir}/evidence/{save_name}/rgb_front"
                failure_cfg["frame_rate"] = 2.0
                run_config["failure_extract"] = failure_cfg
            run_config["output_root"] = str(output_root)
            run_config.pop("experiment", None)
            run_config["protocol"] = protocol
            if args.skip_failed_routes:
                evaluation_cfg = dict(run_config.get("evaluation", {}))
                evaluation_cfg["skip_failed_routes"] = True
                run_config["evaluation"] = evaluation_cfg
                run_config["round_log_template"] = "{round_dir}/partial_ckpt.json"
                retention_cfg = dict(run_config.get("evidence_retention", {}))
                retention_cfg["prune_new_frames_after_route"] = True
                run_config["evidence_retention"] = retention_cfg
            adaptation_cfg = dict(run_config.get("adaptation", {}))
            adaptation_cfg["enabled"] = protocol == "downstream_adaptation"
            run_config["adaptation"] = adaptation_cfg
            if protocol == "downstream_adaptation":
                evaluation_cfg = dict(run_config.get("evaluation", {}))
                evaluation_cfg["use_held_out_routes"] = True
                run_config["evaluation"] = evaluation_cfg
            dry_run_overrides: dict[str, Any] = {}
            if args.dry_run:
                interpretation_cfg = dict(run_config.get("interpretation", {}))
                if str(interpretation_cfg.get("backend", "")).strip().lower() != "rule_based":
                    dry_run_overrides["interpretation.backend"] = "rule_based"
                    interpretation_cfg["backend"] = "rule_based"
                    interpretation_cfg["fallback_to_rule_based"] = True
                    run_config["interpretation"] = interpretation_cfg
                retrieval_cfg = dict(run_config.get("retrieval", {}))
                if str(retrieval_cfg.get("backend", "lexical")).strip().lower() != "lexical":
                    dry_run_overrides["retrieval.backend"] = "lexical"
                    retrieval_cfg["backend"] = "lexical"
                    run_config["retrieval"] = retrieval_cfg

            run_id = (
                f"{protocol}_{model_name}"
                if strategies == ["full"]
                else f"{protocol}_{model_name}_{strategy}"
            )
            generated_config = output_root / f"config_{run_id}.json"
            generated_config.write_text(json.dumps(run_config, indent=2), encoding="utf-8")
            run_dir = output_root / run_id
            command = [
                sys.executable,
                "-m",
                "core.closed_loop_runner",
                "--config",
                str(generated_config),
                "--mode",
                strategy,
            ]
            if args.resume_existing and run_dir.exists():
                if not (run_dir / "run_manifest.json").is_file():
                    raise FileNotFoundError(f"Cannot resume a run without run_manifest.json: {run_dir}")
                command.extend(["--resume-run", str(run_dir)])
            else:
                command.extend(["--run-id", run_id])
            if args.dry_run:
                command.append("--dry-run")
            record: dict[str, Any] = {
                "model": model_name,
                "strategy": strategy,
                "protocol": protocol,
                "run_id": run_id,
                "command": command,
                "synthetic": bool(args.dry_run),
                "measured": False,
            }
            if dry_run_overrides:
                record["dry_run_overrides"] = dry_run_overrides
            try:
                subprocess.run(command, cwd=str(ROOT), check=True)
                summary_path = output_root / run_id / "summary.json"
                record["summary_path"] = str(summary_path)
                if not summary_path.is_file():
                    raise FileNotFoundError(f"Run summary not found: {summary_path}")
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                if not isinstance(summary, dict):
                    raise ValueError("Run summary must be a JSON object")
                record["summary"] = summary
                record["synthetic"] = bool(args.dry_run) or any(
                    bool(row.get("evaluation_synthetic"))
                    for row in summary.get("rounds", [])
                    if isinstance(row, dict)
                )
                record["measured"] = not record["synthetic"]
                record["metrics"] = None
                rounds = [row for row in summary.get("rounds", []) if isinstance(row, dict)]
                coverages = [
                    row["benchmark_coverage"]
                    for row in rounds
                    if isinstance(row.get("benchmark_coverage"), dict)
                ]
                record["benchmark_coverage"] = coverages
                record["post_adaptation_coverage"] = [
                    row["post_adaptation_coverage"]
                    for row in rounds
                    if isinstance(row.get("post_adaptation_coverage"), dict)
                ]
                record["official_complete"] = (
                    not record["synthetic"]
                    and bool(rounds)
                    and (not args.skip_failed_routes or bool(coverages))
                    and all(row.get("official_benchmark_complete") is True for row in rounds)
                    and all(coverage["skipped_count"] == 0 for coverage in
                            coverages + record["post_adaptation_coverage"])
                    and (protocol != "downstream_adaptation" or strategy != "full"
                         or all(row.get("adaptation_executed")
                                and row.get("post_adaptation_metrics") is not None for row in rounds))
                )
                record["status"] = (
                    "completed_partial"
                    if record["measured"] and not record["official_complete"]
                    else "completed"
                )
                if record["measured"]:
                    key = "metrics" if record["official_complete"] else "partial_metrics"
                    record[key] = _summary_metrics(summary)
            except (subprocess.CalledProcessError, OSError, json.JSONDecodeError, ValueError) as exc:
                record["status"] = "failed"
                if isinstance(exc, subprocess.CalledProcessError):
                    record["returncode"] = exc.returncode
                record["error"] = str(exc)
                manifest["runs"].append(record)
                if args.stop_on_error:
                    manifest_path = output_root / "suite_manifest.json"
                    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
                    raise
                continue
            manifest["runs"].append(record)

    manifest_path = output_root / "suite_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"manifest": str(manifest_path), "runs": len(manifest["runs"])}, indent=2))


if __name__ == "__main__":
    main()
