from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import os
import re
import socket
import sys
import xml.etree.ElementTree as ET
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .bench2drive_adapter import build_eval_command, calculate_official_motion_metrics, parse_metrics, run_eval_command
from .partial_benchmark_metrics import build_partial_checkpoint
from .route_skip_runner import run_remaining_routes
from .scenario_materialize import materialize_adaptation_inputs, materialize_round_inputs
from .scenic_policy_adapter import (
    DEFAULT_RESULT_FILE,
    build_native_policy_plan,
    native_policy_config,
    run_native_policy_manifest,
    validate_native_policy_config,
)
from .failure_extract import extract_failures_from_log, generate_mock_failures
from .failure_interpret import DEFAULT_PROMPT_PATH, interpret_failures
from .interpretation_backend import validate_interpretation_config
from .failure_struct import structure_failures
from .scenario_bank import append_evolved_to_bank, ensure_seed_bank, load_scenario_bank, save_scenario_bank
from .scenario_evolve import evolve_from_anchors, random_perturb_from_anchors
from .scenario_retrieve import retrieve_anchor_scenarios
PROTOCOLS = {"fixed_policy_evolution", "downstream_adaptation"}
from .policy_adaptation import (
    build_adaptation_manifest,
    build_expert_generation_command,
    select_adaptation_scenarios,
    run_expert_generation,
    write_adaptation_manifest,
    write_proxy_expert_demos,
)


MODES = {
    "full",
    "static",
    "retrieval",
    "recon_escalation",
    "random",
    "wo_struct",
    "wo_escalation",
}
DEFAULT_METRIC_KEYS = ["DS", "SR", "Eff", "Comf", "CR"]
OFFICIAL_AGENT_SCRIPT_HINTS = [
    "leaderboard/scripts/run_evaluation_debug.sh -> leaderboard/team_code/your_team_agent.py",
    "leaderboard/scripts/run_evaluation_multi_uniad.sh -> team_code/uniad_b2d_agent.py",
    "leaderboard/scripts/run_evaluation_multi_vad.sh -> team_code/vad_b2d_agent.py",
    "leaderboard/scripts/run_evaluation_multi_tcp.sh -> team_code/tcp_b2d_agent.py",
    "leaderboard/scripts/run_evaluation_multi_admlp.sh -> team_code/admlp_b2d_agent.py",
    "leaderboard/run_leaderboard.sh -> leaderboard/leaderboard/autoagents/human_agent.py",
    "local fallback -> leaderboard/leaderboard/autoagents/npc_agent.py",
]
MODEL_AGENT_ALIASES = {
    "uniad": {
        "agent": "leaderboard/team_code/uniad_b2d_agent.py",
        "agent_config": "../Bench2DriveZoo/adzoo/uniad/configs/stage2_e2e/base_e2e_b2d.py+YOUR_UNIAD_CHECKPOINT_PATH",
    },
    "uniad_tiny": {
        "agent": "leaderboard/team_code/uniad_b2d_agent.py",
        "agent_config": "../Bench2DriveZoo/adzoo/uniad/configs/stage2_e2e/tiny_e2e_b2d.py+YOUR_UNIAD_CHECKPOINT_PATH",
    },
    "vad": {
        "agent": "leaderboard/team_code/vad_b2d_agent.py",
        "agent_config": "../Bench2DriveZoo/adzoo/vad/configs/VAD/VAD_base_e2e_b2d.py+YOUR_VAD_CHECKPOINT_PATH",
    },
    "admlp": {
        "agent": "leaderboard/team_code/admlp_b2d_agent.py",
        "agent_config": "YOUR_ADMLP_CHECKPOINT_PATH",
    },
    "tcp": {
        "agent": "leaderboard/team_code/tcp_b2d_agent.py",
        "agent_config": "YOUR_TCP_CHECKPOINT_PATH",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Failure-driven scenario evolution runner")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true", help="Use synthetic failures/metrics")
    parser.add_argument("--mode", choices=sorted(MODES), default="full")
    parser.add_argument("--output-root", type=Path, help="Override config.output_root")
    parser.add_argument("--run-id", help="Use a fixed run directory name instead of a generated timestamp")
    parser.add_argument("--resume-run", type=Path, help="Resume an interrupted run from an existing output directory")
    return parser.parse_args()


def load_config(config_path: Path) -> dict[str, Any]:
    suffix = config_path.suffix.lower()
    text = config_path.read_text(encoding="utf-8")
    if suffix == ".json":
        return json.loads(text)
    if suffix in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore
        except ModuleNotFoundError as exc:
            raise RuntimeError("YAML config requires pyyaml. Use JSON config or install pyyaml.") from exc
        return yaml.safe_load(text)
    raise ValueError(f"Unsupported config format: {config_path}")


def validate_config(cfg: dict[str, Any], *, allow_incomplete_adaptation: bool = False) -> None:
    protocol = str(cfg.get("protocol", "downstream_adaptation")).strip()
    if protocol not in PROTOCOLS:
        raise ValueError(f"config.protocol must be one of {sorted(PROTOCOLS)}")
    if protocol == "fixed_policy_evolution":
        scenic_cfg = cfg.get("materialization", {}).get("scenic", {})
        if not isinstance(scenic_cfg, dict) or not bool(scenic_cfg.get("enabled", False)):
            raise ValueError("fixed_policy_evolution requires materialization.scenic.enabled=true")
        policy_cfg = scenic_cfg.get("policy_execution", {})
        if not allow_incomplete_adaptation and (
            not isinstance(policy_cfg, dict) or not bool(policy_cfg.get("enabled", False))
        ):
            raise ValueError(
                "fixed_policy_evolution requires native Scenic CARLA policy execution; "
                "set materialization.scenic.policy_execution.enabled=true"
            )
        adaptation_cfg = cfg.get("adaptation", {})
        if isinstance(adaptation_cfg, dict) and bool(adaptation_cfg.get("enabled", False)):
            raise ValueError("fixed_policy_evolution requires adaptation.enabled=false")
    if not str(cfg.get("output_root", "")).strip():
        raise ValueError("config.output_root is required")
    if not str(cfg.get("scenario_bank_path", "")).strip():
        bootstrap = cfg.get("scenario_bank_bootstrap", {})
        if not isinstance(bootstrap, dict) or str(bootstrap.get("source", "")).strip().lower() not in {
            "scenic", "scenic_source", "scenic_programs",
        }:
            raise ValueError(
                "config.scenario_bank_path is required unless scenario_bank_bootstrap.source=scenic"
            )

    num_rounds = int(cfg.get("num_rounds", 0))
    if num_rounds <= 0:
        raise ValueError("config.num_rounds must be a positive integer")

    retrieval = cfg.get("retrieval", {})
    if int(retrieval.get("top_k", 0)) <= 0:
        raise ValueError("config.retrieval.top_k must be a positive integer")

    evolution = cfg.get("evolution", {})
    if float(evolution.get("escalation_strength", 0.0)) < 0.0:
        raise ValueError("config.evolution.escalation_strength must be >= 0")

    interpretation = cfg.get("interpretation", {})
    prompt_path = Path(str(interpretation.get("prompt_path", DEFAULT_PROMPT_PATH)))
    validate_interpretation_config(interpretation, prompt_path=prompt_path)

    failure_extract = cfg.get("failure_extract", {})
    failure_format = str(failure_extract.get("format", "")).strip().lower()
    if failure_format and failure_format not in {"jsonl", "json", "bench2drive_checkpoint"}:
        raise ValueError("config.failure_extract.format must be `jsonl`, `json`, or `bench2drive_checkpoint`")

    scenic_cfg = cfg.get("materialization", {}).get("scenic", {})
    scenic_source_dir = str(scenic_cfg.get("source_dir", "")).strip() if isinstance(scenic_cfg, dict) else ""
    if scenic_source_dir and not Path(scenic_source_dir).expanduser().is_dir():
        raise FileNotFoundError(f"Configured Scenic source directory not found: {scenic_source_dir}")
    validate_native_policy_config(cfg)
    evaluation = cfg.get("evaluation", {})
    if isinstance(evaluation, dict) and bool(evaluation.get("use_held_out_routes", False)):
        held_out_raw = str(evaluation.get("held_out_routes_xml", "")).strip()
        if not held_out_raw:
            raise ValueError(
                "evaluation.held_out_routes_xml is required when use_held_out_routes=true"
            )
        if not Path(held_out_raw).expanduser().exists():
            raise FileNotFoundError(f"Configured evaluation routes XML not found: {held_out_raw}")
        if bool(evaluation.get("include_evolved_scenarios", False)):
            raise ValueError(
                "evaluation.include_evolved_scenarios must be false: evolved Scenic scenarios "
                "belong to PDM-Lite adaptation, not benchmark feedback/re-evaluation"
            )
    bootstrap = cfg.get("scenario_bank_bootstrap", {})
    if isinstance(bootstrap, dict) and bool(bootstrap.get("include_scenic_source_bank", False)):
        bootstrap_source_dir = str(bootstrap.get("scenic_source_dir", "")).strip()
        if not bootstrap_source_dir:
            raise ValueError(
                "scenario_bank_bootstrap.scenic_source_dir is required when "
                "include_scenic_source_bank=true"
            )
        if not Path(bootstrap_source_dir).expanduser().is_dir():
            raise FileNotFoundError(f"Configured Scenic source bank directory not found: {bootstrap_source_dir}")
    adaptation = cfg.get("adaptation", {})
    if (
        isinstance(adaptation, dict)
        and bool(adaptation.get("enabled", False))
        and not allow_incomplete_adaptation
    ):
        expert = adaptation.get("expert", {})
        if not isinstance(expert, dict):
            expert = {}
        backend = str(expert.get("backend", "external")).strip().lower()
        if backend not in {"proxy", "kinematic_proxy", "synthetic"} and not str(
            expert.get("command_template", "")
        ).strip():
            raise ValueError(
                "Live adaptation requires adaptation.expert.command_template for PDM-Lite"
            )
        profiles = adaptation.get("models", {})
        if not isinstance(profiles, dict):
            profiles = {}
        bench_cfg = cfg.get("bench2drive", {})
        active_model = str(bench_cfg.get("agent", "")).strip().lower() if isinstance(bench_cfg, dict) else ""
        train_all = bool(adaptation.get("train_all_models", False))
        selected_profiles = profiles if train_all else {active_model: profiles.get(active_model)}
        for model_name, profile in selected_profiles.items():
            if not isinstance(profile, dict) or not str(profile.get("train_command", "")).strip():
                raise ValueError(
                    f"Live adaptation requires adaptation.models.{model_name}.train_command"
                )


def resolve_metric_keys(cfg: dict[str, Any]) -> list[str]:
    metrics_cfg = cfg.get("metrics", {})
    if isinstance(metrics_cfg, dict):
        raw_keys = metrics_cfg.get("keys")
        if isinstance(raw_keys, list) and raw_keys:
            return [str(x) for x in raw_keys]

        config_path = metrics_cfg.get("config_path")
        if config_path:
            payload = load_config(Path(str(config_path)))
            raw_metrics = payload.get("metrics")
            if isinstance(raw_metrics, list) and raw_metrics:
                return [str(x) for x in raw_metrics]

    return list(DEFAULT_METRIC_KEYS)


def resolve_collision_rate_scale(cfg: dict[str, Any]) -> str:
    metrics_cfg = cfg.get("metrics", {})
    if isinstance(metrics_cfg, dict):
        scale = str(metrics_cfg.get("collision_rate_scale", "")).strip().lower()
        if not scale and metrics_cfg.get("config_path"):
            metrics_path = Path(str(metrics_cfg["config_path"]))
            if metrics_path.exists():
                metrics_payload = load_config(metrics_path)
                if isinstance(metrics_payload, dict):
                    scale = str(metrics_payload.get("collision_rate_scale", "percent")).strip().lower()
        if scale in {"fraction", "ratio", "unit", "percent", "%"}:
            return "fraction" if scale in {"fraction", "ratio", "unit"} else "percent"
    return "percent"


def resolve_bench2drive_root(cfg: dict[str, Any]) -> Path:
    eval_cfg = cfg.get("bench2drive", {})
    candidates = [
        eval_cfg.get("root"),
        os.environ.get("BENCH2DRIVE_ROOT"),
        Path(__file__).resolve().parents[2] / "Bench2Drive",
        "Bench2Drive",
        "third_party/Bench2Drive",
    ]
    for candidate in candidates:
        if not candidate:
            continue
        root = Path(str(candidate)).expanduser()
        resolved = normalize_bench2drive_root(root)
        if resolved is not None:
            return resolved.resolve()
    raise FileNotFoundError(
        "Bench2Drive not found. Set bench2drive.root, export BENCH2DRIVE_ROOT, "
        "or run `bash scripts/setup_bench2drive.sh ../Bench2Drive`."
    )


def normalize_bench2drive_root(root: Path) -> Path | None:
    if not root.exists():
        return None

    candidates = [
        root,
        root / "Bench2Drive",
    ]
    for candidate in candidates:
        if (candidate / "leaderboard").exists() and (candidate / "scenario_runner").exists():
            return candidate
    return root if root.exists() else None


def validate_bench2drive_runtime(cfg: dict[str, Any]) -> Path:
    bench_root = resolve_bench2drive_root(cfg)
    eval_cfg = cfg.get("bench2drive", {})
    cmd_template = str(eval_cfg.get("eval_cmd", "")).strip()
    if not cmd_template:
        raise ValueError("bench2drive.eval_cmd is required for non-dry-run mode")

    current_py = sys.version_info
    if current_py < (3, 8) or current_py >= (3, 10):
        raise RuntimeError(
            "Bench2Drive-first single-environment mode expects Python 3.8 or 3.9. "
            f"Current interpreter is {current_py.major}.{current_py.minor}. "
            "Recreate the uv environment with a Bench2Drive-compatible Python, for example "
            "`uv venv --python 3.8` or `uv python install 3.8`."
        )

    if importlib.util.find_spec("pkg_resources") is None:
        raise RuntimeError(
            "Bench2Drive evaluator requires `pkg_resources`, but it is not available in the current Python environment. "
            "Install `setuptools` into the active uv environment and run `uv sync` again."
        )

    evaluator_path = None
    known_candidates = [
        bench_root / "leaderboard" / "leaderboard" / "leaderboard_evaluator.py",
        bench_root / "leaderboard" / "leaderboard_evaluator.py",
    ]
    for candidate in known_candidates:
        if candidate.exists():
            evaluator_path = candidate
            break

    if evaluator_path is None:
        raise FileNotFoundError(
            "Bench2Drive evaluator not found under the configured bench2drive.root. "
            f"Checked: {', '.join(str(path) for path in known_candidates)}"
        )

    resolve_carla_runtime_source(cfg)

    agent_path = resolve_agent_path(cfg, bench_root)
    if agent_path is None:
        raise ValueError(
            "bench2drive.agent is required for non-dry-run mode because leaderboard_evaluator.py requires `--agent`. "
            "No official candidate agent file was found under the Bench2Drive root. "
            f"Checked: {', '.join(str(path) for path in discover_official_agent_candidates(bench_root))}. "
            f"Official script conventions: {'; '.join(OFFICIAL_AGENT_SCRIPT_HINTS)}."
        )
    if not agent_path.exists():
        raise FileNotFoundError(f"Bench2Drive agent file not found: {agent_path}")

    validate_agent_config_value(cfg, bench_root)
    validate_agent_runtime_dependencies(cfg, bench_root)
    # The route supervisor owns the restart path and validates CARLA before
    # every route; a crashed prior server is an expected resume condition.
    if not bool(cfg.get("evaluation", {}).get("skip_failed_routes", False)):
        validate_carla_server(cfg)
    return bench_root


def validate_carla_server(cfg: dict[str, Any]) -> None:
    carla_cfg = cfg.get("carla", {})
    if not isinstance(carla_cfg, dict) or not bool(carla_cfg.get("require_server", False)):
        return
    host = str(carla_cfg.get("host", "127.0.0.1"))
    port = int(carla_cfg.get("port", 2000))
    timeout_s = float(carla_cfg.get("connection_timeout_s", 3.0))
    try:
        with socket.create_connection((host, port), timeout=timeout_s):
            return
    except OSError as exc:
        raise RuntimeError(
            f"CARLA server is not reachable at {host}:{port}. Start the configured CARLA instance before live evaluation."
        ) from exc


def resolve_agent_path(cfg: dict[str, Any], bench_root: Path) -> Path | None:
    bench_cfg = cfg.get("bench2drive", {})
    raw = str(bench_cfg.get("agent_path") or bench_cfg.get("agent", "")).strip()
    if raw:
        raw = resolve_agent_alias(raw)
        path = Path(raw).expanduser()
        return path if path.is_absolute() else (bench_root / path)

    for candidate in discover_official_agent_candidates(bench_root):
        if candidate.exists():
            return candidate
    return None


def resolve_agent_config_path(cfg: dict[str, Any], bench_root: Path) -> Path | None:
    raw = str(cfg.get("bench2drive", {}).get("agent_config", "")).strip()
    if not raw:
        agent_raw = str(cfg.get("bench2drive", {}).get("agent", "")).strip().lower()
        alias_payload = MODEL_AGENT_ALIASES.get(agent_raw)
        if alias_payload:
            raw = alias_payload["agent_config"]
        else:
            return None
    if "+" in raw:
        return None
    path = Path(raw).expanduser()
    return path if path.is_absolute() else (bench_root / path)


def resolve_agent_config_value(cfg: dict[str, Any], bench_root: Path) -> str:
    raw = str(cfg.get("bench2drive", {}).get("agent_config", "")).strip()
    if not raw:
        agent_raw = str(cfg.get("bench2drive", {}).get("agent", "")).strip().lower()
        alias_payload = MODEL_AGENT_ALIASES.get(agent_raw)
        if alias_payload:
            raw = alias_payload["agent_config"]
    if not raw:
        return ""
    if "+" not in raw:
        path = Path(raw).expanduser()
        return str((path if path.is_absolute() else (bench_root / path)).resolve())

    config_part, checkpoint_part = raw.split("+", 1)
    resolved_config = Path(config_part).expanduser()
    resolved_checkpoint = Path(checkpoint_part).expanduser()
    resolved_config = (resolved_config if resolved_config.is_absolute() else (bench_root / resolved_config)).resolve()
    resolved_checkpoint = (resolved_checkpoint if resolved_checkpoint.is_absolute() else (bench_root / resolved_checkpoint)).resolve()
    return f"{resolved_config}+{resolved_checkpoint}"


def resolve_agent_alias(raw: str) -> str:
    alias_payload = MODEL_AGENT_ALIASES.get(raw.lower())
    if alias_payload:
        return alias_payload["agent"]
    return raw


def validate_agent_config_value(cfg: dict[str, Any], bench_root: Path) -> None:
    raw = str(cfg.get("bench2drive", {}).get("agent_config", "")).strip()
    agent_raw = str(cfg.get("bench2drive", {}).get("agent", "")).strip().lower()
    alias_payload = MODEL_AGENT_ALIASES.get(agent_raw)
    if not raw and alias_payload:
        raw = alias_payload["agent_config"]
    if not raw:
        return

    if "+" in raw:
        config_part, checkpoint_part = raw.split("+", 1)
        if "YOUR_" in checkpoint_part:
            raise ValueError(
                f"bench2drive.agent={agent_raw} requires a real checkpoint path in bench2drive.agent_config. "
                f"Current value is still a placeholder: {raw}"
            )
        config_path = Path(config_part).expanduser()
        checkpoint_path = Path(checkpoint_part).expanduser()
        config_path = (config_path if config_path.is_absolute() else (bench_root / config_path)).resolve()
        checkpoint_path = (checkpoint_path if checkpoint_path.is_absolute() else (bench_root / checkpoint_path)).resolve()
        if not config_path.exists():
            raise FileNotFoundError(f"Bench2Drive agent config file not found: {config_path}")
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Bench2Drive checkpoint file not found: {checkpoint_path}")
        return

    path = Path(raw).expanduser()
    path = (path if path.is_absolute() else (bench_root / path)).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Bench2Drive agent config file not found: {path}")


def validate_agent_runtime_dependencies(cfg: dict[str, Any], bench_root: Path) -> None:
    agent_raw = str(cfg.get("bench2drive", {}).get("agent", "")).strip().lower()
    if agent_raw not in {"uniad", "uniad_tiny", "vad"}:
        return

    zoo_root = resolve_bench2drive_zoo_root(cfg, bench_root)
    missing_modules: list[str] = []
    source_fallbacks = {
        "mmcv": zoo_root / "mmcv" if zoo_root is not None else None,
        "Bench2DriveZoo": zoo_root / "adzoo" if zoo_root is not None else None,
    }
    for module_name, source_path in source_fallbacks.items():
        source_available = source_path is not None and source_path.exists()
        if importlib.util.find_spec(module_name) is None and not source_available:
            missing_modules.append(module_name)

    if missing_modules:
        raise RuntimeError(
            "The active uv environment is missing model runtime dependencies required by "
            f"`bench2drive.agent={agent_raw}`: {', '.join(missing_modules)}. "
            "Run `bash scripts/setup_uniad_uv.sh` and ensure the Bench2DriveZoo editable install succeeds before retrying."
        )


def discover_official_agent_candidates(bench_root: Path) -> list[Path]:
    return [
        bench_root / "leaderboard" / "team_code" / "your_team_agent.py",
        bench_root / "team_code" / "uniad_b2d_agent.py",
        bench_root / "team_code" / "vad_b2d_agent.py",
        bench_root / "team_code" / "tcp_b2d_agent.py",
        bench_root / "team_code" / "admlp_b2d_agent.py",
        bench_root / "leaderboard" / "leaderboard" / "autoagents" / "npc_agent.py",
        bench_root / "leaderboard" / "leaderboard" / "autoagents" / "dummy_agent.py",
        bench_root / "leaderboard" / "leaderboard" / "autoagents" / "human_agent.py",
    ]


def resolve_bench2drive_zoo_root(cfg: dict[str, Any], bench_root: Path) -> Path | None:
    raw = str(cfg.get("bench2drive", {}).get("zoo_root", "")).strip()
    if raw:
        candidate = Path(raw).expanduser()
        return candidate if candidate.is_absolute() else (bench_root / candidate)

    sibling = bench_root.parent / "Bench2DriveZoo"
    if sibling.exists():
        return sibling
    return None


def build_bench2drive_runtime_env(cfg: dict[str, Any], bench_root: Path) -> dict[str, str]:
    bench_cfg = cfg.get("bench2drive", {})
    if not isinstance(bench_cfg, dict):
        bench_cfg = {}
    leaderboard_root = bench_root / "leaderboard"
    scenario_runner_root = bench_root / "scenario_runner"
    team_code_root = leaderboard_root / "team_code"
    zoo_root = resolve_bench2drive_zoo_root(cfg, bench_root)
    carla_source = resolve_carla_runtime_source(cfg)
    carla_cfg = cfg.get("carla", {})

    extra_pythonpath_raw = bench_cfg.get("extra_pythonpath", [])
    if isinstance(extra_pythonpath_raw, (str, Path)):
        extra_pythonpath_raw = [extra_pythonpath_raw]
    if not isinstance(extra_pythonpath_raw, list):
        raise ValueError("bench2drive.extra_pythonpath must be a list of paths")
    extra_pythonpath: list[str] = []
    for raw_path in extra_pythonpath_raw:
        path = Path(str(raw_path)).expanduser()
        if not path.is_absolute():
            path = bench_root / path
        extra_pythonpath.append(str(path.resolve()))

    pythonpath_entries = [
        *extra_pythonpath,
        str(leaderboard_root),
        str(team_code_root),
        str(scenario_runner_root),
    ]
    if zoo_root is not None:
        # Bench2DriveZoo is often a source checkout rather than an installed
        # distribution. Its top-level ``mmcv`` package needs the checkout
        # itself on PYTHONPATH; retain the parent for sibling-import layouts.
        pythonpath_entries = [str(zoo_root), str(zoo_root.parent), *pythonpath_entries]
    env = {
        "SCENARIO_RUNNER_ROOT": str(scenario_runner_root),
        "LEADERBOARD_ROOT": str(leaderboard_root),
    }
    configured_environment = bench_cfg.get("environment", {})
    if configured_environment is not None:
        if not isinstance(configured_environment, dict):
            raise ValueError("bench2drive.environment must be an object")
        env.update(
            {
                str(key): str(value)
                for key, value in configured_environment.items()
                if str(key).strip()
            }
        )
    if zoo_root is not None:
        env["BENCH2DRIVE_ZOO_ROOT"] = str(zoo_root)
        env["BENCH2DRIVE_ZOO_PARENT"] = str(zoo_root.parent)

    if carla_source["source"] == "pythonapi_artifact":
        pythonapi_root = Path(str(carla_source["pythonapi_root"]))
        pythonapi_dir = pythonapi_root / "PythonAPI"
        pythonpath_entries = [
            str(pythonapi_dir),
            str(pythonapi_dir / "carla"),
            str(carla_source["artifact"]),
            *pythonpath_entries,
        ]
        env["CARLA_ROOT"] = str(pythonapi_root)
        env["CARLA_PYTHON_ARTIFACT"] = str(carla_source["artifact"])
    else:
        env["CARLA_PYTHON_SOURCE"] = "installed-package"
        pythonapi_root_raw = str(carla_cfg.get("pythonapi_root", "")).strip()
        if pythonapi_root_raw:
            pythonapi_root = Path(pythonapi_root_raw)
            pythonapi_dir = pythonapi_root / "PythonAPI"
            pythonpath_entries = [
                str(pythonapi_dir),
                str(pythonapi_dir / "carla"),
                *pythonpath_entries,
            ]
            env["CARLA_ROOT"] = str(pythonapi_root)

    existing_pythonpath = os.environ.get("PYTHONPATH", "")
    if existing_pythonpath:
        pythonpath_entries.append(existing_pythonpath)

    env["PYTHONPATH"] = os.pathsep.join(pythonpath_entries)
    return env


def select_carla_python_artifact(dist_dir: Path) -> Path:
    candidates = sorted(
        path for path in dist_dir.iterdir()
        if path.name.startswith("carla-")
        and path.suffix in {".egg", ".whl"}
        and re.search(r"carla-0\.9\.(14|15)", path.name)
    )
    if not candidates:
        raise RuntimeError(f"No Bench2Drive-compatible CARLA Python artifacts found under {dist_dir}")

    current_tag = f"cp{sys.version_info.major}{sys.version_info.minor}"
    current_py_tag = f"py{sys.version_info.major}.{sys.version_info.minor}"
    compatible = [
        path for path in candidates
        if current_tag in path.name or current_py_tag in path.name
    ]
    if not compatible:
        raise RuntimeError(
            f"No CARLA artifact compatible with Python {sys.version_info.major}.{sys.version_info.minor} "
            f"was found under {dist_dir}; install carla==0.9.15 into the active environment "
            "rather than loading a wheel built for another Python ABI."
        )
    return max(compatible, key=lambda path: ("carla-0.9.15" in path.name, path.name))


def resolve_carla_runtime_source(cfg: dict[str, Any]) -> dict[str, Any]:
    if _carla_binding_is_usable():
        return {"source": "installed_package"}

    carla_cfg = cfg.get("carla", {})
    pythonapi_root_raw = str(carla_cfg.get("pythonapi_root", "")).strip()
    if not pythonapi_root_raw:
        raise ValueError(
            "No installed `carla` package was found in the current environment, and carla.pythonapi_root is not configured. "
            "Either `pip install carla` into the active uv environment or point carla.pythonapi_root to a local CARLA directory that contains PythonAPI/."
        )

    pythonapi_root = Path(pythonapi_root_raw)
    pythonapi_dir = pythonapi_root / "PythonAPI"
    if not pythonapi_dir.exists():
        raise FileNotFoundError(f"CARLA PythonAPI directory not found: {pythonapi_dir}")

    dist_dir = pythonapi_dir / "carla" / "dist"
    if not dist_dir.exists():
        raise FileNotFoundError(
            "CARLA PythonAPI dist directory not found under the configured carla.pythonapi_root. "
            f"Checked: {dist_dir}"
        )
    try:
        artifact = select_carla_python_artifact(dist_dir)
    except RuntimeError as exc:
        raise RuntimeError(f"{exc} Available files: {', '.join(str(path.name) for path in sorted(dist_dir.iterdir()))}") from exc

    return {
        "source": "pythonapi_artifact",
        "pythonapi_root": pythonapi_root,
        "artifact": artifact,
    }


def _carla_binding_is_usable() -> bool:
    """Distinguish a real CARLA binding from an empty namespace package.

    Some Bench2Drive checkouts contain ``PythonAPI/carla`` as a namespace
    directory.  ``find_spec('carla')`` reports that directory as installed even
    though it has no ``Client`` or actor classes; treating it as usable hides
    the configured local CARLA artifact and fails later with opaque imports.
    """
    try:
        spec = importlib.util.find_spec("carla")
        if spec is None:
            return False
        # Keep this helper compatible with tests and embedders which provide a
        # lightweight sentinel from find_spec instead of a real ModuleSpec.
        if not hasattr(spec, "loader"):
            return True
        if spec.loader is None:
            return False
        module = importlib.import_module("carla")
        return all(hasattr(module, name) for name in ("Client", "Vehicle", "TrafficLightState"))
    except (ImportError, AttributeError, ValueError):
        return False


def resolve_run_root(cfg: dict[str, Any], args: argparse.Namespace) -> Path:
    if args.resume_run:
        run_root = args.resume_run.expanduser().resolve()
        if not run_root.exists():
            raise FileNotFoundError(f"Resume run directory not found: {run_root}")
        return run_root

    output_root = args.output_root or Path(str(cfg["output_root"]))
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    run_root = (output_root / run_id).expanduser().resolve()
    if run_root.exists():
        raise FileExistsError(f"Run directory already exists: {run_root}")
    return run_root


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def collect_resume_state(run_root: Path, num_rounds: int) -> tuple[int, list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]] | None]:
    all_rounds: list[dict[str, Any]] = []
    all_structured: list[dict[str, Any]] = []
    all_evolved: list[dict[str, Any]] = []
    scenario_bank: list[dict[str, Any]] | None = None

    for round_idx in range(num_rounds):
        round_dir = run_root / f"round{round_idx}"
        round_summary_path = round_dir / "round_summary.json"
        if not round_summary_path.exists():
            return round_idx, all_rounds, all_structured, all_evolved, scenario_bank

        all_rounds.append(json.loads(round_summary_path.read_text(encoding="utf-8")))

        structured_path = round_dir / "failures_structured.json"
        if structured_path.exists():
            all_structured.extend(json.loads(structured_path.read_text(encoding="utf-8")))

        evolved_path = round_dir / "evolved_scenarios.json"
        if evolved_path.exists():
            all_evolved.extend(json.loads(evolved_path.read_text(encoding="utf-8")))

        bank_path = round_dir / "scenario_bank_post_round.json"
        if bank_path.exists():
            scenario_bank = load_scenario_bank(bank_path)

    return num_rounds, all_rounds, all_structured, all_evolved, scenario_bank


def write_run_manifest(
    run_root: Path,
    *,
    args: argparse.Namespace,
    metric_keys: list[str],
    interpretation_prompt_path: Path,
    cfg: dict[str, Any],
    failure_extract_cfg: dict[str, Any],
    status: str,
    completed_rounds: int,
) -> None:
    manifest = {
        "project": "Failure-Driven Scenario Evolution",
        "protocol": str(cfg.get("protocol", "downstream_adaptation")),
        "protocol_order": (
            ["native_scenic_evaluation", "benchmark_failure_feedback", "failure_interpretation",
             "clip_scenic_retrieval", "scenic_evolution"]
            if str(cfg.get("protocol")) == "fixed_policy_evolution"
            and bool(cfg.get("evaluation", {}).get("use_held_out_routes", False))
            else ["native_scenic_evaluation", "failure_interpretation",
                  "clip_scenic_retrieval", "scenic_evolution"]
            if str(cfg.get("protocol")) == "fixed_policy_evolution"
            else ["benchmark_evaluation", "failure_feedback", "failure_interpretation",
                  "clip_scenic_retrieval", "scenic_evolution", "pdm_lite_expert_data",
                  "model_finetune", "benchmark_re_evaluation"]
        ),
        "config_path": str(args.config),
        "dry_run": bool(args.dry_run),
        "mode": args.mode,
        "metric_keys": metric_keys,
        "interpretation_prompt_path": str(interpretation_prompt_path),
        "interpretation_backend": str(cfg.get("interpretation", {}).get("backend", "rule_based")),
        "failure_extract": {
            "format": failure_extract_cfg.get("format", "jsonl"),
            "records_path": failure_extract_cfg.get("records_path"),
        },
        "evaluation": cfg.get("evaluation", {}),
        "retrieval_backend": cfg.get("retrieval", {}).get("backend", "lexical"),
        "collision_rate_scale": resolve_collision_rate_scale(cfg),
        "status": status,
        "completed_rounds": completed_rounds,
        "run_root": str(run_root),
        "resumed": bool(args.resume_run),
    }
    write_json(run_root / "run_manifest.json", manifest)


def validate_official_benchmark_complete(checkpoint_path: Path, routes_xml: Path) -> int:
    """Reject partial official scores even when the evaluator exits successfully."""
    expected = len(ET.parse(routes_xml).getroot().findall("route"))
    if expected < 1:
        raise ValueError(f"No routes in benchmark evaluation XML: {routes_xml}")
    payload = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    checkpoint = payload.get("_checkpoint", {})
    progress = checkpoint.get("progress", [])
    records = checkpoint.get("records", [])
    if (payload.get("entry_status") != "Finished"
            or progress != [expected, expected]
            or not isinstance(records, list) or len(records) != expected):
        raise RuntimeError(
            f"Incomplete benchmark evaluation: expected {expected} finished routes, "
            f"entry_status={payload.get('entry_status')!r}, progress={progress!r}, "
            f"records={len(records) if isinstance(records, list) else 'invalid'}; "
            f"inspect {checkpoint_path}"
        )
    # The evaluator's 4000-tick limit yields TickRuntime as a scored route
    # failure, not a simulator crash. Keep its measured partial route score.
    execution_errors = [
        str(record.get("status", ""))
        for record in records
        if isinstance(record, dict) and any(
            marker in str(record.get("status", ""))
            for marker in ("Simulation crashed", "Agent crashed", "sensors were invalid")
        )
    ]
    if execution_errors:
        raise RuntimeError(
            f"Benchmark route execution failed: {execution_errors[0]}; "
            f"checkpoint {checkpoint_path} cannot be reported as a completed score"
        )
    return expected


def run_eval_round(
    round_idx: int,
    out_dir: Path,
    *,
    dry_run: bool,
    cfg: dict[str, Any],
    scenario_file: Path,
    metric_keys: list[str],
    collision_rate_scale: str = "percent",
    stage: str = "baseline",
    agent_config_override: str | Path | None = None,
) -> dict[str, float]:
    out_dir.mkdir(parents=True, exist_ok=True)
    if dry_run:
        adaptation_bonus = 0.7 if str(stage).strip().lower() in {"post_adaptation", "retest"} else 0.0
        base = 34.0 + (round_idx * 1.3) + adaptation_bonus
        metrics = {k: 0.0 for k in metric_keys}
        collision_rate = 0.08 - round_idx * 0.002 - (0.003 if adaptation_bonus else 0.0)
        if collision_rate_scale == "percent":
            collision_rate *= 100.0
        metrics.update(
            {
                "DS": base,
                "SR": 12.0 + round_idx * 0.9,
                "Eff": 98.0 + round_idx * 0.4,
                "CR": collision_rate,
                "Comf": 70.0,
            }
        )
        write_json(out_dir / "metrics.json", metrics)
        write_json(
            out_dir / "evaluation_contract.json",
            {
                "stage": stage,
                "scenario_file": str(scenario_file.resolve()),
                "official_benchmark_feedback_only": bool(
                    cfg.get("evaluation", {}).get("use_held_out_routes", False)
                    if isinstance(cfg.get("evaluation", {}), dict)
                    else False
                ),
            },
        )
        return metrics

    bench_root = validate_bench2drive_runtime(cfg)
    eval_cfg = cfg.get("bench2drive", {})
    cmd_template = str(eval_cfg.get("eval_cmd", "")).strip()
    if not cmd_template:
        raise ValueError("bench2drive.eval_cmd is required for non-dry-run mode")

    cmd = build_eval_command(
        cmd_template,
        bench2drive_root=bench_root,
        round_id=round_idx,
        scenario_file=scenario_file,
        out_dir=out_dir,
        host=str(cfg.get("carla", {}).get("host", "127.0.0.1")),
        port=int(cfg.get("carla", {}).get("port", 2000)),
        traffic_manager_port=int(cfg.get("carla", {}).get("tm_port", 8000)),
        track=str(cfg.get("bench2drive", {}).get("track", "SENSORS")),
        agent=str(resolve_agent_path(cfg, bench_root) or ""),
        agent_config=(
            str(agent_config_override)
            if agent_config_override is not None
            else resolve_agent_config_value(cfg, bench_root)
        ),
        gpu_rank=int(cfg.get("bench2drive", {}).get("gpu_rank", 0)),
        resume=bool(cfg.get("bench2drive", {}).get("resume", True)),
    )
    (out_dir / "eval_command.sh").write_text(f"{cmd}\n", encoding="utf-8")
    runtime_env = build_bench2drive_runtime_env(cfg, bench_root)
    failure_extract_cfg = cfg.get("failure_extract", {})
    if isinstance(failure_extract_cfg, dict) and bool(failure_extract_cfg.get("capture_agent_evidence", False)):
        evidence_template = str(
            failure_extract_cfg.get("evidence_root_template", "{out_dir}/evidence")
        )
        evidence_root = Path(
            evidence_template.format(
                round_id=round_idx,
                run_root=str(out_dir.parent),
                round_dir=str(out_dir),
                out_dir=str(out_dir),
                scenario_file=str(scenario_file),
            )
        ).expanduser()
        runtime_env.update(
            {
                "IS_BENCH2DRIVE": "True",
                "ROUTES": str(scenario_file.resolve()),
                "SAVE_PATH": str(evidence_root.resolve()),
            }
        )
    write_json(out_dir / "eval_environment.json", runtime_env)
    evaluation_cfg = cfg.get("evaluation", {})
    skip_failed_routes = bool(evaluation_cfg.get("skip_failed_routes", False))
    coverage: dict[str, Any] | None = None
    if skip_failed_routes:
        carla_cfg = cfg.get("carla", {})
        attempts = run_remaining_routes(
            cmd,
            bench_root=bench_root,
            scenario_file=scenario_file,
            output_dir=out_dir,
            environment=runtime_env,
            host=str(carla_cfg.get("host", "127.0.0.1")),
            port=int(carla_cfg.get("port", 2000)),
            server_launcher=Path(
                carla_cfg.get("server_launcher", Path(__file__).resolve().parents[2] / "carla/CarlaUE4.sh")
            ).resolve(),
            server_gpu_adapter=int(carla_cfg.get("graphicsadapter", 0)),
            route_timeout_s=int(evaluation_cfg.get("route_timeout_s", 1800)),
            failure_extract_config=(
                dict(failure_extract_cfg)
                if bool(cfg.get("evidence_retention", {}).get("prune_new_frames_after_route", False))
                else None
            ),
            evidence_max_images=int(cfg.get("interpretation", {}).get("max_images", 3)),
        )
        metrics_file = out_dir / "partial_ckpt.json"
        combined = build_partial_checkpoint(
            attempts["records"], attempts["skipped"], attempts["expected"], metrics_file
        )
        coverage = combined["audit"]
    else:
        run_eval_command(cmd, cwd=bench_root, env=runtime_env, log_file=out_dir / "evaluation.log")
        metrics_file = out_dir / str(eval_cfg.get("metrics_file", "metrics.json"))
    if not metrics_file.is_file():
        raise FileNotFoundError(f"Bench2Drive evaluation produced no checkpoint: {metrics_file}")
    if bool(evaluation_cfg.get("verify_route_completion", False)) and (
        not skip_failed_routes or not coverage["skipped_count"]
    ):
        validate_official_benchmark_complete(metrics_file, scenario_file)
    strict = bool(evaluation_cfg.get("verify_route_completion", False))
    motion_keys = {"Eff", "Comf"}.intersection(metric_keys) if strict else set()
    checkpoint_keys = [key for key in metric_keys if key not in motion_keys]
    metrics = parse_metrics(
        metrics_file,
        keys=checkpoint_keys,
        collision_rate_scale=collision_rate_scale,
        strict=strict,
    )
    if motion_keys:
        evidence_root = runtime_env.get("SAVE_PATH")
        if not evidence_root:
            raise RuntimeError("Official Eff/Comf require captured Bench2Drive agent metric_info.json")
        motion = calculate_official_motion_metrics(metrics_file, Path(evidence_root), bench_root)
        metrics.update({key: motion[key] for key in motion_keys})
    write_json(out_dir / "metrics.json", metrics)
    write_json(
        out_dir / "evaluation_contract.json",
        {
            "stage": stage,
            "scenario_file": str(scenario_file.resolve()),
            "official_benchmark_feedback_only": bool(
                cfg.get("evaluation", {}).get("use_held_out_routes", False)
                if isinstance(cfg.get("evaluation", {}), dict)
                else False
            ),
            "checkpoint_file": str(metrics_file.resolve()),
            "official_complete": coverage is None or coverage["skipped_count"] == 0,
            "coverage": coverage,
        },
    )
    return metrics


def extract_round_failures(
    round_idx: int,
    *,
    dry_run: bool,
    cfg: dict[str, Any],
    run_root: Path,
    round_dir: Path,
    evaluation_dir: Path | None = None,
):
    if dry_run:
        return generate_mock_failures()

    source_dir = evaluation_dir or round_dir
    template = str(cfg.get("round_log_template", "{round_dir}/ckpt.json"))
    log_path = Path(
        template.format(round_id=round_idx, run_root=run_root, round_dir=source_dir, out_dir=source_dir)
    )
    if bool(cfg.get("evaluation", {}).get("skip_failed_routes", False)):
        contract_file = source_dir / "evaluation_contract.json"
        if contract_file.is_file():
            contract = json.loads(contract_file.read_text(encoding="utf-8"))
            log_path = Path(contract["checkpoint_file"])
    failure_extract_cfg = dict(cfg.get("failure_extract", {}))
    failure_extract_cfg["__log_file__"] = str(log_path)
    evidence_template = str(failure_extract_cfg.get("evidence_root_template", "")).strip()
    if evidence_template and not str(failure_extract_cfg.get("metric_info_root", "")).strip():
        failure_extract_cfg["metric_info_root"] = evidence_template.format(
            round_id=round_idx,
            run_root=str(run_root),
            round_dir=str(source_dir),
            out_dir=str(source_dir),
        )
    failures = extract_failures_from_log(log_path, failure_extract_cfg)
    return failures


def build_summary(mode: str, metric_keys: list[str], all_rounds: list[dict[str, Any]]) -> dict[str, Any]:
    if not all_rounds:
        return {
            "mode": mode,
            "num_rounds": 0,
            "metrics_keys": metric_keys,
            "rounds": [],
            "improvement": {k: 0.0 for k in metric_keys},
        }

    comparable = not any(
        row.get("official_benchmark_complete") is False
        or any(
            isinstance(row.get(key), dict) and row[key].get("skipped_count", 0)
            for key in ("benchmark_coverage", "post_adaptation_coverage")
        )
        for row in all_rounds
    )
    first_metrics = all_rounds[0].get("baseline_metrics", all_rounds[0].get("metrics", {}))
    last_metrics = all_rounds[-1].get("metrics", {})
    improvement = (
        {k: float(last_metrics.get(k, 0.0)) - float(first_metrics.get(k, 0.0)) for k in metric_keys}
        if comparable else None
    )

    return {
        "mode": mode,
        "num_rounds": len(all_rounds),
        "metrics_keys": metric_keys,
        "rounds": all_rounds,
        "improvement": improvement,
    }


def _active_model_name(cfg: dict[str, Any]) -> str:
    bench_cfg = cfg.get("bench2drive", {})
    if isinstance(bench_cfg, dict):
        agent = str(bench_cfg.get("agent", "")).strip().lower()
        if agent in {"uniad", "vad", "admlp", "tcp"}:
            return agent
    return ""


def _format_training_command(
    template: str,
    *,
    model_name: str,
    selected_path: Path,
    expert_data_path: Path,
    output_dir: Path,
    seed: int,
    round_id: int,
    previous_checkpoint: str,
    checkpoint_output: Path,
    scenic_program_dir: Path | None,
    scenic_execution_manifest: Path | None,
    routes_xml: Path | None,
) -> str:
    values = {
        "model_name": model_name,
        "scenario_bank": str(selected_path.resolve()),
        "expert_demos": str(expert_data_path.resolve()),
        "expert_data": str(expert_data_path.resolve()),
        "output_dir": str(output_dir.resolve()),
        "seed": seed,
        "round_id": round_id,
        "previous_checkpoint": previous_checkpoint,
        "checkpoint_output": str(checkpoint_output.resolve()),
        "scenic_program_dir": str(scenic_program_dir.resolve()) if scenic_program_dir else "",
        "scenic_execution_manifest": (
            str(scenic_execution_manifest.resolve()) if scenic_execution_manifest else ""
        ),
        "routes_xml": str(routes_xml.resolve()) if routes_xml else "",
    }
    return template.format(**values)


def prepare_adaptation_artifacts(
    run_root: Path,
    scenario_bank: list[dict[str, Any]],
    cfg: dict[str, Any],
    *,
    round_dir: Path | None = None,
    round_id: int | None = None,
    dry_run: bool = False,
    adaptation_rows: list[dict[str, Any]] | None = None,
    previous_checkpoint: str | None = None,
) -> dict[str, Any] | None:
    """Run one Scenic -> PDM-Lite -> fine-tuning adaptation stage.

    ``bench2drive220.xml`` is feedback and re-evaluation material only, never
    sent to expert-data or training commands. Native Scenic expert collection
    consumes its execution manifest; route-XML collectors are rejected for
    native Scenic scenarios.
    """
    adaptation_cfg = cfg.get("adaptation", {})
    if not isinstance(adaptation_cfg, dict) or not bool(adaptation_cfg.get("enabled", False)):
        return None
    evaluation_cfg = cfg.get("evaluation", {})
    if not isinstance(evaluation_cfg, dict):
        evaluation_cfg = {}
    held_out = Path(str(evaluation_cfg.get("held_out_routes_xml", ""))).expanduser()
    if not held_out.exists():
        raise FileNotFoundError(f"Adaptation held-out routes XML not found: {held_out}")
    if adaptation_rows is None:
        selected_source = select_adaptation_scenarios(
            scenario_bank,
            max_scenarios=adaptation_cfg.get("max_scenarios"),
            seed=int(adaptation_cfg.get("seed", 2026)),
        )
    else:
        selected_source = select_adaptation_scenarios(
            list(adaptation_rows),
            max_scenarios=adaptation_cfg.get("max_scenarios"),
            seed=int(adaptation_cfg.get("seed", 2026)),
        )
    if not selected_source:
        return None

    base_output_dir = round_dir or run_root
    output_dir = base_output_dir / str(adaptation_cfg.get("output_dir", "adaptation"))
    if round_dir is not None:
        # Avoid nesting round0/round0 when output_dir is configured as a
        # relative name.  A round's adaptation artifacts belong inside that
        # round directory.
        output_dir = base_output_dir / str(adaptation_cfg.get("output_dir", "adaptation"))
    manifest = build_adaptation_manifest(
        selected_source,
        held_out_routes_xml=held_out,
        output_dir=output_dir,
        config=adaptation_cfg,
        round_id=round_id,
    )
    selected_path = output_dir / "adaptation_scenarios.json"
    write_json(selected_path, selected_source)

    materialized = materialize_adaptation_inputs(
        selected_source,
        output_dir / "scenario_inputs",
        cfg,
    )
    manifest["scenic_materialization"] = {
        key: str(value)
        for key, value in materialized.items()
        if key in {"scenic_program_dir", "scenic_execution_manifest", "scenic_library_index", "bench2drive_xml"}
    }

    expert_cfg = adaptation_cfg.get("expert", {})
    if not isinstance(expert_cfg, dict):
        expert_cfg = {}
    backend = str(expert_cfg.get("backend", "external")).strip().lower()
    demos_path = output_dir / str(expert_cfg.get("demos_file", "expert_demos.jsonl"))
    expert_status = "planned"
    expert_command = ""
    if dry_run:
        write_proxy_expert_demos(selected_source, demos_path)
        expert_status = "dry_run_proxy"
    elif backend in {"proxy", "kinematic_proxy", "synthetic"}:
        if not bool(expert_cfg.get("allow_proxy", False)):
            raise ValueError(
                "A proxy expert is not a paper reproduction. Set adaptation.expert.allow_proxy=true "
                "only for an explicit non-reproducibility smoke test."
            )
        write_proxy_expert_demos(selected_source, demos_path)
        expert_status = "proxy"
    else:
        expert_template = str(expert_cfg.get("command_template", "")).strip()
        if not expert_template:
            raise ValueError(
                "adaptation.expert.command_template is required for live Scenic/PDM-Lite data generation"
            )
        if "{routes_xml}" in expert_template and (
            materialized.get("bench2drive_xml") is None
            or any(
                row.get("config", {}).get("scenic_program_format") == "scenic_native_v1"
                for row in selected_source
            )
        ):
            raise ValueError(
                "PDM-Lite route-XML collection cannot collect native Scenic programs. "
                "Use a native Scenic expert collector with {scenic_execution_manifest}; "
                "the official Bench2Drive routes are evaluation-only and cannot be "
                "substituted as training data."
            )
        expert_command = build_expert_generation_command(
            expert_template,
            scenario_bank_path=selected_path,
            output_dir=output_dir,
            seed=int(adaptation_cfg.get("seed", 2026)),
            scenic_program_dir=materialized.get("scenic_program_dir"),
            scenic_execution_manifest=materialized.get("scenic_execution_manifest"),
            routes_xml=materialized.get("bench2drive_xml"),
            demos_file=demos_path,
            round_id=round_id,
            expert_camera_profile=_active_model_name(cfg),
            carla_host=str(cfg.get("carla", {}).get("host", "")) if isinstance(cfg.get("carla", {}), dict) else None,
            carla_port=int(cfg.get("carla", {}).get("port", 0)) if isinstance(cfg.get("carla", {}), dict) else None,
            carla_tm_port=int(cfg.get("carla", {}).get("tm_port", 0)) if isinstance(cfg.get("carla", {}), dict) else None,
            carla_pythonapi_root=str(cfg.get("carla", {}).get("pythonapi_root", "")) if isinstance(cfg.get("carla", {}), dict) else None,
            carla_root=str(cfg.get("carla", {}).get("root", "")) if isinstance(cfg.get("carla", {}), dict) else None,
        )
        (output_dir / "expert_command.sh").write_text(expert_command + "\n", encoding="utf-8")
        run_expert_generation(expert_command, cwd=Path(__file__).resolve().parents[1])
        expert_status = "completed"
    if not demos_path.exists() and not str(expert_cfg.get("data_path", "")).strip():
        raise FileNotFoundError(
            "PDM-Lite command completed without producing adaptation.expert.demos_file: "
            f"{demos_path}"
        )
    expert_data_path = (
        Path(str(expert_cfg["data_path"])).expanduser()
        if str(expert_cfg.get("data_path", "")).strip()
        else demos_path
    )
    if not expert_data_path.is_absolute():
        expert_data_path = (output_dir / expert_data_path).resolve()
    if not expert_data_path.exists():
        raise FileNotFoundError(
            "PDM-Lite command completed without producing adaptation expert data: "
            f"{expert_data_path}"
        )

    model_profiles = adaptation_cfg.get("models", {})
    if not isinstance(model_profiles, dict):
        model_profiles = {}
    active_model = _active_model_name(cfg)
    train_all = bool(adaptation_cfg.get("train_all_models", False))
    if active_model and not train_all and active_model in model_profiles:
        selected_models = {active_model: model_profiles[active_model]}
    else:
        selected_models = model_profiles

    checkpoint_paths: dict[str, str] = {}
    training_rows: list[dict[str, Any]] = []
    for model_name, profile in selected_models.items():
        if not isinstance(profile, dict):
            profile = {}
        checkpoint_output = output_dir / str(
            profile.get("checkpoint_output", f"checkpoints/{model_name}_round{round_id or 0}.pth")
        )
        train_template = str(profile.get("train_command", "")).strip()
        train_status = "planned"
        train_command = ""
        if dry_run:
            write_json(
                checkpoint_output,
                {
                    "model": str(model_name),
                    "round": round_id,
                    "source": "dry_run_finetune_placeholder",
                    "previous_checkpoint": previous_checkpoint or "",
                },
            )
            train_status = "dry_run_placeholder"
        else:
            if not train_template:
                raise ValueError(
                    f"adaptation.models.{model_name}.train_command is required for live fine-tuning"
                )
            train_command = _format_training_command(
                train_template,
                model_name=str(model_name),
                selected_path=selected_path,
                expert_data_path=expert_data_path,
                output_dir=output_dir,
                seed=int(adaptation_cfg.get("seed", 2026)),
                round_id=int(round_id or 0),
                previous_checkpoint=str(previous_checkpoint or profile.get("checkpoint", "")),
                checkpoint_output=checkpoint_output,
                scenic_program_dir=materialized.get("scenic_program_dir"),
                scenic_execution_manifest=materialized.get("scenic_execution_manifest"),
                routes_xml=materialized.get("bench2drive_xml"),
            )
            (output_dir / f"train_{model_name}.sh").write_text(train_command + "\n", encoding="utf-8")
            run_expert_generation(train_command, cwd=Path(__file__).resolve().parents[1])
            train_status = "completed"
        if not checkpoint_output.exists():
            raise FileNotFoundError(
                f"Fine-tuning command completed without producing checkpoint_output: {checkpoint_output}"
            )
        checkpoint_paths[str(model_name)] = str(checkpoint_output.resolve())
        training_rows.append(
            {
                "model": str(model_name),
                "status": train_status,
                "checkpoint": str(checkpoint_output.resolve()),
                "command_file": str(output_dir / f"train_{model_name}.sh") if train_command else None,
            }
        )

    manifest["manifest_path"] = str(output_dir / "adaptation_manifest.json")
    manifest["selected_scenarios_path"] = str(selected_path)
    manifest["expert_generation"] = {
        "status": expert_status,
        "command_file": str(output_dir / "expert_command.sh") if expert_command else None,
        "data_path": str(expert_data_path.resolve()),
        "demos_file": str(demos_path.resolve()),
    }
    manifest["fine_tuning"] = {
        "status": "completed" if training_rows else "not_run",
        "models": training_rows,
        "checkpoint_paths": checkpoint_paths,
    }
    write_adaptation_manifest(manifest, output_dir / "adaptation_manifest.json")
    return manifest


def checkpoint_for_active_model(manifest: dict[str, Any] | None, cfg: dict[str, Any]) -> str | None:
    if not manifest:
        return None
    checkpoints = manifest.get("fine_tuning", {}).get("checkpoint_paths", {})
    if not isinstance(checkpoints, dict):
        return None
    active_model = _active_model_name(cfg)
    checkpoint = checkpoints.get(active_model) if active_model else None
    if checkpoint is None and len(checkpoints) == 1:
        checkpoint = next(iter(checkpoints.values()))
    return str(checkpoint) if checkpoint else None


def build_agent_config_override(
    cfg: dict[str, Any],
    checkpoint_path: str | None,
) -> str | None:
    """Replace only the checkpoint portion of Bench2Drive agent config."""
    if not checkpoint_path:
        return None
    checkpoint = str(Path(checkpoint_path).expanduser().resolve())
    raw = str(cfg.get("bench2drive", {}).get("agent_config", "")).strip()
    if "+" in raw:
        config_part, _ = raw.split("+", 1)
        return f"{config_part}+{checkpoint}"
    return checkpoint


def checkpoint_from_agent_config(cfg: dict[str, Any], agent_config: str | None) -> str | None:
    """Keep Zoo's config+checkpoint agent syntax out of model training commands."""
    if not agent_config:
        return None
    configured = str(cfg.get("bench2drive", {}).get("agent_config", ""))
    if "+" in configured:
        config_path = configured.split("+", 1)[0]
        prefix = f"{config_path}+"
        if not agent_config.startswith(prefix):
            raise ValueError(f"previous agent config does not match the active model: {agent_config}")
        return agent_config[len(prefix):]
    return agent_config

def validate_native_execution_complete(execution: dict[str, Any], round_id: int) -> None:
    programs = execution.get("programs", [])
    if not isinstance(programs, list) or not programs:
        raise RuntimeError(
            f"fixed_policy_evolution round {round_id} has no native Scenic execution outcomes"
        )
    incomplete = [
        row for row in programs
        if not isinstance(row, dict) or row.get("status") != "completed"
    ]
    if incomplete:
        first = incomplete[0]
        detail = (
            f"{first.get('scenario_id', 'unknown')}: "
            f"{first.get('error_type', first.get('status', 'invalid_record'))}: "
            f"{first.get('error', first.get('reason', ''))}"
            if isinstance(first, dict) else "invalid result record"
        )
        raise RuntimeError(
            f"fixed_policy_evolution round {round_id} has {len(incomplete)} incomplete "
            f"native executions (errors/skipped are not successes); first: {detail}. "
            f"Inspect {execution.get('result_file', 'native policy results')}."
        )



def native_policy_metrics(
    execution: dict[str, Any],
    *,
    collision_rate_scale: str,
    synthetic: bool = False,
) -> dict[str, float]:
    """Return collision rate from completed native Scenic executions only."""
    if synthetic:
        return {"CR": 0.0}
    programs = execution.get("programs", [])
    completed = [
        row for row in programs
        if isinstance(row, dict) and row.get("status") == "completed"
    ] if isinstance(programs, list) else []
    collisions = sum(1 for row in completed if int(row.get("collisions", 0)) > 0)
    cr = collisions / len(completed) if completed else 0.0
    if collision_rate_scale == "percent":
        cr *= 100.0
    return {"CR": cr}


def execute_native_policy_stage(
    manifest_path: Path,
    config_path: Path,
    cfg: dict[str, Any],
    output_dir: Path,
    *,
    dry_run: bool,
    resume: bool,
) -> dict[str, Any]:
    if dry_run:
        return run_native_policy_manifest(manifest_path, cfg, output_dir, dry_run=True)

    policy_cfg = native_policy_config(cfg)
    result_path = output_dir / str(policy_cfg.get("result_file_name", DEFAULT_RESULT_FILE))
    if resume and result_path.exists():
        previous = json.loads(result_path.read_text(encoding="utf-8"))
        expected = build_native_policy_plan(manifest_path, cfg)["programs"]
        completed = previous.get("programs", [])
        if (
            previous.get("status") == "completed"
            and Path(str(previous.get("manifest", ""))).resolve() == manifest_path.resolve()
            and len(completed) == len(expected)
            and all(
                row.get("status") == "completed"
                and row.get("scenario_id") == program["scenario_id"]
                and Path(str(row.get("program_path", ""))).resolve()
                == Path(str(program["program_path"])).resolve()
                and Path(str(row.get("trace_file", ""))).is_file()
                for row, program in zip(completed, expected)
            )
            and Path(str(previous.get("failure_file", ""))).is_file()
        ):
            return previous

    output_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-m",
        "core.scenic_policy_adapter",
        "--manifest",
        str(manifest_path.resolve()),
        "--config",
        str(config_path.resolve()),
        "--output-dir",
        str(output_dir.resolve()),
    ]
    if resume:
        command.append("--resume-completed")
    with (output_dir / "execution.log").open("w", encoding="utf-8") as log:
        subprocess.run(
            command,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )
    return json.loads(result_path.read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    if args.dry_run:
        # Synthetic orchestration does not require a live VLM or CLIP encoder.
        cfg["interpretation"] = {**cfg.get("interpretation", {}), "backend": "rule_based"}
        cfg["retrieval"] = {**cfg.get("retrieval", {}), "backend": "lexical"}
    validate_config(cfg, allow_incomplete_adaptation=args.dry_run)
    metric_keys = resolve_metric_keys(cfg)
    collision_rate_scale = resolve_collision_rate_scale(cfg)
    fixed_protocol = str(cfg.get("protocol", "downstream_adaptation")) == "fixed_policy_evolution"
    benchmark_feedback = fixed_protocol and bool(cfg.get("evaluation", {}).get("use_held_out_routes", False))
    if fixed_protocol and not benchmark_feedback:
        # A native Scenic execution has no official Bench2Drive route scores.
        metric_keys = ["CR"]

    use_structure = args.mode != "wo_struct"
    use_escalation = args.mode != "wo_escalation"
    evolution_enabled = args.mode != "static"
    run_root = resolve_run_root(cfg, args)
    run_root.mkdir(parents=True, exist_ok=True)

    num_rounds = int(cfg.get("num_rounds", 2))
    top_k = int(cfg.get("retrieval", {}).get("top_k", 3))
    escalation = float(cfg.get("evolution", {}).get("escalation_strength", 0.25))
    random_seed = int(cfg.get("seed", 2026))
    interpretation_prompt_path = Path(str(cfg.get("interpretation", {}).get("prompt_path", DEFAULT_PROMPT_PATH)))
    failure_extract_cfg = cfg.get("failure_extract", {})

    scenario_bank_path = Path(cfg.get("scenario_bank_path", run_root / "scenario_bank.json"))
    if args.resume_run:
        start_round, all_rounds, all_structured, all_evolved, resumed_bank = collect_resume_state(run_root, num_rounds)
        scenario_bank = resumed_bank if resumed_bank is not None else ensure_seed_bank(
            scenario_bank_path,
            cfg.get("scenario_bank_bootstrap"),
        )
        raw_failures_all_path = run_root / "failures_raw_all.json"
        all_failures = json.loads(raw_failures_all_path.read_text(encoding="utf-8")) if raw_failures_all_path.exists() else []
    else:
        start_round = 0
        scenario_bank = ensure_seed_bank(scenario_bank_path, cfg.get("scenario_bank_bootstrap"))
        all_rounds = []
        all_failures = []
        all_structured = []
        all_evolved = []

    current_agent_config_override: str | None = None
    if not fixed_protocol and all_rounds and isinstance(all_rounds[-1], dict):
        previous_override = all_rounds[-1].get("agent_config_override")
        if previous_override:
            current_agent_config_override = str(previous_override)

    write_run_manifest(
        run_root,
        args=args,
        metric_keys=metric_keys,
        interpretation_prompt_path=interpretation_prompt_path,
        cfg=cfg,
        failure_extract_cfg=failure_extract_cfg,
        status="running",
        completed_rounds=start_round,
    )

    for rid in range(start_round, num_rounds):
        round_dir = run_root / f"round{rid}"
        round_dir.mkdir(parents=True, exist_ok=True)
        round_inputs = materialize_round_inputs(scenario_bank, round_dir, cfg)
        # This is the only route file used by the benchmark evaluator.  When
        # the paper config enables the feedback benchmark, it is a copy of the
        # original bench2drive220.xml and never contains evolved routes.
        scenario_file = round_inputs.get("evaluation_xml")
        if scenario_file is None:
            scenario_file = round_inputs.get("bench2drive_xml", round_inputs["json"])
        bank_size_before = len(scenario_bank)

        native_execution: dict[str, Any] | None = None
        native_metrics: dict[str, float] | None = None
        if fixed_protocol:
            manifest_path = round_inputs.get("scenic_execution_manifest")
            if manifest_path is None:
                raise RuntimeError(
                    f"fixed_policy_evolution round {rid} produced no native Scenic execution manifest"
                )
            baseline_dir = round_dir / "native_policy_evaluation"
            native_execution = execute_native_policy_stage(
                manifest_path,
                args.config,
                cfg,
                baseline_dir,
                dry_run=args.dry_run,
                resume=bool(args.resume_run),
            )
            if args.dry_run:
                native_failures = generate_mock_failures()
            else:
                validate_native_execution_complete(native_execution, rid)
                failure_file = Path(str(native_execution.get("failure_file", "")))
                if not failure_file.exists():
                    raise RuntimeError(
                        "Native Scenic policy execution produced no failure records file: "
                        f"{failure_file}"
                    )
                native_failures = extract_failures_from_log(failure_file, {"format": "jsonl"})
            native_metrics = native_policy_metrics(
                native_execution,
                collision_rate_scale=collision_rate_scale,
                synthetic=args.dry_run,
            )
            write_json(baseline_dir / "metrics.json", native_metrics)
            write_json(
                baseline_dir / "evaluation_contract.json",
                {
                    "protocol": "fixed_policy_evolution",
                    "source": "native_scenic_policy_execution",
                    "synthetic": bool(args.dry_run),
                    "scenario_manifest": str(manifest_path.resolve()),
                },
            )
            if benchmark_feedback:
                # The official routes supply failure feedback and official
                # scores; Scenic is the separate, cumulative exposure pool.
                benchmark_dir = round_dir / "benchmark_evaluation"
                baseline_metrics = run_eval_round(
                    rid,
                    benchmark_dir,
                    dry_run=args.dry_run,
                    cfg=cfg,
                    scenario_file=scenario_file,
                    metric_keys=metric_keys,
                    collision_rate_scale=collision_rate_scale,
                    stage="fixed_policy_feedback",
                )
                failures = extract_round_failures(
                    rid,
                    dry_run=args.dry_run,
                    cfg=cfg,
                    run_root=run_root,
                    round_dir=round_dir,
                    evaluation_dir=benchmark_dir,
                )
            else:
                baseline_metrics = native_metrics
                failures = native_failures
        else:
            baseline_dir = round_dir / "benchmark_evaluation"
            baseline_metrics = run_eval_round(
                rid,
                baseline_dir,
                dry_run=args.dry_run,
                cfg=cfg,
                scenario_file=scenario_file,
                metric_keys=metric_keys,
                collision_rate_scale=collision_rate_scale,
                stage="baseline",
                agent_config_override=current_agent_config_override,
            )
            failures = extract_round_failures(
                rid,
                dry_run=args.dry_run,
                cfg=cfg,
                run_root=run_root,
                round_dir=round_dir,
                evaluation_dir=baseline_dir,
            )
        semantics = interpret_failures(
            failures,
            prompt_path=interpretation_prompt_path,
            config=cfg.get("interpretation"),
        )
        structured = structure_failures(failures, semantics, use_structure=use_structure)

        evolved = []
        retrieval_trace = []
        anchor_score_total = 0.0
        anchor_count = 0
        unique_anchor_ids: set[str] = set()
        for sf in structured:
            anchors = retrieve_anchor_scenarios(
                sf,
                scenario_bank,
                top_k=top_k,
                retrieval_config=cfg.get("retrieval", {}),
            )
            retrieval_trace.append(
                {
                    "failure_key": f"{sf.event.route_id}:{sf.event.timestamp_s:.2f}",
                    "interaction_pattern": sf.interaction_pattern,
                    "target_failure_type": sf.event.failure_type,
                    "retrieval_backend": cfg.get("retrieval", {}).get("backend", "lexical"),
                    "anchor_candidates": [anchor.to_dict() for anchor in anchors],
                }
            )
            anchor_score_total += sum(anchor.score for anchor in anchors)
            anchor_count += len(anchors)
            unique_anchor_ids.update(anchor.scenario_id for anchor in anchors)
            if not evolution_enabled:
                continue
            if args.mode == "random":
                evolved.extend(
                    random_perturb_from_anchors(
                        sf,
                        anchors,
                        strength=escalation,
                        seed=random_seed + rid,
                        generation_tag=f"round{rid}",
                    )
                )
            else:
                evolved.extend(
                    evolve_from_anchors(
                        sf,
                        anchors,
                        escalation_strength=escalation,
                        use_escalation=use_escalation and args.mode != "retrieval",
                        reconstruction=args.mode != "retrieval",
                        generation_tag=f"round{rid}",
                    )
                )

        failures_payload = [failure.to_dict() for failure in failures]
        semantics_payload = [
            {
                "failure_key": failure_key,
                "semantics": semantics_row,
            }
            for failure_key, semantics_row in semantics.items()
        ]
        structured_payload = [x.to_dict() for x in structured]
        evolved_candidates_payload = [x.to_dict() for x in evolved]
        rejected_evolved = []
        evolved_payload = []
        for candidate in evolved_candidates_payload:
            modifications = candidate.get("modifications", {})
            validation_errors = modifications.get("scenic_validation_errors", []) if isinstance(modifications, dict) else []
            if isinstance(validation_errors, list) and validation_errors:
                rejected_evolved.append(
                    {
                        "evolved_id": candidate.get("evolved_id"),
                        "errors": [str(error) for error in validation_errors],
                    }
                )
                continue
            evolved_payload.append(candidate)
        write_json(round_dir / "failures_raw.json", failures_payload)
        write_json(round_dir / "failures_semantics.json", semantics_payload)
        write_json(round_dir / "failures_structured.json", structured_payload)
        write_json(round_dir / "retrieval_trace.json", retrieval_trace)
        write_json(round_dir / "evolved_scenarios.json", evolved_payload)
        write_json(round_dir / "evolved_scenarios_rejected.json", rejected_evolved)

        scenario_bank = append_evolved_to_bank(scenario_bank, evolved_payload)
        save_scenario_bank(round_dir / "scenario_bank_post_round.json", scenario_bank)

        adaptation_manifest = None
        post_adaptation_metrics: dict[str, float] | None = None
        post_adaptation_dir: Path | None = None
        adapted_override = current_agent_config_override
        adaptation_cfg = cfg.get("adaptation", {})
        adaptation_enabled = (
            not fixed_protocol
            and isinstance(adaptation_cfg, dict)
            and bool(adaptation_cfg.get("enabled", False))
        )
        if adaptation_enabled and evolution_enabled and not evolved_payload:
            raise RuntimeError(
                f"Round {rid} produced no executable evolved scenarios for downstream adaptation"
            )
        if adaptation_enabled and evolved_payload:
            evolved_ids = {str(row.get("evolved_id", "")) for row in evolved_payload}
            adaptation_rows = [
                row
                for row in scenario_bank
                if str(row.get("scenario_id", "")) in evolved_ids
            ]
            adaptation_manifest = prepare_adaptation_artifacts(
                run_root,
                scenario_bank,
                cfg,
                round_dir=round_dir,
                round_id=rid,
                dry_run=args.dry_run,
                adaptation_rows=adaptation_rows,
                previous_checkpoint=checkpoint_from_agent_config(cfg, current_agent_config_override),
            )
            checkpoint_path = checkpoint_for_active_model(adaptation_manifest, cfg)
            if not checkpoint_path:
                raise RuntimeError(f"Round {rid} produced no adapted checkpoint for the active model")
            adapted_override = build_agent_config_override(cfg, checkpoint_path)
            post_adaptation_dir = round_dir / "benchmark_re_evaluation"
            post_adaptation_metrics = run_eval_round(
                rid,
                post_adaptation_dir,
                dry_run=args.dry_run,
                cfg=cfg,
                scenario_file=scenario_file,
                metric_keys=metric_keys,
                collision_rate_scale=collision_rate_scale,
                stage="post_adaptation",
                agent_config_override=adapted_override,
            )
            current_agent_config_override = adapted_override

        metrics = post_adaptation_metrics or baseline_metrics
        metric_delta: dict[str, float] = {}
        if not fixed_protocol:
            metric_delta = {
                key: float(metrics.get(key, 0.0)) - float(baseline_metrics.get(key, 0.0))
                for key in metric_keys
                if key in metrics and key in baseline_metrics
            }

        round_summary = {
            "round": rid,
            "strategy": args.mode,
            "protocol": str(cfg.get("protocol", "downstream_adaptation")),
            "evaluation_synthetic": bool(args.dry_run),
            "scenario_file": str(scenario_file) if not fixed_protocol or benchmark_feedback else None,
            "official_benchmark_routes": str(scenario_file) if not fixed_protocol or benchmark_feedback else None,
            "evaluation_route_policy": (
                "bench2drive220_failure_feedback_and_separate_native_scenic_exposure"
                if benchmark_feedback else
                "native_scenic_scenario_bank_fixed_policy" if fixed_protocol else
                "fixed_bench2drive220_feedback_and_re_evaluation"
            ),
            "baseline_metrics": baseline_metrics,
            "post_adaptation_metrics": post_adaptation_metrics,
            "metrics": metrics,
            "adaptation_executed": adaptation_manifest is not None,
            "agent_config_override": adapted_override,
            "num_failures": len(failures),
            "num_structured": len(structured),
            "num_evolved": len(evolved),
            "num_evolved_accepted": len(evolved_payload),
            "num_evolved_rejected": len(rejected_evolved),
            "num_unique_anchor_sources": len(unique_anchor_ids),
            "avg_anchor_score": (anchor_score_total / anchor_count) if anchor_count else 0.0,
            "scenario_bank_size_before": bank_size_before,
            "scenario_bank_size_after": len(scenario_bank),
            "scenic_native_scenarios": sum(
                1
                for row in scenario_bank
                if str(row.get("config", {}).get("scenic_program_format", "")).strip() == "scenic_native_v1"
            ),
        }
        if not args.dry_run and (benchmark_feedback or not fixed_protocol):
            evaluation_path = benchmark_dir if fixed_protocol else baseline_dir
            evaluation_contract = json.loads(
                (evaluation_path / "evaluation_contract.json").read_text(encoding="utf-8")
            )
            if evaluation_contract.get("coverage") is not None:
                round_summary["benchmark_coverage"] = evaluation_contract["coverage"]
            complete = bool(evaluation_contract["official_complete"])
            if post_adaptation_dir is not None:
                post_contract = json.loads(
                    (post_adaptation_dir / "evaluation_contract.json").read_text(encoding="utf-8")
                )
                if post_contract.get("coverage") is not None:
                    round_summary["post_adaptation_coverage"] = post_contract["coverage"]
                complete = complete and bool(post_contract["official_complete"])
            round_summary["official_benchmark_complete"] = complete
            if not complete:
                round_summary["evaluation_route_policy"] += "_partial_coverage"
                metric_delta = {}
        if fixed_protocol:
            round_summary["native_evaluation_dir"] = str(baseline_dir)
            round_summary["native_metrics"] = native_metrics
            if benchmark_feedback:
                round_summary["benchmark_evaluation_dir"] = str(benchmark_dir)
                round_summary["benchmark_used_for_failure_feedback"] = True
                round_summary["benchmark_strictly_held_out_for_adaptation"] = False
        else:
            round_summary["benchmark_evaluation_dir"] = str(baseline_dir)
            round_summary["metric_delta_after_adaptation"] = metric_delta
        if fixed_protocol and native_execution is not None:
            programs = native_execution.get("programs", [])
            programs = programs if isinstance(programs, list) else []
            round_summary["native_execution_counts"] = {
                "total": len(programs),
                "completed": sum(
                    1 for row in programs if isinstance(row, dict) and row.get("status") == "completed"
                ),
                "errors": sum(
                    1 for row in programs if isinstance(row, dict) and row.get("status") == "error"
                ),
                "skipped": sum(
                    1 for row in programs if isinstance(row, dict) and row.get("status") == "skipped"
                ),
                "collision_scenarios": sum(
                    1
                    for row in programs
                    if isinstance(row, dict)
                    and row.get("status") == "completed"
                    and int(row.get("collisions", 0)) > 0
                ),
            }
            round_summary["native_execution_status"] = native_execution.get("status")
        if post_adaptation_dir is not None:
            round_summary["benchmark_re_evaluation_dir"] = str(post_adaptation_dir)
        if adaptation_manifest is not None:
            round_summary["adaptation"] = {
                "manifest_path": adaptation_manifest.get("manifest_path"),
                "num_scenarios": adaptation_manifest.get("num_scenarios", 0),
                "expert_generation": adaptation_manifest.get("expert_generation", {}),
                "fine_tuning": adaptation_manifest.get("fine_tuning", {}),
            }
        write_json(round_dir / "round_summary.json", round_summary)

        all_failures.extend(failures_payload)
        all_rounds.append(round_summary)
        all_structured.extend(structured_payload)
        all_evolved.extend(evolved_payload)

        write_run_manifest(
            run_root,
            args=args,
            metric_keys=metric_keys,
            interpretation_prompt_path=interpretation_prompt_path,
            cfg=cfg,
            failure_extract_cfg=failure_extract_cfg,
            status="running",
            completed_rounds=rid + 1,
        )

    save_scenario_bank(run_root / "scenario_bank_final.json", scenario_bank)
    write_json(run_root / "failures_raw_all.json", all_failures)
    write_json(run_root / "failure_db.json", all_structured)
    write_json(run_root / "evolved_scenarios_all.json", all_evolved)

    summary = build_summary(args.mode, metric_keys, all_rounds)
    adaptation_rounds = [
        round_row.get("adaptation")
        for round_row in all_rounds
        if isinstance(round_row, dict) and round_row.get("adaptation")
    ]
    summary["adaptation"] = {
        "rounds": adaptation_rounds,
        "num_rounds": len(adaptation_rounds),
    }
    write_json(run_root / "summary.json", summary)
    write_run_manifest(
        run_root,
        args=args,
        metric_keys=metric_keys,
        interpretation_prompt_path=interpretation_prompt_path,
        cfg=cfg,
        failure_extract_cfg=failure_extract_cfg,
        status="completed",
        completed_rounds=len(all_rounds),
    )

    print(f"[fd_set] run root: {run_root}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
