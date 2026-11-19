"""Execute native Scenic programs with a Bench2Drive autonomous agent.

The repository has two deliberately separate execution paths:

* Bench2Drive route XML is evaluated by the official leaderboard evaluator;
* native Scenic programs are evaluated by Scenic's CARLA simulator.

This module connects the second path to the same agent class used by the first
path.  It targets Scenic 2.0.0, which is the version pinned by this project and
uses ``Simulation.run(maxSteps)``.  Imports of Scenic/CARLA/Bench2Drive are
kept inside the live execution path so that materialization, dry-runs, and unit
tests remain usable without a running CARLA server.
"""

from __future__ import annotations

import importlib.util
import json
import os
import random
import sys
import traceback
from collections import OrderedDict
from pathlib import Path
from typing import Any


NATIVE_POLICY_SCHEMA = "fdset.scenic_policy_execution.v1"
DEFAULT_RESULT_FILE = "native_scenic_policy_results.json"
DEFAULT_FAILURE_FILE = "native_scenic_policy_failures.jsonl"

# Scenic 2 CARLA IDs removed or renamed in CARLA 0.9.15. Isetta is no
# longer shipped; Microlino is the available microcar replacement.
_SCENIC_CARLA_BLUEPRINTS = {
    "vehicle.bmw.isetta": "vehicle.micro.microlino",
    "vehicle.dodge_charger.police": "vehicle.dodge.charger_police",
    "vehicle.lincoln.mkz2017": "vehicle.lincoln.mkz_2017",
    "vehicle.mercedes-benz.coupe": "vehicle.mercedes.coupe",
    "vehicle.mini.cooperst": "vehicle.mini.cooper_s",
    "vehicle.mustang.mustang": "vehicle.ford.mustang",
}


class NativePolicyExecutionError(RuntimeError):
    """Raised when native Scenic policy execution cannot be initialized."""


def native_policy_config(cfg: dict[str, Any]) -> dict[str, Any]:
    """Return the normalized native-policy configuration.

    Configuration lives under ``materialization.scenic.policy_execution`` so
    the materialized execution manifest and its executor share one switch.
    """
    materialization = cfg.get("materialization", {})
    scenic_cfg = materialization.get("scenic", {}) if isinstance(materialization, dict) else {}
    policy_cfg = scenic_cfg.get("policy_execution", {}) if isinstance(scenic_cfg, dict) else {}
    return dict(policy_cfg) if isinstance(policy_cfg, dict) else {}


def validate_native_policy_config(cfg: dict[str, Any]) -> None:
    """Validate configuration which can be checked without contacting CARLA."""
    policy_cfg = native_policy_config(cfg)
    if not bool(policy_cfg.get("enabled", False)):
        return

    max_steps = int(policy_cfg.get("max_steps", 600))
    if max_steps <= 0:
        raise ValueError("materialization.scenic.policy_execution.max_steps must be positive")

    generate_iterations = int(policy_cfg.get("generate_max_iterations", 2000))
    if generate_iterations <= 0:
        raise ValueError(
            "materialization.scenic.policy_execution.generate_max_iterations must be positive"
        )

    max_programs = policy_cfg.get("max_programs")
    if max_programs is not None and int(max_programs) <= 0:
        raise ValueError("materialization.scenic.policy_execution.max_programs must be positive or null")

    route_points = int(policy_cfg.get("route_points", 120))
    if route_points < 2:
        raise ValueError("materialization.scenic.policy_execution.route_points must be at least 2")

    timestep = float(policy_cfg.get("timestep", 0.1))
    if timestep <= 0:
        raise ValueError("materialization.scenic.policy_execution.timestep must be positive")


def load_native_execution_manifest(path: Path) -> dict[str, Any]:
    """Load and structurally validate a materialized native Scenic manifest."""
    if not path.exists():
        raise FileNotFoundError(f"Native Scenic execution manifest not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Native Scenic execution manifest must be a JSON object")
    if payload.get("schema_version") != "fdset.scenic_execution_manifest.v1":
        raise ValueError(
            "Unsupported native Scenic execution manifest schema: "
            f"{payload.get('schema_version')}"
        )
    programs = payload.get("programs")
    if not isinstance(programs, list):
        raise ValueError("Native Scenic execution manifest must contain a programs list")
    for index, program in enumerate(programs):
        if not isinstance(program, dict):
            raise ValueError(f"Native Scenic manifest program {index} must be an object")
        if not str(program.get("scenario_id", "")).strip():
            raise ValueError(f"Native Scenic manifest program {index} is missing scenario_id")
        if not str(program.get("program_path", "")).strip():
            raise ValueError(f"Native Scenic manifest program {index} is missing program_path")
    return payload


def build_native_policy_plan(manifest_path: Path, cfg: dict[str, Any]) -> dict[str, Any]:
    """Build a no-side-effect execution plan for inspection and dry-runs."""
    manifest = load_native_execution_manifest(manifest_path)
    policy_cfg = native_policy_config(cfg)
    programs = manifest.get("programs", [])
    max_programs = policy_cfg.get("max_programs")
    if max_programs is not None:
        programs = programs[: int(max_programs)]

    plan_programs = []
    for program in programs:
        path = Path(str(program["program_path"])).expanduser()
        plan_programs.append(
            {
                "scenario_id": str(program["scenario_id"]),
                "program_path": str(path),
                "program_exists": path.exists(),
                "native_execution_ready": bool(program.get("native_execution_ready", False)),
                "map": str(program.get("map", "")),
                "agent": "PDM-Lite DataAgent" if policy_cfg.get("expert_mode") else str(cfg.get("bench2drive", {}).get("agent", "")),
                "max_steps": int(policy_cfg.get("max_steps", 600)),
                "timestep": float(policy_cfg.get("timestep", 0.1)),
            }
        )
    return {
        "schema_version": NATIVE_POLICY_SCHEMA,
        "status": "planned",
        "manifest": str(manifest_path.resolve()),
        "enabled": bool(policy_cfg.get("enabled", False)),
        "program_count": len(plan_programs),
        "programs": plan_programs,
        "execution_contract": {
            "scenario_runtime": "Scenic 2.0.0 CarlaSimulator",
            "policy_api": (
                "PDM-Lite DataAgent.run_step(input_data, timestamp, sensors)"
                if policy_cfg.get("expert_mode")
                else "Bench2Drive AutonomousAgent.run_step(input_data, timestamp)"
            ),
            "ego_role_name": "hero",
            "route_source": "Scenic sampled fdset_route_goal via CARLA GlobalRoutePlanner; otherwise map waypoints from ego spawn",
        },
    }


def run_native_policy_manifest(
    manifest_path: Path,
    cfg: dict[str, Any],
    output_dir: Path,
    *,
    dry_run: bool = False,
    resume_completed: bool = False,
) -> dict[str, Any]:
    """Execute or plan all native Scenic programs in a manifest.

    The live path is intentionally opt-in through the config.  ``dry_run``
    never imports CARLA or an agent module and never connects to the server.
    """
    validate_native_policy_config(cfg)
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = load_native_execution_manifest(manifest_path)
    policy_cfg = native_policy_config(cfg)
    result_path = output_dir / str(policy_cfg.get("result_file_name", DEFAULT_RESULT_FILE))
    failure_path = output_dir / str(policy_cfg.get("failure_file_name", DEFAULT_FAILURE_FILE))

    if dry_run or not bool(policy_cfg.get("enabled", False)):
        plan = build_native_policy_plan(manifest_path, cfg)
        plan["status"] = "dry_run" if dry_run else "disabled"
        if not dry_run:
            plan["note"] = (
                "Native Scenic policy execution is disabled. Set "
                "materialization.scenic.policy_execution.enabled=true for live execution."
            )
        result_path.write_text(json.dumps(plan, indent=2, ensure_ascii=False), encoding="utf-8")
        return plan

    previous_rows: dict[str, dict[str, Any]] = {}
    previous_failures: dict[str, list[dict[str, Any]]] = {}
    if resume_completed and result_path.is_file() and failure_path.is_file():
        previous = json.loads(result_path.read_text(encoding="utf-8"))
        if Path(str(previous.get("manifest", ""))).resolve() == manifest_path.resolve():
            for row in previous.get("programs", []):
                if isinstance(row, dict) and row.get("status") == "completed":
                    previous_rows[str(row.get("scenario_id", ""))] = row
            for line in failure_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    record = json.loads(line)
                    previous_failures.setdefault(str(record.get("route_id", "")), []).append(record)

    original_environment = dict(os.environ)
    if bool(policy_cfg.get("expert_mode", False)):
        # Import Torch before Scenic's CARLA/geometry runtime. Loading
        # libtorch_cpu later can make its C++ random_device initializer fail
        # after Scenic has initialized the simulator.
        import torch
    try:
        runtime = _prepare_live_runtime(cfg)
        random.seed(int(policy_cfg.get("seed", cfg.get("seed", 2026))))
        try:
            import numpy as np

            np.random.seed(int(policy_cfg.get("seed", cfg.get("seed", 2026))))
        except Exception:
            pass
        programs = manifest.get("programs", [])
        max_programs = policy_cfg.get("max_programs")
        if max_programs is not None:
            programs = programs[: int(max_programs)]

        results: list[dict[str, Any]] = []
        failure_records: list[dict[str, Any]] = []
        previous_agent = (
            previous.get("runtime", {}).get("agent_path")
            if resume_completed and result_path.is_file() and failure_path.is_file()
            else None
        )
        if previous_agent != str(runtime["agent_path"]):
            previous_rows.clear()
            previous_failures.clear()
        for index, program in enumerate(programs):
            old = previous_rows.get(str(program.get("scenario_id", "")))
            if (
                old is not None
                and Path(str(old.get("program_path", ""))).resolve()
                == Path(str(program.get("program_path", ""))).resolve()
                and Path(str(old.get("trace_file", ""))).is_file()
            ):
                results.append(old)
                failure_records.extend(previous_failures.get(str(program["scenario_id"]), []))
                continue
            if not bool(program.get("native_execution_ready", False)):
                results.append(
                    {
                        "scenario_id": str(program.get("scenario_id", f"program_{index}")),
                        "status": "skipped",
                        "reason": "manifest marks program as not native_execution_ready",
                    }
                )
                continue
            try:
                previous_cwd = Path.cwd()
                try:
                    # Bench2Drive agent configs use paths such as
                    # ``Bench2DriveZoo/...`` relative to the evaluator root.
                    os.chdir(str(runtime["bench_root"]))
                    row_result, row_failures = _run_one_program(
                        program,
                        cfg,
                        policy_cfg,
                        runtime,
                        output_dir,
                        index=index,
                    )
                finally:
                    os.chdir(str(previous_cwd))
                results.append(row_result)
                failure_records.extend(row_failures)
            except Exception as exc:  # keep the remaining scenarios diagnosable
                results.append(
                    {
                        "scenario_id": str(program.get("scenario_id", f"program_{index}")),
                        "program_path": str(program.get("program_path", "")),
                        "status": "error",
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "traceback": traceback.format_exc(),
                    }
                )
    finally:
        # Agent modules cache SAVE_PATH/ROUTES at import time, but the parent
        # process must not leak those native-run values into the next official
        # Bench2Drive round.
        os.environ.clear()
        os.environ.update(original_environment)

    _write_jsonl(failure_path, failure_records)
    summary = {
        "schema_version": NATIVE_POLICY_SCHEMA,
        "status": "completed",
        "manifest": str(manifest_path.resolve()),
        "result_file": str(result_path.resolve()),
        "failure_file": str(failure_path.resolve()),
        "program_count": len(results),
        "completed": sum(1 for row in results if row.get("status") == "completed"),
        "skipped": sum(1 for row in results if row.get("status") == "skipped"),
        "errors": sum(1 for row in results if row.get("status") == "error"),
        "collisions": len(failure_records),
        "programs": results,
        "runtime": runtime["description"],
    }
    result_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def _prepare_live_runtime(cfg: dict[str, Any]) -> dict[str, Any]:
    """Prepare isolated Bench2Drive or local PDM-Lite execution dependencies."""
    from .closed_loop_runner import (
        build_bench2drive_runtime_env,
        resolve_agent_config_value,
        resolve_agent_path,
        resolve_bench2drive_root,
        resolve_carla_runtime_source,
        validate_agent_config_value,
        validate_agent_runtime_dependencies,
        validate_carla_server,
    )

    if importlib.util.find_spec("scenic.syntax") is None:
        raise NativePolicyExecutionError(
            "Scenic runtime is missing: install scenic==2.0.0 in a compatible Python "
            "environment; the project's scenic/ directory contains programs, not the runtime."
        )
    expert_mode = bool(native_policy_config(cfg).get("expert_mode", False))
    if expert_mode:
        from pdm_lite.scenic_resources import (
            AGENT_ROOT, DATA_AGENT, DEFAULT_AGENT_CONFIG, LEADERBOARD_ROOT, SCENARIO_RUNNER_ROOT,
        )

        bench_root = Path(__file__).resolve().parents[1]
        agent_path = DATA_AGENT.resolve()
        agent_config = str(DEFAULT_AGENT_CONFIG.resolve())
        validate_carla_server(cfg)
        carla_source = resolve_carla_runtime_source(cfg)
        carla_cfg = cfg.get("carla", {})
        pythonapi = Path(str(carla_cfg.get("pythonapi_root", ""))).expanduser() / "PythonAPI" / "carla"
        if not (pythonapi / "agents").is_dir():
            raise NativePolicyExecutionError(f"CARLA navigation agents source not found: {pythonapi / 'agents'}")
        runtime_env = {
            "LEADERBOARD_ROOT": str(LEADERBOARD_ROOT),
            "SCENARIO_RUNNER_ROOT": str(SCENARIO_RUNNER_ROOT),
            "PYTHONPATH": os.pathsep.join(map(str, (AGENT_ROOT, LEADERBOARD_ROOT, SCENARIO_RUNNER_ROOT, pythonapi))),
        }
    else:
        bench_root = resolve_bench2drive_root(cfg)
        agent_path = resolve_agent_path(cfg, bench_root)
        if agent_path is None or not agent_path.exists():
            raise NativePolicyExecutionError(f"Bench2Drive agent file not found: {agent_path}")
        validate_agent_config_value(cfg, bench_root)
        validate_agent_runtime_dependencies(cfg, bench_root)
        validate_carla_server(cfg)
        carla_source = resolve_carla_runtime_source(cfg)
        runtime_env = build_bench2drive_runtime_env(cfg, bench_root)

    _prepend_pythonpath(runtime_env.get("PYTHONPATH", ""))
    for key, value in runtime_env.items():
        os.environ[key] = str(value)

    resolved_agent_config = agent_config if expert_mode else resolve_agent_config_value(cfg, bench_root)
    return {
        "bench_root": bench_root,
        "agent_path": agent_path,
        "agent_config": resolved_agent_config,
        # Official Bench2Drive agents use the final ``+``-separated field as
        # their evidence-directory name.  Passing the resolved absolute
        # checkpoint path directly therefore makes SAVE_PATH point at the
        # checkpoint itself (and mkdir raises FileExistsError).  Keep the
        # normal absolute value for regular execution, and provide a setup
        # value with an explicit safe final field for evidence capture.
        "agent_config_for_evidence": (
            resolved_agent_config if expert_mode
            else _agent_config_with_evidence_save_name(resolved_agent_config)
        ),
        "expert_mode": expert_mode,
        "runtime_env": runtime_env,
        "carla_source": carla_source,
        "description": {
            "expert_root" if expert_mode else "bench2drive_root": str(bench_root),
            "agent_path": str(agent_path),
            "carla_source": str(carla_source.get("source", "unknown")),
        },
    }


def _prepend_pythonpath(raw: str) -> None:
    entries = [entry for entry in raw.split(os.pathsep) if entry]
    for entry in reversed(entries):
        if entry not in sys.path:
            sys.path.insert(0, entry)


def _agent_config_with_evidence_save_name(agent_config: str) -> str:
    """Add a safe agent evidence name without changing loaded config paths.

    Bench2Drive's official agents parse ``path_to_conf_file`` as either
    ``config+checkpoint`` or ``checkpoint`` and use the last ``+``-separated
    component as ``save_name``.  Native Scenic resolves those paths to
    absolute values for reliable loading, but an absolute final component is
    unsafe when the agents join it to ``SAVE_PATH``.  Appending a basename as
    a third component preserves the first two components used for loading and
    gives all supported agents a relative evidence directory name.
    """
    value = str(agent_config or "").strip()
    if not value:
        return value
    source = value.rsplit("+", 1)[-1]
    save_name = _safe_name(Path(source).name) or "agent"
    return f"{value}+{save_name}"


def _run_one_program(
    program: dict[str, Any],
    cfg: dict[str, Any],
    policy_cfg: dict[str, Any],
    runtime: dict[str, Any],
    output_dir: Path,
    *,
    index: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Compile, sample, and execute one native Scenic program."""
    program_path = Path(str(program["program_path"])).expanduser().resolve()
    if not program_path.exists():
        raise FileNotFoundError(f"Native Scenic program not found: {program_path}")

    carla_cfg = cfg.get("carla", {})
    if not isinstance(carla_cfg, dict):
        carla_cfg = {}
    host = str(carla_cfg.get("host", "127.0.0.1"))
    port = int(carla_cfg.get("port", 2000))
    tm_port = int(carla_cfg.get("tm_port", port + 6000))
    timeout = float(carla_cfg.get("connection_timeout_s", 10.0))
    timestep = float(policy_cfg.get("timestep", 0.1))
    max_steps = int(policy_cfg.get("max_steps", 600))
    verbosity = int(policy_cfg.get("verbosity", 0))

    # Scenic 2 evaluates the source program while generating the scene.  The
    # params below override only simulator connection/runtime values; the map
    # and scenario geometry stay in the source program.
    from scenic.syntax.translator import scenarioFromFile

    scenic_params: dict[str, Any] = {
        "address": host,
        "port": port,
        "timeout": timeout,
        "render": bool(policy_cfg.get("render", False)),
        "timestep": timestep,
    }
    scenario = scenarioFromFile(str(program_path), params=scenic_params)
    scene, generation_iterations = scenario.generate(
        maxIterations=int(policy_cfg.get("generate_max_iterations", 2000)),
        verbosity=verbosity,
    )
    if not scene.objects:
        raise NativePolicyExecutionError("Scenic program generated no objects")

    # Scenic guarantees the ego object is first.  Removing its Scenic behavior
    # is the hand-off point: all other Scenic agents continue to run normally,
    # while Bench2Drive owns the ego vehicle control loop.
    ego = scene.egoObject or scene.objects[0]
    ego.behavior = None
    ego.rolename = "hero"
    _prepare_scene_objects_for_carla(scene, ego)

    map_name = str(program.get("map", "")).strip() or str(scene.params.get("carla_map", "")).strip()
    map_name = map_name.strip("'\"") or None
    map_path = _resolve_map_path(program_path, scene.params, program)
    # The supplied Scenic source bank uses a custom OpenDRIVE map which is
    # not packaged in the running CARLA server.  Scenic's CARLA simulator can
    # load that artifact directly; use it by default when available, while
    # retaining an opt-out for installations which pre-load a named map.
    prefer_opendrive = bool(policy_cfg.get("prefer_opendrive_map", True))
    simulator_map = None if prefer_opendrive and map_path.suffix.lower() == ".xodr" else map_name
    opendrive_identity = str(map_path.resolve()) if simulator_map is None else None
    reuse_loaded_world = (
        opendrive_identity is not None
        and runtime.get("loaded_opendrive_map") == opendrive_identity
    )
    record_dir = ""
    if bool(policy_cfg.get("record", False)):
        record_dir = str((output_dir / "carla_recordings" / _safe_name(str(program["scenario_id"]))).resolve())
        Path(record_dir).mkdir(parents=True, exist_ok=True)

    evidence_root = output_dir / "native_scenic_evidence" / _safe_name(str(program["scenario_id"]))
    capture_agent_evidence = bool(policy_cfg.get("capture_agent_evidence", False))
    if capture_agent_evidence:
        evidence_root.mkdir(parents=True, exist_ok=True)
    # Bench2Drive agents read these values at module import time.  Setting them
    # before loading the module enables the same metric/image sidecars as the
    # official evaluator.  Some official agents use an absolute checkpoint
    # path as ``save_name``; leave SAVE_PATH unset by default so that a native
    # smoke test does not attempt to mkdir at the checkpoint location.
    expert_mode = bool(policy_cfg.get("expert_mode", False))
    os.environ["IS_BENCH2DRIVE"] = "0" if expert_mode else "True"
    os.environ["ROUTES"] = str(program_path)
    if expert_mode:
        dataset_dir = output_dir / "pdm_lite_dataset"
        dataset_dir.mkdir(parents=True, exist_ok=True)
        os.environ.update({
            "DATAGEN": "1", "SAVE_PATH": str(dataset_dir),
            "FDSET_EXPERT_CAMERA_PROFILE": str(policy_cfg.get("camera_profile", "admlp")),
            "TOWN": "FDSET_SCENIC", "REPETITION": "0",
            "DEBUG_CHALLENGE": "0", "TMP_VISU": "0", "HISTOGRAM": "0", "TP_STATS": "0",
            "FDSET_BEV_MAP_DIR": str((output_dir / "bev_maps" / _safe_name(str(program["scenario_id"]))).resolve()),
            "FDSET_SPEED_LIMIT_DIR": str((output_dir / "speed_limits" / _safe_name(str(program["scenario_id"]))).resolve()),
        })
    elif capture_agent_evidence:
        os.environ["SAVE_PATH"] = str(evidence_root.resolve())
    else:
        os.environ.pop("SAVE_PATH", None)

    agent_class = _load_agent_class(runtime["agent_path"])
    simulator_cls, _ = _build_policy_simulator_classes(expert_mode=expert_mode)
    simulator = None
    simulation = None
    trace_path = output_dir / "native_scenic_policy_traces" / f"{_safe_name(str(program['scenario_id']))}.jsonl"
    try:
        agent_config = runtime["agent_config"]
        if capture_agent_evidence:
            agent_config = runtime["agent_config_for_evidence"]
        simulator = simulator_cls(
            carla_map=simulator_map,
            map_path=str(map_path),
            reuse_loaded_opendrive_world=reuse_loaded_world,
            address=host,
            port=port,
            timeout=timeout,
            render=bool(policy_cfg.get("render", False)),
            record=record_dir,
            timestep=timestep,
            traffic_manager_port=tm_port,
            agent_class=agent_class,
            agent_config=agent_config,
            agent_host=host,
            agent_port=port,
            selected_track="MAP" if expert_mode else str(cfg.get("bench2drive", {}).get("track", "SENSORS")),
            agent_debug=bool(policy_cfg.get("agent_debug", False)),
            route_points=int(policy_cfg.get("route_points", 120)),
            scenario_id=str(program["scenario_id"]),
            expert_mode=expert_mode,
            route_index=index,
        )
        if opendrive_identity is not None:
            runtime["loaded_opendrive_map"] = opendrive_identity
        result = simulator.simulate(
            scene,
            maxSteps=max_steps,
            maxIterations=1,
            verbosity=verbosity,
        )
        simulation = simulator.last_simulation
        if result is None:
            raise NativePolicyExecutionError("Scenic rejected all generated native simulations")
        collisions = list(getattr(simulation, "collision_events", []))
        trace = list(getattr(simulation, "policy_trace", []))
        speeds_by_frame = {
            int(row["frame"]): float(row["ego_speed_mps"])
            for row in trace
            if "frame" in row and "ego_speed_mps" in row
        }
        failure_records = [
            _collision_failure_record(
                program,
                collision,
                timestep=timestep,
                scene_tags=["scenic", "native", map_name or "unknown_map"],
                ego_speed=speeds_by_frame.get(int(collision.get("frame", -1)), 0.0),
            )
            for collision in collisions
        ]
        _write_jsonl(trace_path, trace)
        row = {
            "scenario_id": str(program["scenario_id"]),
            "program_path": str(program_path),
            "status": "completed",
            "generation_iterations": generation_iterations,
            "termination_reason": str(result.terminationReason),
            "steps": len(trace),
            "collisions": len(collisions),
            "trace_file": str(trace_path.resolve()),
            "map": map_name,
            "agent": str(runtime["agent_path"]),
        }
        return row, failure_records
    finally:
        # Scenic 2 does not destroy a simulator after simulate().
        if simulation is None and simulator is not None:
            simulation = simulator.last_simulation
        if simulation is not None and getattr(simulation, "policy_trace", None):
            _write_jsonl(trace_path, list(simulation.policy_trace))
        if simulator is not None:
            simulator.destroy()


def _resolve_map_path(program_path: Path, scene_params: dict[str, Any], program: dict[str, Any]) -> Path:
    raw = str(scene_params.get("map", "")).strip().strip("'\"")
    if raw:
        path = Path(raw).expanduser()
        if path.exists():
            return path.resolve()
    references = program.get("map_artifacts", [])
    if isinstance(references, list) and references:
        for reference in references:
            candidate = Path(str(reference)).expanduser()
            if candidate.exists():
                return candidate.resolve()
    # ``carla_map`` is normally set for the supplied source bank, so this is
    # only a diagnostic fallback for custom OpenDRIVE programs.
    return (program_path.parent / "../maps/unknown.xodr").resolve()


def _prepare_scene_objects_for_carla(scene, ego) -> None:
    """Fill CARLA-specific properties for programs using the generic model.

    The original Scenic CARLA model declares ``blueprint``, ``physics`` and
    ``rolename`` on its object types.  Several of the legacy challenge
    programs use only ``scenic.domains.driving.model`` instead, so their cars
    are valid Scenic objects but are not directly spawnable by
    ``CarlaSimulation``.  Keep the source programs unchanged and provide the
    simulator defaults at the native execution boundary.
    """
    scenic_blueprints = sys.modules.get("scenic.simulators.carla.blueprints")
    if scenic_blueprints is not None:
        # Scenic's FollowTrajectoryBehavior also checks carModels to choose
        # vehicle PID gains. Keep the replacement IDs classified as cars.
        car_models = scenic_blueprints.carModels
        for mapped_id in _SCENIC_CARLA_BLUEPRINTS.values():
            if mapped_id not in car_models:
                car_models.append(mapped_id)
    for obj in scene.objects:
        object_type = type(obj).__name__.lower()
        try:
            getattr(obj, "blueprint")
        except AttributeError:
            if "bicycle" in object_type:
                blueprint = "vehicle.bh.crossbike"
            elif "motorcycle" in object_type:
                blueprint = "vehicle.harley-davidson.low_rider"
            elif "walker" in object_type or "pedestrian" in object_type:
                blueprint = "walker.pedestrian.0001"
            else:
                blueprint = "vehicle.lincoln.mkz_2017"
            obj.blueprint = blueprint

        obj.blueprint = _SCENIC_CARLA_BLUEPRINTS.get(obj.blueprint, obj.blueprint)

        for prop, default in (
            ("rolename", None),
            ("physics", True),
            ("carlaActor", None),
            ("carlaController", None),
            ("_control", None),
        ):
            try:
                getattr(obj, prop)
            except AttributeError:
                setattr(obj, prop, default)
                # ``Simulation.updateObjects`` replaces the dynamic proxy
                # after each tick.  A class-level fallback keeps simulator
                # bookkeeping attributes available on the replacement copy.
                setattr(type(obj), prop, default)

        # Generic driving.Vehicle does not implement the small control
        # protocol consumed by RegulatedControlAction.  Install the protocol
        # on its class once; CARLA-model vehicles already provide it.
        if "car" in object_type or "vehicle" in object_type or "bicycle" in object_type:
            _install_generic_vehicle_control_protocol(type(obj))

    # DynamicScenario caches its agent list when the Scenic scene is created.
    # Remove the hand-off ego from that cache as well as from scene.behavior,
    # otherwise Scenic starts the ego behavior before the policy loop.
    dynamic_scenario = getattr(scene, "dynamicScenario", None)
    if dynamic_scenario is not None and hasattr(dynamic_scenario, "_agents"):
        dynamic_scenario._agents = [agent for agent in dynamic_scenario._agents if agent is not ego]


def _install_generic_vehicle_control_protocol(vehicle_type) -> None:
    """Add CARLA-backed control methods to generic Scenic vehicle classes."""
    if not hasattr(vehicle_type, "_fdset_get_control"):
        def _fdset_get_control(self):
            control = getattr(self, "_control", None)
            if control is None:
                control = self.carlaActor.get_control()
                self._control = control
            return control

        def _fdset_set_throttle(self, value):
            self._fdset_get_control().throttle = value

        def _fdset_set_steering(self, value):
            self._fdset_get_control().steer = value

        def _fdset_set_braking(self, value):
            self._fdset_get_control().brake = value

        def _fdset_set_handbrake(self, value):
            self._fdset_get_control().hand_brake = value

        def _fdset_set_reverse(self, value):
            self._fdset_get_control().reverse = value

        vehicle_type._fdset_get_control = _fdset_get_control
        vehicle_type.setThrottle = _fdset_set_throttle
        vehicle_type.setSteering = _fdset_set_steering
        vehicle_type.setBraking = _fdset_set_braking
        vehicle_type.setHandbrake = _fdset_set_handbrake
        vehicle_type.setReverse = _fdset_set_reverse
        vehicle_type.control = property(_fdset_get_control)


def _load_agent_class(agent_path: Path) -> type:
    module_name = f"fdset_native_agent_{abs(hash(str(agent_path.resolve())))}"
    spec = importlib.util.spec_from_file_location(module_name, str(agent_path))
    if spec is None or spec.loader is None:
        raise NativePolicyExecutionError(f"Unable to load Bench2Drive agent module: {agent_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    entry_point = getattr(module, "get_entry_point", None)
    class_name = entry_point() if callable(entry_point) else ""
    if not class_name or not hasattr(module, class_name):
        raise NativePolicyExecutionError(
            f"Agent module {agent_path} does not expose get_entry_point() -> class"
        )
    return getattr(module, class_name)


def _build_policy_simulator_classes(*, expert_mode: bool = False) -> tuple[type, type]:
    """Create Scenic-2-compatible simulator subclasses lazily."""
    # Bind Scenario Runner before Scenic/leaderboard imports.  CARLA also
    # exposes an ``agents`` package, and the mixed import graph can otherwise
    # leave ``srunner`` partially initialized during deferred imports.
    from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
    from srunner.scenariomanager.timer import GameTime

    try:
        import carla
        from scenic.core.simulators import EndScenarioAction, EndSimulationAction
        from scenic.simulators.carla.simulator import CarlaSimulation, CarlaSimulator
    except Exception as exc:
        raise NativePolicyExecutionError(
            "Native Scenic policy execution requires the Scenic 2 CARLA runtime and "
            "a compatible CARLA Python binding."
        ) from exc

    if expert_mode:
        from leaderboard.autoagents.agent_wrapper_local import AgentWrapperFactory, validate_sensor_configuration
    else:
        from leaderboard.autoagents.agent_wrapper import AgentWrapperFactory, validate_sensor_configuration
    from leaderboard.autoagents.autonomous_agent import Track
    from leaderboard.utils.route_manipulation import _get_latlon_ref, location_route_to_gps
    from agents.navigation.local_planner import RoadOption

    class PolicyCarlaSimulation(CarlaSimulation):
        def __init__(self, scene, client, tm, timestep, render, record, scenario_number, **kwargs):
            self._agent_class = kwargs.pop("agent_class")
            self._agent_config = kwargs.pop("agent_config")
            self._agent_host = kwargs.pop("agent_host")
            self._agent_port = kwargs.pop("agent_port")
            self._selected_track = kwargs.pop("selected_track")
            self._agent_debug = kwargs.pop("agent_debug")
            self._route_points = kwargs.pop("route_points")
            self._route_goal = kwargs.pop("route_goal", None)
            self._scenario_id = kwargs.pop("scenario_id")
            self._expert_mode = kwargs.pop("expert_mode")
            self._route_index = kwargs.pop("route_index")
            self.policy_agent = None
            self.agent_wrapper = None
            self.collision_sensor = None
            self.collision_events: list[dict[str, Any]] = []
            self.policy_trace: list[dict[str, Any]] = []
            self._destroyed = False
            super().__init__(
                scene,
                client,
                tm,
                timestep,
                render=render,
                record=record,
                scenario_number=scenario_number,
                verbosity=kwargs.pop("verbosity", 0),
            )
            try:
                self._setup_policy_runtime()
            except Exception:
                # The Scenic 2 constructor has already spawned the scene
                # actors before this adapter is initialized.  If agent setup
                # fails, release those actors immediately so the next manifest
                # entry does not inherit a half-created world.
                self.destroy()
                raise

        def actionsAreCompatible(self, agent, actions):
            """Accept legacy driving actions at the CARLA execution boundary.

            The challenge source bank mixes Scenic's generic driving model with
            the CARLA model.  In Scenic 2.0 the resulting class identity check
            for ``RegulatedControlAction`` can fail after that model hand-off,
            even though the action's ``applyTo`` implementation is valid for
            the spawned CARLA vehicle.  The native adapter owns the simulator
            contract here, so validate that entries are executable actions and
            let the normal Scenic action application perform the actual write.
            """
            return all(callable(getattr(action, "applyTo", None)) for action in actions)

        def mutableProperties(self, obj):
            # CARLA actor handles are simulator-owned state, not Scenic
            # properties.  Preserve them when Scenic refreshes its dynamic
            # proxy after each world tick, including for generic driving.Car
            # objects which do not declare these fields in their model.
            return super().mutableProperties(obj) | {
                "carlaActor",
                "carlaController",
                "_control",
            }

        def _setup_policy_runtime(self) -> None:
            ego_obj = self.objects[0]
            ego_actor = ego_obj.carlaActor
            if ego_actor is None:
                raise NativePolicyExecutionError("Scenic ego object has no CARLA actor")

            CarlaDataProvider.set_client(self.client)
            CarlaDataProvider.set_world(self.world)
            CarlaDataProvider.set_traffic_manager_port(self.tm.get_port())
            # Scenic owns actor creation, while Bench2Drive discovers hero via
            # CarlaDataProvider's pool.  Register only this ego actor here;
            # Scenic remains responsible for destroying all scene actors.
            try:
                CarlaDataProvider.register_actor(ego_actor, ego_actor.get_transform())
            except KeyError:
                pass
            CarlaDataProvider._carla_actor_pool[ego_actor.id] = ego_actor

            GameTime.restart()
            self._tick_callback_id = self.world.on_tick(self._on_carla_tick)
            route_goal = getattr(self, "_route_goal", None)
            route = _build_route(
                ego_actor, self.world, self._route_points, RoadOption, carla,
                route_goal=route_goal,
            )
            lat_ref, lon_ref = _get_latlon_ref(self.world)
            gps_route = location_route_to_gps(route, lat_ref, lon_ref)

            if self._expert_mode:
                self.policy_agent = self._agent_class(self._agent_config, self._route_index)
            else:
                self.policy_agent = self._agent_class(
                    self._agent_host, self._agent_port, self._agent_debug,
                )
            self.policy_agent.set_global_plan(gps_route, route)
            if self._expert_mode:
                self.policy_agent.setup(self._agent_config, self._route_index, self.tm)
            else:
                self.policy_agent.setup(self._agent_config)
            agent_track = getattr(self.policy_agent, "track", Track.SENSORS)
            if not isinstance(agent_track, Track):
                agent_track = Track(agent_track)
            selected_track = getattr(agent_track, "value", agent_track)
            if self._selected_track:
                selected_track = self._selected_track
            if not bool(os.environ.get("SAVE_PATH")):
                # Bench2Drive's official agents may declare a BEV sensor at a
                # radius larger than Scenic's default 3m limit.  The official
                # evaluator permits up to 100m; use the same validation rule
                # without enabling the agent's optional image sidecars.
                import leaderboard.autoagents.agent_wrapper as agent_wrapper_module

                agent_wrapper_module.MAX_ALLOWED_RADIUS_SENSOR = 100.0
            sensors = self.policy_agent.sensors()
            if self._expert_mode:
                # PDM's data logger adds its own RGB views to the model's
                # genuine camera suite. Validate all sensors, but use the
                # collector's eight-camera limit instead of the four-camera
                # submission limit; restore it before any other agent runs.
                import leaderboard.autoagents.agent_wrapper_local as local_wrapper

                limits = local_wrapper.QUALIFIER_SENSORS_LIMITS
                rgb_limit = limits["sensor.camera.rgb"]
                try:
                    limits["sensor.camera.rgb"] = local_wrapper.SENSORS_LIMITS["sensor.camera.rgb"]
                    validate_sensor_configuration(sensors, agent_track, selected_track)
                finally:
                    limits["sensor.camera.rgb"] = rgb_limit
            else:
                validate_sensor_configuration(sensors, agent_track, selected_track)
            self.agent_wrapper = AgentWrapperFactory.get_wrapper(self.policy_agent)
            # Bench2Drive's Scenic-era wrapper keeps this list as a class
            # attribute until cleanup.  Make it instance-local so multiple
            # native programs can run in one adapter process.
            self.agent_wrapper._sensors_list = []
            if self._expert_mode:
                self.agent_wrapper.sensor_list_names = []
            self.agent_wrapper.setup_sensors(ego_actor)

            collision_bp = self.world.get_blueprint_library().find("sensor.other.collision")
            self.collision_sensor = self.world.spawn_actor(
                collision_bp,
                carla.Transform(),
                attach_to=ego_actor,
            )
            self.collision_sensor.listen(self._on_collision)
            # The wrapper's warm-up ticks have already populated sensor data;
            # one callback-driven tick ensures the policy starts at a clean
            # current GameTime frame.
            self.world.tick()

        def _on_carla_tick(self, snapshot) -> None:
            GameTime.on_carla_tick(snapshot.timestamp)
            CarlaDataProvider.on_carla_tick()

        def _on_collision(self, event) -> None:
            other = getattr(event, "other_actor", None)
            self.collision_events.append(
                {
                    "frame": int(getattr(event, "frame", GameTime.get_frame())),
                    "timestamp_s": float(GameTime.get_time()),
                    "other_actor": getattr(other, "type_id", "unknown"),
                    "intensity": _vector_magnitude(getattr(event, "normal_impulse", None)),
                }
            )

        def _run_policy_step(self) -> None:
            frame = GameTime.get_frame()
            if not frame:
                snapshot = self.world.get_snapshot()
                self._on_carla_tick(snapshot)
                frame = GameTime.get_frame()
            input_data = self.policy_agent.sensor_interface.get_data(frame)
            timestamp = float(GameTime.get_time())
            control = (
                self.policy_agent.run_step(input_data, timestamp, self.agent_wrapper.sensor_list_names)
                if self._expert_mode
                else self.policy_agent.run_step(input_data, timestamp)
            )
            if control is None or not hasattr(control, "steer"):
                raise NativePolicyExecutionError(
                    f"Agent returned an invalid control at frame {frame}: {control!r}"
                )
            control.manual_gear_shift = False
            self.objects[0].carlaActor.apply_control(control)
            velocity = self.objects[0].carlaActor.get_velocity()
            speed = _vector_magnitude(velocity)
            self.policy_trace.append(
                {
                    "scenario_id": self._scenario_id,
                    "frame": int(frame),
                    "timestamp_s": timestamp,
                    "steer": float(control.steer),
                    "throttle": float(control.throttle),
                    "brake": float(control.brake),
                    "hand_brake": bool(control.hand_brake),
                    "reverse": bool(control.reverse),
                    "ego_speed_mps": speed,
                }
            )

        def run(self, maxSteps):
            """Scenic 2 loop with the official agent inserted before each tick."""
            import scenic.syntax.veneer as veneer

            if self.currentTime > 0:
                raise RuntimeError("tried to run a Simulation which has already run")
            action_sequence = []
            veneer.beginSimulation(self)
            dynamic_scenario = self.scene.dynamicScenario
            try:
                dynamic_scenario._start()
                self.updateObjects()
                termination_reason = None
                while maxSteps is None or self.currentTime < maxSteps:
                    if self.verbosity >= 3:
                        print(f"    Time step {self.currentTime}:")
                    termination_reason = dynamic_scenario._step()
                    dynamic_scenario._checkAlwaysRequirements()
                    new_reason = dynamic_scenario._runMonitors()
                    if new_reason is not None:
                        termination_reason = new_reason
                    if termination_reason is not None:
                        break
                    termination_reason = dynamic_scenario._checkSimulationTerminationConditions()
                    if termination_reason is not None:
                        break

                    all_actions = OrderedDict()
                    for agent in self.scheduleForAgents():
                        behavior = agent.behavior
                        if behavior is None:
                            continue
                        if not behavior._runningIterator:
                            behavior.start(agent)
                        actions = behavior.step()
                        if isinstance(actions, EndSimulationAction):
                            termination_reason = str(actions)
                            break
                        if isinstance(actions, EndScenarioAction):
                            termination_reason = str(actions)
                            break
                        if not isinstance(actions, tuple):
                            raise NativePolicyExecutionError(
                                f"Scenic behavior returned non-tuple actions: {actions!r}"
                            )
                        if len(actions) == 1 and isinstance(actions[0], (list, tuple)):
                            actions = tuple(actions[0])
                        if not self.actionsAreCompatible(agent, actions):
                            raise NativePolicyExecutionError(
                                f"Scenic agent {agent} returned incompatible actions {actions!r}"
                            )
                        all_actions[agent] = actions
                    if termination_reason is not None:
                        break

                    self.executeActions(all_actions)
                    self._run_policy_step()
                    self.step()
                    self.updateObjects()
                    self.currentTime += 1
                    self.trajectory.append(self.currentState())
                    action_sequence.append(all_actions)

                if termination_reason is None:
                    termination_reason = f"reached time limit ({maxSteps} steps)"
                from scenic.core.simulators import SimulationResult

                return SimulationResult(self.trajectory, action_sequence, termination_reason)
            finally:
                # Always unwind Scenic's global veneer state, even if an
                # actor cleanup callback raises.  The manifest runner is
                # intentionally per-program fault tolerant, so one malformed
                # legacy scene must not poison compilation of the next one.
                try:
                    self.destroy()
                finally:
                    try:
                        for obj in self.scene.objects:
                            from scenic.core.object_types import disableDynamicProxyFor

                            disableDynamicProxyFor(obj)
                        for agent in self.agents:
                            if agent.behavior is not None:
                                agent.behavior.stop()
                        for monitor in self.scene.monitors:
                            monitor.stop()
                    finally:
                        veneer.endSimulation(self)

        def destroy(self):
            if self._destroyed:
                return
            self._destroyed = True
            if self.agent_wrapper is not None:
                try:
                    self.agent_wrapper.cleanup()
                except Exception:
                    pass
                self.agent_wrapper = None
            if self.collision_sensor is not None:
                try:
                    self.collision_sensor.stop()
                    self.collision_sensor.destroy()
                except Exception:
                    pass
                self.collision_sensor = None
            if self.policy_agent is not None:
                try:
                    self.policy_agent.destroy()
                except Exception:
                    pass
                self.policy_agent = None
            if self._tick_callback_id is not None:
                try:
                    self.world.remove_on_tick(self._tick_callback_id)
                except Exception:
                    pass
                self._tick_callback_id = None
            ego_actor = getattr(self.objects[0], "carlaActor", None) if self.objects else None
            try:
                super().destroy()
            finally:
                if ego_actor is not None:
                    CarlaDataProvider._carla_actor_pool.pop(ego_actor.id, None)
                    for mapping_name in (
                        "_actor_velocity_map",
                        "_actor_location_map",
                        "_actor_transform_map",
                    ):
                        getattr(CarlaDataProvider, mapping_name, {}).pop(ego_actor, None)

    class PolicyCarlaSimulator(CarlaSimulator):
        def __init__(self, *args, **kwargs):
            reuse_loaded_world = kwargs.pop("reuse_loaded_opendrive_world", False)
            self._policy_kwargs = {
                "agent_class": kwargs.pop("agent_class"),
                "agent_config": kwargs.pop("agent_config"),
                "agent_host": kwargs.pop("agent_host"),
                "agent_port": kwargs.pop("agent_port"),
                "selected_track": kwargs.pop("selected_track"),
                "agent_debug": kwargs.pop("agent_debug"),
                "route_points": kwargs.pop("route_points"),
                "scenario_id": kwargs.pop("scenario_id"),
                "expert_mode": kwargs.pop("expert_mode"),
                "route_index": kwargs.pop("route_index"),
            }
            tm_port = kwargs.pop("traffic_manager_port", None)
            if reuse_loaded_world:
                from scenic.domains.driving.simulators import DrivingSimulator

                DrivingSimulator.__init__(self)
                self.client = carla.Client(kwargs["address"], kwargs["port"])
                self.client.set_timeout(kwargs["timeout"])
                self.world = self.client.get_world()
                if not self.world.get_map().name.endswith("/OpenDriveMap"):
                    raise NativePolicyExecutionError(
                        "Previously loaded OpenDRIVE world was replaced before Scenic reuse"
                    )
                self.timestep = kwargs["timestep"]
                self.tm = self.client.get_trafficmanager()
                self.tm.set_synchronous_mode(True)
                settings = self.world.get_settings()
                settings.synchronous_mode = True
                settings.fixed_delta_seconds = self.timestep
                self.world.apply_settings(settings)
                self.render = kwargs["render"]
                self.record = kwargs["record"]
                self.scenario_number = 0
            else:
                super().__init__(*args, **kwargs)
            if tm_port is not None:
                self.tm.set_synchronous_mode(False)
                self.tm = self.client.get_trafficmanager(int(tm_port))
                self.tm.set_synchronous_mode(True)
                self._policy_tm_port = int(tm_port)
            else:
                self._policy_tm_port = self.tm.get_port()
            self.last_simulation = None

        def createSimulation(self, scene, verbosity=0):
            self.scenario_number += 1
            policy_kwargs = dict(self._policy_kwargs)
            policy_kwargs["route_goal"] = getattr(scene, "params", {}).get("fdset_route_goal")
            simulation = PolicyCarlaSimulation(
                scene,
                self.client,
                self.tm,
                self.timestep,
                render=self.render,
                record=self.record,
                scenario_number=self.scenario_number,
                verbosity=verbosity,
                **policy_kwargs,
            )
            self.last_simulation = simulation
            return simulation

    return PolicyCarlaSimulator, PolicyCarlaSimulation


def _build_route(
    actor, world, route_points: int, road_option_type, carla_module, *, route_goal=None,
):
    carla_map = world.get_map()
    start = actor.get_location()
    driving_lane = carla_map.get_waypoint(
        start,
        project_to_road=True,
        lane_type=carla_module.LaneType.Driving,
    )
    if driving_lane is None:
        raise NativePolicyExecutionError(
            "Could not project Scenic ego spawn onto a driving lane"
        )

    if route_goal is None:
        # Preserve the legacy route exactly for Scenic programs without a
        # sampled navigation goal.
        def waypoint_key(waypoint):
            location = waypoint.transform.location
            return (
                waypoint.road_id,
                waypoint.section_id,
                waypoint.lane_id,
                round(float(location.x), 2),
                round(float(location.y), 2),
                round(float(location.z), 2),
            )

        route = [(driving_lane.transform, road_option_type.LANEFOLLOW)]
        visited = {waypoint_key(driving_lane)}
        for _ in range(route_points - 1):
            candidates = driving_lane.next(2.0)
            if not candidates:
                break
            next_waypoint = candidates[0]
            key = waypoint_key(next_waypoint)
            if key in visited and len(candidates) > 1:
                next_waypoint = candidates[1]
                key = waypoint_key(next_waypoint)
            if key in visited:
                break
            visited.add(key)
            route.append((next_waypoint.transform, road_option_type.LANEFOLLOW))
            driving_lane = next_waypoint
        if len(route) < 2:
            raise NativePolicyExecutionError(
                "Could not build a two-point route from the Scenic ego spawn"
            )
        return route

    try:
        goal_x = float(route_goal.x)
        goal_y = float(route_goal.y)
        goal_z = float(getattr(route_goal, "z", start.z))
    except (AttributeError, TypeError, ValueError) as exc:
        raise NativePolicyExecutionError(
            "Scenic fdset_route_goal must be a sampled world point with x/y coordinates"
        ) from exc
    goal = carla_module.Location(x=goal_x, y=-goal_y, z=goal_z)
    goal_lane = carla_map.get_waypoint(
        goal,
        project_to_road=True,
        lane_type=carla_module.LaneType.Driving,
    )
    if goal_lane is None:
        raise NativePolicyExecutionError(
            "Scenic fdset_route_goal does not project onto a driving lane"
        )
    projected_goal = goal_lane.transform.location
    if (
        (float(projected_goal.x) - goal.x) ** 2
        + (float(projected_goal.y) - goal.y) ** 2
        + (float(projected_goal.z) - goal.z) ** 2
    ) ** 0.5 > 3.0:
        raise NativePolicyExecutionError(
            "Scenic fdset_route_goal is not on a driving lane"
        )

    try:
        from agents.navigation.global_route_planner import GlobalRoutePlanner

        try:
            planner = GlobalRoutePlanner(carla_map, 2.0)
        except TypeError:
            from agents.navigation.global_route_planner_dao import GlobalRoutePlannerDAO

            planner = GlobalRoutePlanner(GlobalRoutePlannerDAO(carla_map, 2.0))
        setup = getattr(planner, "setup", None)
        if callable(setup):
            setup()
        planned = planner.trace_route(start, goal)
    except Exception as exc:
        raise NativePolicyExecutionError(
            f"CARLA could not navigate from Scenic ego spawn to fdset_route_goal: {exc}"
        ) from exc
    if not planned or len(planned) < 2:
        raise NativePolicyExecutionError(
            "CARLA GlobalRoutePlanner returned fewer than two points for fdset_route_goal"
        )
    route = [(waypoint.transform, option) for waypoint, option in planned]
    if len(route) < 2:
        raise NativePolicyExecutionError(
            "CARLA GlobalRoutePlanner returned fewer than two route points"
        )
    return route


def _collision_failure_record(
    program: dict[str, Any],
    collision: dict[str, Any],
    *,
    timestep: float,
    scene_tags: list[str],
    ego_speed: float = 0.0,
) -> dict[str, Any]:
    timestamp_value = collision.get("timestamp_s")
    timestamp_s = (
        float(timestamp_value)
        if timestamp_value is not None
        else float(collision.get("frame", 0)) * timestep
    )
    return {
        "route_id": str(program.get("scenario_id", "native_scenic")),
        "timestamp_s": timestamp_s,
        "event_type": "collision",
        "ego_speed": ego_speed,
        "note": f"native Scenic ego collision with {collision.get('other_actor', 'unknown')}",
        "scene_tags": scene_tags,
        "risk_level": "high",
        "metadata": {
            "source": "native_scenic_policy",
            "frame": collision.get("frame"),
            "intensity": collision.get("intensity", 0.0),
        },
    }


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _vector_magnitude(vector: Any) -> float:
    if vector is None:
        return 0.0
    return float(
        (
            float(getattr(vector, "x", 0.0)) ** 2
            + float(getattr(vector, "y", 0.0)) ** 2
            + float(getattr(vector, "z", 0.0)) ** 2
        )
        ** 0.5
    )


def _safe_name(value: str) -> str:
    return "".join(char if char.isalnum() or char in {"-", "_", "."} else "_" for char in value)


def main() -> None:
    """Small standalone entry point for one materialized manifest."""
    import argparse

    parser = argparse.ArgumentParser(description="Run native Scenic programs with a Bench2Drive agent")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume-completed", action="store_true")
    args = parser.parse_args()

    from .closed_loop_runner import load_config

    result = run_native_policy_manifest(
        args.manifest,
        load_config(args.config),
        args.output_dir,
        dry_run=args.dry_run,
        resume_completed=args.resume_completed,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
