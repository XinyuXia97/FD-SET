"""PDM-Lite paths and map resources for native Scenic collection."""

from __future__ import annotations

from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
AGENT_ROOT = PACKAGE_ROOT / "agent"
DATA_AGENT = AGENT_ROOT / "data_agent.py"
DEFAULT_AGENT_CONFIG = AGENT_ROOT / "expert_agent_config.json"
RUNTIME_ROOT = PACKAGE_ROOT / "runtime"
LEADERBOARD_ROOT = RUNTIME_ROOT / "leaderboard_autopilot"
SCENARIO_RUNNER_ROOT = RUNTIME_ROOT / "scenario_runner_autopilot"


def ensure_speed_limit_map(carla_map, output_dir: Path) -> Path:
    """Use measured CARLA speed limits only for an identical sampled road network.

    For unmatched OpenDRIVE maps, retain PDM-Lite's speed-sign lookup and its
    50 km/h fallback when the map has no speed-limit signs.
    """
    import carla
    import numpy as np

    output_dir.mkdir(parents=True, exist_ok=True)
    map_name = carla_map.name.split("/")[-1]
    path = output_dir / f"{map_name}_speed_limits.npy"
    if path.exists():
        return path
    waypoints = carla_map.generate_waypoints(1.0)
    if not waypoints:
        raise RuntimeError(f"CARLA map {map_name} has no driving waypoints for PDM-Lite")
    locations = np.asarray(
        [[wp.transform.location.x, wp.transform.location.y, wp.transform.location.z] for wp in waypoints],
        dtype=np.float64,
    )
    references = Path(__file__).resolve().parent / "agent" / "speed_limits"
    for reference_path in sorted(references.glob("*_speed_limits.npy")):
        reference = np.load(reference_path, allow_pickle=True).item()
        if np.array_equal(reference["locations"], locations):
            speeds = np.asarray(reference["speed_limits"], dtype=np.float64)
            if speeds.shape != (len(waypoints),) or not np.isfinite(speeds).all() or (speeds <= 0).any():
                raise RuntimeError(f"Invalid PDM-Lite speed-limit cache for {reference_path}")
            np.save(path, {"locations": locations, "speed_limits": speeds})
            return path
    signs = carla_map.get_all_landmarks_of_type(carla.LandmarkType.MaximumSpeed)
    if signs:
        limits = []
        for waypoint in waypoints:
            nearby = waypoint.get_landmarks_of_type(4000, carla.LandmarkType.MaximumSpeed)
            if not nearby:
                raise RuntimeError(
                    f"Speed-sign map {map_name} has an unsigned road; cannot infer its PDM-Lite speed limit"
                )
            limits.append(float(nearby[0].value))
        speeds = np.asarray(limits, dtype=np.float64)
    else:
        speeds = np.full(len(waypoints), 50.0, dtype=np.float64)
    if not np.isfinite(speeds).all() or (speeds <= 0).any():
        raise RuntimeError(f"Invalid CARLA speed-limit landmarks on {map_name}")
    np.save(path, {"locations": locations, "speed_limits": speeds})
    return path
