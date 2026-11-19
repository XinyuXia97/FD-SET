from __future__ import annotations

import json
from pathlib import Path

from core.scenario_materialize import build_bench2drive_routes_artifacts, build_bench2drive_routes_xml, materialize_round_inputs


def test_build_bench2drive_routes_xml_from_scenario_bank() -> None:
    scenario_bank = [
        {
            "scenario_id": "b2d_1_HighwayCutIn_1",
            "tags": ["bench2drive", "Town12", "HighwayCutIn"],
            "pattern": "adversarial_merge",
            "config": {
                "ego_speed": 9.5,
                "merge_offset": 30.0,
                "bench2drive_spec": {
                    "route": {
                        "id": "1",
                        "town": "Town12",
                        "waypoints": [
                            {"x": 1.0, "y": 2.0, "z": 3.0},
                            {"x": 4.0, "y": 5.0, "z": 6.0},
                        ],
                    },
                    "scenario": {
                        "name": "HighwayCutIn_1",
                        "type": "HighwayCutIn",
                        "trigger_point": {"x": 1.0, "y": 2.0, "z": 3.0, "yaw": 90.0},
                        "parameters": [
                            {"tag": "flow_speed", "attrs": {"value": "8.3"}},
                            {"tag": "source_dist_interval", "attrs": {"from": "15", "to": "30"}},
                        ],
                    },
                },
            },
        }
    ]

    xml_payload = build_bench2drive_routes_xml(scenario_bank)

    assert xml_payload is not None
    assert 'route id="1" town="Town12"' in xml_payload
    assert 'scenario name="b2d_1_HighwayCutIn_1" type="HighwayCutIn"' in xml_payload
    assert 'flow_speed value="9.5"' in xml_payload


def test_materialization_applies_type_aware_updates_and_writes_trace(tmp_path: Path) -> None:
    scenario_bank = [
        {
            "scenario_id": "b2d_2_ParkingCrossingPedestrian_1",
            "tags": ["bench2drive", "Town12", "ParkingCrossingPedestrian"],
            "pattern": "pedestrian_intrusion",
            "config": {
                "spawn_gap": 18.0,
                "ped_speed": 3.0,
                "bench2drive_profile": {
                    "scenario_family": "pedestrian",
                    "scenario_type": "ParkingCrossingPedestrian",
                },
                "bench2drive_spec": {
                    "route": {
                        "id": "2",
                        "town": "Town12",
                        "waypoints": [{"x": 1.0, "y": 2.0, "z": 3.0}],
                    },
                    "scenario": {
                        "name": "ParkingCrossingPedestrian_1",
                        "type": "ParkingCrossingPedestrian",
                        "trigger_point": {"x": 1.0, "y": 2.0, "z": 3.0, "yaw": 90.0},
                        "parameters": [
                            {"tag": "distance", "attrs": {"value": "50"}},
                            {"tag": "bicycle_speed", "attrs": {"value": "8"}},
                        ],
                    },
                },
            },
        }
    ]

    xml_payload, trace = build_bench2drive_routes_artifacts(scenario_bank)

    assert xml_payload is not None
    assert 'distance value="18"' in xml_payload
    assert 'bicycle_speed value="3"' in xml_payload
    assert trace[0]["scenario_family"] == "pedestrian"
    assert {item["tag"] for item in trace[0]["applied_parameter_updates"]} == {"distance", "bicycle_speed"}

    outputs = materialize_round_inputs(
        scenario_bank,
        tmp_path,
        {
            "materialization": {
                "bench2drive": {
                    "enabled": True,
                    "route_file_name": "routes.xml",
                    "trace_file_name": "trace.json",
                }
            }
        },
    )

    assert outputs["bench2drive_xml"].name == "routes.xml"
    trace_payload = json.loads(outputs["bench2drive_trace"].read_text(encoding="utf-8"))
    assert trace_payload[0]["scenario_id"] == "b2d_2_ParkingCrossingPedestrian_1"


def test_materialization_preserves_real_frequency_and_actor_controls() -> None:
    scenario_bank = [
        {
            "scenario_id": "b2d_3_HazardAtSideLaneTwoWays_1",
            "tags": ["bench2drive", "Town12", "HazardAtSideLaneTwoWays"],
            "pattern": "pedestrian_intrusion",
            "config": {
                "spawn_gap": 22.0,
                "ped_speed": 6.0,
                "ped_travel_dist": 75.0,
                "traffic_frequency": 95.0,
                "bench2drive_profile": {
                    "scenario_family": "pedestrian",
                    "scenario_type": "HazardAtSideLaneTwoWays",
                },
                "bench2drive_spec": {
                    "route": {
                        "id": "3",
                        "town": "Town12",
                        "waypoints": [{"x": 1.0, "y": 2.0, "z": 3.0}],
                    },
                    "scenario": {
                        "name": "HazardAtSideLaneTwoWays_1",
                        "type": "HazardAtSideLaneTwoWays",
                        "trigger_point": {"x": 1.0, "y": 2.0, "z": 3.0, "yaw": 90.0},
                        "parameters": [
                            {"tag": "distance", "attrs": {"value": "50"}},
                            {"tag": "frequency", "attrs": {"value": "80"}},
                            {"tag": "bicycle_drive_distance", "attrs": {"value": "50"}},
                            {"tag": "bicycle_speed", "attrs": {"value": "5"}},
                        ],
                    },
                },
            },
        }
    ]

    xml_payload, trace = build_bench2drive_routes_artifacts(scenario_bank)

    assert xml_payload is not None
    assert 'distance value="22"' in xml_payload
    assert 'frequency value="95"' in xml_payload
    assert 'bicycle_drive_distance value="75"' in xml_payload
    assert 'bicycle_speed value="6"' in xml_payload
    assert {item["tag"] for item in trace[0]["applied_parameter_updates"]} >= {
        "distance",
        "frequency",
        "bicycle_drive_distance",
        "bicycle_speed",
    }


def test_materialization_handles_junction_and_signal_specific_parameters() -> None:
    scenario_bank = [
        {
            "scenario_id": "b2d_4_InvadingTurn_1",
            "tags": ["bench2drive", "Town12", "InvadingTurn"],
            "pattern": "junction_conflict",
            "config": {
                "ego_speed": 8.0,
                "obstacle_dist": 90.0,
                "lane_offset": 0.25,
                "bench2drive_profile": {
                    "scenario_family": "junction",
                    "scenario_type": "InvadingTurn",
                },
                "bench2drive_spec": {
                    "route": {
                        "id": "4",
                        "town": "Town12",
                        "waypoints": [{"x": 1.0, "y": 2.0, "z": 3.0}],
                    },
                    "scenario": {
                        "name": "InvadingTurn_1",
                        "type": "InvadingTurn",
                        "trigger_point": {"x": 1.0, "y": 2.0, "z": 3.0, "yaw": 90.0},
                        "parameters": [
                            {"tag": "distance", "attrs": {"value": "100"}},
                            {"tag": "offset", "attrs": {"value": "0.2"}},
                        ],
                    },
                },
            },
        },
        {
            "scenario_id": "b2d_5_OppositeVehicleRunningRedLight_1",
            "tags": ["bench2drive", "Town12", "OppositeVehicleRunningRedLight"],
            "pattern": "signal_violation",
            "config": {
                "turn_direction": "right",
                "bench2drive_profile": {
                    "scenario_family": "signal",
                    "scenario_type": "OppositeVehicleRunningRedLight",
                },
                "bench2drive_spec": {
                    "route": {
                        "id": "5",
                        "town": "Town12",
                        "waypoints": [{"x": 1.0, "y": 2.0, "z": 3.0}],
                    },
                    "scenario": {
                        "name": "OppositeVehicleRunningRedLight_1",
                        "type": "OppositeVehicleRunningRedLight",
                        "trigger_point": {"x": 1.0, "y": 2.0, "z": 3.0, "yaw": 90.0},
                        "parameters": [
                            {"tag": "direction", "attrs": {"value": "left"}},
                        ],
                    },
                },
            },
        },
    ]

    xml_payload, trace = build_bench2drive_routes_artifacts(scenario_bank)

    assert xml_payload is not None
    assert 'distance value="90"' in xml_payload
    assert 'offset value="0.25"' in xml_payload
    assert 'direction value="right"' in xml_payload
    assert trace[0]["scenario_family"] == "junction"
    assert trace[1]["scenario_family"] == "signal"
