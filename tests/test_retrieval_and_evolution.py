from __future__ import annotations

from core.models import FailureEvent, StructuredFailure
from core.scenario_evolve import evolve_from_anchors
from core.scenario_retrieve import retrieve_anchor_scenarios


def test_retrieval_and_evolution_include_provenance_metadata() -> None:
    failure = StructuredFailure(
        event=FailureEvent(
            route_id="route_001",
            timestamp_s=12.0,
            failure_type="collision",
            ego_speed=8.0,
            description="cut-in vehicle merged aggressively",
            scene_tags=["urban", "merge"],
            risk_level="high",
        ),
        actors=["ego", "vehicle"],
        interaction_pattern="adversarial_merge",
        constraints={"ego_speed_min": 6.0, "ego_speed_max": 10.0},
        semantic_evidence=None,
    )
    scenario_bank = [
        {
            "scenario_id": "seed_merge_01",
            "tags": ["urban", "merge"],
            "pattern": "adversarial_merge",
            "config": {
                "ego_speed": 7.5,
                "merge_offset": 16.0,
                "bench2drive_profile": {
                    "scenario_family": "merge",
                },
                "bench2drive_spec": {
                    "route": {"id": "1", "town": "Town12", "waypoints": []},
                    "scenario": {
                        "name": "HighwayCutIn_1",
                        "type": "HighwayCutIn",
                        "trigger_point": {"x": 0.0, "y": 0.0, "z": 0.0, "yaw": 0.0},
                        "parameters": [
                            {"tag": "flow_speed", "attrs": {"value": "8.0"}},
                            {"tag": "source_dist_interval", "attrs": {"from": "15", "to": "30"}},
                        ],
                    },
                },
            },
        }
    ]

    anchors = retrieve_anchor_scenarios(failure, scenario_bank, top_k=1)
    evolved = evolve_from_anchors(failure, anchors, escalation_strength=0.2, use_escalation=True)

    assert anchors[0].score_breakdown["tag_overlap"] == 2.0
    assert anchors[0].score_breakdown["pattern_bonus"] == 2.0
    assert anchors[0].score_breakdown["speed_bonus"] == 1.0
    assert anchors[0].score_breakdown["text_bonus"] > 0.0
    assert anchors[0].score_breakdown["control_bonus"] > 0.0
    assert evolved[0].metadata is not None
    assert evolved[0].metadata["anchor_score"] > 5.0
    assert evolved[0].metadata["applied_escalation_strength"] == 0.2
    assert evolved[0].modifications["bench2drive_parameter_overrides"]["flow_speed"]["value"] == "9"
    assert evolved[0].modifications["bench2drive_parameter_overrides"]["source_dist_interval"]["to"] == "12.8"


def test_evolution_generates_real_parameter_overrides_for_pedestrian_and_frequency_controls() -> None:
    failure = StructuredFailure(
        event=FailureEvent(
            route_id="route_002",
            timestamp_s=25.0,
            failure_type="collision_pedestrian",
            ego_speed=6.0,
            description="pedestrian crossed from side lane",
            scene_tags=["urban", "pedestrian"],
            risk_level="high",
        ),
        actors=["ego", "pedestrian"],
        interaction_pattern="pedestrian_intrusion",
        constraints={"ego_speed_min": 4.0, "ego_speed_max": 8.0},
        semantic_evidence=None,
    )
    scenario_bank = [
        {
            "scenario_id": "seed_hazard_01",
            "tags": ["urban", "pedestrian"],
            "pattern": "pedestrian_intrusion",
            "config": {
                "ego_speed": 6.5,
                "spawn_gap": 20.0,
                "ped_speed": 8.0,
                "ped_travel_dist": 50.0,
                "traffic_frequency": 80.0,
                "bench2drive_profile": {"scenario_family": "pedestrian"},
                "bench2drive_spec": {
                    "route": {"id": "2", "town": "Town12", "waypoints": []},
                    "scenario": {
                        "name": "HazardAtSideLaneTwoWays_1",
                        "type": "HazardAtSideLaneTwoWays",
                        "trigger_point": {"x": 0.0, "y": 0.0, "z": 0.0, "yaw": 0.0},
                        "parameters": [
                            {"tag": "distance", "attrs": {"value": "50"}},
                            {"tag": "frequency", "attrs": {"value": "80"}},
                            {"tag": "bicycle_drive_distance", "attrs": {"value": "50"}},
                            {"tag": "bicycle_speed", "attrs": {"value": "8"}},
                        ],
                    },
                },
            },
        }
    ]

    anchors = retrieve_anchor_scenarios(failure, scenario_bank, top_k=1)
    evolved = evolve_from_anchors(failure, anchors, escalation_strength=0.25, use_escalation=True)

    overrides = evolved[0].modifications["bench2drive_parameter_overrides"]
    assert overrides["distance"]["value"] == "15"
    assert overrides["frequency"]["value"] == "100"
    assert overrides["bicycle_drive_distance"]["value"] == "62.5"
    assert overrides["bicycle_speed"]["value"] == "9"


def test_evolution_generates_junction_and_control_overrides() -> None:
    junction_failure = StructuredFailure(
        event=FailureEvent(
            route_id="route_003",
            timestamp_s=30.0,
            failure_type="blocked",
            ego_speed=7.0,
            description="intersection turning conflict with blocked lane",
            scene_tags=["urban", "junction"],
            risk_level="high",
        ),
        actors=["ego", "vehicle"],
        interaction_pattern="junction_conflict",
        constraints={"ego_speed_min": 5.0, "ego_speed_max": 9.0},
        semantic_evidence=None,
    )
    junction_bank = [
        {
            "scenario_id": "seed_invading_turn_01",
            "tags": ["urban", "junction"],
            "pattern": "junction_conflict",
            "config": {
                "ego_speed": 7.5,
                "obstacle_dist": 100.0,
                "lane_offset": 0.2,
                "bench2drive_profile": {"scenario_family": "junction"},
                "bench2drive_spec": {
                    "route": {"id": "3", "town": "Town12", "waypoints": []},
                    "scenario": {
                        "name": "InvadingTurn_1",
                        "type": "InvadingTurn",
                        "trigger_point": {"x": 0.0, "y": 0.0, "z": 0.0, "yaw": 0.0},
                        "parameters": [
                            {"tag": "distance", "attrs": {"value": "100"}},
                            {"tag": "offset", "attrs": {"value": "0.2"}},
                        ],
                    },
                },
            },
        }
    ]

    anchors = retrieve_anchor_scenarios(junction_failure, junction_bank, top_k=1)
    evolved = evolve_from_anchors(junction_failure, anchors, escalation_strength=0.25, use_escalation=True)
    junction_overrides = evolved[0].modifications["bench2drive_parameter_overrides"]
    assert junction_overrides["distance"]["value"] == "75"
    assert junction_overrides["offset"]["value"] == "0.25"

    control_failure = StructuredFailure(
        event=FailureEvent(
            route_id="route_004",
            timestamp_s=31.0,
            failure_type="control_loss",
            ego_speed=6.5,
            description="control loss during sharp maneuver",
            scene_tags=["urban", "control"],
            risk_level="high",
        ),
        actors=["ego"],
        interaction_pattern="control_instability",
        constraints={"ego_speed_min": 5.0, "ego_speed_max": 8.0},
        semantic_evidence=None,
    )
    control_bank = [
        {
            "scenario_id": "seed_control_01",
            "tags": ["urban", "control"],
            "pattern": "control_instability",
            "config": {
                "ego_speed": 6.0,
                "lane_offset": 0.15,
                "bench2drive_profile": {"scenario_family": "control"},
                "bench2drive_spec": {
                    "route": {"id": "4", "town": "Town12", "waypoints": []},
                    "scenario": {
                        "name": "ControlLoss_1",
                        "type": "ControlLoss",
                        "trigger_point": {"x": 0.0, "y": 0.0, "z": 0.0, "yaw": 0.0},
                        "parameters": [
                            {"tag": "offset", "attrs": {"value": "0.15"}},
                        ],
                    },
                },
            },
        }
    ]

    anchors = retrieve_anchor_scenarios(control_failure, control_bank, top_k=1)
    evolved = evolve_from_anchors(control_failure, anchors, escalation_strength=0.2, use_escalation=True)
    control_overrides = evolved[0].modifications["bench2drive_parameter_overrides"]
    assert control_overrides["offset"]["value"] == "0.18"


def test_scenic_descriptions_disambiguate_equally_tagged_scenario_anchors() -> None:
    failure = StructuredFailure(
        event=FailureEvent(
            route_id="route_001",
            timestamp_s=12.0,
            failure_type="collision",
            ego_speed=5.0,
            description="pedestrian crosses unexpectedly",
            scene_tags=["urban"],
            risk_level="high",
        ),
        actors=["ego"],
        interaction_pattern="generic_failure",
        constraints={},
        semantic_evidence=None,
    )
    candidates = [
        {
            "scenario_id": "scenic_01",
            "tags": ["urban"],
            "pattern": "generic_failure",
            "config": {"scenic_description": "vehicle changes lanes ahead"},
        },
        {
            "scenario_id": "scenic_02",
            "tags": ["urban"],
            "pattern": "generic_failure",
            "config": {"scenic_description": "pedestrian crosses unexpectedly"},
        },
    ]

    result = retrieve_anchor_scenarios(failure, candidates, top_k=1)
    assert result[0].scenario_id == "scenic_02"
    assert result[0].score_breakdown["text_bonus"] > 0
