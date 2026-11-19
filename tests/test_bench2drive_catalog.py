from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from core.bench2drive_catalog import (
    IGNORED_PARAMETER_TAGS,
    known_scenario_types,
    supported_parameter_tags,
)
from core.closed_loop_runner import resolve_bench2drive_root


def test_bench2drive_catalog_covers_official_training_and_eval_xml() -> None:
    known_types = known_scenario_types()
    known_tags = supported_parameter_tags() | set(IGNORED_PARAMETER_TAGS)

    try:
        bench_root = resolve_bench2drive_root({})
    except FileNotFoundError:
        pytest.skip("Bench2Drive checkout is not available")
    xml_paths = [
        bench_root / "leaderboard" / "data" / "routes_training.xml",
        bench_root / "leaderboard" / "data" / "bench2drive220.xml",
    ]

    seen_types: set[str] = set()
    seen_tags: set[str] = set()
    for xml_path in xml_paths:
        root = ET.parse(xml_path).getroot()
        for scenario in root.iter("scenario"):
            scenario_type = str(scenario.attrib.get("type", ""))
            if scenario_type:
                seen_types.add(scenario_type)
            for child in scenario:
                if child.tag == "trigger_point":
                    continue
                seen_tags.add(child.tag)

    assert seen_types <= known_types
    assert seen_tags <= known_tags
