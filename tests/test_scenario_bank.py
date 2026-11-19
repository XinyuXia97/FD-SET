from __future__ import annotations

import pytest

from core.scenario_bank import append_evolved_to_bank, validate_scenario_bank


def test_validate_scenario_bank_rejects_duplicate_ids() -> None:
    with pytest.raises(ValueError, match="Duplicate scenario_id"):
        validate_scenario_bank(
            [
                {
                    "scenario_id": "dup",
                    "tags": ["urban"],
                    "pattern": "generic_failure",
                    "config": {},
                },
                {
                    "scenario_id": "dup",
                    "tags": ["urban"],
                    "pattern": "generic_failure",
                    "config": {},
                },
            ]
        )


def test_append_evolved_to_bank_deduplicates_and_normalizes_tags() -> None:
    bank = [
        {
            "scenario_id": "seed_01",
            "tags": ["urban"],
            "pattern": "generic_failure",
            "config": {},
        }
    ]
    evolved = [
        {
            "evolved_id": "seed_01_evo_0",
            "modifications": {
                "interaction_pattern": "adversarial_merge",
                "tags": ["urban", "urban", "merge"],
            },
        },
        {
            "evolved_id": "seed_01_evo_0",
            "modifications": {
                "interaction_pattern": "adversarial_merge",
                "tags": ["urban", "merge"],
            },
        },
    ]

    updated = append_evolved_to_bank(bank, evolved)

    assert len(updated) == 2
    assert updated[-1]["tags"] == ["urban", "merge"]
