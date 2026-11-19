from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .scenic_library import build_scenic_source_bank, merge_scenic_source_bank


def _normalize_scenario(row: dict[str, Any], *, index: int) -> dict[str, Any]:
    scenario_id = str(row.get("scenario_id", "")).strip()
    if not scenario_id:
        raise ValueError(f"Scenario at index {index} is missing scenario_id")

    raw_tags = row.get("tags", [])
    if not isinstance(raw_tags, list):
        raise ValueError(f"Scenario {scenario_id} has invalid tags; expected a list")
    tags = [str(tag) for tag in raw_tags]

    pattern = str(row.get("pattern", "generic_failure")).strip() or "generic_failure"

    raw_config = row.get("config", {})
    if not isinstance(raw_config, dict):
        raise ValueError(f"Scenario {scenario_id} has invalid config; expected an object")

    return {
        "scenario_id": scenario_id,
        "tags": list(dict.fromkeys(tags)),
        "pattern": pattern,
        "config": dict(raw_config),
    }


def validate_scenario_bank(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    for index, row in enumerate(rows):
        normalized_row = _normalize_scenario(row, index=index)
        scenario_id = str(normalized_row["scenario_id"])
        if scenario_id in seen_ids:
            raise ValueError(f"Duplicate scenario_id found in scenario bank: {scenario_id}")
        seen_ids.add(scenario_id)
        normalized.append(normalized_row)

    return normalized


def load_scenario_bank(path: Path) -> list[dict]:
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Scenario bank must be a JSON list: {path}")
    return validate_scenario_bank(payload)


def save_scenario_bank(path: Path, scenarios: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    normalized = validate_scenario_bank(scenarios)
    path.write_text(json.dumps(normalized, indent=2), encoding="utf-8")


def ensure_seed_bank(
    path: Path, bootstrap_cfg: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Create or reopen the Scenic seed bank for a run."""


    bootstrap_cfg = bootstrap_cfg or {}
    if path.exists():
        scenarios = load_scenario_bank(path)
        scenarios = _merge_configured_scenic_source_bank(scenarios, bootstrap_cfg)
        save_scenario_bank(path, scenarios)
        if not bool(bootstrap_cfg.get("allow_eval_routes", False)):
            leaked = [
                row.get("scenario_id", "")
                for row in scenarios
                if isinstance(row.get("config", {}).get("bench2drive_profile"), dict)
                and row["config"]["bench2drive_profile"].get("dataset_split") == "evaluation"
            ]
            if leaked:
                raise ValueError(
                    "Scenario bank contains official evaluation scenarios while evaluation-route leakage is disabled: "
                    + ", ".join(str(item) for item in leaked[:5])
                )
        return scenarios

    source = str(bootstrap_cfg.get("source", "")).strip().lower()
    if source in {"scenic", "scenic_source", "scenic_programs"}:
        scenic_dir_raw = str(
            bootstrap_cfg.get("scenic_source_dir", bootstrap_cfg.get("source_dir", ""))
        ).strip()
        if not scenic_dir_raw:
            raise ValueError(
                "scenario_bank_bootstrap.scenic_source_dir is required when source=scenic"
            )
        scenic_dir = Path(scenic_dir_raw).expanduser()
        scenarios = build_scenic_source_bank(scenic_dir)
        if not scenarios:
            raise ValueError(f"No Scenic programs found in bootstrap directory: {scenic_dir}")
        save_scenario_bank(path, scenarios)
        return scenarios

    if source:
        raise ValueError(f"Unsupported scenario-bank bootstrap source: {source!r}")

    seed = [
        {
            "scenario_id": "seed_merge_01",
            "tags": ["urban", "merge", "dense_traffic"],
            "pattern": "adversarial_merge",
            "config": {"npc_speed": 9.0, "merge_offset": 16.0, "ego_speed": 7.5},
        },
        {
            "scenario_id": "seed_ped_02",
            "tags": ["urban", "pedestrian", "occlusion"],
            "pattern": "pedestrian_intrusion",
            "config": {"ped_speed": 1.8, "ego_speed": 8.0, "spawn_gap": 12.0},
        },
        {
            "scenario_id": "seed_static_03",
            "tags": ["urban", "parked_vehicle"],
            "pattern": "late_response",
            "config": {"ego_speed": 7.0, "obstacle_dist": 19.0},
        },
    ]
    save_scenario_bank(path, seed)
    return seed


def _merge_configured_scenic_source_bank(
    scenarios: list[dict[str, Any]],
    bootstrap_cfg: dict[str, Any],
) -> list[dict[str, Any]]:
    if not bool(bootstrap_cfg.get("include_scenic_source_bank", False)):
        return scenarios
    source_dir_raw = str(bootstrap_cfg.get("scenic_source_dir", "")).strip()
    if not source_dir_raw:
        raise ValueError(
            "scenario_bank_bootstrap.scenic_source_dir is required when "
            "include_scenic_source_bank=true"
        )
    source_dir = Path(source_dir_raw).expanduser()
    return merge_scenic_source_bank(scenarios, source_dir)


def append_evolved_to_bank(bank: list[dict], evolved_payloads: list[dict]) -> list[dict]:
    existing_ids = {b.get("scenario_id") for b in bank}
    for row in evolved_payloads:
        sid = row.get("evolved_id")
        if sid in existing_ids:
            continue
        modifications = dict(row.get("modifications", {}))
        tags = modifications.get("tags", [])
        metadata = row.get("metadata", {})
        if not isinstance(metadata, dict):
            metadata = {}
        # An evolved row may inherit ``scenario_role=scenario_bank`` from a
        # native Scenic anchor.  The role must be overwritten so the current
        # round's Scenic/PDM-Lite adaptation selector cannot silently skip it.
        modifications["scenario_role"] = "evolved"
        modifications.setdefault("scenario_origin", "scenic_evolution")
        modifications.setdefault(
            "evolution_provenance",
            {
                "source_scenario_id": row.get("source_scenario_id"),
                "evolved_id": sid,
                "rationale": row.get("rationale", ""),
                "metadata": metadata,
            },
        )
        bank.append(
            {
                "scenario_id": sid,
                "tags": list(dict.fromkeys(str(x) for x in tags)),
                "pattern": modifications.get("interaction_pattern", "generic_failure"),
                "config": modifications,
            }
        )
        existing_ids.add(sid)
    return validate_scenario_bank(bank)
