from __future__ import annotations

from pathlib import Path
from core.scenario_bank import ensure_seed_bank


def test_ensure_seed_bank_bootstraps_from_scenic_source_idempotently(tmp_path: Path) -> None:
    scenic_dir = tmp_path / "scenic"
    scenic_dir.mkdir()
    (scenic_dir / "carlaChallenge1.scenic").write_text(
        "param map = localPath('../maps/Town.xodr')\n"
        "model scenic.domains.driving.model\n"
        "ego = Car at spawn\n",
        encoding="utf-8",
    )
    cfg = {
        "source": "scenic",
        "scenic_source_dir": str(scenic_dir),
    }
    bank_path = tmp_path / "bootstrap" / "seed_bank.json"
    bank = ensure_seed_bank(bank_path, cfg)
    assert [row["scenario_id"] for row in bank] == ["scenic_carlaChallenge1"]
    assert ensure_seed_bank(bank_path, cfg) == bank
