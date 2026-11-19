from __future__ import annotations

from pathlib import Path

from reports.make_table import render_table, resolve_run_dir


def test_render_table_uses_dynamic_metric_keys() -> None:
    summary = {
        "metrics_keys": ["DS", "SR", "CR", "Comf", "L2m"],
        "rounds": [
            {
                "round": 0,
                "metrics": {"DS": 34.0, "SR": 12.0, "CR": 8.0, "Comf": 70.0, "L2m": 0.0},
            },
            {
                "round": 1,
                "metrics": {"DS": 35.3, "SR": 12.9, "CR": 7.8, "Comf": 70.0, "L2m": 0.0},
            },
        ],
        "improvement": {"DS": 1.3, "SR": 0.9, "CR": -0.2, "Comf": 0.0, "L2m": 0.0},
    }

    rendered = render_table(summary)

    assert "| Round | DS | SR | CR | Comf | L2m |" in rendered
    assert "- L2m: +0.00" in rendered


def test_resolve_run_dir_accepts_run_name(tmp_path: Path, monkeypatch) -> None:
    artifacts_root = tmp_path / "artifacts"
    run_dir = artifacts_root / "sample_run"
    run_dir.mkdir(parents=True)
    monkeypatch.chdir(tmp_path)

    resolved = resolve_run_dir(Path("artifacts"), Path("sample_run"))

    assert resolved == run_dir
