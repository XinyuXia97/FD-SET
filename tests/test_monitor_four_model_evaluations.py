from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest

from scripts import monitor_four_model_evaluations as monitor


def test_monitor_alerts_on_missing_runner_and_distinguishes_ticking_from_stalled(tmp_path: Path,
                                                                                   monkeypatch) -> None:
    run = tmp_path / "official220_vad/vad"
    (run / "baseline").mkdir(parents=True)
    adapted = run / "adapted"
    adapted.mkdir()
    (run / "baseline/ckpt.json").write_text(json.dumps({
        "entry_status": "Finished", "_checkpoint": {"progress": [220, 220]}}))
    checkpoint = adapted / "ckpt.json"
    checkpoint.write_text(json.dumps({
        "entry_status": "Started", "_checkpoint": {"progress": [13, 220]}}))
    log = adapted / "evaluator.log"
    log.write_text("=== [Agent] -- Game time = 50.000\n")
    monkeypatch.setattr(monitor, "active_runners", lambda: {("vad", run.parent.resolve(), None): 4321})
    monkeypatch.setattr(monitor, "runner_started", lambda pid: time.time() - 3600)
    seen: dict[str, tuple[str, int, float]] = {}
    assert monitor.inspect(tmp_path, ["vad"], 60, 3600, seen)[0]["progress"] == [13, 220]

    now = time.time()
    os.utime(log, (now - 65, now - 65))
    monkeypatch.setattr(monitor, "runner_started", lambda pid: now)
    assert monitor.inspect(tmp_path, ["vad"], 60, 3600, seen)[0]["heartbeat_age_s"] == 0
    monkeypatch.setattr(monitor, "runner_started", lambda pid: now - 3600)
    with pytest.raises(RuntimeError, match="no evaluator activity"):
        monitor.inspect(tmp_path, ["vad"], 60, 3600, seen)

    os.utime(log, (now, now))
    os.utime(checkpoint, (now, now))  # Rewriting a checkpoint does not advance the route.
    seen["vad"] = ("adapted", 13, now - 3605)
    with pytest.raises(RuntimeError, match="no route checkpoint advancement"):
        monitor.inspect(tmp_path, ["vad"], 60, 3600, seen)

    checkpoint.write_text(json.dumps({
        "entry_status": "Started", "_checkpoint": {"progress": [14, 220]}}))
    assert monitor.inspect(tmp_path, ["vad"], 60, 3600, seen)[0]["progress"] == [14, 220]

    monkeypatch.setattr(monitor, "active_runners", lambda: {})
    with pytest.raises(RuntimeError, match="runner absent"):
        monitor.inspect(tmp_path, ["vad"], 60, 3600, seen)


def test_monitor_selects_adapted_when_baseline_is_blocked(tmp_path: Path, monkeypatch) -> None:
    model = tmp_path / "official220_tcp/tcp"
    for variant, progress in (("baseline", 145), ("adapted", 3)):
        run = model / variant
        run.mkdir(parents=True)
        (run / "ckpt.json").write_text(json.dumps({
            "entry_status": "Started", "_checkpoint": {"progress": [progress, 220]}}))
        (run / "evaluator.log").write_text("=== [Agent] -- Game time = 50.000\n")
    runners = {("tcp", model.parent.resolve(), "baseline"): 444,
               ("tcp", model.parent.resolve(), "adapted"): 333}
    monkeypatch.setattr(monitor, "runner_started", lambda pid: time.time() - 3600)
    monkeypatch.setattr(monitor, "active_runners", lambda: runners)
    status = monitor.inspect(tmp_path, ["tcp"], 60, 3600, variant_only="adapted")
    assert [(row["variant"], row["progress"], row["pid"]) for row in status] == [
        ("adapted", [3, 220], 333)]
    del runners["tcp", model.parent.resolve(), "adapted"]
    with pytest.raises(RuntimeError, match="runner absent"):
        monitor.inspect(tmp_path, ["tcp"], 60, 3600, variant_only="adapted")
