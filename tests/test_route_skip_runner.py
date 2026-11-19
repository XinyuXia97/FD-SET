from __future__ import annotations

import json
import os
import signal
import socket
import sys
import time
from pathlib import Path

import pytest

from core import route_skip_runner
from core.bench2drive_adapter import parse_metrics
from core.partial_benchmark_metrics import build_partial_checkpoint


def _record(route_id: str, status: str, composed: float) -> dict:
    return {
        "route_id": f"RouteScenario_{route_id}_rep0",
        "save_name": f"RouteScenario_{route_id}_rep0_test",
        "status": status,
        "scores": {"score_composed": composed, "score_route": composed, "score_penalty": 1.0},
        "infractions": {"collisions_vehicle": [], "collisions_pedestrian": [], "collisions_layout": []},
    }


def test_simulator_death_skips_one_route_then_scores_following_routes_and_resumes(tmp_path, monkeypatch):
    route_ids = ("11", "12", "13", "14")
    xml = tmp_path / "routes.xml"
    xml.write_text("<routes>" + "".join(f'<route id="{route_id}" town="Town12" />' for route_id in route_ids) + "</routes>")
    output = tmp_path / "evaluation"
    output.mkdir()
    original = output / "ckpt.json"
    original.write_text(json.dumps({"entry_status": "Started", "_checkpoint": {
        "progress": [1, 4], "records": [_record("11", "Failed - TickRuntime", 37.0)]
    }}))
    original_bytes = original.read_bytes()
    server_pids = tmp_path / "servers.txt"
    launcher = tmp_path / "fake_carla"
    launcher.write_text(
        "#!/usr/bin/env python3\n"
        "import os, socket, sys\n"
        "port = int(next(a.split('=', 1)[1] for a in sys.argv if a.startswith('-carla-rpc-port=')))\n"
        "sock = socket.socket()\n"
        "sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)\n"
        "sock.bind(('127.0.0.1', port))\n"
        "sock.listen()\n"
        "with open(os.environ['TEST_SERVER_PIDS'], 'a') as log: log.write(str(os.getpid()) + '\\n')\n"
        "while True:\n"
        "    connection, _ = sock.accept()\n"
        "    connection.close()\n"
    )
    launcher.chmod(0o755)
    evaluator = tmp_path / "fake_evaluator.py"
    evaluator.write_text(
        "import argparse, json, os, signal, sys\n"
        "p = argparse.ArgumentParser()\n"
        "p.add_argument('--checkpoint', required=True)\n"
        "p.add_argument('--routes-subset', required=True)\n"
        "a, _ = p.parse_known_args()\n"
        "id = a.routes_subset\n"
        "status = 'Failed - Simulation crashed' if id == '12' else ('Failed - TickRuntime' if id == '14' else 'Completed')\n"
        "if id == '12':\n"
        "    with open(os.environ['TEST_SERVER_PIDS']) as log: pid = int(log.readlines()[-1])\n"
        "    os.kill(pid, signal.SIGTERM)\n"
        "score = 0.0 if id == '12' else (71.0 if id == '13' else 18.0)\n"
        "record = {'route_id': f'RouteScenario_{id}_rep0', 'save_name': f'RouteScenario_{id}_rep0_test', "
        "'status': status, 'scores': {'score_composed': score, 'score_route': score, 'score_penalty': 1.0}, "
        "'infractions': {'collisions_vehicle': [], 'collisions_pedestrian': [], 'collisions_layout': []}}\n"
        "with open(a.checkpoint, 'w') as out: json.dump({'entry_status': 'Crashed' if id == '12' else 'Finished', "
        "'_checkpoint': {'progress': [1, 1], 'records': [record]}}, out)\n"
        "sys.exit(1 if id == '12' else 0)\n"
    )
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]

    def probe_server(host, rpc_port, _env, _cwd):
        try:
            with socket.create_connection((host, rpc_port), timeout=0.1):
                return "healthy"
        except OSError:
            return "unbound"

    monkeypatch.setattr(route_skip_runner, "_probe_server", probe_server)
    command = f'{sys.executable} {evaluator} --routes {xml} --checkpoint {original} --resume True'
    env = {"TEST_SERVER_PIDS": str(server_pids)}
    try:
        result = route_skip_runner.run_remaining_routes(
            command, bench_root=tmp_path, scenario_file=xml, output_dir=output,
            environment=env, host="127.0.0.1", port=port,
            server_launcher=launcher, server_gpu_adapter=5, route_timeout_s=15,
        )
        assert [item["index"] for item in result["records"]] == [0, 2, 3]
        assert [item["index"] for item in result["skipped"]] == [1]
        assert original.read_bytes() == original_bytes
        assert len(server_pids.read_text().splitlines()) == 2
        combined = build_partial_checkpoint(result["records"], result["skipped"], 4, output / "partial_ckpt.json")
        assert combined["entry_status"] == "Partial"
        assert combined["audit"]["measured_count"] == 3
        assert parse_metrics(output / "partial_ckpt.json", keys=["DS"], strict=True)["DS"] == 42.0
        assert route_skip_runner.run_remaining_routes(
            command, bench_root=tmp_path, scenario_file=xml, output_dir=output,
            environment=env, host="127.0.0.1", port=port,
            server_launcher=launcher, server_gpu_adapter=5, route_timeout_s=15,
        )["skipped"] == result["skipped"]
        assert len(server_pids.read_text().splitlines()) == 2
    finally:
        if server_pids.exists():
            for raw_pid in server_pids.read_text().splitlines():
                try:
                    os.killpg(int(raw_pid), signal.SIGTERM)
                except ProcessLookupError:
                    pass


def test_evaluator_stops_waiting_after_carla_port_remains_unbound(tmp_path, monkeypatch):
    monkeypatch.setattr(route_skip_runner, "_SERVER_LOSS_GRACE_S", 1)
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    start = time.monotonic()
    timed_out, returncode = route_skip_runner._run_evaluator(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        tmp_path, os.environ.copy(), tmp_path / "evaluator.log", 15,
        tmp_path, "127.0.0.1", port,
    )
    assert not timed_out
    assert returncode == -signal.SIGTERM
    assert time.monotonic() - start < 8


def test_partial_checkpoint_rejects_unscored_crash_and_unexplained_gap(tmp_path):
    with pytest.raises(ValueError, match="valid scored result"):
        build_partial_checkpoint(
            [{"index": 0, "record": _record("11", "Failed - Simulation crashed", 0.0)}],
            [], 1, tmp_path / "bad.json",
        )
    with pytest.raises(ValueError, match="neither measured nor skipped"):
        build_partial_checkpoint(
            [{"index": 0, "record": _record("11", "Completed", 80.0)}],
            [], 2, tmp_path / "gap.json",
        )
