from __future__ import annotations

import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from core.closed_loop_runner import build_summary, build_agent_config_override, checkpoint_from_agent_config
from scripts import run_paper_suite


PAPER_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "paper_reproduction.json"


def _synthetic_config(tmp_path: Path) -> Path:
    config = json.loads(PAPER_CONFIG.read_text(encoding="utf-8"))
    config["output_root"] = str(tmp_path / "suite")
    config["num_rounds"] = 1
    routes_xml = tmp_path / "benchmark_routes.xml"
    routes_xml.write_text(
        '<routes><route id="0" town="Town01"><waypoints>'
        '<position x="0" y="0" z="0" pitch="0" roll="0" yaw="0"/>'
        '<position x="30" y="0" z="0" pitch="0" roll="0" yaw="0"/>'
        '</waypoints><scenarios><scenario name="HighwayCutIn_1" type="HighwayCutIn">'
        '<trigger_point x="0" y="0" z="0" yaw="0"/>'
        '<flow_speed value="8"/></scenario></scenarios></route></routes>',
        encoding="utf-8",
    )
    config["evaluation"]["held_out_routes_xml"] = str(routes_xml)
    config_path = tmp_path / "paper_config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    return config_path


def test_paper_default_dry_run_retests_same_routes_after_evolved_scenic_adaptation(tmp_path, monkeypatch):
    config_path = _synthetic_config(tmp_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["output_root"] = "suite"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    suite = tmp_path / "suite"
    monkeypatch.setattr(
        sys, "argv", ["run_paper_suite.py", "--config", str(config_path), "--dry-run", "--stop-on-error"]
    )

    run_paper_suite.main()

    manifest = json.loads((suite / "suite_manifest.json").read_text(encoding="utf-8"))
    assert len(manifest["runs"]) == 4
    assert manifest["protocol"] == "downstream_adaptation"
    assert manifest["models"] == ["uniad", "vad", "admlp", "tcp"]
    assert all(row["status"] == "completed" for row in manifest["runs"])
    record = next(row for row in manifest["runs"] if row["model"] == "admlp")
    assert record["status"] == "completed"
    assert record["synthetic"] is True
    assert record["measured"] is False
    assert record["metrics"] is None
    assert record["official_complete"] is False

    round_dir = suite / record["run_id"] / "round0"
    assert (suite / record["run_id"] / "scenario_bank.json").is_file()
    result = json.loads((round_dir / "round_summary.json").read_text(encoding="utf-8"))
    assert result["evaluation_synthetic"] is True
    assert result["adaptation_executed"] is True
    assert result["num_evolved_accepted"] > 0
    assert result["post_adaptation_metrics"] is not None
    assert result["agent_config_override"] != str(
        json.loads(config_path.read_text(encoding="utf-8"))["bench2drive"]["agent_config"]
    )
    assert result["metric_delta_after_adaptation"]
    assert (round_dir / "benchmark_evaluation" / "metrics.json").is_file()
    assert (round_dir / "benchmark_re_evaluation" / "metrics.json").is_file()
    source_routes = ET.parse(tmp_path / "benchmark_routes.xml").getroot()
    evaluated_routes = ET.parse(result["scenario_file"]).getroot()
    assert [(node.tag, node.attrib) for node in source_routes.iter()] == [
        (node.tag, node.attrib) for node in evaluated_routes.iter()
    ]
    baseline_contract = json.loads(
        (round_dir / "benchmark_evaluation" / "evaluation_contract.json").read_text(encoding="utf-8")
    )
    retest_contract = json.loads(
        (round_dir / "benchmark_re_evaluation" / "evaluation_contract.json").read_text(encoding="utf-8")
    )
    assert baseline_contract["stage"] == "baseline"
    assert retest_contract["stage"] == "post_adaptation"
    assert baseline_contract["scenario_file"] == retest_contract["scenario_file"] == result["scenario_file"]

    adaptation = json.loads((round_dir / "adaptation" / "adaptation_manifest.json").read_text(encoding="utf-8"))
    assert adaptation["adaptation_source"] == "scenic_evolved_scenarios_only"
    assert adaptation["benchmark_role"] == "feedback_and_re_evaluation_only"
    assert adaptation["num_scenarios"] > 0
    assert adaptation["num_scenarios"] <= result["num_evolved_accepted"]
    assert adaptation["expert_generation"]["status"] == "dry_run_proxy"
    evolved = json.loads((round_dir / "evolved_scenarios.json").read_text(encoding="utf-8"))
    assert set(adaptation["scenario_ids"]) <= {row["evolved_id"] for row in evolved}
    selected = json.loads((round_dir / "adaptation" / "adaptation_scenarios.json").read_text(encoding="utf-8"))
    assert {row["scenario_id"] for row in selected} == set(adaptation["scenario_ids"])
    assert all(row["config"]["scenario_role"] == "evolved" for row in selected)


def test_next_round_trains_from_previous_round_checkpoint(tmp_path, monkeypatch):
    config_path = _synthetic_config(tmp_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["num_rounds"] = 2
    config_path.write_text(json.dumps(config), encoding="utf-8")
    monkeypatch.setattr(
        sys, "argv", ["run_paper_suite.py", "--config", str(config_path), "--dry-run",
                     "--model", "admlp", "--stop-on-error"]
    )
    run_paper_suite.main()

    run_root = tmp_path / "suite" / "downstream_adaptation_admlp"
    first = json.loads((run_root / "round0" / "round_summary.json").read_text(encoding="utf-8"))
    second = json.loads((run_root / "round1" / "round_summary.json").read_text(encoding="utf-8"))
    placeholder = json.loads(Path(second["agent_config_override"]).read_text(encoding="utf-8"))
    assert placeholder["previous_checkpoint"] == first["agent_config_override"]
    assert second["scenario_bank_size_before"] == first["scenario_bank_size_after"]
    assert first["post_adaptation_metrics"] is not None
    assert second["post_adaptation_metrics"] is not None


@pytest.mark.parametrize("family", ("uniad", "vad"))
def test_zoo_next_round_training_reads_checkpoint_not_agent_config_pair(family):
    config = json.loads(PAPER_CONFIG.read_text(encoding="utf-8"))
    config["bench2drive"].update(config["model_profiles"][family])
    override = build_agent_config_override(config, "/tmp/zoo-round0.pth")
    assert override is not None and override.endswith("+/tmp/zoo-round0.pth")
    assert checkpoint_from_agent_config(config, override) == "/tmp/zoo-round0.pth"
    with pytest.raises(ValueError, match="active model"):
        checkpoint_from_agent_config(config, "other_config.py+/tmp/zoo-round0.pth")


def test_live_model_without_training_command_rejected_before_creating_artifacts(tmp_path, monkeypatch):
    output_root = tmp_path / "unsupported"
    config = json.loads(PAPER_CONFIG.read_text(encoding="utf-8"))
    config["adaptation"]["models"]["tcp"]["train_command"] = ""
    config_path = tmp_path / "missing_train_command.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    monkeypatch.setattr(
        sys, "argv", ["run_paper_suite.py", "--config", str(config_path),
                     "--output-root", str(output_root), "--model", "tcp"]
    )
    with pytest.raises(ValueError, match=r"train_command.*tcp"):
        run_paper_suite.main()
    assert not output_root.exists()


def test_post_adaptation_missing_route_disables_improvement_even_if_baseline_is_complete():
    rows = [{
        "round": 0,
        "baseline_metrics": {"DS": 45.0},
        "metrics": {"DS": 52.0},
        "benchmark_coverage": {"expected": 220, "measured_count": 220, "skipped_count": 0},
        "post_adaptation_coverage": {"expected": 220, "measured_count": 219, "skipped_count": 1},
    }]
    assert build_summary("full", ["DS"], rows)["improvement"] is None
