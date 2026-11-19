"""Load an unchanged Bench2DriveZoo agent with locally rebuilt CUDA extensions.

The Zoo checkout includes binaries compiled for a different PyTorch ABI.  The
matching binaries live in this project's artifacts; only the three extension
modules are replaced, not the agent, model architecture, or checkpoint.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from pdm_lite.zoo_runtime import prepare_zoo_runtime

prepare_zoo_runtime(Path(os.environ["BENCH2DRIVE_ZOO_ROOT"]))


_agent_name = os.environ.get("FDSET_ZOO_AGENT_NAME", "")
if _agent_name not in {"uniad", "vad"}:
    raise ValueError("FDSET_ZOO_AGENT_NAME must be uniad or vad")
_zoo_root = Path(os.environ["BENCH2DRIVE_ZOO_ROOT"])
_agent_file = _zoo_root / "team_code" / f"{_agent_name}_b2d_agent.py"
_spec = importlib.util.spec_from_file_location(f"fdset_original_{_agent_name}_agent", _agent_file)
if _spec is None or _spec.loader is None:
    raise ImportError(f"Cannot load the original Bench2DriveZoo agent: {_agent_file}")
_agent_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_agent_module)
OriginalAgent = getattr(_agent_module, _agent_module.get_entry_point())


class ZooCudaAgent(OriginalAgent):
    """Original Zoo agent with rebuilt extensions and Zoo-relative setup paths."""

    def setup(self, path_to_conf_file: str) -> None:
        config_path, checkpoint_path, save_name = path_to_conf_file.split("+", 2)
        config_path = Path(config_path).resolve()
        checkpoint_path = Path(checkpoint_path).resolve()
        # The original agents resolve plugins and UniAD motion anchors relative
        # to the Zoo checkout's parent, not the evaluator's Bench2Drive cwd.
        evaluator_cwd = Path.cwd()
        try:
            os.chdir(_zoo_root.resolve().parent)
            # Preserve the evaluator's per-route name below SAVE_PATH instead
            # of treating the absolute checkpoint path as a directory.
            super().setup(f"{config_path}+{checkpoint_path}+{save_name}")
        finally:
            os.chdir(evaluator_cwd)

    def save(self, tick_data) -> None:
        if os.environ.get("FDSET_METRICS_ONLY") != "1":
            super().save(tick_data)
            return
        # Keep the stock per-tick motion evidence without encoding seven debug
        # PNGs or dumping controller metadata; inference and controls are unchanged.
        with (self.save_path / "metric_info.json").open("w", encoding="utf-8") as stream:
            json.dump(self.metric_info, stream, indent=4)



def get_entry_point() -> str:
    return "ZooCudaAgent"
