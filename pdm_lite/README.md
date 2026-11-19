# Native Scenic PDM-Lite expert and Zoo adapters

FD-SET vendors the PDM-Lite DataAgent, required BEV/speed-limit resources and the matching local Leaderboard/ScenarioRunner runtime. The native Scenic expert uses `scenic_collector.py`; **official Bench2Drive route XML is never an expert training input**. CARLA server, PythonAPI/navigation agents and the original Zoo model code/checkpoints remain external dependencies. See the [paper release guide](../README.md) for setup.

## Native Scenic collection

The paper runner passes each round's accepted evolved programs in a `scenic_execution_manifest.json` to the PDM-Lite expert. For an independent collection with a running CARLA server:

```bash
.venv/bin/python pdm_lite/scenic_collector.py \
  --manifest artifacts/<run-id>/round0/adaptation/scenario_inputs/scenic_execution_manifest.json \
  --output-dir artifacts/<run-id>/round0/adaptation/pdm_lite \
  --demos-file artifacts/<run-id>/round0/adaptation/expert_demos.jsonl \
  --camera-profile uniad --carla-port 3000 --traffic-manager-port 3050
```

`--camera-profile uniad` records six real Zoo RGB cameras and IMU, supplying both UniAD/VAD and TCP's three-view inputs. `admlp` needs measured pose, speed, acceleration, navigation and controls but cannot supply camera-based training labels. The collector saves source scenario IDs, sensor files and measured control/trajectory records. On custom OpenDRIVE maps, BEV and speed-limit resources derive from the active world; existing speed-limit maps are reused only for identical sampled road geometry. A `--dry-run` does not generate real measurements.

For full 20-scenario × 3-seed independent-bank admission, run `scripts/audit_scenic_bank.py --camera-profile uniad` followed by:

```bash
.venv/bin/python scripts/prepare_scenic_training_demos.py \
  --audit artifacts/scenic_bank_audit/audit.json \
  --output-dir artifacts/four_model_training/uniad \
  --required-camera-profile uniad
```

The gate excludes entire collided/failed runs, excludes stop-sign/red-light hazard frames from green-light-start demonstrations, and retains only contiguous 0.5-second segments of at least 11 measurements. Each scenario/seed/continuous segment has its own training episode; trajectories cannot cross runs or gaps. It checks required sensor and measurement files and consistent audit source/runtime fingerprints. Fingerprint consistency inside an audit does **not** prove its sources still match a subsequently edited checkout; rerun or preserve the exact source and runtime snapshot before publishing scores.

## Actual Zoo weight adaptation

All adapters load the **original** model implementation and previous-round compatible checkpoint. Measured ego trajectories supervise UniAD/VAD planning heads while the perception, map and occupancy heads stay frozen; they do not fabricate actor/map/occupancy labels. TCP consumes measured three-view images, trajectories, speed and control actions. AD-MLP consumes measured 5-history/6-future windows and near `command` (not distant `next_command`). An unchanged checkpoint or empty gradient update is an error.

Run from the repository root with admitted model-compatible demonstrations and the matching Zoo dependencies:

```bash
.venv/bin/python -m pdm_lite.bev_planning_adapter adapt \
  --family uniad --demos artifacts/four_model_training/uniad/expert_demos.jsonl \
  --zoo-root ../Bench2DriveZoo --baseline ../Bench2DriveZoo/ckpts/uniad_base_b2d.pth \
  --output artifacts/four_model_training/checkpoints/uniad_adapted.pth \
  --epochs 1 --batch-size 1 --learning-rate 0.00001 --device cuda
# Repeat with --family vad, the VAD baseline and VAD-compatible demos.
.venv/bin/python -m pdm_lite.tcp_adapter adapt \
  --demos artifacts/four_model_training/tcp/expert_demos.jsonl \
  --zoo-root ../Bench2DriveZoo_1 --baseline ../Bench2DriveZoo/ckpts/tcp_b2d.ckpt \
  --output artifacts/four_model_training/checkpoints/tcp_adapted.ckpt \
  --epochs 1 --batch-size 1 --learning-rate 0.00001 --device cuda
.venv/bin/python -m pdm_lite.admlp_adapter adapt \
  --demos artifacts/four_model_training/admlp/expert_demos.jsonl \
  --zoo-root ../Bench2DriveZoo_1 --baseline ../Bench2DriveZoo/ckpts/admlp_b2d.ckpt \
  --output artifacts/four_model_training/checkpoints/admlp_adapted.ckpt \
  --epochs 1 --batch-size 8 --learning-rate 0.00001 --device cpu
```

The full paper runner selects the active model and previous-round weights automatically. Standalone adapter commands validate model/dataset compatibility; their successful execution alone does not establish an official score. The separate `scripts/evaluate_four_models_official220.py` runs original/adapted checkpoints through the same unmodified 220-route XML and calculates Eff/Comf with Bench2Drive's own tool **only after complete route coverage**. A smoke route and dry-run placeholders are not paper results.

`pdm_lite/requirements.txt` records the source project's historical environment. Installing it wholesale may override this repository's pinned Scenic/PyTorch/CARLA-compatible versions; use the top-level extras and a separately verified Zoo CUDA ABI instead.
