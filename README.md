# FD-SET

Failure-Driven Scenario Evolution for autonomous-driving safety evaluation. FD-SET connects failure collection, vision–language interpretation, structured scenario retrieval, Scenic-based evolution, and closed-loop evaluation. It supports UniAD, VAD, AD-MLP, and TCP with Bench2Drive and CARLA.

## Workflow

1. Evaluate a driving policy on the Bench2Drive 220-route benchmark and collect failure records, camera evidence, and driving metrics.
2. Interpret each failure with Qwen3-VL. The model receives selected event-centered camera frames and structured event context, and returns semantic fields in JSON.
3. Convert the semantic fields into an interaction pattern, agent roles, and scenario constraints. Retrieve the top matching Scenic scenarios using frozen CLIP text embeddings.
4. Reconstruct retrieved native Scenic programs and adjust failure-sensitive parameters within bounded ranges. Add accepted variants to the scenario bank.
5. In the `downstream_adaptation` protocol, collect expert demonstrations on evolved Scenic scenarios with PDM-Lite, update the driving model from its previous checkpoint, and evaluate it again on the same Bench2Drive routes. Repeat for five rounds.

The `fixed_policy_evolution` protocol evaluates evolving native Scenic scenarios with fixed driving-policy weights. The official benchmark routes, Scenic scenario bank, and expert demonstration scenarios remain separate inputs.

## Components

| Path | Function |
|---|---|
| `core/failure_extract.py` | Extract failure events and evidence from evaluation checkpoints. |
| `core/failure_interpret.py`, `core/interpretation_backend.py`, `prompts/vlm_prompt.txt` | Generate and normalize structured failure semantics with Qwen3-VL. |
| `core/failure_struct.py` | Build the scenario-level failure representation. |
| `core/scenario_retrieve.py`, `core/retrieval_encoder.py` | Encode structured descriptions and retrieve Scenic anchors with CLIP. |
| `core/scenario_evolve.py`, `core/scenario_program.py` | Reconstruct source-backed Scenic programs and adjust scenario parameters. |
| `pdm_lite/` | Execute Scenic scenarios, collect expert demonstrations, and adapt the four driving-model families. |
| `core/closed_loop_runner.py`, `scripts/run_paper_suite.py` | Run the multi-round evaluation and adaptation protocols. |
| `reports/` | Summarize metrics, failures, and scenario-bank evolution. |

`scenic/` contains the native Scenic scenario sources; `maps/` contains their OpenDRIVE maps. `configs/paper_reproduction.json` defines the paper protocols, model profiles, and runtime paths.

## Requirements

- Python 3.8, Scenic 2, CARLA 0.9.15 and its compatible PythonAPI.
- Bench2Drive, its official 220-route XML, Bench2DriveZoo agents and baseline checkpoints.
- Local Ollama with `qwen3-vl:8b-instruct`; local CLIP `openai/clip-vit-base-patch32` weights in `clip/`.
- PyTorch/CUDA and the Zoo CUDA extensions compatible with the selected models.

The default configuration expects sibling directories `../Bench2Drive`, `../Bench2DriveZoo`, `../Bench2DriveZoo_1`, and `../carla`. Update `configs/paper_reproduction.json` when using other locations.

```bash
uv venv --python /usr/bin/python3.8 --seed .venv
uv pip install --python .venv/bin/python --index-url https://download.pytorch.org/whl/cu118 \
  'torch==2.0.1+cu118' 'torchvision==0.15.2+cu118'
uv pip install --python .venv/bin/python -e '.[dev,paper,expert,admlp]'
uv pip install --python .venv/bin/python 'carla==0.9.15'
```

Download the frozen CLIP revision configured for the paper protocol:

```bash
.venv/bin/python -c 'from huggingface_hub import snapshot_download; snapshot_download(repo_id="openai/clip-vit-base-patch32", revision="3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268", local_dir="clip", allow_patterns=["config.json", "preprocessor_config.json", "tokenizer_config.json", "vocab.json", "merges.txt", "special_tokens_map.json", "pytorch_model.bin"])'
```

## Run

Start CARLA on the port specified in `configs/paper_reproduction.json` and make `qwen3-vl:8b-instruct` available through Ollama. Run from the repository root:

```bash
.venv/bin/python scripts/run_paper_suite.py --config configs/paper_reproduction.json \
  --output-root artifacts/paper_reproduction
```

To run one model, add `--model uniad`, `--model vad`, `--model admlp`, or `--model tcp`. Resume an interrupted suite with `--resume-existing`. For fixed-policy experiments, use `--protocol fixed_policy_evolution` and a strategy such as `--strategy full`, `--strategy retrieval`, `--strategy recon_escalation`, `--strategy random`, or `--strategy static`.

For a separate baseline/adapted comparison on official routes:

```bash
.venv/bin/python scripts/evaluate_four_models_official220.py \
  --bench-root ../Bench2Drive --carla-launcher ../carla/CarlaUE4.sh \
  --checkpoints artifacts/four_model_training/checkpoints \
  --output artifacts/four_model_evaluation/official220 --gpu 0 --graphics-adapter 0
```

## Outputs

Each run writes to `<output-root>/<run-id>/`. A round includes `benchmark_evaluation/`, `failures_raw.json`, `failures_semantics.json`, `failures_structured.json`, `retrieval_trace.json`, `evolved_scenarios.json`, `scenario_bank_post_round.json`, `adaptation/`, `benchmark_re_evaluation/`, and `round_summary.json` as applicable to the selected protocol. The run root contains `run_manifest.json` and `summary.json`.

## Tests

```bash
.venv/bin/python -m pytest -q
```
