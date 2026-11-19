#!/usr/bin/env bash
set -euo pipefail

BENCH2DRIVE_ZOO_ROOT="${BENCH2DRIVE_ZOO_ROOT:-../Bench2DriveZoo}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu118}"

if ! command -v uv >/dev/null 2>&1; then
  echo "[setup] uv is required but not found in PATH." >&2
  exit 1
fi

if [ ! -d "$BENCH2DRIVE_ZOO_ROOT" ]; then
  echo "[setup] Bench2DriveZoo root not found: $BENCH2DRIVE_ZOO_ROOT" >&2
  echo "[setup] Export BENCH2DRIVE_ZOO_ROOT=/abs/path/to/Bench2DriveZoo and retry." >&2
  exit 1
fi

cat <<EOF
[setup] UniAD environment bootstrap via uv
[setup] Bench2DriveZoo root: $BENCH2DRIVE_ZOO_ROOT
[setup] Torch index url: $TORCH_INDEX_URL

[setup] System prerequisites still required outside uv:
  - Python 3.8
  - CUDA 11.8 toolkit
  - GCC 9.4 recommended
  - CUDA_HOME pointing at your CUDA install
EOF

uv sync --extra dev --extra paper
uv pip install --index-url "$TORCH_INDEX_URL" torch torchvision torchaudio
uv pip install ninja packaging
uv pip install --no-build-isolation -e "$BENCH2DRIVE_ZOO_ROOT"
uv pip install "Pillow<10"

cat <<EOF
[setup] UniAD Python dependencies installed into the current uv environment.
[setup] Next:
  1. Ensure BENCH2DRIVE_ROOT points to your external Bench2Drive checkout if needed.
  2. Fill bench2drive.agent_config with:
     <uniad_config.py>+<uniad_checkpoint.pth>
  3. Run:
     uv run python scripts/run_paper_suite.py --config configs/paper_reproduction.json
EOF
