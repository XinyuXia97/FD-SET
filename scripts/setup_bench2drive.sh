#!/usr/bin/env bash
set -euo pipefail

TARGET_DIR="${1:-../Bench2Drive}"
DEFAULT_REPO_URL="https://github.com/Thinklab-SJTU/Bench2Drive.git"
REPO_URL="${BENCH2DRIVE_GIT_URL:-$DEFAULT_REPO_URL}"

if [ -d "$TARGET_DIR/.git" ]; then
  echo "[setup] Bench2Drive already exists at $TARGET_DIR"
  exit 0
fi

mkdir -p "$(dirname "$TARGET_DIR")"

echo "[setup] Bench2Drive canonical repo: $DEFAULT_REPO_URL"
echo "[setup] Cloning Bench2Drive from: $REPO_URL"
if git clone "$REPO_URL" "$TARGET_DIR"; then
  echo "[setup] Done: $TARGET_DIR"
else
  echo "[setup] Clone failed."
  echo "[setup] You can set BENCH2DRIVE_GIT_URL to an internal mirror and retry:"
  echo "         BENCH2DRIVE_GIT_URL=<mirror_url> bash scripts/setup_bench2drive.sh"
  exit 1
fi
