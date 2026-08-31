#!/usr/bin/env bash
# =============================================================================
#  run_generate_image.sh — caption -> FLUX.1-schnell -> PNG -> .excalidraw
#
#  Rung 0 of the image side of Module 3. Caption is a DUMMY default for now
#  (see generate_image.py:DEFAULT_CAPTION) — pass --caption to override, and
#  later this gets called per compressed fact headline once wired to the KG.
#
#  Needs a CUDA GPU (department cluster, not this laptop) and this module's
#  own venv (see requirements.txt — torch/diffusers live ONLY here, kept out of
#  modules/kg-agent-memory's env on purpose).
#
#  Usage:
#    bash scripts/run_generate_image.sh
#    bash scripts/run_generate_image.sh --caption "Finance Ops locks go/no-go"
# =============================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${REPO_ROOT}/logs"
OUT_DIR="${REPO_ROOT}/output"
PY="${REPO_ROOT}/.venv/bin/python"
[[ -x "$PY" ]] || { echo "❌ no repo venv — see requirements.txt"; exit 1; }

mkdir -p "$LOG_DIR" "$OUT_DIR"
STAMP="$(date +%Y%m%d_%H%M%S)"
IMAGE_OUT="${OUT_DIR}/image_${STAMP}.png"
SCENE_OUT="${OUT_DIR}/image_${STAMP}.excalidraw"

{
  echo "🖼️  generating image — $(date '+%Y-%m-%d %H:%M:%S')"
  GEN_OUT="$(PYTHONUNBUFFERED=1 "$PY" -u "${REPO_ROOT}/generate_image.py" --out "$IMAGE_OUT" "$@")"
  echo "$GEN_OUT"
  CAPTION="$(echo "$GEN_OUT" | sed -n 's/^caption : //p')"

  PYTHONUNBUFFERED=1 "$PY" -u "${REPO_ROOT}/image_to_excalidraw.py" \
    --image "$IMAGE_OUT" --caption "$CAPTION" --out "$SCENE_OUT"

  echo "✅ image : $IMAGE_OUT"
  echo "✅ scene : $SCENE_OUT  (open in Excalidraw or the VS Code extension)"
} 2>&1 | tee "${LOG_DIR}/generate_image_${STAMP}.log"
