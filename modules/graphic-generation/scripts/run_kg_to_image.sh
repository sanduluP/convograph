#!/usr/bin/env bash
# =============================================================================
#  run_kg_to_image.sh — TemporalKG -> caption (local LLM) -> FLUX -> .excalidraw
#
#  Test 2 of the image side of Module 3. Test 1 (generate_image.py's
#  DEFAULT_CAPTION / a hand-typed --caption) is untouched by this — this
#  script only adds a KG-grounded caption step in front of the SAME
#  caption -> FLUX -> Excalidraw pipeline (scripts/run_on_unicorn.sh),
#  reused unchanged.
#
#  Caption generation runs HERE, on the laptop, against local ollama
#  (qwen2.5:3b-instruct — see kg_to_caption.py for why not a reasoning model).
#  Image generation still runs on unicorn, which is the only thing with a
#  CUDA GPU.
#
#  Usage:
#    bash scripts/run_kg_to_image.sh                        # samples/sample_kg.json
#    bash scripts/run_kg_to_image.sh --kg path/to/real_kg.json
# =============================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${REPO_ROOT}/logs"
OUT_DIR="${REPO_ROOT}/output"
mkdir -p "$LOG_DIR" "$OUT_DIR"

STAMP="$(date +%Y%m%d_%H%M%S)"
REQUEST_OUT="${OUT_DIR}/kg_request_${STAMP}.json"

{
  echo "🧠 KG -> caption (local ollama, qwen2.5:3b-instruct) — $(date '+%Y-%m-%d %H:%M:%S')"
  CAP_OUT="$(python3 "${REPO_ROOT}/kg_to_caption.py" --out "$REQUEST_OUT" "$@")"
  echo "$CAP_OUT"
  CAPTION="$(echo "$CAP_OUT" | sed -n 's/^caption : //p')"
  echo "✅ request : $REQUEST_OUT"

  echo "🚀 caption -> FLUX -> Excalidraw on unicorn (reusing run_on_unicorn.sh)"
  bash "${REPO_ROOT}/scripts/run_on_unicorn.sh" --caption "$CAPTION"
} 2>&1 | tee "${LOG_DIR}/kg_to_image_${STAMP}.log"
