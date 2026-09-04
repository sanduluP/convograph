#!/usr/bin/env bash
# =============================================================================
#  run_generate_image.sh — caption(s) -> FLUX.1-schnell -> PNG(s)
#
#  TWO MODES:
#    bash scripts/run_generate_image.sh --caption "Finance Ops locks go/no-go"
#        ONE image, plus a per-image .excalidraw scene. Original behaviour.
#
#    bash scripts/run_generate_image.sh --captions-file captions.txt
#        MANY images, ONE model load. Writes output/batch_<stamp>/ containing
#        image_000.png ... and manifest.jsonl (see generate_image.py's docstring
#        for the manifest's fields).
#
#  Batch mode does NOT emit per-image .excalidraw scenes: its caller composes
#  every image onto ONE canvas (ui/compose_board.py), so a scene per image would
#  be written and immediately thrown away.
#
#  Needs a CUDA GPU (department cluster, not this laptop) and this module's own
#  venv (see requirements.txt — torch/diffusers live ONLY here, kept out of
#  modules/kg-agent-memory's env on purpose). --dry-run needs neither.
# =============================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${REPO_ROOT}/logs"
OUT_DIR="${REPO_ROOT}/output"
PY="${REPO_ROOT}/.venv/bin/python"
[[ -x "$PY" ]] || { echo "❌ no repo venv — see requirements.txt"; exit 1; }

# ── Where the 31 GB of FLUX weights live ─────────────────────────────────────
# NEVER inside the repo, and NEVER in $HOME. Three reasons, all of which bit us:
#
#   1. scripts/sync_to_cluster.sh rsyncs with --delete and excludes only
#      .gitignore entries, .git/ and .venv/. A cache inside REPO_ROOT would be
#      DELETED by the next sync and silently re-downloaded — 31 GB, every time.
#   2. On unicorn $HOME is ~95% full (~93 GB free), and HuggingFace defaults to
#      ~/.cache/huggingface — the weights would land on the nearly-full root
#      filesystem instead of the 14 TB /scratch.
#   3. It is SHARED. Faris, Rahul and Priyabanta are all in group `dsa`, so one
#      copy serves all three rather than three copies of 32 GB. The directory is
#      setgid (2775) so anything created inside inherits the group; without that
#      bit, a file one of us writes is unreadable to the other two.
#
# (/scratch/huggingface is the machine-wide cache but is root-owned and not
# writable by us, and we cannot create at the /scratch top level either.)
#
# Preference order: an explicit HF_HOME wins; then the shared dsa cache; then a
# sibling of the repo, which works on any other host (Pegasus is a separate
# filesystem island and has no /scratch/faris).
DSA_SHARED_MODELS="/scratch/faris/models/huggingface"
if [[ -n "${HF_HOME:-}" ]]; then
  :                                          # caller decided; respect it
elif [[ -d "$DSA_SHARED_MODELS" ]]; then
  export HF_HOME="$DSA_SHARED_MODELS"
else
  export HF_HOME="$(dirname "$REPO_ROOT")/hf-cache"
fi
mkdir -p "$HF_HOME"

# The HF token is deliberately NOT in HF_HOME: that directory is group-readable,
# and a credential does not belong in a shared tree. It stays in the private
# ~/.cache/huggingface/token, which the huggingface_hub library also checks.

mkdir -p "$LOG_DIR" "$OUT_DIR"
STAMP="$(date +%Y%m%d_%H%M%S)"

# Decide the mode by scanning our own arguments, so the caller uses one script
# either way and nothing downstream needs to know which path it took.
BATCH_DIR=""
for arg in "$@"; do
  if [[ "$arg" == "--captions-file" ]]; then
    BATCH_DIR="${OUT_DIR}/batch_${STAMP}"
    break
  fi
done

if [[ -n "$BATCH_DIR" ]]; then
  # ── batch: one model load, many images ────────────────────────────────────
  {
    echo "🖼️  batch generation — $(date '+%Y-%m-%d %H:%M:%S')"
    echo "   out: ${BATCH_DIR}"
    PYTHONUNBUFFERED=1 "$PY" -u "${REPO_ROOT}/generate_image.py" \
      --out-dir "$BATCH_DIR" "$@"
    # The caller parses this exact line to locate the manifest.
    echo "✅ batch-dir : ${BATCH_DIR}"
  } 2>&1 | tee "${LOG_DIR}/generate_image_${STAMP}.log"
else
  # ── single: unchanged, still emits a per-image scene ──────────────────────
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
fi
