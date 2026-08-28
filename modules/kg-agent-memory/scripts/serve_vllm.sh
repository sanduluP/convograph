#!/bin/bash
# ---------------------------------------------------------------------------
# serve_vllm.sh — Start vLLM serving Qwen3-30B-A3B-Instruct-2507-FP8 in the
# BACKGROUND from the GroupMemBench cluster venv, wait until it's ready, return.
# Timestamped logs under logs/serve_vllm/. (Adapted from the kggen-eval drill.)
#
# Run inside a GPU job (see scripts/srun_submit.sh). Blocks ~3-8 min on model
# load, then prints READY and exits 0 — vLLM keeps serving in the background.
#
# WHY this model: Instruct (reliable structured output for Graphiti extraction —
# a Thinking model's <think> traces break JSON/tool parsing), MoE 8/128 experts
# active (~3B active → high concurrency, the whole reason we left the slow API),
# FP8 e4m3 (fits one L40S/H100). Served on Ada/Hopper only (rule 8).
#
# Stop it with:  kill "$(cat logs/serve_vllm/serve_vllm.pid)"   (or exit the job)
# ---------------------------------------------------------------------------
set -euo pipefail
source "$(dirname "$0")/lib/logging.sh"

VENV="/fscratch/abuali/venvs/groupmembench"
# Model is env-overridable so we can A/B a newer checkpoint without editing code:
#   VLLM_MODEL_DIR=/fscratch/abuali/models/Qwen3.6-35B-A3B-FP8 bash scripts/serve_vllm.sh
MODEL_DIR="${VLLM_MODEL_DIR:-/fscratch/abuali/models/Qwen3-30B-A3B-Instruct-2507-FP8}"
# The id clients pass as --agent-model. Defaults to the checkpoint's folder name
# (prefixed "Qwen/") so it always matches whatever MODEL_DIR we served.
SERVED_MODEL_NAME="${VLLM_SERVED_NAME:-Qwen/$(basename "$MODEL_DIR")}"
PORT="${VLLM_PORT:-8000}"
MAX_LEN="${VLLM_MAX_LEN:-32768}"   # model supports 256k; 32k is ample for QA + extraction, saves KV cache

[ -d "$MODEL_DIR" ]     || { echo "❌ Model not found: $MODEL_DIR"; exit 1; }
[ -x "$VENV/bin/vllm" ] || { echo "❌ vllm not in venv ($VENV) — run scripts/setup_cluster_env.sh"; exit 1; }

LOG_FILE="$(log_path serve_vllm)"
# PID file is PORT-SCOPED: a self-contained ingest job runs TWO vLLM servers on
# one node (the chat model on 8000 and the embedder on 8001). A single fixed
# pid-file name would let the second server clobber the first one's PID, so the
# job could never stop them independently.
PID_FILE="$(dirname "$LOG_FILE")/serve_vllm_${PORT}.pid"

echo "🖥️  Node : $(hostname)  (IP: $(hostname --ip-address 2>/dev/null | awk '{print $1}'))"
echo "📂 Model: $MODEL_DIR"
echo "🔌 Port : $PORT   |   📏 max-model-len: $MAX_LEN"
echo "📄 Log  : $LOG_FILE"

# Launch vLLM detached. FP8 quant is auto-detected from the checkpoint's
# config.json (quantization_config), so --dtype auto is correct here.
# EXTRA_ARGS lets a caller bolt on flags (e.g. --limit-mm-per-prompt) without
# editing this script. Word-splitting is intentional here, hence no quotes below.
EXTRA_ARGS="${VLLM_EXTRA_ARGS:-}"
[ -n "$EXTRA_ARGS" ] && echo "🔧 extra args: $EXTRA_ARGS"

# shellcheck disable=SC2086  # EXTRA_ARGS must word-split into separate flags
nohup "$VENV/bin/vllm" serve "$MODEL_DIR" \
    --port "$PORT" \
    --tensor-parallel-size 1 \
    --dtype auto \
    --max-model-len "$MAX_LEN" \
    --enable-chunked-prefill \
    --served-model-name "$SERVED_MODEL_NAME" \
    $EXTRA_ARGS \
    >> "$LOG_FILE" 2>&1 &
PID=$!
echo "$PID" > "$PID_FILE"
echo "🚀 vLLM starting in background (PID $PID). Loading ~3-8 min…  (follow: tail -f $LOG_FILE)"

# Wait until the server answers, or fail fast if the process dies.
for _ in $(seq 1 90); do
    if curl -sf "http://localhost:${PORT}/v1/models" >/dev/null 2>&1; then
        echo "✅ vLLM READY → http://localhost:${PORT}/v1   (model: $SERVED_MODEL_NAME)"
        exit 0
    fi
    if ! kill -0 "$PID" 2>/dev/null; then
        echo "❌ vLLM exited during startup — last 30 log lines:"; tail -n 30 "$LOG_FILE"; exit 1
    fi
    sleep 10
done
echo "⚠️  Timed out (~15 min) waiting for vLLM. Inspect: tail -f $LOG_FILE"
exit 1
