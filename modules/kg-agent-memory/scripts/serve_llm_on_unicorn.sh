#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════════
#  serve_llm_on_unicorn.sh — run the KG-extraction LLM on unicorn's H100
#  instead of the laptop's CPU. Driven from the laptop over ssh; nothing needs
#  to be checked out on unicorn.
#
#  WHY
#  ---
#  ui_ingest.py defaults to a LOCAL ollama. This laptop has NO NVIDIA GPU, so
#  Graphiti's ~10-20 extraction calls per episode run on CPU: measured
#  2026-09-06, ONE 3-line transcript took ~10 minutes. That is what makes the
#  Streamlit demo feel broken. The same work on an H100 is seconds.
#
#  WHY OUR OWN OLLAMA AND NOT THE ONE ALREADY RUNNING THERE
#  --------------------------------------------------------
#  unicorn has a shared ollama on :11434 holding ~118 GB of other people's
#  models, and its store sits on the root filesystem, which is 96% full. Adding
#  ours there would be antisocial and risky. So we run a SECOND instance on a
#  different port with its models on /scratch, per the storage rule (code in
#  $HOME, weights on scratch) - and in the same group-shared `dsa` tree as the
#  FLUX weights, so Rahul and Priyabanta get one copy rather than three.
#
#  MODEL CHOICE
#  ------------
#  qwen3:4b-INSTRUCT, deliberately not qwen3:4b. The plain tag is the hybrid
#  build with thinking on: it narrates its deliberation and exhausts the token
#  budget, returning an empty string on short-output prompts (see this module's
#  CLAUDE.md). The instruct build answers directly.
#
#  Usage:
#      bash scripts/serve_llm_on_unicorn.sh start     # start + pull + verify
#      bash scripts/serve_llm_on_unicorn.sh status
#      bash scripts/serve_llm_on_unicorn.sh stop
#      bash scripts/serve_llm_on_unicorn.sh tunnel    # print the tunnel command
# ═══════════════════════════════════════════════════════════════════════════════
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${HERE}/logs"; mkdir -p "$LOG_DIR"
LOG="${LOG_DIR}/serve_llm_on_unicorn.log"

HOST="${UNICORN_HOST:-unicorn}"
# 11434 is the SHARED instance. Ours gets its own port so the two never collide.
PORT="${UNICORN_OLLAMA_PORT:-11435}"
# Group-shared, on scratch — same tree and same reasoning as the FLUX weights.
MODELS_DIR="${UNICORN_OLLAMA_MODELS:-/scratch/faris/models/ollama}"
CHAT_MODEL="${CHAT_MODEL:-qwen3:4b-instruct}"
# ⚠️ CAP THE CONTEXT WINDOW. ollama otherwise honours the model's advertised
# maximum - 262144 tokens for this one - and sizes the KV cache for it. Measured
# 2026-09-06: that made a 4B model report as **43 GB**, which did not fit beside
# FLUX's 37.7 GB, so 27% of the layers spilled to CPU and a one-sentence reply
# took 24 s on an H100. Graphiti's extraction prompts are a few thousand tokens,
# so 16K is generous and keeps the whole model resident on the GPU.
CONTEXT_LENGTH="${CONTEXT_LENGTH:-16384}"
EMBED_MODEL="${EMBED_MODEL:-bge-m3}"

ACTION="${1:-start}"

remote() { ssh -o BatchMode=yes "$HOST" "$@"; }

case "$ACTION" in
  tunnel)
    echo "ssh -N -L ${PORT}:localhost:${PORT} ${HOST} &"
    echo "# then, before launching Streamlit:"
    echo "export GRAPHITI_LLM_BASE_URL=http://localhost:${PORT}/v1"
    echo "export GRAPHITI_LLM_MODEL=${CHAT_MODEL}"
    echo "export GRAPHITI_EMBED_BASE_URL=http://localhost:${PORT}/v1"
    echo "export GRAPHITI_EMBED_MODEL=${EMBED_MODEL}"
    exit 0;;

  status)
    remote "curl -s --max-time 5 http://localhost:${PORT}/api/tags" \
      || echo "not running on ${HOST}:${PORT}"
    exit 0;;

  stop)
    # Kill by PID FILE, never by pattern. The first attempt matched on
    # "OLLAMA_HOST=..." — an ENVIRONMENT variable, which never appears in a
    # command line — so it killed nothing, the old server kept running, and the
    # "restart" silently reused the already-loaded 256K-context model.
    #
    # The fallback matches our MODELS DIR, which does appear in the llama-server
    # argv. Both are scoped to processes we own: the shared instance on :11434
    # runs as a different user entirely, so it can never be caught by either.
    remote "if [ -f '${MODELS_DIR}/../ollama_${PORT}.pid' ]; then
              kill \$(cat '${MODELS_DIR}/../ollama_${PORT}.pid') 2>/dev/null || true
              rm -f '${MODELS_DIR}/../ollama_${PORT}.pid'
            fi
            pkill -u \$(id -u) -f '${MODELS_DIR}' 2>/dev/null || true
            sleep 2; echo done"
    echo "🛑 stopped our ollama on ${HOST}:${PORT} (shared :11434 untouched)"
    exit 0;;
esac

{
  echo "🧠 KG-extraction LLM on unicorn   $(date '+%Y-%m-%d %H:%M:%S')"
  echo "   host   : ${HOST}:${PORT}   (shared instance on :11434 is left alone)"
  echo "   models : ${MODELS_DIR}"
  echo "   chat   : ${CHAT_MODEL}"
  echo "   embed  : ${EMBED_MODEL}"
  echo "   ctx    : ${CONTEXT_LENGTH} tokens (capped — see the comment in this script)"
  echo ""

  echo "📁 preparing the group-shared model dir..."
  # setgid so anything pulled inherits group dsa and Rahul/Priyabanta can use it.
  remote "mkdir -p '${MODELS_DIR}' && chgrp dsa '${MODELS_DIR}' 2>/dev/null || true; chmod 2775 '${MODELS_DIR}' 2>/dev/null || true; ls -ld '${MODELS_DIR}'"

  if remote "curl -s --max-time 4 http://localhost:${PORT}/api/tags >/dev/null 2>&1"; then
    echo "✅ already serving on :${PORT}"
  else
    echo "🚀 starting ollama on :${PORT}..."
    # setsid + nohup so it outlives this ssh session.
    remote "OLLAMA_MODELS='${MODELS_DIR}' OLLAMA_HOST=127.0.0.1:${PORT} \
            OLLAMA_CONTEXT_LENGTH='${CONTEXT_LENGTH}' \
            setsid nohup ollama serve > '${MODELS_DIR}/../ollama_${PORT}.log' 2>&1 < /dev/null &
            echo \$! > '${MODELS_DIR}/../ollama_${PORT}.pid'
            sleep 3; echo \"started pid \$(cat '${MODELS_DIR}/../ollama_${PORT}.pid')\""
    for i in $(seq 1 20); do
      remote "curl -sf --max-time 3 http://localhost:${PORT}/api/tags >/dev/null 2>&1" && break
      sleep 2
    done
    remote "curl -sf --max-time 3 http://localhost:${PORT}/api/tags >/dev/null 2>&1" \
      || { echo "❌ did not come up — see ${MODELS_DIR}/../ollama_${PORT}.log"; exit 1; }
    echo "✅ up"
  fi

  for m in "${CHAT_MODEL}" "${EMBED_MODEL}"; do
    echo ""
    echo "📥 pulling ${m} (skipped if already present)..."
    remote "OLLAMA_MODELS='${MODELS_DIR}' OLLAMA_HOST=127.0.0.1:${PORT} ollama pull '${m}'" 2>&1 | tail -2
  done

  echo ""
  echo "🔎 verifying the chat model ANSWERS (a hybrid/thinking build returns empty)..."
  remote "OLLAMA_HOST=127.0.0.1:${PORT} curl -s --max-time 120 http://localhost:${PORT}/api/chat -d '{\"model\":\"${CHAT_MODEL}\",\"messages\":[{\"role\":\"user\",\"content\":\"Reply with exactly: OK\"}],\"stream\":false}'" \
    | python3 -c "import sys,json; c=json.load(sys.stdin)['message']['content'].strip(); print('   reply:', repr(c[:80])); print('   ✅ answers directly' if c else '   ❌ EMPTY — this is a thinking build, use the -instruct tag')"

  echo ""
  echo "✅ ready. Now, on the laptop:"
  echo ""
  bash "$0" tunnel | sed 's/^/     /'
} 2>&1 | tee "$LOG"
