#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# run_kuzu_check.sh — can we drop the Neo4j SERVER and store the KG in a FILE?
#
# If this passes, the whole ingest fits in ONE self-contained Pegasus SLURM job:
# no Neo4j server, no SSH tunnel, no VPN dependency, no laptop babysitting.
#
# Needs only: the tunnel (or any OpenAI-compatible LLM) + serv-3306 embeddings.
# Takes ~2 minutes — 3 tiny episodes, not a real ingest.
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

PY="${PY:-${ROOT_DIR}/.venv/bin/python}"
[[ -x "${PY}" ]] || { echo "❌ no python at ${PY}" >&2; exit 1; }

MODEL_ID=${MODEL_ID:-"Qwen/Qwen3-30B-A3B-Instruct-2507-FP8"}
LLM_URL=${LLM_URL:-"http://localhost:8000/v1"}

mkdir -p logs
LOG="${ROOT_DIR}/logs/kuzu_check.log"

echo "🔎 [kuzu-check] LLM=${LLM_URL} model=${MODEL_ID}"
curl -sf -m 15 "${LLM_URL}/models" >/dev/null \
  || { echo "❌ LLM endpoint unreachable at ${LLM_URL} (tunnel/VPN?)" >&2; exit 1; }
echo "   ✅ LLM reachable"

PYTHONUNBUFFERED=1 PYTHONPATH="${ROOT_DIR}" \
GRAPHITI_LLM_BASE_URL="${LLM_URL}" \
GRAPHITI_LLM_MODEL="${MODEL_ID}" \
SEMAPHORE_LIMIT="${SEMAPHORE_LIMIT:-10}" \
  "${PY}" baselines/graphiti/kuzu_check.py 2>&1 | tee "${LOG}"
