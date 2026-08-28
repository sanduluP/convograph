#!/bin/bash
# ---------------------------------------------------------------------------
# cluster_baseline_job.sh — ONE self-contained GPU job (step 2a):
#   1) serve vLLM (Qwen3-30B-Instruct) on this node,
#   2) re-run the FULL BM25 baseline against it (localhost:8000),
#   3) stop vLLM and exit.
#
# This gives the apples-to-apples BM25 number on the NEW model (the 36.9% we have
# was on qwen2.5:32b) — the fair baseline our Graphiti retriever must beat — AND
# proves the whole vLLM path works, all WITHOUT Neo4j (that comes in step 2b).
#
# Submit it with (broad FP8-safe partition, ~1 GPU):
#   bash scripts/srun_submit.sh all gmb_bm25 8 1 96G 4 scripts/cluster_baseline_job.sh
# ---------------------------------------------------------------------------
set -euo pipefail
cd "$(dirname "$0")/.."          # repo root
REPO_ROOT="$(pwd)"

VENV="/fscratch/abuali/venvs/groupmembench"
PY="${VENV}/bin/python"
[ -x "$PY" ] || { echo "❌ no cluster venv — run scripts/setup_cluster_env.sh first"; exit 1; }

# Which checkpoint to baseline on. Override to A/B a different model:
#   VLLM_MODEL_DIR=/fscratch/abuali/models/Qwen3.6-35B-A3B-FP8 ...
MODEL_DIR="${VLLM_MODEL_DIR:-/fscratch/abuali/models/Qwen3-30B-A3B-Instruct-2507-FP8}"
MODEL_ID="${VLLM_SERVED_NAME:-Qwen/$(basename "$MODEL_DIR")}"

# 1️⃣  Serve the model (blocks until READY, then backgrounds).
echo "════════ 1/3  serving vLLM ════════"
VLLM_MODEL_DIR="$MODEL_DIR" VLLM_SERVED_NAME="$MODEL_ID" bash scripts/serve_vllm.sh

# 2️⃣  Re-run the BM25 baseline against the LOCAL vLLM. run_eval.sh reads ENV_FILE
#     from the environment, so we point it at .env.cluster (localhost:8000) and
#     use the served model name for both agent and judge.
echo "════════ 2/3  BM25 re-baseline on ${MODEL_ID} ════════"
export ENV_FILE=".env.cluster"
export LLM_PROVIDER="openai"

# Hand the run scripts the CLUSTER interpreter explicitly. (Do NOT symlink
# .venv/bin/python at $VENV/bin/python: Python looks for pyvenv.cfg beside the
# symlink, finds none, and silently falls back to the base interpreter WITHOUT
# site-packages — that's what made the 2026-07-22 run die with
# "No module named 'openai'" and report N/A for all six question types.)
export PY="$PY"

DOMAIN="${DOMAIN:-Finance}" \
AGENT_MODEL="$MODEL_ID" JUDGE_MODEL="$MODEL_ID" \
  bash scripts/run_bm25_baseline.sh

# 3️⃣  Stop vLLM.
echo "════════ 3/3  stopping vLLM ════════"
PIDF="${REPO_ROOT}/logs/serve_vllm/serve_vllm.pid"
[ -f "$PIDF" ] && kill "$(cat "$PIDF")" 2>/dev/null || true
echo "✅ cluster_baseline_job done — results in results/bm25_${DOMAIN:-Finance}/"
