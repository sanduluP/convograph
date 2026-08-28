#!/bin/bash
# ---------------------------------------------------------------------------
# setup_cluster_env.sh — One-time: create the GroupMemBench Python env on Pegasus,
# fully ISOLATED (rule 1). Installs vLLM (to SERVE Qwen3-30B-Instruct) + the
# benchmark harness deps (to RUN the eval) into ONE venv on /fscratch (home dirs
# have tiny quotas; torch/CUDA wheels are large).
#
# STEP 2a needs only: vllm + requirements.txt (openai/numpy/tqdm/rank-bm25).
# STEP 2b (Graphiti retriever) will additionally install graphiti-core==0.29.2
# + neo4j — added later, once the vLLM + BM25-rebaseline path is proven.
#
# Run on a COMPUTE node (heavy CUDA install), e.g. grab one interactively:
#   bash scripts/slurm_pty.sh all gmb_setup 8 1 0 16G 1     # if you have slurm_pty
# or just run this as a one-shot job via srun_submit.sh. Then:
#   bash scripts/setup_cluster_env.sh
# ---------------------------------------------------------------------------
set -euo pipefail

VENV="/fscratch/abuali/venvs/groupmembench"
REPO_ROOT="/home/abuali/projects/GroupMemBench"

echo "🐍 Creating venv at ${VENV}"
python3 -m venv "${VENV}"
"${VENV}/bin/python" -m pip install -q --upgrade pip setuptools wheel

echo "📦 Installing vLLM (CUDA wheels; ~10-20 min) — pins torch/CUDA"
"${VENV}/bin/pip" install -q vllm

echo "📦 Installing the GroupMemBench harness requirements"
"${VENV}/bin/pip" install -q --requirement "${REPO_ROOT}/requirements.txt"

echo "🔎 Verifying imports"
"${VENV}/bin/python" - <<'PY'
import importlib
for mod in ("vllm", "openai", "numpy", "tqdm", "rank_bm25"):
    importlib.import_module(mod); print(f"   ✅ import ok: {mod}")
PY

echo ""
echo "✅ Env ready: ${VENV}"
echo "   vllm:   ${VENV}/bin/vllm"
echo "   python: ${VENV}/bin/python"
