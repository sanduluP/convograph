#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# setup_env.sh — create THIS repo's own isolated .venv (rule 1) and install the
# LIGHT dependency set needed for the step-1 harness smoke test.
#
# Deliberately light: step 1 only needs the GroupMemBench harness deps
# (openai, numpy, tqdm, rank-bm25). Graphiti (heavy, needs its own version pin
# + Neo4j) is installed LATER in step 2, once the plumbing is proven — so an
# install hiccup here can never be blamed on Graphiti.
#
# All output is tee'd to logs/setup_env.log so the run is inspectable after the
# fact (rule 3), and Python runs unbuffered so the log updates live.
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

# Resolve repo root regardless of where the script is invoked from.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

# Persist every line of stdout+stderr to a log the user can tail / open in VSCode.
mkdir -p logs
LOG="${ROOT_DIR}/logs/setup_env.log"
exec > >(tee "${LOG}") 2>&1

echo "🚀 [setup_env] repo root: ${ROOT_DIR}"

# 1️⃣  Create the isolated virtual environment IN THIS REPO (never reuse another repo's env).
if [[ -d .venv ]]; then
  echo "♻️  [setup_env] .venv already exists — reusing it"
else
  echo "🧱 [setup_env] creating .venv with $(python3 --version)"
  python3 -m venv .venv
fi

# 2️⃣  Upgrade pip tooling so wheels resolve cleanly.
echo "⬆️  [setup_env] upgrading pip/setuptools/wheel"
./.venv/bin/python -m pip install --quiet --upgrade pip setuptools wheel

# 3️⃣  Install the LIGHT harness deps (the benchmark's own requirements.txt).
echo "📦 [setup_env] installing GroupMemBench harness requirements"
./.venv/bin/python -m pip install --requirement requirements.txt

# 4️⃣  Sanity-check the imports the smoke test relies on.
echo "🔎 [setup_env] verifying key imports"
PYTHONUNBUFFERED=1 ./.venv/bin/python - <<'PY'
import importlib
for mod in ("openai", "numpy", "tqdm", "rank_bm25"):
    importlib.import_module(mod)
    print(f"   ✅ import ok: {mod}")
PY

echo "✅ [setup_env] done — light venv ready for the step-1 smoke test"
echo "   next: bash baselines/graphiti/run_eval.sh   (after choosing a questions file)"
