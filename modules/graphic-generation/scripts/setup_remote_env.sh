#!/usr/bin/env bash
# setup_remote_env.sh — one-time: create this module's venv on unicorn.
#
# Run THIS ON THE REMOTE HOST (unicorn), after syncing code there with
# sync_to_cluster.sh. From your laptop:
#
#   bash scripts/sync_to_cluster.sh
#   ssh sandulu@serv-7101.kl.dfki.de
#   cd /scratch/mpatil/sandulu/convograph-graphic-generation
#   bash scripts/setup_remote_env.sh
#
# Installs torch + diffusers into its OWN venv (mirrors modules/kg-agent-memory's
# house rule: this is the one module allowed to have torch, kept isolated so its
# pins never collide with that module's deliberately torch-free env).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${REPO_ROOT}/.venv"

echo "🐍 Creating venv at ${VENV}"
python3 -m venv "${VENV}"
"${VENV}/bin/python" -m pip install -q --upgrade pip setuptools wheel

# torch specifically, from the CUDA-tagged index, NOT plain PyPI: a bare
# `pip install torch` grabs the newest default build, which can bundle a CUDA
# runtime newer than the host's driver supports. When that happens torch
# imports fine but torch.cuda.is_available() silently returns False rather
# than erroring — cu124 is broadly compatible with any driver >=CUDA 12.4
# (forward-compatible), which covers this host (driver 570.x / CUDA 12.8) and
# most others without needing to detect the exact driver version.
echo "📦 Installing torch (CUDA 12.4 wheel — a few minutes)"
"${VENV}/bin/pip" install -q torch --index-url https://download.pytorch.org/whl/cu124

echo "📦 Installing the rest of requirements.txt"
"${VENV}/bin/pip" install -q --requirement "${REPO_ROOT}/requirements.txt"

echo "🔎 Verifying imports + CUDA visibility"
"${VENV}/bin/python" - <<'PY'
import importlib
for mod in ("torch", "diffusers", "transformers", "neo4j", "PIL"):
    importlib.import_module(mod)
    print(f"   ✅ import ok: {mod}")

import torch
print(f"   CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"   device: {torch.cuda.get_device_name(0)}")
else:
    print("   ⚠️  no CUDA device visible — generate_image.py will fail on .to('cuda')")
PY

echo ""
echo "✅ Env ready: ${VENV}"
echo "   Next: bash scripts/run_generate_image.sh"
