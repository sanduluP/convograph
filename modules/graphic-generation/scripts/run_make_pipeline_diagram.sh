#!/usr/bin/env bash
# =============================================================================
#  run_make_pipeline_diagram.sh — the module 3 pipeline as one landscape diagram.
#
#  ONE-SHOT. Per CLAUDE.md rule 12 the .excalidraw becomes Faris's the moment it
#  exists: he aligns it by hand in the app and none of that is visible to a
#  regenerator. The script refuses to overwrite; to rebuild, move the old file
#  aside first and keep the edited one.
#
#  USAGE
#      bash scripts/run_make_pipeline_diagram.sh
# =============================================================================
set -euo pipefail

MODULE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${MODULE_ROOT}/logs"
LOG_FILE="${LOG_DIR}/make_pipeline_diagram.log"
VENV_PY="${MODULE_ROOT}/../../ui/.venv/bin/python"   # needs nothing but stdlib

mkdir -p "${LOG_DIR}"
{
  echo "═══════════════════════════════════════════════════════════════════"
  echo "📐 MODULE 3 PIPELINE DIAGRAM — $(date '+%Y-%m-%d %H:%M:%S')"
  echo "═══════════════════════════════════════════════════════════════════"
  PYTHONUNBUFFERED=1 "${VENV_PY}" -u "${MODULE_ROOT}/analysis/make_pipeline_diagram.py"
  echo "🧹 done"
} 2>&1 | tee -a "${LOG_FILE}"
echo
echo "📝 log: ${LOG_FILE}"
