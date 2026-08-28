#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════════
#  run_plot_graph_shape.sh — render the figure that explains Module 2's finding.
#
#  Needs the merged full-Finance KG running locally:
#      docker start neo4j-gmb-full     (bolt localhost:7688, neo4j/graphiti123)
#  Use scripts/run_open_merged_kg.sh first if the container does not exist yet.
# ═══════════════════════════════════════════════════════════════════════════════
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT_DIR="$(pwd)"
mkdir -p logs figures

PY="${ROOT_DIR}/.venv/bin/python"
[[ -x "${PY}" ]] || { echo "❌ no repo venv at .venv — this repo uses its OWN env"; exit 1; }

{
  echo "🎨 rendering graph-shape figure — $(date '+%Y-%m-%d %H:%M:%S')"
  echo "   neo4j : ${NEO4J_URI:-bolt://localhost:7688}"
  echo ""
  PYTHONUNBUFFERED=1 "${PY}" -u analysis/plot_graph_shape.py
  echo ""
  echo "✅ done — figures/graph_shape.png"
} 2>&1 | tee "${ROOT_DIR}/logs/plot_graph_shape.log"
