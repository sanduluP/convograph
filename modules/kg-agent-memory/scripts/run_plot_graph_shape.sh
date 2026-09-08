#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════════
#  run_plot_graph_shape.sh — render the figure that explains Module 2's finding.
#
#  Needs a merged KG running locally:
#      docker start neo4j-gmb-full           (bolt 7688) — ingested WITH speakers
#      docker start neo4j-gmb-speaker_free   (bolt 7689) — speaker-free
#  Use scripts/run_open_merged_kg.sh first if the container does not exist yet.
#
#  SHAPE_OUT names the figure. Set it when rendering a SECOND graph, or the two
#  overwrite each other and the comparison is lost:
#      NEO4J_URI=bolt://localhost:7689 SHAPE_OUT=figures/graph_shape_speaker_free.png \
#        bash scripts/run_plot_graph_shape.sh
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
  # PYTHONPATH so `from analysis.…` resolves: run as a file, sys.path[0] is
  # analysis/, not the module root, and the import fails.
  PYTHONUNBUFFERED=1 PYTHONPATH="${ROOT_DIR}" "${PY}" -u analysis/plot_graph_shape.py
  echo ""
  echo "✅ done — ${SHAPE_OUT:-figures/graph_shape.png}"
} 2>&1 | tee "${ROOT_DIR}/logs/plot_graph_shape.log"
