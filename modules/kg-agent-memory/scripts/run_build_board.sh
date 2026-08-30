#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════════
#  run_build_board.sh — Module 3, rung 0: draw ONE board from the knowledge graph.
#
#  The smallest artifact that answers "can a temporal KG be drawn as a graphic
#  recording?". Static, no icons, no live input — those are later rungs. The value
#  is having something real to look at instead of arguing about a hypothetical.
#
#  Needs a Neo4j holding the merged graph:
#     docker start neo4j-gmb-full          # local  → bolt://localhost:7688
#  or point it at the hosted instance:
#     NEO4J_URI=neo4j+s://<id>.databases.neo4j.io NEO4J_USER=<id> \
#     NEO4J_PASSWORD=... bash scripts/run_build_board.sh
#
#  Knobs:  BOARD_TOPICS (default 4)   BOARD_FACTS_PER_TOPIC (default 5)
#          → 4 x 5 = 20 cards, the low end of what a readable board holds.
# ═══════════════════════════════════════════════════════════════════════════════
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs excalidraw/board

PY="./.venv/bin/python"; [[ -x "$PY" ]] || { echo "❌ no repo venv — see requirements.txt"; exit 1; }

{
  echo "🎨 building board — $(date '+%Y-%m-%d %H:%M:%S')"
  echo "   neo4j  : ${NEO4J_URI:-bolt://localhost:7688}"
  echo "   topics : ${BOARD_TOPICS:-4} × ${BOARD_FACTS_PER_TOPIC:-5} facts"
  echo "   style  : ${BOARD_STYLE:-graphic-recording}   (see styles/board_styles.json)"
  echo ""
  PYTHONUNBUFFERED=1 "$PY" -u analysis/build_board.py
  echo ""
  echo "✅ open excalidraw/board/board.excalidraw in Excalidraw or the VS Code extension"
} 2>&1 | tee logs/build_board.log
