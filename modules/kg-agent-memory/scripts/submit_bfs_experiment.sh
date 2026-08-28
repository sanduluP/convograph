#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════════
#  submit_bfs_experiment.sh — re-score the EXISTING full-Finance KG with graph
#  traversal (BFS) switched ON, changing nothing else.
#
#  THE QUESTION
#  ------------
#  Reading the source on 2026-08-13 established that our "KG retrieval" never
#  walks the graph: the default recipe EDGE_HYBRID_SEARCH_RRF is BM25 over the
#  fact SENTENCE plus cosine over that same sentence's embedding. Structure is
#  never read. That explains why every structural fix we tried (entity stitching,
#  group_id namespacing, channel scoping) measured exactly zero.
#
#  This run switches on the traversal that was missing, and ONLY that.
#
#  WHAT WE EXPECT — AND WHY A NULL RESULT IS STILL VALUABLE
#  --------------------------------------------------------
#  BFS is seeded from the SOURCE NODES of the first-pass hits, and 94 % of our
#  facts originate at a person, so the seeds collapse onto the 12 `User_N` hubs
#  (`User_13` has degree 17,814). A walk from a node connected to everything
#  returns everything, which is not a neighbourhood. So the honest prediction is
#  "flat or slightly worse".
#
#  That is precisely why it is worth one hour: a flat result is DIRECT evidence
#  that the graph's SHAPE is the ceiling, which settles the open question of
#  whether to keep tuning retrieval or to go fix extraction.
#
#  ⚠️ DEPTH — DO NOT RAISE THIS CASUALLY
#  --------------------------------------
#  The walk is one Cypher variable-length match:
#      MATCH path = (origin)-[:RELATES_TO|MENTIONS*1..DEPTH]->(:Entity) ... LIMIT n
#  Neo4j expands EVERY path before LIMIT applies. Graphiti's own default is 3;
#  on a graph whose seeds are degree-17,814 hubs that is a combinatorial explosion
#  which can hang the query or OOM the database. We start at DEPTH=1.
#
#  RESULTS ARE WRITTEN TO SEPARATE FILES — THE EXISTING SCORES ARE NOT TOUCHED
#  ---------------------------------------------------------------------------
#  RESULT_TAG=bfs<depth> appends a suffix to every output path, so this run lands in
#      results/cluster/graphiti_gmb_finance_full_<qtype>_bfs1.jsonl
#  while yesterday's baseline stays at
#      results/cluster/graphiti_gmb_finance_full_<qtype>.jsonl
#  Without the tag the second run would silently destroy the first run's evidence.
#
#  Usage:
#    bash scripts/submit_bfs_experiment.sh
#    GRAPHITI_BFS_DEPTH=2 bash scripts/submit_bfs_experiment.sh    # after depth 1 is proven safe
#    QTYPES="knowledge_update" bash scripts/submit_bfs_experiment.sh
# ═══════════════════════════════════════════════════════════════════════════════
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

STORE="${STORE:-/fscratch/abuali/neo4j/merged/gmb_finance_full}"
GROUP_ID="${GROUP_ID:-gmb_finance_full}"
WINDOW="${WINDOW:-5}"
DEPTH="${GRAPHITI_BFS_DEPTH:-1}"

# All six, so the comparison against yesterday's baseline is complete rather than
# a cherry-picked column.
QTYPES="${QTYPES:-multi_hop knowledge_update term_ambiguity user_implicit temporal abstention}"

# The tag that keeps this run's results separate from the baseline's.
TAG="${RESULT_TAG:-bfs${DEPTH}}"

# BFS adds a graph query per search on top of the existing two channels, so the
# per-question cost rises. 10 h is deliberate headroom, not an estimate.
WALLTIME="${WALLTIME:-10}"

echo "🔎 BFS traversal experiment — same graph, same questions, traversal ON"
echo "   store      : ${STORE}"
echo "   group      : ${GROUP_ID}"
echo "   bfs depth  : ${DEPTH}   ⚠️  Graphiti's own default is 3 — dangerous here (hub degree 17,814)"
echo "   result tag : _${TAG}    ← baseline files are NOT overwritten"
echo "   qtypes     : ${QTYPES}"
echo "   walltime   : ${WALLTIME} h"
echo ""

for QT in ${QTYPES}; do
  QFILE="${ROOT_DIR}/questions/Finance/${QT}.jsonl"
  [[ -f "${QFILE}" ]] || { echo "❌ no such question file: ${QFILE}"; exit 1; }
  echo "   ✓ $(printf '%-16s' "${QT}") $(wc -l < "${QFILE}") questions"
done
echo ""

# Refuse to start if the tag would collide with an existing result file — losing a
# measured run to a silent overwrite is the exact failure RESULT_TAG exists to stop.
for QT in ${QTYPES}; do
  OUT="${ROOT_DIR}/results/cluster/graphiti_${GROUP_ID}_${QT}_${TAG}.jsonl"
  if [[ -f "${OUT}" ]]; then
    echo "❌ ${OUT} already exists — pick a different RESULT_TAG or move it aside."
    exit 1
  fi
done

echo "📤 submitting …"
STORE_ROOT="${STORE}" \
GROUP_ID="${GROUP_ID}" \
WINDOW="${WINDOW}" \
QTYPE="${QTYPES}" \
DOMAIN=Finance \
CORPUS=0 \
RETRIEVER=graphiti \
RETRIEVE_ONLY=1 \
INGEST_ONLY=0 \
CONTROL_BM25=0 \
STRICT_PROMPT=1 \
RESULT_TAG="${TAG}" \
GRAPHITI_BFS=1 \
GRAPHITI_BFS_DEPTH="${DEPTH}" \
  nohup bash scripts/srun_submit.sh all "gmb_bfs${DEPTH}" 8 1 96G "${WALLTIME}" \
    scripts/cluster_ingest_job.sh > /tmp/submit_bfs.log 2>&1 &

wait
cat /tmp/submit_bfs.log
echo ""
echo "✅ submitted — watch with: squeue -u abuali"
echo "   new results : results/cluster/graphiti_${GROUP_ID}_<qtype>_${TAG}.jsonl"
echo "   baseline    : results/cluster/graphiti_${GROUP_ID}_<qtype>.jsonl  (untouched)"
