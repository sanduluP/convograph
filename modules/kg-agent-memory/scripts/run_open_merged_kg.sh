#!/usr/bin/env bash
# =============================================================================
#  run_open_merged_kg.sh — pull the merged full-corpus KG down from Pegasus and
#  open it in a LOCAL Neo4j Browser, so Faris (and Rahul) can EYEBALL the graph.
# =============================================================================
#
#  WHY THIS EXISTS
#  ---------------
#  The knowledge graph lives on the cluster as a raw Neo4j *store directory* on
#  /fscratch. A store directory is not browsable by itself — it is only data
#  files. To look at it you need a Neo4j SERVER pointed at that directory, and
#  the cluster's compute nodes are not something we can keep a browser open on.
#
#  So: copy the store to the laptop once, then run a throwaway Neo4j container
#  on top of it. From then on the graph is one `docker start` away.
#
#  IMPORTANT — the Neo4j version must MATCH the one that wrote the store.
#  The cluster used Neo4j 5.26.0, so we run the `neo4j:5.26` image. A newer
#  Neo4j would try to UPGRADE the store format on first boot, which rewrites it.
#
#  PORTS: we deliberately use 7475/7688 instead of the usual 7474/7687 so this
#  never collides with the `neo4j-graphiti` container used for the self-tests.
#
#  Usage:
#      bash scripts/run_open_merged_kg.sh              # stitched merge (default)
#      STORE=nostitch bash scripts/run_open_merged_kg.sh
#      SKIP_SYNC=1    bash scripts/run_open_merged_kg.sh   # already downloaded
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------- configuration
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${REPO_ROOT}/logs"
LOG_FILE="${LOG_DIR}/open_merged_kg.log"

# Which merged store to open. `full` = the entity-stitched merge (names unified
# across shards); `nostitch` = the raw union, kept as the control experiment.
STORE="${STORE:-full}"
case "${STORE}" in
  full)     REMOTE_STORE="/fscratch/abuali/neo4j/merged/gmb_finance_full" ;;
  nostitch) REMOTE_STORE="/fscratch/abuali/neo4j/merged/gmb_finance_nostitch" ;;
  *) echo "❌ unknown STORE='${STORE}' (expected: full | nostitch)"; exit 1 ;;
esac

# Where the store lands locally, and what the container is called. Both are
# suffixed with ${STORE} so the two merges can coexist side by side.
LOCAL_ROOT="${HOME}/code/DSA_HiWi/neo4j-merged/${STORE}"
CONTAINER="neo4j-gmb-${STORE}"
HTTP_PORT="${HTTP_PORT:-7475}"
BOLT_PORT="${BOLT_PORT:-7688}"
NEO4J_IMAGE="neo4j:5.26"          # MUST match the cluster's 5.26.0 store format
NEO4J_PASSWORD="${NEO4J_PASSWORD:-graphiti123}"   # same as the cluster's dev pw
SKIP_SYNC="${SKIP_SYNC:-0}"

mkdir -p "${LOG_DIR}"

# Everything from here on is tee'd to a persistent log (house rule 3).
{
  echo "════════════════════════════════════════════════════════════════"
  echo "🧠 Opening merged KG locally   ($(date '+%F %T'))"
  echo "   store     : ${STORE}"
  echo "   remote    : pegasus:${REMOTE_STORE}"
  echo "   local     : ${LOCAL_ROOT}"
  echo "   container : ${CONTAINER}  (http :${HTTP_PORT}, bolt :${BOLT_PORT})"
  echo "════════════════════════════════════════════════════════════════"

  # -------------------------------------------------------------- 1. download
  if [[ "${SKIP_SYNC}" == "1" ]]; then
    echo "⏭️  SKIP_SYNC=1 — using the copy already on disk"
  else
    echo "📥 rsyncing the store from Pegasus (a few GB — this is the slow part)…"
    mkdir -p "${LOCAL_ROOT}"
    # --delete keeps the local copy an exact mirror: a half-synced store would
    # boot into a corrupt database, which is far worse than re-downloading.
    rsync -ah --info=progress2 --delete \
      "pegasus:${REMOTE_STORE}/data/" "${LOCAL_ROOT}/data/"
    echo "✅ download complete: $(du -sh "${LOCAL_ROOT}/data" | cut -f1)"
  fi

  # ------------------------------------------------------- 2. (re)create server
  # Remove any previous container of the same name so a re-run is idempotent.
  # The DATA lives in ${LOCAL_ROOT}, not in the container, so this is safe.
  if docker ps -a --format '{{.Names}}' | grep -qx "${CONTAINER}"; then
    echo "🧹 removing the previous '${CONTAINER}' container (data is untouched)"
    docker rm -f "${CONTAINER}" >/dev/null
  fi

  echo "🚀 starting Neo4j ${NEO4J_IMAGE} on the downloaded store…"
  docker run -d \
    --name "${CONTAINER}" \
    -p "${HTTP_PORT}:7474" \
    -p "${BOLT_PORT}:7687" \
    -v "${LOCAL_ROOT}/data:/data" \
    -e NEO4J_AUTH="neo4j/${NEO4J_PASSWORD}" \
    -e NEO4J_server_memory_heap_max__size=4G \
    -e NEO4J_server_memory_pagecache_size=2G \
    "${NEO4J_IMAGE}" >/dev/null

  # ------------------------------------------------------------ 3. wait for it
  # `docker run` returns immediately; Neo4j needs ~15-40 s before HTTP answers.
  echo -n "⏳ waiting for Neo4j to accept connections "
  for _ in $(seq 1 60); do
    if curl -sf "http://localhost:${HTTP_PORT}" >/dev/null 2>&1; then
      echo " ✅"
      break
    fi
    echo -n "."
    sleep 2
  done

  # ------------------------------------------------------------- 4. quick census
  # Prove the store actually loaded, rather than trusting that it did.
  echo "📊 census:"
  docker exec "${CONTAINER}" cypher-shell -u neo4j -p "${NEO4J_PASSWORD}" \
    "MATCH (n) RETURN labels(n)[0] AS label, count(*) AS n ORDER BY n DESC;" \
    2>/dev/null || echo "   (still warming up — retry in a few seconds)"

  echo
  echo "════════════════════════════════════════════════════════════════"
  echo "🎉 Open in your browser:  http://localhost:${HTTP_PORT}"
  echo "   connect URL : bolt://localhost:${BOLT_PORT}"
  echo "   user / pass : neo4j / ${NEO4J_PASSWORD}"
  echo
  echo "   Starter queries are in: analysis/eyeball_kg.cypher"
  echo "   Stop it later with    : docker stop ${CONTAINER}"
  echo "   Start it again with   : docker start ${CONTAINER}"
  echo "════════════════════════════════════════════════════════════════"
} 2>&1 | tee -a "${LOG_FILE}"
