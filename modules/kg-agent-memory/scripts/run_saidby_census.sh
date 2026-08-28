#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════════
#  run_saidby_census.sh — EARLY sanity check on the speaker-framing fix.
#
#  THE POINT: we do NOT need the full corpus to answer "did the fix change the
#  graph's shape?". Shape is a structural property, measurable on any sizeable
#  sample. Five of eight shards are already extracted, so we can read the answer
#  today instead of waiting for the last three.
#
#  WHY A SHARD STORE IS A PERFECT A/B
#  ----------------------------------
#  The shard store path is keyed on SHARD name, not on group. So the original
#  run (group gmb_finance_full, "User_5: <text>") and the new run (group
#  gmb_finance_saidby, "[said by User_5] <text>") BOTH live inside the same
#  physical store, over the SAME window range. Same messages, same model, same
#  prompts — only the framing differs. Neo4j keeps them apart by group_id.
#  (That co-location was an accident of the path scheme, but it hands us a
#  controlled comparison for free.)
#
#  It starts a READ-ONLY Neo4j on its own ports against a completed shard,
#  runs analysis/saidby_census.cypher, and stops the server again.
#
#  Usage (on the Pegasus login node):
#     bash scripts/run_saidby_census.sh          # shard s1
#     SHARD=s3 bash scripts/run_saidby_census.sh
# ═══════════════════════════════════════════════════════════════════════════════
set -euo pipefail
cd "$(dirname "$0")/.."
REPO_ROOT="$(pwd)"

FS_ROOT=${FS_ROOT:-/fscratch/abuali}
SHARD=${SHARD:-s1}
STORE="${FS_ROOT}/neo4j/shards/${SHARD}"
NEO4J_HOME="${FS_ROOT}/neo4j/neo4j-community-5.26.0"
export JAVA_HOME="${FS_ROOT}/conda_envs/java21"
export PATH="${JAVA_HOME}/bin:${PATH}"
PASS=${NEO4J_PASSWORD:-graphiti123}

mkdir -p logs
LOG="${REPO_ROOT}/logs/saidby_census_${SHARD}.log"

# Refuse to touch a shard that a running job still holds — s2/s4/s6 are being
# re-ingested right now and Neo4j takes an exclusive lock on its data directory.
BUSY=$(squeue -u "$USER" -h -o "%j" 2>/dev/null | grep -oE "s[0-9]+$" | sort -u || true)
if echo "$BUSY" | grep -qx "${SHARD}"; then
  echo "❌ shard ${SHARD} is being written by a running job — pick a finished one."
  exit 1
fi
[[ -d "${STORE}/data" ]] || { echo "❌ no store at ${STORE}/data"; exit 1; }

{
  echo "🔬 saidby census — $(date '+%Y-%m-%d %H:%M:%S')"
  echo "   shard : ${SHARD}"
  echo "   store : ${STORE}"
  echo ""

  # Own conf dir + own ports, so this never collides with a job's Neo4j.
  CONF="${STORE}/conf_census"
  mkdir -p "${CONF}" "${STORE}/logs_census" "${STORE}/run_census"
  BOLT=$(python3 -c 'import socket;s=socket.socket();s.bind(("127.0.0.1",0));print(s.getsockname()[1]);s.close()')
  HTTP=$(python3 -c 'import socket;s=socket.socket();s.bind(("127.0.0.1",0));print(s.getsockname()[1]);s.close()')
  cat > "${CONF}/neo4j.conf" <<CONFEOF
server.directories.data=${STORE}/data
server.directories.logs=${STORE}/logs_census
server.directories.run=${STORE}/run_census
server.bolt.listen_address=:${BOLT}
server.http.listen_address=:${HTTP}
server.memory.heap.initial_size=1G
server.memory.heap.max_size=2G
server.memory.pagecache.size=512m
CONFEOF
  export NEO4J_CONF="${CONF}"

  echo "🚀 starting Neo4j (bolt ${BOLT}) …"
  "${NEO4J_HOME}/bin/neo4j" start >/dev/null 2>&1 || true

  # Wait for bolt rather than sleeping a fixed amount.
  for i in $(seq 1 60); do
    "${NEO4J_HOME}/bin/cypher-shell" -a "bolt://localhost:${BOLT}" -u neo4j -p "${PASS}" \
      "RETURN 1;" >/dev/null 2>&1 && break
    sleep 3
  done

  echo "📊 running census …"
  echo ""
  "${NEO4J_HOME}/bin/cypher-shell" -a "bolt://localhost:${BOLT}" -u neo4j -p "${PASS}" \
      --format plain < "${REPO_ROOT}/analysis/saidby_census.cypher"

  echo ""
  echo "🧹 stopping Neo4j …"
  "${NEO4J_HOME}/bin/neo4j" stop >/dev/null 2>&1 || true
  echo "✅ done"
} 2>&1 | tee "${LOG}"
