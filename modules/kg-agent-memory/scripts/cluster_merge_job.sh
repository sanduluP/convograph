#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# cluster_merge_job.sh — glue the 8 per-shard Graphiti graphs into ONE graph.
#
# THE POINT: scripts/submit_shards.sh ingested the 30,000 Finance messages as 8
# parallel jobs, each writing to its OWN Neo4j store under
# ${FS_ROOT}/neo4j/shards/<shard>/data. That is 8 partial graphs. This job turns
# them into a single Finance knowledge graph that the QA run (and the eventual
# visualization) can query.
#
# HOW
#   Phase 1  for each shard: start ITS Neo4j alone → dump nodes+edges to JSONL →
#            stop it. Serial on purpose: one store, one process, no lock races.
#   Phase 2  start the MERGED Neo4j (a fresh store) → import every dump →
#            collapse cross-shard duplicate entities → rebuild indices → report.
#
# NO GPU IS NEEDED. This is pure Neo4j I/O plus Python — the embeddings are
# carried inside the dumps, so nothing is re-embedded and no model is loaded.
# Submit it with GPUS=0 across a BROAD partition list (rule 8) and it schedules
# immediately on whatever node is free.
#
# ‼️  DO NOT RUN WHILE ANY gmb_s* INGEST JOB IS STILL RUNNING. Two processes on
#     two nodes opening the same /fscratch store would corrupt it — NFS does not
#     reliably enforce Neo4j's store lock. scripts/run_merge_shards.sh checks
#     squeue and refuses; run this script through that wrapper.
#
# Knobs: SHARD_LIST, GROUP_ID, EXPECTED_WINDOWS, SKIP_EXPORT, STITCH.
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail
cd "$(dirname "$0")/.."
REPO_ROOT="$(pwd)"

FS_ROOT=${FS_ROOT:-/fscratch/abuali}
VENV="${FS_ROOT}/venvs/groupmembench"
PY="${VENV}/bin/python"
[ -x "$PY" ] || { echo "❌ no cluster venv at ${VENV}"; exit 1; }

# --- Neo4j (same read-only install every shard used) --------------------------
NEO4J_VERSION=${NEO4J_VERSION:-5.26.0}
NEO4J_HOME="${FS_ROOT}/neo4j/neo4j-community-${NEO4J_VERSION}"
JAVA_HOME="${FS_ROOT}/conda_envs/java21"
NEO4J_PASSWORD=${NEO4J_PASSWORD:-graphiti123}
export JAVA_HOME
export PATH="${JAVA_HOME}/bin:${PATH}"
[ -x "${NEO4J_HOME}/bin/neo4j" ] || { echo "❌ Neo4j missing at ${NEO4J_HOME}"; exit 1; }

# --- knobs --------------------------------------------------------------------
SHARD_LIST=${SHARD_LIST:-"s1 s2 s3 s4 s5 s6 s7 s8"}
GROUP_ID=${GROUP_ID:-gmb_finance_full}
# 6,002 rather than 6,000 because _build_windows never lets an episode straddle a
# channel boundary, so each of the 6 Finance channels ends with a short window.
EXPECTED_WINDOWS=${EXPECTED_WINDOWS:-6002}
DUMP_DIR=${DUMP_DIR:-"${FS_ROOT}/neo4j/dumps/${GROUP_ID}"}
MERGED_ROOT=${MERGED_ROOT:-"${FS_ROOT}/neo4j/merged/${GROUP_ID}"}
# SKIP_EXPORT=1 reuses dumps from a previous attempt (phase 1 is the slow part).
SKIP_EXPORT=${SKIP_EXPORT:-0}
# STITCH=0 keeps cross-shard duplicate entities separate (union only).
STITCH=${STITCH:-1}
# FRESH=1 deletes the merged store first, so the import starts from nothing.
# Default 1: the import MERGEs on uuid and is idempotent, but a stale store from
# an aborted attempt could still hold entities that a later stitch already
# collapsed, and reasoning about that is not worth the 20 minutes it saves.
FRESH=${FRESH:-1}

# Ports are chosen at runtime — compute nodes are shared and 7687 may be taken.
free_port() {
  "${PY}" - <<'PYEOF'
import socket
s = socket.socket(); s.bind(("127.0.0.1", 0))
print(s.getsockname()[1]); s.close()
PYEOF
}

echo "🖥️  [merge] node=$(hostname)"
echo "🧩 [merge] shards      : ${SHARD_LIST}"
echo "🏷️  [merge] group_id    : ${GROUP_ID}"
echo "📁 [merge] dumps       : ${DUMP_DIR}"
echo "📁 [merge] merged store: ${MERGED_ROOT}"

# ─────────────────────────────────────────────────────────────────────────────
# Helpers — one Neo4j at a time, always through these two functions
# ─────────────────────────────────────────────────────────────────────────────

# write_conf <store_root> <bolt_port> <http_port> <label>
# Builds an isolated conf dir pointing at <store_root>. We regenerate it on every
# run rather than reusing what the ingest job left behind, because the ports in
# that file were free on a DIFFERENT node months of jobs ago — they mean nothing
# here.
write_conf() {
  local root="$1" bolt="$2" http="$3" label="$4"
  mkdir -p "${root}"/{conf,data,logs,run}
  # Start from the base conf so JVM/tuning settings stay in sync with the install,
  # then append our block (and drop any previous copy of it).
  sed '/^# --- groupmembench settings/,$d' "${NEO4J_HOME}/conf/neo4j.conf" \
    > "${root}/conf/neo4j.conf"
  cat >> "${root}/conf/neo4j.conf" <<EOF
# --- groupmembench settings (${label}) ---
server.default_listen_address=127.0.0.1
server.bolt.listen_address=127.0.0.1:${bolt}
server.http.listen_address=127.0.0.1:${http}
server.directories.data=${root}/data
server.directories.logs=${root}/logs
server.directories.run=${root}/run
server.memory.heap.initial_size=4g
server.memory.heap.max_size=16g
server.memory.pagecache.size=8g
EOF
  # Neo4j 5 refuses to boot from a conf dir missing these two log configs.
  cp -n "${NEO4J_HOME}/conf/server-logs.xml" "${root}/conf/" 2>/dev/null || true
  cp -n "${NEO4J_HOME}/conf/user-logs.xml"   "${root}/conf/" 2>/dev/null || true
}

# start_neo4j <bolt_uri> <label>  — start and BLOCK until bolt actually answers
# (`neo4j start` returns long before the database is ready to serve).
start_neo4j() {
  local uri="$1" label="$2"
  "${NEO4J_HOME}/bin/neo4j" start
  for i in $(seq 1 60); do
    if "${PY}" - <<PYEOF 2>/dev/null
from neo4j import GraphDatabase
d = GraphDatabase.driver("${uri}", auth=("neo4j", "${NEO4J_PASSWORD}"))
d.verify_connectivity(); d.close()
PYEOF
    then echo "✅ [merge] ${label} Neo4j ready after ${i} checks"; return 0; fi
    sleep 5
  done
  echo "❌ [merge] ${label} Neo4j never became ready"
  tail -40 "${NEO4J_CONF%/conf}/logs/neo4j.log" 2>/dev/null || true
  return 1
}

stop_neo4j() { "${NEO4J_HOME}/bin/neo4j" stop >/dev/null 2>&1 || true; }

# Never leak a running Neo4j holding a lock on a /fscratch store, whatever kills
# us (error, SLURM timeout, Ctrl-C).
trap 'echo "🧹 [merge] stopping Neo4j …"; stop_neo4j; echo "🧹 [merge] done"' EXIT

# ═══ PHASE 1 — dump each shard ═══════════════════════════════════════════════
if [[ "${SKIP_EXPORT}" == "1" ]]; then
  echo "⏭️  [merge] SKIP_EXPORT=1 — reusing existing dumps in ${DUMP_DIR}"
else
  echo "════════ phase 1/2  exporting shards ════════"
  mkdir -p "${DUMP_DIR}"
  for shard in ${SHARD_LIST}; do
    SHARD_ROOT="${FS_ROOT}/neo4j/shards/${shard}"
    if [[ ! -d "${SHARD_ROOT}/data" ]]; then
      echo "❌ [merge] no store for shard ${shard} at ${SHARD_ROOT}/data"; exit 1
    fi
    BOLT=$(free_port); HTTP=$(free_port)
    write_conf "${SHARD_ROOT}" "${BOLT}" "${HTTP}" "shard ${shard}"
    export NEO4J_CONF="${SHARD_ROOT}/conf"
    echo "📤 [merge] shard ${shard} → bolt ${BOLT}"
    start_neo4j "bolt://localhost:${BOLT}" "shard ${shard}"
    PYTHONUNBUFFERED=1 "${PY}" -u -m baselines.graphiti.merge_shards export \
      --bolt-uri "bolt://localhost:${BOLT}" \
      --neo4j-password "${NEO4J_PASSWORD}" \
      --group-id "${GROUP_ID}" \
      --shard "${shard}" \
      --dump-dir "${DUMP_DIR}"
    stop_neo4j
    # Give the JVM a moment to release the store before the next start.
    sleep 5
  done
  echo "✅ [merge] all shards exported"
  ls -lh "${DUMP_DIR}"
fi

# ═══ PHASE 2 — import everything into one store ══════════════════════════════
echo "════════ phase 2/2  building the merged graph ════════"
if [[ "${FRESH}" == "1" && -d "${MERGED_ROOT}/data" ]]; then
  echo "🧹 [merge] FRESH=1 — removing previous merged store"
  rm -rf "${MERGED_ROOT}/data"
fi

BOLT=$(free_port); HTTP=$(free_port)
write_conf "${MERGED_ROOT}" "${BOLT}" "${HTTP}" "merged ${GROUP_ID}"
export NEO4J_CONF="${MERGED_ROOT}/conf"
# A brand-new store has no password yet; set-initial-password only works before
# the first start and exits non-zero afterwards, which is expected on a re-run.
"${NEO4J_HOME}/bin/neo4j-admin" dbms set-initial-password "${NEO4J_PASSWORD}" 2>/dev/null \
  && echo "🔐 [merge] merged-store password initialised" \
  || echo "ℹ️  [merge] merged-store password already set"
echo "📥 [merge] merged store → bolt ${BOLT}"
start_neo4j "bolt://localhost:${BOLT}" "merged"

STITCH_FLAG=()
[[ "${STITCH}" == "1" ]] || STITCH_FLAG+=(--no-stitch)

# Belt and braces for the index rebuild: graphiti-core constructs OpenAI clients
# eagerly, and any one of them missing a key aborts the whole run. No model is
# ever called here (we only issue index DDL), so a placeholder is correct — but
# it must be non-empty or the OpenAI SDK raises before we reach any of our code.
export OPENAI_API_KEY="${OPENAI_API_KEY:-unused-merge-only}"

PYTHONUNBUFFERED=1 "${PY}" -u -m baselines.graphiti.merge_shards import \
  --bolt-uri "bolt://localhost:${BOLT}" \
  --neo4j-password "${NEO4J_PASSWORD}" \
  --group-id "${GROUP_ID}" \
  --dump-dir "${DUMP_DIR}" \
  --expected-windows "${EXPECTED_WINDOWS}" \
  --stats-json "${REPO_ROOT}/results/merge_census_${GROUP_ID}.json" \
  "${STITCH_FLAG[@]}"

echo ""
echo "🎉 [merge] ALL DONE"
echo "   merged store : ${MERGED_ROOT}/data"
echo "   census       : ${REPO_ROOT}/results/merge_census_${GROUP_ID}.json"
echo "   next         : run the full-corpus QA against this store, and gap-fill"
echo "                  the missing windows listed above with a --resume pass."
