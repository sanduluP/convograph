#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# setup_cluster_neo4j.sh — ONE-TIME install of Neo4j + a JRE on /fscratch, so a
# SLURM job can run its own Neo4j on the compute node.
#
# WHY: today the KG lives in Neo4j on Faris's LAPTOP while the LLM runs on the
# cluster, so every long ingest depends on the VPN staying up for hours. Putting
# Neo4j on the compute node makes the whole run self-contained: submit the job,
# close the laptop, no VPN.
#
# WHY NOT A CONTAINER: SLURM jobs here already run inside an enroot/pyxis
# container, and nesting another one is a fight. Neo4j Community is just a
# tarball + a JVM, so we run it as a plain process instead. No root needed.
#
# RUN THIS ON THE LOGIN NODE — compute nodes have restricted egress (they cannot
# reach neo4j.org), the login node has full internet. /fscratch is shared, so
# whatever we download here is visible to every compute node later.
#
# Idempotent: re-running skips anything already installed.
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

FS_ROOT=${FS_ROOT:-/fscratch/abuali}
NEO4J_VERSION=${NEO4J_VERSION:-5.26.0}
NEO4J_HOME="${FS_ROOT}/neo4j/neo4j-community-${NEO4J_VERSION}"
JAVA_ENV="${FS_ROOT}/conda_envs/java21"
CONDA="${FS_ROOT}/miniforge3/bin/conda"
NEO4J_PASSWORD=${NEO4J_PASSWORD:-graphiti123}

mkdir -p "${FS_ROOT}/neo4j" "${FS_ROOT}/logs"
LOG="${FS_ROOT}/logs/setup_cluster_neo4j.log"

{
echo "🏗️  [setup] Neo4j ${NEO4J_VERSION} → ${NEO4J_HOME}"

# --- 1) Java 21 (Neo4j 5.26 needs JDK 17 or 21) ------------------------------
if [[ -x "${JAVA_ENV}/bin/java" ]]; then
  echo "✅ [setup] java already present: $("${JAVA_ENV}/bin/java" -version 2>&1 | head -1)"
else
  echo "☕ [setup] creating conda env with OpenJDK 21 …"
  "${CONDA}" create -y -p "${JAVA_ENV}" -c conda-forge openjdk=21 >/dev/null
  echo "✅ [setup] java installed: $("${JAVA_ENV}/bin/java" -version 2>&1 | head -1)"
fi

# --- 2) Neo4j Community tarball ----------------------------------------------
if [[ -x "${NEO4J_HOME}/bin/neo4j" ]]; then
  echo "✅ [setup] Neo4j already unpacked at ${NEO4J_HOME}"
else
  TARBALL="${FS_ROOT}/neo4j/neo4j-community-${NEO4J_VERSION}-unix.tar.gz"
  if [[ ! -f "${TARBALL}" ]]; then
    echo "📥 [setup] downloading Neo4j ${NEO4J_VERSION} …"
    curl -fL --retry 3 -o "${TARBALL}" \
      "https://dist.neo4j.org/neo4j-community-${NEO4J_VERSION}-unix.tar.gz"
  fi
  echo "📦 [setup] unpacking …"
  tar xzf "${TARBALL}" -C "${FS_ROOT}/neo4j"
  echo "✅ [setup] unpacked"
fi

# --- 3) Configure ------------------------------------------------------------
# Keep the DATA dir OUTSIDE the install dir so the KG survives a Neo4j upgrade
# and is trivially copyable/rsyncable back to the laptop for eyeballing.
DATA_DIR="${FS_ROOT}/neo4j/data"
mkdir -p "${DATA_DIR}" "${FS_ROOT}/neo4j/logs" "${FS_ROOT}/neo4j/run"

CONF="${NEO4J_HOME}/conf/neo4j.conf"
# Rewrite our settings idempotently: strip any previous block, then append.
sed -i '/^# --- groupmembench settings ---/,$d' "${CONF}" 2>/dev/null || true
cat >> "${CONF}" <<EOF
# --- groupmembench settings ---
# Bind to localhost ONLY: the ingest runs on the same node, and we do not want to
# expose an unauthenticated-ish DB on a shared cluster network.
server.default_listen_address=127.0.0.1
server.bolt.listen_address=127.0.0.1:7687
server.http.listen_address=127.0.0.1:7474
# Data outside the install dir → survives upgrades, easy to copy to the laptop.
server.directories.data=${DATA_DIR}
server.directories.logs=${FS_ROOT}/neo4j/logs
server.directories.run=${FS_ROOT}/neo4j/run
# Modest heap/pagecache: the job asks for 96G, Neo4j needs far less than vLLM.
server.memory.heap.initial_size=2g
server.memory.heap.max_size=8g
server.memory.pagecache.size=4g
EOF
echo "✅ [setup] conf written (bolt 127.0.0.1:7687, data ${DATA_DIR})"

# --- 4) Initial password -----------------------------------------------------
# Only settable while the DB has never started; harmless (non-zero) afterwards.
export JAVA_HOME="${JAVA_ENV}"
export PATH="${JAVA_ENV}/bin:${PATH}"
if "${NEO4J_HOME}/bin/neo4j-admin" dbms set-initial-password "${NEO4J_PASSWORD}" 2>/dev/null; then
  echo "🔐 [setup] initial password set"
else
  echo "ℹ️  [setup] password already initialised (fine)"
fi

echo "🎉 [setup] done"
echo "   NEO4J_HOME=${NEO4J_HOME}"
echo "   JAVA_HOME=${JAVA_ENV}"
echo "   data=${DATA_DIR}"
} 2>&1 | tee "${LOG}"
