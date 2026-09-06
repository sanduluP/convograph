#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════════
#  setup_cluster_java.sh — (re)create the JVM that Neo4j needs on Pegasus.
#
#  WHY THIS FILE EXISTS
#  --------------------
#  Neo4j is a JVM application. Every job here that starts a Neo4j store — the
#  shard ingests and cluster_merge_job.sh — needs a Java runtime, and there is
#  NO system java on Pegasus. It lives in a conda env on /fscratch.
#
#  /fscratch is scratch: no backups, and things get cleaned up. On 2026-09-06 an
#  unrelated cleanup removed conda_envs/java21, and the next merge died with
#      Error: JAVA_HOME is not defined correctly.
#  ...40 minutes into a 6-hour job. This script makes recovery one command
#  instead of an archaeology session, and is safe to re-run: if a working java
#  is already there it exits immediately.
#
#  Neo4j 5.26 supports Java 17 and 21; we pin 21.
#
#  Usage:
#      bash scripts/setup_cluster_java.sh          # create if missing
#      FORCE=1 bash scripts/setup_cluster_java.sh  # rebuild even if present
# ═══════════════════════════════════════════════════════════════════════════════
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${REPO_ROOT}/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/setup_cluster_java.log"

FS_ROOT="${FS_ROOT:-/fscratch/abuali}"
JAVA_HOME="${JAVA_HOME_TARGET:-${FS_ROOT}/conda_envs/java21}"
# openjdk 21 from conda-forge — same major version Neo4j 5.26 is tested against.
JAVA_SPEC="${JAVA_SPEC:-openjdk=21}"
CONDA="${CONDA:-${FS_ROOT}/miniforge3/bin/mamba}"

{
  echo "☕ Neo4j JVM setup   $(date '+%Y-%m-%d %H:%M:%S')"
  echo "   target : ${JAVA_HOME}"
  echo "   spec   : ${JAVA_SPEC}"
  echo ""

  # Idempotent: a working java means there is nothing to do. Re-installing would
  # cost minutes and risks disturbing a running job that is using it.
  if [[ "${FORCE:-0}" != "1" ]] && "${JAVA_HOME}/bin/java" -version >/dev/null 2>&1; then
    echo "✅ java already present and working:"
    "${JAVA_HOME}/bin/java" -version 2>&1 | sed 's/^/   /'
    echo "   (FORCE=1 to rebuild anyway)"
    exit 0
  fi

  if [[ ! -x "${CONDA}" ]]; then
    echo "❌ no mamba/conda at ${CONDA}"
    echo "   Set CONDA=/path/to/mamba, or install miniforge on ${FS_ROOT}."
    exit 1
  fi

  echo "📥 creating the env (a few minutes)..."
  # -y so it never blocks on a prompt inside a batch job.
  "${CONDA}" create -y -p "${JAVA_HOME}" -c conda-forge ${JAVA_SPEC}

  echo ""
  echo "🔎 verifying..."
  "${JAVA_HOME}/bin/java" -version 2>&1 | sed 's/^/   /'

  echo ""
  echo "✅ done — cluster_merge_job.sh and the ingest jobs will find it at"
  echo "   JAVA_HOME=${JAVA_HOME}"
} 2>&1 | tee "$LOG_FILE"
