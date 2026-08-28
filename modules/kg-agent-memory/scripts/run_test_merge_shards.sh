#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# run_test_merge_shards.sh — prove the shard-merge mechanics on a tiny synthetic
# two-shard graph BEFORE trusting them with ~40 h of real ingest.
#
# Runs against the LOCAL Docker Neo4j (neo4j-graphiti) on the throwaway group
# `merge_selftest`, which it deletes on the way in and on the way out — so it can
# never touch a real graph. No GPU, no LLM, a couple of seconds.
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail
cd "$(dirname "$0")/.."
REPO_ROOT="$(pwd)"

LOG_DIR="${REPO_ROOT}/logs"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/test_merge_shards.log"

echo "📝 logging to ${LOG_FILE}"

{
  echo ""
  echo "════════════════════════════════════════════════════════════════"
  echo "🧪 merge self-test   $(date '+%Y-%m-%d %H:%M:%S')"
  echo "════════════════════════════════════════════════════════════════"

  # Rule 1: this repo's OWN venv, never another project's.
  PY="${REPO_ROOT}/.venv/bin/python"
  [ -x "${PY}" ] || { echo "❌ no venv at ${REPO_ROOT}/.venv"; exit 1; }

  # The test needs the local Neo4j up; starting it is idempotent.
  if ! docker ps --filter name=neo4j-graphiti --format '{{.Names}}' | grep -q neo4j-graphiti; then
    echo "🐳 starting neo4j-graphiti container …"
    docker start neo4j-graphiti
    sleep 15
  fi
  echo "🐳 Neo4j container is up"

  echo "⚙️  running merge self-test …"
  PYTHONUNBUFFERED=1 "${PY}" -u analysis/test_merge_shards.py
  RC=$?

  if [ "${RC}" -eq 0 ]; then
    echo "✅ merge self-test PASSED"
  else
    echo "❌ merge self-test FAILED (exit ${RC})"
  fi
  exit "${RC}"
} 2>&1 | tee -a "${LOG_FILE}"

# Propagate the Python exit code through the pipe, not tee's.
exit "${PIPESTATUS[0]}"
