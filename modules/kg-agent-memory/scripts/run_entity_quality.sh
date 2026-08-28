#!/usr/bin/env bash
# =============================================================================
#  run_entity_quality.sh — what ARE the entities Graphiti extracted?
#
#  Descriptive audit of the merged Finance KG's entity layer: what kind of thing
#  each entity name is, what a fact actually connects (person→concept vs
#  concept→concept), hub nodes, and near-duplicates.
#
#  Needs the local merged-KG container:  docker start neo4j-gmb-full
# =============================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${REPO_ROOT}/logs"
LOG_FILE="${LOG_DIR}/entity_quality.log"
PY="${REPO_ROOT}/.venv/bin/python"      # this repo's OWN venv (house rule 1)

mkdir -p "${LOG_DIR}"

{
  echo "════════════════════════════════════════════════════════════════"
  echo "🔬 Entity-layer audit   ($(date '+%F %T'))"
  echo "════════════════════════════════════════════════════════════════"

  if ! docker ps --format '{{.Names}}' | grep -qx "neo4j-gmb-full"; then
    echo "❌ container 'neo4j-gmb-full' is not running."
    echo "   start it with:  docker start neo4j-gmb-full"
    exit 1
  fi
  echo "✅ merged-KG container is up"

  PYTHONUNBUFFERED=1 "${PY}" -u "${REPO_ROOT}/analysis/entity_quality.py"
  echo "✅ done"
} 2>&1 | tee -a "${LOG_FILE}"
