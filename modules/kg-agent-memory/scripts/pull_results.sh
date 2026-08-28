#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════════
#  pull_results.sh — copy scored result JSONLs back DOWN from Pegasus.
#
#  WHY THIS EXISTS
#  ---------------
#  `sync_to_cluster.sh` is a one-way PUSH, and it excludes `results/` (via
#  .gitignore) precisely so that `--delete` can never wipe cluster-side outputs.
#  The consequence is that nothing brings results back — every scoring run so far
#  was fetched by hand. This script is that missing half.
#
#  It is deliberately NOT --delete: local results are the record of everything we
#  have ever measured, including runs whose cluster-side copies have since been
#  cleaned up. Never let a pull destroy them.
#
#  Usage:
#    bash scripts/pull_results.sh            # fetch results/cluster/
#    ALL=1 bash scripts/pull_results.sh      # fetch all of results/
# ═══════════════════════════════════════════════════════════════════════════════
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

CLUSTER_HOST="${CLUSTER_HOST:-pegasus}"
CLUSTER_ROOT="${CLUSTER_ROOT:-/home/abuali/projects/GroupMemBench}"

# Default to just results/cluster/ — that is where the job script writes, and it
# keeps the transfer small. ALL=1 widens it when you want everything.
SUBDIR="${SUBDIR:-results/cluster/}"
[[ "${ALL:-0}" == "1" ]] && SUBDIR="results/"

echo "📥 pulling ${SUBDIR} from ${CLUSTER_HOST}"
mkdir -p "${ROOT_DIR}/${SUBDIR}"

rsync --archive --verbose --compress --human-readable \
      "${CLUSTER_HOST}:${CLUSTER_ROOT}/${SUBDIR}" \
      "${ROOT_DIR}/${SUBDIR}"

echo ""
echo "✅ done — local copy at ${ROOT_DIR}/${SUBDIR}"
