#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════════
#  run_bm25_reproduction.sh — pull the BM25 results down from Pegasus and build the
#  "do we reproduce the paper's BM25 row?" table.
#
#  This is the instrument check that has to pass before any Graphiti number can be
#  placed next to Table 2 of the GroupMemBench paper. See analysis/bm25_reproduction.py
#  for the reasoning; this wrapper just runs it with logging so the result is on disk.
#
#  Usage:
#    bash scripts/run_bm25_reproduction.sh              # pull, then analyse
#    SKIP_PULL=1 bash scripts/run_bm25_reproduction.sh  # analyse what is already local
# ═══════════════════════════════════════════════════════════════════════════════
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

mkdir -p logs
LOG="${ROOT_DIR}/logs/bm25_reproduction.log"

# Rule 1: this repo's OWN venv, never another project's.
PY="${ROOT_DIR}/.venv/bin/python"
[[ -x "${PY}" ]] || PY="python3"

# Human-facing notes live in the Obsidian vault, NOT in a repo-local reports/ dir.
# This is a DSA HiWi project, so it belongs under the vault's DSA folder — that is
# the one place Faris actually reads, and a markdown table buried in the repo is a
# table nobody sees.
VAULT="${VAULT:-/home/faris/Documents/Obsidian Vault/🧮  DSA}"
REPORT="${REPORT:-${VAULT}/📊 BM25 reproduction — ours vs GroupMemBench.md}"
mkdir -p "$(dirname "${REPORT}")"
# Never overwrite a previous result in place — archive it first. The archive lives
# beside the note inside the vault, so the history travels with the note.
if [[ -f "${REPORT}" ]]; then
  ARCHIVE_DIR="$(dirname "${REPORT}")/archive"
  mkdir -p "${ARCHIVE_DIR}"
  STAMP="$(date +%Y-%m-%d-%H%M%S)"
  cp "${REPORT}" "${ARCHIVE_DIR}/BM25 reproduction ${STAMP}.md"
  echo "🗄️  archived previous note → ${ARCHIVE_DIR}/BM25 reproduction ${STAMP}.md"
fi

{
  echo "🔬 BM25 reproduction check — $(date '+%Y-%m-%d %H:%M:%S')"
  echo ""

  if [[ "${SKIP_PULL:-0}" != "1" ]]; then
    echo "📥 pulling results from the cluster …"
    ALL=1 bash "${ROOT_DIR}/scripts/pull_results.sh"
    echo ""
  else
    echo "⏭️  SKIP_PULL=1 — analysing the local copy as-is"
    echo ""
  fi

  echo "📊 building the comparison table …"
  PYTHONUNBUFFERED=1 "${PY}" -u "${ROOT_DIR}/analysis/bm25_reproduction.py" \
      --results-root "${ROOT_DIR}/results" \
      --out "${REPORT}"
  echo ""
  echo "✅ done — note written into the Obsidian vault:"
  echo "   ${REPORT}"
} 2>&1 | tee "${LOG}"
