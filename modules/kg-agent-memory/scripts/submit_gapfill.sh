#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════════
#  submit_gapfill.sh — ingest the ~192 windows that are MISSING from the merged
#  full-Finance KG, bringing coverage from 96.8 % to (hopefully) ~100 %.
#
#  WHY
#  ---
#  During the 8-shard ingest, 192 of 6,002 windows failed because the extraction
#  LLM returned unparseable JSON (repetition loop -> truncated string). They were
#  skipped so one bad window could not kill a 13 h job. The result is a graph with
#  a stated 96.8 % coverage, which forces every score we report to carry a "lower
#  bound" caveat. This job removes that caveat.
#
#  HOW
#  ---
#  It runs the SAME ingest against the MERGED store with --resume and NO
#  --window-range. Resume reads back the episode names already present
#  (`gmb_finance_full_w{N}`) and skips them with no LLM call, so the job only pays
#  for the windows that are actually absent. 192 windows at ~44 s is ~2.4 h.
#
#  THE ONE KNOB THAT HAD TO CHANGE
#  -------------------------------
#  GRAPHITI_MAX_CONSECUTIVE_FAILURES is normally 5: five failures in a row means
#  the LLM endpoint or the network died, and continuing would silently produce a
#  holey graph. That heuristic is WRONG here, because this job's entire workload is
#  the windows that already failed once — a run of failures is expected content
#  behaviour, not an outage. Raised to 25: still an unmistakable outage signal out
#  of only ~192 attempts, but tolerant of a clustered batch of genuinely hard
#  windows.
#
#  NOTE ON DETERMINISM: the retries are not byte-identical replays. Temperature is
#  0.2 (not 0), which is exactly the fix that stopped repetition loops from
#  reproducing themselves verbatim — so a window that failed once has a real
#  chance of parsing this time.
#
#  Usage:  bash scripts/submit_gapfill.sh
# ═══════════════════════════════════════════════════════════════════════════════
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

STORE="${STORE:-/fscratch/abuali/neo4j/merged/gmb_finance_full}"
GROUP_ID="${GROUP_ID:-gmb_finance_full}"
WINDOW="${WINDOW:-5}"
WALLTIME="${WALLTIME:-8}"   # ~2.4 h of work; 8 h leaves room for a slow card

echo "🩹 gap-fill submission"
echo "   store    : ${STORE}"
echo "   group    : ${GROUP_ID}"
echo "   window   : ${WINDOW}  (must match the original ingest, or the window"
echo "              numbering shifts and resume matches the wrong episodes)"
echo "   walltime : ${WALLTIME} h"
echo ""

STORE_ROOT="${STORE}" \
GROUP_ID="${GROUP_ID}" \
WINDOW="${WINDOW}" \
CORPUS=0 \
RESUME=1 \
RETRIEVE_ONLY=0 \
INGEST_ONLY=1 \
CONTROL_BM25=0 \
GRAPHITI_MAX_CONSECUTIVE_FAILURES=25 \
  nohup bash scripts/srun_submit.sh all gmb_gapfill 8 1 96G "${WALLTIME}" \
    scripts/cluster_ingest_job.sh > /tmp/submit_gapfill.log 2>&1 &

wait
cat /tmp/submit_gapfill.log
echo ""
echo "✅ submitted — watch with: squeue -u abuali"
