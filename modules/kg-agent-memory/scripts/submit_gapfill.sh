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

# ── the extraction configuration MUST match the original ingest ──────────────
# A gap-fill adds episodes to an EXISTING graph, so any setting that changes what
# the extractor produces has to be the same one the graph was built with. The
# speaker-free corpus was ingested with GRAPHITI_EXCLUDE_SPEAKERS=1 and
# STRICT_PROMPT=1 (see submit_speaker_free_full.sh); filling its gaps without
# those flags would insert 229 windows full of person-rooted facts into the one
# graph whose entire value is that it has none. Nothing would error — the graph
# would just quietly stop meaning what we say it means.
EXCLUDE_SPEAKERS="${EXCLUDE_SPEAKERS:-0}"
STRICT_PROMPT="${STRICT_PROMPT:-0}"

# The guard, because remembering the flag is not a plan.
if [[ "${GROUP_ID}" == *speaker_free* && "${EXCLUDE_SPEAKERS}" != "1" ]]; then
  echo "🚫 '${GROUP_ID}' is a speaker-free graph but EXCLUDE_SPEAKERS=${EXCLUDE_SPEAKERS}."
  echo "   Filling it without that flag would add person-rooted facts to the one"
  echo "   graph whose value is that it has none. Re-run with:"
  echo "       EXCLUDE_SPEAKERS=1 STRICT_PROMPT=1 GROUP_ID=${GROUP_ID} \\"
  echo "       STORE=${STORE} bash scripts/submit_gapfill.sh"
  exit 1
fi
# And the reverse: never strip speakers from a graph that was built with them.
if [[ "${GROUP_ID}" != *speaker_free* && "${EXCLUDE_SPEAKERS}" == "1" ]]; then
  echo "🚫 EXCLUDE_SPEAKERS=1 but '${GROUP_ID}' was ingested WITH speakers."; exit 1
fi

# SLURM job names must be readable in squeue (CLAUDE.md rule 8) - one gap-fill
# looks exactly like another otherwise.
JOB_NAME="gapfill_${GROUP_ID}"

echo "🩹 gap-fill submission"
echo "   store    : ${STORE}"
echo "   group    : ${GROUP_ID}"
echo "   job      : ${JOB_NAME}"
echo "   window   : ${WINDOW}  (must match the original ingest, or the window"
echo "              numbering shifts and resume matches the wrong episodes)"
echo "   speakers : $([[ "${EXCLUDE_SPEAKERS}" == "1" ]] && echo "EXCLUDED (speaker-free)" || echo "included")"
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
GRAPHITI_EXCLUDE_SPEAKERS="${EXCLUDE_SPEAKERS}" \
STRICT_PROMPT="${STRICT_PROMPT}" \
DOMAIN="${DOMAIN:-Finance}" \
  nohup bash scripts/srun_submit.sh all "${JOB_NAME}" 8 1 96G "${WALLTIME}" \
    scripts/cluster_ingest_job.sh > "/tmp/submit_${JOB_NAME}.log" 2>&1 &

wait
cat "/tmp/submit_${JOB_NAME}.log"
echo ""
echo "✅ submitted — watch with: squeue -u abuali"
