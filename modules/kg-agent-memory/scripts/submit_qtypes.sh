#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════════
#  submit_qtypes.sh — score the merged full-Finance KG on the FIVE question types
#  we have not measured yet, so we can fill in a full six-column row comparable to
#  Table 2 of the GroupMemBench paper.
#
#  WHY
#  ---
#  So far we only ever scored `knowledge_update` (Graphiti 9/32 = 28.1 %). Table 2
#  of the paper reports SIX categories and an average, so a single column cannot be
#  placed next to it. These jobs produce the other five.
#
#  WHY WE ALSO RE-RUN BM25 (CONTROL_BM25=1)
#  ----------------------------------------
#  We already have BM25 numbers for all six types in `results/bm25_Finance/` from
#  the 2026-07-22 run. We deliberately do NOT reuse them. On `knowledge_update`
#  that older run scored 40.6 % while the cluster run scored 46.9 % on the SAME
#  questions over the SAME corpus — a 6-point gap that can only come from a config
#  difference (agent/judge model, prompt, top-k). Pairing a Graphiti number from
#  this job against a BM25 number from a different job would silently smuggle that
#  gap into the comparison. CONTROL_BM25=1 makes both rows come out of the same
#  job, the same vLLM process and the same judge, so the ONLY difference between
#  them is the retrieval method. That is the whole point of a control.
#
#  WHY RETRIEVE_ONLY=1
#  -------------------
#  The graph already exists at STORE (built by the 8-shard ingest + merge). This
#  flag says "ingest nothing, just map episodes back to message indices and run
#  QA", which turns a multi-hour job into a short one. It also implies --resume
#  inside the retriever, so a scoring run can never wipe the graph it is scoring.
#
#  WHY ONE JOB FOR ALL TYPES, NOT ONE JOB EACH
#  -------------------------------------------
#  The first attempt (2026-08-12) submitted five concurrent jobs, one per question
#  type, all pointed at the SAME merged store. Four of them died instantly with
#  "❌ Neo4j never became ready": **Neo4j takes an EXCLUSIVE lock on its data
#  directory**, so the first job to start wins it and the rest cannot open the
#  database at all. Only `abstention` survived.
#
#  Concurrency here would require giving each job its own COPY of the store, which
#  means duplicating a multi-GB database per question type for no benefit. Instead
#  cluster_ingest_job.sh now accepts QTYPE as a space-separated LIST and loops over
#  it internally — one Neo4j, one vLLM, all types scored in sequence. That is also
#  cheaper: the ~30 GB model is loaded once instead of five times.
#
#  Usage:
#    bash scripts/submit_qtypes.sh              # all five remaining types
#    QTYPES="temporal" bash scripts/submit_qtypes.sh   # just one
# ═══════════════════════════════════════════════════════════════════════════════
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

# The merged, entity-stitched full-Finance graph — the same store the 28.1 %
# knowledge_update number was measured on. Scoring anything else would break
# comparability with the column we already have.
STORE="${STORE:-/fscratch/abuali/neo4j/merged/gmb_finance_full}"
GROUP_ID="${GROUP_ID:-gmb_finance_full}"

# WINDOW must match the original ingest. It is not used to build anything here,
# but the retriever uses it to reconstruct the episode->message-index mapping, and
# a mismatch would map facts back to the wrong messages.
WINDOW="${WINDOW:-5}"

# `knowledge_update` is deliberately absent — already measured, twice.
QTYPES="${QTYPES:-multi_hop temporal term_ambiguity user_implicit abstention}"

# Generous: the longest type is multi_hop (48 questions) and CONTROL_BM25 doubles
# the LLM calls. Most of a short job is still the vLLM warm-up.
WALLTIME="${WALLTIME:-8}"

echo "🎯 scoring the merged full-Finance KG on the remaining question types"
echo "   store    : ${STORE}"
echo "   group    : ${GROUP_ID}"
echo "   window   : ${WINDOW}"
echo "   qtypes   : ${QTYPES}"
echo "   walltime : ${WALLTIME} h  (per job)"
echo "   control  : BM25 re-run inside each job, so both rows share a judge"
echo ""

# Fail loudly rather than submitting a job that will die on a missing file: a
# typo'd qtype would otherwise burn a GPU allocation to discover it.
for QT in ${QTYPES}; do
  QFILE="${ROOT_DIR}/questions/Finance/${QT}.jsonl"
  [[ -f "${QFILE}" ]] || { echo "❌ no such question file: ${QFILE}"; exit 1; }
  echo "   ✓ $(printf '%-16s' "${QT}") $(wc -l < "${QFILE}") questions"
done
echo ""

# ONE job, with the whole list handed to it. The job script loops internally.
echo "📤 submitting a single job covering all of the above …"
STORE_ROOT="${STORE}" \
GROUP_ID="${GROUP_ID}" \
WINDOW="${WINDOW}" \
QTYPE="${QTYPES}" \
DOMAIN=Finance \
CORPUS=0 \
RETRIEVER=graphiti \
RETRIEVE_ONLY=1 \
INGEST_ONLY=0 \
CONTROL_BM25=1 \
STRICT_PROMPT=1 \
  nohup bash scripts/srun_submit.sh all gmb_qtypes 8 1 96G "${WALLTIME}" \
    scripts/cluster_ingest_job.sh > /tmp/submit_qtypes.log 2>&1 &

wait
cat /tmp/submit_qtypes.log
echo ""
echo "✅ all submitted — watch with: squeue -u abuali"
echo "   results land in results/cluster/{graphiti,bm25}_${GROUP_ID}_<qtype>.jsonl"
