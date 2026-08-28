#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════════
#  submit_bm25_domains.sh — run the BM25 baseline on the domains we have not
#  measured yet, so we can compute a FOUR-DOMAIN average directly comparable to
#  the BM25 row of Table 2 in the GroupMemBench paper.
#
#  WHY THIS IS THE FIRST EXPERIMENT, NOT A SIDE QUEST
#  --------------------------------------------------
#  We wanted to place our Graphiti number next to the paper's Table 2. That is only
#  sound if our harness measures the same thing theirs does. BM25 is the instrument
#  check: it appears in both tables and is deterministic and parameter-free, so any
#  gap between their BM25 row and ours comes from the harness and the domain, never
#  from the retriever.
#
#  On Finance alone (measured 2026-08-01, Qwen3-30B) that gap is large:
#      macro 56.44 % for us vs 42.20 % for the paper — +14.2 points,
#      with per-column swings from +39.2 (user_implicit) to -17.2 (temporal).
#  A single domain cannot tell us whether that is "our harness is stronger" or
#  "Finance happens to be easy". Running the other three settles it, and gives a
#  four-domain average that lines up against their published 43.22 exactly.
#
#  WHY FINANCE IS NOT IN THE LIST
#  ------------------------------
#  It has already been run on the CURRENT model (results/bm25_Finance/, 2026-08-01,
#  113/214 = 52.80 % micro). Re-running it would only overwrite a good result with
#  a statistically identical one. NOTE: the copy that sat in the local repo until
#  2026-08-12 was an OLDER run on qwen2.5:32b (36.9 % micro) — it has been archived
#  to results/archive/. Do not confuse the two; the model changed, not the code.
#
#  COST
#  ----
#  Retrieval is BM25 (CPU, instant). The cost is two LLM calls per question, run
#  serially inside the job. Finance's 214 questions took ~19 min plus ~10 min of
#  vLLM warm-up, and the three remaining domains are all smaller. They run as three
#  CONCURRENT jobs, so wall-clock is one domain, not three.
#
#  Usage:
#    bash scripts/submit_bm25_domains.sh
#    DOMAINS="Healthcare" bash scripts/submit_bm25_domains.sh   # just one
# ═══════════════════════════════════════════════════════════════════════════════
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

DOMAINS="${DOMAINS:-Healthcare Manufacturing Technology}"
# Finance's whole run fit in ~30 min; 4 h absorbs a slow node or a queue wait
# without risking a walltime kill partway through a domain.
WALLTIME="${WALLTIME:-4}"

echo "📊 BM25 baseline across domains (instrument check vs the paper's Table 2)"
echo "   domains  : ${DOMAINS}"
echo "   walltime : ${WALLTIME} h (per job, jobs run concurrently)"
echo ""

# Validate before submitting: a missing corpus or question dir would otherwise
# burn a GPU allocation just to discover a typo.
for D in ${DOMAINS}; do
  CONV="${ROOT_DIR}/data/final/${D}/synthetic_domain_channels_rolevariants_${D}.json"
  QDIR="${ROOT_DIR}/questions/${D}"
  [[ -f "${CONV}" ]] || { echo "❌ missing corpus: ${CONV}"; exit 1; }
  [[ -d "${QDIR}" ]] || { echo "❌ missing questions: ${QDIR}"; exit 1; }
  NQ=$(cat "${QDIR}"/*.jsonl | wc -l)
  echo "   ✓ $(printf '%-14s' "${D}") ${NQ} questions"
done
echo ""

for D in ${DOMAINS}; do
  echo "📤 submitting ${D} …"
  # cluster_baseline_job.sh serves vLLM, loops all six question types via
  # run_bm25_baseline.sh, then stops vLLM. DOMAIN is the only knob it needs.
  DOMAIN="${D}" \
    nohup bash scripts/srun_submit.sh all "gmb_bm25_${D}" 8 1 96G "${WALLTIME}" \
      scripts/cluster_baseline_job.sh \
      > "/tmp/submit_bm25_${D}.log" 2>&1 &
  # Stagger so three jobs do not all hit the model directory at the same instant.
  sleep 3
done

wait
echo ""
for D in ${DOMAINS}; do
  echo "──────── ${D} ────────"
  cat "/tmp/submit_bm25_${D}.log"
done
echo ""
echo "✅ submitted — watch with: squeue -u abuali"
echo "   results land in results/bm25_<DOMAIN>/<qtype>.jsonl"
echo "   pull them back with: bash scripts/pull_results.sh   (ALL=1)"
