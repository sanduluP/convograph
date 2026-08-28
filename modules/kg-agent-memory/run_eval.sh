#!/usr/bin/env bash
# Run RAG baselines (bm25, text_embedding_3_large) on the GroupMemBench
# question set for the Finance and Technology domains, then summarise the
# accuracy per (baseline, question type).
#
# Pipeline:
#   qa         — for each (domain × baseline × qtype) cell, invoke the
#                baseline's run_eval.sh on the pre-converted question file at
#                questions/<Domain>/<qtype>.jsonl, writing a result JSONL to
#                results/<Domain>/<baseline>__<qtype>.jsonl.
#   summarize  — call task_synthesis/summarize_typed_eval.py per domain.
#
# Usage:
#   bash run_eval.sh
#
#   # Override domains, baselines, or question types
#   DOMAINS="Finance" BASELINES="bm25" QTYPES="multi_hop temporal" bash run_eval.sh
#
#   # Re-run only summarisation (skip QA)
#   PHASE=summarize bash run_eval.sh
#
#   # Force a rerun even if the output JSONL already exists
#   FORCE_QA_RERUN=1 bash run_eval.sh

set -uo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

# ── Config ──────────────────────────────────────────────────────────────────
DOMAINS_STR="${DOMAINS:-Finance Technology Healthcare Manufacturing}"
BASELINES_STR="${BASELINES:-bm25 text_embedding_3_large}"
QTYPES_STR="${QTYPES:-multi_hop knowledge_update temporal user_implicit term_ambiguity abstention}"
PHASES_STR="${PHASE:-qa,summarize}"

QUESTIONS_ROOT="${QUESTIONS_ROOT:-questions}"
DATA_ROOT="${DATA_ROOT:-data/final}"
RESULTS_ROOT="${RESULTS_ROOT:-results}"

ENV_FILE="${ENV_FILE:-.env}"
FORCE_QA_RERUN="${FORCE_QA_RERUN:-0}"

# Per-baseline embedding store roots. text_embedding_3_large caches embeddings
# under STORE_DIR keyed by (domain, baseline) so re-runs across qtypes are free.
# bm25 has no persistent store (rebuild is sub-second on CPU).
STORE_ROOT_TEXT_EMBEDDING_3_LARGE="${STORE_ROOT_TEXT_EMBEDDING_3_LARGE:-${REPO_ROOT}/stores}"

# ── Parse arrays ────────────────────────────────────────────────────────────
read -r -a DOMAINS <<<"$DOMAINS_STR"
read -r -a BASELINES <<<"$BASELINES_STR"
read -r -a QTYPES <<<"$QTYPES_STR"
IFS=',' read -r -a PHASES <<<"$PHASES_STR"

has_phase() {
    local target="$1" p
    for p in "${PHASES[@]}"; do [ "$p" = "$target" ] && return 0; done
    return 1
}

upper() { echo "$1" | tr '[:lower:]' '[:upper:]'; }

store_dir_for() {
    local baseline="$1" domain="$2"
    if [ "$baseline" = "bm25" ]; then echo ""; return; fi
    local UC_B; UC_B=$(upper "$baseline")
    local root_var="STORE_ROOT_${UC_B}"
    local root="${!root_var:-${REPO_ROOT}/stores}"
    echo "${root}/${baseline}_${domain}_eval_store"
}

conv_path_for() {
    local domain="$1"
    echo "${REPO_ROOT}/${DATA_ROOT}/${domain}/synthetic_domain_channels_rolevariants_${domain}.json"
}

# ── Phase: qa ───────────────────────────────────────────────────────────────
phase_qa() {
    echo "============================================================"
    echo "Phase: qa"
    echo "============================================================"
    local D B T CONV STORE QF OUT LOG rc n
    for D in "${DOMAINS[@]}"; do
        CONV="$(conv_path_for "$D")"
        if [ ! -f "$CONV" ]; then
            echo "  [skip] $D — conversation JSON missing: $CONV" >&2
            continue
        fi
        mkdir -p "${RESULTS_ROOT}/${D}/logs"
        for B in "${BASELINES[@]}"; do
            STORE="$(store_dir_for "$B" "$D")"
            for T in "${QTYPES[@]}"; do
                QF="${QUESTIONS_ROOT}/${D}/${T}.jsonl"
                OUT="${RESULTS_ROOT}/${D}/${B}__${T}.jsonl"
                LOG="${RESULTS_ROOT}/${D}/logs/qa__${B}__${T}.log"
                if [ ! -f "$QF" ]; then
                    echo "  [$D/$B] [skip] $T — no input ($QF)"
                    continue
                fi
                if [ "$FORCE_QA_RERUN" != "1" ] && [ -s "$OUT" ]; then
                    n="$(wc -l <"$OUT" | tr -d ' ')"
                    echo "  [$D/$B] [skip-qa] $T — already $n lines (FORCE_QA_RERUN=1 to redo)"
                    continue
                fi
                echo "  [$D/$B] [run] $T -> $OUT"
                rc=0
                case "$B" in
                    bm25)
                        env \
                            CONVERSATION_JSON="$CONV" QUESTIONS_JSONL="$QF" \
                            OUTPUT_JSONL="$OUT" ENV_FILE="$ENV_FILE" \
                            bash baselines/bm25/run_eval.sh >"$LOG" 2>&1 || rc=$?
                        ;;
                    text_embedding_3_large)
                        env \
                            CONVERSATION_JSON="$CONV" QUESTIONS_JSONL="$QF" \
                            OUTPUT_JSONL="$OUT" STORE_DIR="$STORE" ENV_FILE="$ENV_FILE" \
                            bash baselines/text_embedding_3_large/run_eval.sh >"$LOG" 2>&1 || rc=$?
                        ;;
                    *)
                        echo "  [$D/$B] [error] unknown baseline: $B"; rc=2
                        ;;
                esac
                if [ "$rc" -ne 0 ]; then
                    echo "  [$D/$B] [warn] $T failed (rc=$rc) — see $LOG"
                fi
            done
        done
    done
}

# ── Phase: summarize ────────────────────────────────────────────────────────
phase_summarize() {
    echo
    echo "============================================================"
    echo "Phase: summarize"
    echo "============================================================"
    local baselines_csv qtypes_csv D
    baselines_csv=$(IFS=,; echo "${BASELINES[*]}")
    qtypes_csv=$(IFS=,; echo "${QTYPES[*]}")
    for D in "${DOMAINS[@]}"; do
        if [ -d "${RESULTS_ROOT}/${D}" ]; then
            PYTHONPATH="$REPO_ROOT" python task_synthesis/summarize_typed_eval.py \
                --results-dir "${RESULTS_ROOT}/${D}" \
                --baselines "$baselines_csv" \
                --question-types "$qtypes_csv" \
                --out-markdown "${RESULTS_ROOT}/${D}/accuracy.md" \
                --out-tsv "${RESULTS_ROOT}/${D}/accuracy.tsv" || true
            echo "  [$D] -> ${RESULTS_ROOT}/${D}/accuracy.md"
        else
            echo "  [$D] no results dir at ${RESULTS_ROOT}/${D}"
        fi
    done
}

# ── Main ────────────────────────────────────────────────────────────────────
echo "Domains      : ${DOMAINS[*]}"
echo "Baselines    : ${BASELINES[*]}"
echo "Qtypes       : ${QTYPES[*]}"
echo "Phases       : ${PHASES[*]}"
echo "Questions    : ${QUESTIONS_ROOT}/<D>/<T>.jsonl"
echo "Data root    : ${DATA_ROOT}/<D>/synthetic_domain_channels_rolevariants_<D>.json"
echo "Results      : ${RESULTS_ROOT}/<D>/<B>__<T>.jsonl"
for D in "${DOMAINS[@]}"; do
    for B in "${BASELINES[@]}"; do
        SD="$(store_dir_for "$B" "$D")"
        [ -n "$SD" ] && echo "  store ${B}/${D}: $SD"
    done
done
echo

if has_phase qa;        then phase_qa; fi
if has_phase summarize; then phase_summarize; fi

echo
echo "Done."
