#!/bin/bash
# logging.sh — shared, modular, timestamped logging for GroupMemBench cluster
# scripts (copied verbatim from the kggen-eval drill, rule 3).
#
# Usage (inside a script):
#   source "$(dirname "$0")/lib/logging.sh"
#   init_log serve_vllm          # tees all output to logs/serve_vllm/serve_vllm_<ts>.log
#
# Each run gets its own timestamped file, so re-runs never overwrite. The log
# base is <repo>/logs by default, or $GMB_LOG_DIR if set.

init_log() {
    local category="$1"
    local repo_root base log_dir ts
    repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
    base="${GMB_LOG_DIR:-${repo_root}/logs}"
    log_dir="${base}/${category}"
    mkdir -p "$log_dir"
    ts="$(date +%Y%m%d_%H%M%S)"
    export LOG_FILE="${log_dir}/${category}_${ts}.log"
    exec > >(tee -a "$LOG_FILE") 2>&1
    echo "📄 Log → ${LOG_FILE}"
}

# log_path <category> — echo a fresh timestamped log path (creating its dir),
# WITHOUT redirecting the whole script.
log_path() {
    local category="$1" repo_root base log_dir
    repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
    base="${GMB_LOG_DIR:-${repo_root}/logs}"
    log_dir="${base}/${category}"
    mkdir -p "$log_dir"
    echo "${log_dir}/${category}_$(date +%Y%m%d_%H%M%S).log"
}
