#!/usr/bin/env bash
# =============================================================================
#  env_paths.sh — where this module's venv, model weights and code live.
#  Sourced by every script here so the three of them can never drift apart.
#
#  THE RULE (DFKI storage guidelines, https://pegasus.dfki.de/docs/guidelines/storage/):
#
#      $HOME        SOURCE CODE ONLY. 10 GB limit on Pegasus, backed up every 3 h.
#      /netscratch  virtualenvs, model weights, results. No backup. (Pegasus)
#      /scratch     the same role on unicorn, which has no /netscratch.
#
#  So: code in $HOME, venv and weights on scratch. Never the other way round —
#  a 5 GB venv in $HOME eats a backed-up 10 GB quota with files that are
#  reinstallable in minutes.
#
#  ⚠️  ON UNICORN FARIS HAS TWO USERNAMES, and this is not a typo:
#          /home/abuali      ← the account
#          /scratch/faris    ← his scratch tree
#      They do NOT match. Never derive one path from the other by substituting
#      $USER; both are spelled out below.
# =============================================================================

# Every path is overridable so a second person's checkout still works: this
# module is checked out under more than one account.

# ── The venv ────────────────────────────────────────────────────────────────
# Order: explicit MODULE_VENV, then the scratch convention, then a .venv beside
# the code (which is what a laptop checkout or a fresh host will have).
_DSA_VENV="/scratch/faris/venvs/convograph-graphic-generation"
if   [[ -n "${MODULE_VENV:-}" ]];        then VENV="$MODULE_VENV"
elif [[ -x "$_DSA_VENV/bin/python" ]];   then VENV="$_DSA_VENV"
else                                          VENV="${REPO_ROOT}/.venv"
fi
PY="${VENV}/bin/python"

# ── The model weights (~31 GB of FLUX) ──────────────────────────────────────
# NEVER inside the repo: sync_to_cluster.sh rsyncs with --delete and would wipe
# them on the next sync, costing a 31 GB re-download. NEVER in $HOME either —
# see the rule above, and unicorn's $HOME sits on a filesystem that is 96% full.
# The shared copy is group `dsa` (Faris, Rahul, Priyabanta) with the setgid bit,
# so one copy serves all three instead of three copies of the same weights.
_DSA_MODELS="/scratch/faris/models/huggingface"
if   [[ -n "${HF_HOME:-}" ]];      then :                       # caller decided
elif [[ -d "$_DSA_MODELS" ]];      then export HF_HOME="$_DSA_MODELS"
else                                    export HF_HOME="$(dirname "$REPO_ROOT")/hf-cache"
fi
mkdir -p "$HF_HOME"

# The HF token deliberately does NOT live in HF_HOME: that directory is
# group-readable and a credential has no business in a shared tree. It stays in
# the private ~/.cache/huggingface/token, which huggingface_hub also checks.
