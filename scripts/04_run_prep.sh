#!/usr/bin/env bash
# 04_run_prep.sh — run 04_preprocess.py for several subjects in parallel.
# Run it inside tmux; one subject takes ~30-40 min with 8 ITK threads.
#
# Usage:
#   tmux new -s prep
#   bash 04_run_prep.sh sub-032128 sub-032130
#   bash 04_run_prep.sh --force sub-032128        # extra flags are passed through
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${GANMAT_ENV_FILE:-${SCRIPT_DIR}/ganmat_env.sh}"
[ -f "$ENV_FILE" ] || ENV_FILE="${SCRIPT_DIR}/../ganmat_env.sh"
export GANMAT_ENV_QUIET=1
# shellcheck source=/dev/null
source "$ENV_FILE" || { echo "cannot load ganmat_env.sh (set GANMAT_ENV_FILE)"; exit 1; }

FLAGS=(); SUBS=()
for a in "$@"; do case "$a" in sub-*) SUBS+=("$a") ;; *) FLAGS+=("$a") ;; esac; done
[ "${#SUBS[@]}" -gt 0 ] || { echo "usage: bash 04_run_prep.sh [--flags] sub-A sub-B ..."; exit 2; }

export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS="${ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS:-8}"
mkdir -p "${PROJECT_DIR}/logs"
STAMP="$(date +%Y%m%d_%H%M%S)"
declare -A PIDS
for s in "${SUBS[@]}"; do
  log="${PROJECT_DIR}/logs/prep_${s}_${STAMP}.log"
  "$VENV_PY" "${SCRIPT_DIR}/04_preprocess.py" --sub "$s" "${FLAGS[@]}" > "$log" 2>&1 &
  PIDS[$s]=$!
  echo "started $s (pid ${PIDS[$s]}, ${ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS} threads) -> $log"
done

FAILED=0
for s in "${SUBS[@]}"; do
  if wait "${PIDS[$s]}"; then echo "done    $s"; else echo "FAILED  $s (see its log)"; FAILED=1; fi
done
echo "outputs: ${PROJECT_DIR}/derivatives/prep/<sub>/   (QC figures in qc/, summary in manifest.json)"
exit "$FAILED"
