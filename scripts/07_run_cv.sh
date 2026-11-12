#!/usr/bin/env bash
# 07_run_cv.sh — run the folds of one E3 experiment in parallel, one fold per GPU at a time.
# Each run is an ordinary single-GPU run (effective batch 1), so results are identical to
# running the folds one after another; two GPUs simply halve the wall time.
#
# Usage (inside tmux):
#   bash 07_run_cv.sh                                   # official init, folds 0-4 on GPUs 0,1
#   bash 07_run_cv.sh -- --init scratch                 # the from-scratch control
#   bash 07_run_cv.sh --gpus 0,1 --folds 0,1 -- --lr 2e-5
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GPUS="0,1"; FOLDS="0,1,2,3,4"; EXTRA=()
while [ $# -gt 0 ]; do
  case "$1" in
    --gpus)  GPUS="$2"; shift 2 ;;
    --folds) FOLDS="$2"; shift 2 ;;
    --)      shift; EXTRA=("$@"); break ;;
    *)       echo "usage: bash 07_run_cv.sh [--gpus 0,1] [--folds 0,1,2,3,4] [-- <07_finetune.py args>]"; exit 2 ;;
  esac
done
export GANMAT_ENV_QUIET=1
# shellcheck source=/dev/null
source "${GANMAT_ENV_FILE:-${SCRIPT_DIR}/ganmat_env.sh}" || { echo "cannot load ganmat_env.sh"; exit 1; }
unset CUDA_VISIBLE_DEVICES

IFS=, read -ra G <<< "$GPUS"
IFS=, read -ra F <<< "$FOLDS"
STAMP="$(date +%Y%m%d_%H%M%S)"
mkdir -p "${PROJECT_DIR}/logs"
for gi in "${!G[@]}"; do
  (
    for fi in "${!F[@]}"; do
      [ $(( fi % ${#G[@]} )) -eq "$gi" ] || continue
      f="${F[$fi]}"
      log="${PROJECT_DIR}/logs/e3_fold${f}_gpu${G[$gi]}_${STAMP}.log"
      echo "start fold $f on GPU ${G[$gi]} -> $log"
      if CUDA_VISIBLE_DEVICES="${G[$gi]}" "$VENV_PY" "${SCRIPT_DIR}/07_finetune.py" --fold "$f" "${EXTRA[@]}" > "$log" 2>&1; then
        echo "done   fold $f (GPU ${G[$gi]}): $(grep 'best val MAE' "$log" | tail -1)"
      else
        echo "FAILED fold $f (GPU ${G[$gi]}), see $log"
      fi
    done
  ) &
done
wait
echo "all folds finished"
