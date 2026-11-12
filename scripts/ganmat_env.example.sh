# shellcheck shell=bash
# ganmat_env.example.sh — copy to ganmat_env.sh (git-ignored), edit the block below, then
#     source scripts/ganmat_env.sh
# before running any script. It only sets variables; it does not modify files.
#
# Expected layout under WORKSPACE (created by the scripts):
#   data/PRIME-DE_UCDavis_full/   paired T1w/T2w (06_fetch_ucdavis.py)
#   GAN-MAT/                      official GAN-MAT clone + functions/model/model.pth
#   macaque_t2synth/              this project's outputs: derivatives/, logs/
#   software/{fsl,ants}           tools (02_setup_tools.sh installs ANTs here)
#   templates/NMT_v2.1_sym/       NMT v2 macaque template (02_setup_tools.sh)

# ---- EDIT ---------------------------------------------------------------------------
# Option A: a fixed workspace folder
GANMAT_WORKSPACE="${GANMAT_WORKSPACE:-}"
# Option B: locate a data disk by filesystem UUID (robust to changing mount points),
#           with the workspace in GANMAT_USER_DIR on that disk
GANMAT_RESEARCH_UUID="${GANMAT_RESEARCH_UUID:-}"
GANMAT_USER_DIR="${GANMAT_USER_DIR:-}"
# Python with torch, nibabel, scipy, scikit-image, pandas, matplotlib (see requirements.txt)
export VENV_PY="${VENV_PY:-$(command -v python3)}"
# FSL installation (FAST is used for tissue segmentation)
export FSLDIR="${FSLDIR:-${HOME}/fsl}"
# --------------------------------------------------------------------------------------

_ganmat_had_u=0
case $- in *u*) _ganmat_had_u=1; set +u ;; esac

if [ -n "${GANMAT_WORKSPACE}" ]; then
  WORKSPACE="${GANMAT_WORKSPACE}"
elif [ -n "${GANMAT_RESEARCH_UUID}" ]; then
  GANMAT_MOUNT="$(findmnt -rn -o TARGET --source "UUID=${GANMAT_RESEARCH_UUID}" 2>/dev/null | head -n1)"
  if [ -z "${GANMAT_MOUNT}" ]; then
    echo "[ganmat_env] ERROR: disk with UUID ${GANMAT_RESEARCH_UUID} is not mounted." >&2
    export GANMAT_ENV_OK=0
    [ "${_ganmat_had_u}" = 1 ] && set -u
    return 1 2>/dev/null || exit 1
  fi
  export GANMAT_MOUNT
  WORKSPACE="${GANMAT_MOUNT}/${GANMAT_USER_DIR}"
else
  echo "[ganmat_env] ERROR: set GANMAT_WORKSPACE or GANMAT_RESEARCH_UUID in ganmat_env.sh" >&2
  export GANMAT_ENV_OK=0
  [ "${_ganmat_had_u}" = 1 ] && set -u
  return 1 2>/dev/null || exit 1
fi

export WORKSPACE
export DATA_DIR="${DATA_DIR:-${WORKSPACE}/data/PRIME-DE_UCDavis_full}"
export GANMAT_ROOT="${GANMAT_ROOT:-${WORKSPACE}/GAN-MAT}"
export PROJECT_DIR="${PROJECT_DIR:-${WORKSPACE}/macaque_t2synth}"
export SOFTWARE_DIR="${SOFTWARE_DIR:-${WORKSPACE}/software}"
export TEMPLATE_DIR="${TEMPLATE_DIR:-${WORKSPACE}/templates}"

_ganmat_path_append() { case ":${PATH}:" in *":$1:"*) ;; *) PATH="${PATH}:$1" ;; esac; }

# FSL: share/fsl/bin (FSL >= 6.0.6) holds FSL commands only, so FSL's python never
# shadows the project's; appended, not prepended.
if [ -f "${FSLDIR}/etc/fslconf/fsl.sh" ]; then
  . "${FSLDIR}/etc/fslconf/fsl.sh"
  export FSLOUTPUTTYPE=NIFTI_GZ
  if [ -d "${FSLDIR}/share/fsl/bin" ]; then _ganmat_path_append "${FSLDIR}/share/fsl/bin"; else _ganmat_path_append "${FSLDIR}/bin"; fi
fi
if [ -x "${SOFTWARE_DIR}/ants/bin/antsRegistration" ]; then
  export ANTSPATH="${SOFTWARE_DIR}/ants/bin/"
  _ganmat_path_append "${SOFTWARE_DIR}/ants/bin"
fi
export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS="${ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS:-8}"

NMT_DIR=""
for _d in "${TEMPLATE_DIR}"/NMT_v2*/; do [ -d "${_d}" ] && NMT_DIR="${_d%/}"; done
export NMT_DIR
unset _d
export PATH
export GANMAT_ENV_OK=1

if [ "${GANMAT_ENV_QUIET:-0}" != 1 ]; then
  echo "[ganmat_env] WORKSPACE : ${WORKSPACE}"
  echo "[ganmat_env] FSL       : $( [ -x "${FSLDIR}/bin/fast" ] && echo "${FSLDIR}" || echo 'not found')"
  echo "[ganmat_env] ANTs      : $(command -v antsRegistration 2>/dev/null || echo 'not installed')"
  echo "[ganmat_env] NMT       : ${NMT_DIR:-not downloaded}"
  echo "[ganmat_env] python    : ${VENV_PY}"
fi
[ "${_ganmat_had_u}" = 1 ] && set -u
unset _ganmat_had_u
