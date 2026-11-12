#!/usr/bin/env bash
# 01_audit.sh — Step 1a: environment audit for the macaque T1w->T2w project.
#
# Read-only: changes nothing on the system. The only thing it writes is its own
# log, under $PROJECT_DIR/logs/ (or /tmp if the research disk is not mounted).
#
# Usage:  bash 01_audit.sh
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STAMP="$(date +%Y%m%d_%H%M%S)"
N_OK=0; N_WARN=0; N_FAIL=0
ok()      { printf '  [ OK ] %s\n' "$*"; N_OK=$((N_OK+1)); }
warn()    { printf '  [WARN] %s\n' "$*"; N_WARN=$((N_WARN+1)); }
bad()     { printf '  [FAIL] %s\n' "$*"; N_FAIL=$((N_FAIL+1)); }
info()    { printf '         %s\n' "$*"; }
section() { printf '\n== %s ==\n' "$*"; }
have()    { command -v "$1" >/dev/null 2>&1; }

# ---- environment (research disk resolved by UUID) ----------------------------
ENV_OK=1
export GANMAT_ENV_QUIET=1
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/ganmat_env.sh" || ENV_OK=0
# Fallbacks so the audit still completes when the research disk is not mounted.
VENV_PY="${VENV_PY:-$(command -v python3)}"
FSLDIR="${FSLDIR:-${HOME}/fsl}"
NMT_DIR="${NMT_DIR:-}"

LOG_DIR=/tmp
if [ "$ENV_OK" = 1 ] && mkdir -p "${PROJECT_DIR}/logs" 2>/dev/null; then
  LOG_DIR="${PROJECT_DIR}/logs"
fi
LOG="${LOG_DIR}/audit_${STAMP}.log"
exec > >(tee "$LOG") 2>&1
echo "GAN-MAT macaque project — environment audit  ${STAMP}"

# ---- 1. host -----------------------------------------------------------------
section "1. Host"
H="$(hostname)"
info "hostname: $H"
OS_VER="$( . /etc/os-release 2>/dev/null && echo "${VERSION_ID:-?}")"
info "OS: $( . /etc/os-release 2>/dev/null && echo "${PRETTY_NAME:-?}")   kernel: $(uname -r)"
info "CPU: $(lscpu 2>/dev/null | awk -F: '/Model name/{sub(/^ +/,"",$2); print $2; exit}')   threads: $(nproc)"
info "RAM: $(free -g | awk '/^Mem:/{print $2}') GiB total, $(free -g | awk '/^Mem:/{print $7}') GiB available"
info "user: $(id -un)"

# ---- 2. research disk --------------------------------------------------------
section "2. Research disk"
if [ "$ENV_OK" = 1 ]; then
  ok "workspace resolved: ${WORKSPACE}"
  if [ -d "$WORKSPACE" ]; then ok "WORKSPACE exists: $WORKSPACE"; else bad "WORKSPACE missing: $WORKSPACE"; fi
  if [ -w "$WORKSPACE" ]; then ok "WORKSPACE writable by $(id -un)"; else bad "WORKSPACE not writable by $(id -un)"; fi
  AVAIL_G="$(df -BG --output=avail "$GANMAT_MOUNT" 2>/dev/null | tail -1 | tr -dc '0-9')"
  if [ "${AVAIL_G:-0}" -ge 200 ]; then ok "free space: ${AVAIL_G} GiB"
  else warn "free space only ${AVAIL_G:-?} GiB (full-cohort preprocessing + checkpoints need more)"; fi
else
  bad "workspace not resolved (check ganmat_env.sh); most checks below are skipped"
fi

# ---- 3. paired example data --------------------------------------------------
section "3. Paired data (first two subjects with T1w + T2w under DATA_DIR)"
DATA_FILES=()
if [ "$ENV_OK" = 1 ]; then
  n_found=0
  while IFS= read -r t2; do
    t1="$(ls "$(dirname "$t2")"/*_T1w.nii* 2>/dev/null | head -1)"
    [ -n "$t1" ] || continue
    ok "$(basename "$t2" | cut -d_ -f1): T1w + T2w found"; DATA_FILES+=("$t1" "$t2")
    n_found=$((n_found + 1)); [ "$n_found" -ge 2 ] && break
  done < <(find "${DATA_DIR}" -name "sub-*_T2w.nii*" 2>/dev/null | sort)
  [ "$n_found" -gt 0 ] || bad "no paired T1w/T2w under ${DATA_DIR} (run 06_fetch_ucdavis.py)"
fi
if [ "${#DATA_FILES[@]}" -gt 0 ] && [ -x "${VENV_PY:-/nonexistent}" ]; then
  info "full-read integrity check (decompresses every volume; takes ~1 min)..."
  NII_OUT="$("$VENV_PY" - "${DATA_FILES[@]}" <<'PY'
import sys, os
import numpy as np, nibabel as nib
for f in sys.argv[1:]:
    name = os.path.basename(f)
    try:
        img = nib.load(f)
        data = np.asarray(img.dataobj, dtype=np.float32)  # full read catches truncated/corrupt gzip
        ax = "".join(nib.aff2axcodes(img.affine))
        zooms = tuple(round(float(z), 5) for z in img.header.get_zooms()[:3])
        c = img.affine @ np.r_[(np.array(img.shape[:3]) - 1) / 2.0, 1.0]
        fin = bool(np.isfinite(data).all())
        print(f"NIIOK {name} shape={img.shape} zooms={zooms} {ax} "
              f"centre_mm=({c[0]:.1f},{c[1]:.1f},{c[2]:.1f}) "
              f"range=[{data.min():.1f},{data.max():.1f}] finite={fin}")
    except Exception as e:
        print(f"NIIBAD {name} {type(e).__name__}: {e}")
PY
)"
  while read -r tag rest; do
    case "$tag" in
      NIIOK)  case "$rest" in *finite=False*) warn "$rest (contains NaN/Inf)";; *) ok "$rest";; esac ;;
      NIIBAD) bad "unreadable: $rest" ;;
      *)      [ -n "$tag" ] && info "$tag $rest" ;;
    esac
  done <<< "$NII_OUT"
fi

# ---- 4. python ---------------------------------------------------------------
section "4. Python environment"
if [ -L "${HOME}/.venvs" ]; then warn "${HOME}/.venvs is a symlink -> $(readlink "${HOME}/.venvs")"; fi
if [ -x "${VENV_PY:-/nonexistent}" ]; then
  ok "VENV_PY: $VENV_PY ($("$VENV_PY" -V 2>&1))"
  PY_OUT="$("$VENV_PY" - <<'PY'
import importlib
required = ["torch", "numpy", "nibabel"]
optional = ["scipy", "skimage", "SimpleITK", "matplotlib", "pandas"]
for m in required + optional:
    tag = "REQ" if m in required else "OPT"
    try:
        mod = importlib.import_module(m)
        print(f"PYOK {m} {getattr(mod, '__version__', '?')}")
    except Exception as e:
        print(f"PYMISS{tag} {m} {type(e).__name__}")
try:
    import torch
    print(f"CUDA {torch.cuda.is_available()} torch_cuda={torch.version.cuda} devices={torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        print(f"TGPU {i}:{p.name.replace(' ', '_')} {p.total_memory / 2**30:.1f}GiB")
except Exception as e:
    print(f"CUDAERR {type(e).__name__}: {e}")
PY
)"
  while read -r tag a b; do
    case "$tag" in
      PYOK)       ok "$a $b" ;;
      PYMISSREQ)  bad "required package missing: $a ($b)" ;;
      PYMISSOPT)  warn "analysis package missing: $a (02_setup_tools.sh --only python installs it)" ;;
      CUDA)       if [ "$a" = True ]; then ok "torch CUDA available, $b"; else bad "torch CUDA not available ($b)"; fi ;;
      TGPU)       info "torch sees GPU $a $b" ;;
      CUDAERR)    bad "torch/CUDA check failed: $a $b" ;;
    esac
  done <<< "$PY_OUT"
else
  bad "VENV_PY not executable: ${VENV_PY:-unset}"
fi
PY_IN_PATH="$(command -v python 2>/dev/null || true)"
case "$PY_IN_PATH" in
  *fsl*) warn "bare 'python' in PATH is FSL's ($PY_IN_PATH); always call \$VENV_PY explicitly" ;;
  "")    info "no bare 'python' in PATH (fine; scripts use \$VENV_PY)" ;;
  *)     info "bare 'python' in PATH: $PY_IN_PATH (scripts use \$VENV_PY regardless)" ;;
esac

# ---- 5. GPU ------------------------------------------------------------------
section "5. GPUs"
if have nvidia-smi; then
  while IFS= read -r l; do info "GPU $l"; done < <(nvidia-smi \
    --query-gpu=index,name,memory.total,memory.used,utilization.gpu,driver_version \
    --format=csv,noheader 2>/dev/null)
  APPS="$(nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader 2>/dev/null)"
  if [ -n "$APPS" ]; then
    warn "other compute processes are running on the GPUs right now:"
    while IFS= read -r l; do info "$l"; done <<< "$APPS"
  else ok "no GPU compute processes running right now"; fi
else
  bad "nvidia-smi not found"
fi

# ---- 6. FSL ------------------------------------------------------------------
section "6. FSL"
if [ -d "$FSLDIR" ]; then ok "FSLDIR: $FSLDIR"; else bad "FSLDIR does not exist: $FSLDIR"; fi
if [ "$ENV_OK" = 1 ]; then
  if [ -x "${SOFTWARE_DIR}/fsl/bin/flirt" ]; then ok "research-disk FSL copy present: ${SOFTWARE_DIR}/fsl"
  else warn "no FSL copy at ${SOFTWARE_DIR}/fsl"; fi
fi
[ -f "${FSLDIR}/etc/fslversion" ] && info "FSL version: $(head -1 "${FSLDIR}/etc/fslversion")"
for t in bet bet2 flirt fast fslmaths; do
  if have "$t"; then ok "$t: $(command -v "$t")"; else bad "$t not on PATH"; fi
done
if have flirt; then
  FV="$(timeout 30 flirt -version 2>&1 | head -1)"
  case "$FV" in *[Vv]ersion*) ok "flirt runs: $FV" ;; *) bad "flirt does not run: $FV" ;; esac
fi
for s in imglob fsl_sub fslpython; do
  f="${FSLDIR}/bin/$s"
  [ -f "$f" ] || continue
  sb="$(head -n1 "$f" 2>/dev/null)"
  case "$sb" in
    '#!/'*)
      interp="${sb#\#!}"; interp="${interp%% *}"
      if [ -x "$interp" ]; then ok "FSL script shebang resolves ($s -> $interp)"
      else bad "FSL script $s points to missing interpreter $interp (restore that path as a symlink or reinstall FSL)"; fi
      break ;;
  esac
done

# ---- 7. ANTs -----------------------------------------------------------------
section "7. ANTs (primary macaque registration/N4 tool)"
ANTS_MISSING=0
for t in antsRegistration antsApplyTransforms N4BiasFieldCorrection antsRegistrationSyN.sh; do
  have "$t" || ANTS_MISSING=1
done
if [ "$ANTS_MISSING" = 0 ]; then
  ok "ANTs found: $(command -v antsRegistration)"
  info "$(antsRegistration --version 2>&1 | head -2 | tr '\n' ' ')"
else
  bad "ANTs not installed or incomplete (fix: 02_setup_tools.sh --only ants)"
fi

# ---- 8. AFNI (optional) ------------------------------------------------------
section "8. AFNI (optional; needed only for @animal_warper)"
if have tcsh; then ok "tcsh: $(command -v tcsh)"; else warn "tcsh not installed (AFNI installer and @animal_warper need it; requires admin)"; fi
if have 3dinfo; then
  ok "AFNI: $(3dinfo -ver 2>/dev/null | head -1)"
  if have @animal_warper; then ok "@animal_warper present"; else warn "@animal_warper not found"; fi
else
  warn "AFNI not installed (optional; ANTs path does not need it)"
fi
if /sbin/ldconfig -p 2>/dev/null | grep -q 'libgsl\.so'; then
  info "system GSL: $(/sbin/ldconfig -p | grep -o 'libgsl\.so\.[0-9]*' | sort -u | tr '\n' ' ')"
else
  info "no system libgsl found"
fi

# ---- 9. NMT template ---------------------------------------------------------
section "9. NMT v2 macaque template"
if [ -n "${NMT_DIR:-}" ]; then
  ok "NMT: $NMT_DIR"
  find "$NMT_DIR" -maxdepth 2 -name '*.nii.gz' 2>/dev/null | sort | head -n 12 | while IFS= read -r f; do info "${f#"$NMT_DIR"/}"; done
else
  bad "NMT v2 not downloaded (fix: 02_setup_tools.sh --only nmt)"
fi

# ---- 10. GAN-MAT -------------------------------------------------------------
section "10. GAN-MAT repository"
if [ "$ENV_OK" = 1 ]; then
  CKPT="${GANMAT_ROOT}/functions/model/model.pth"
  if [ -f "$CKPT" ]; then ok "checkpoint: $CKPT ($(du -h "$CKPT" | cut -f1))"; else bad "checkpoint missing: $CKPT"; fi
  if [ -f "${GANMAT_ROOT}/functions/model/main.py" ]; then ok "inference entry point: functions/model/main.py"
  else bad "functions/model/main.py missing"; fi
  if [ -d "${GANMAT_ROOT}/.git" ] && have git; then
    info "git HEAD: $(git -C "$GANMAT_ROOT" rev-parse --short HEAD 2>/dev/null || echo '?')   modified files: $(git -C "$GANMAT_ROOT" status --porcelain 2>/dev/null | wc -l)"
  fi
  info "functions/model/: $(ls "${GANMAT_ROOT}/functions/model" 2>/dev/null | tr '\n' ' ')"
  TRAIN_HINTS="$(grep -rlE 'optimizer|\.backward\(|netD' --include='*.py' "$GANMAT_ROOT" 2>/dev/null | head -n 8)"
  if [ -n "$TRAIN_HINTS" ]; then
    info "files that look like training code (for step 2):"
    while IFS= read -r f; do info "  ${f#"$GANMAT_ROOT"/}"; done <<< "$TRAIN_HINTS"
  else
    info "no training-like code found in the repo (optimizer/backward/netD)"
  fi
fi

# ---- 11. utilities -----------------------------------------------------------
section "11. Utilities"
for t in tmux git curl wget unzip tar parallel; do
  if have "$t"; then ok "$t"; else warn "$t not found"; fi
done

# ---- 12. network -------------------------------------------------------------
section "12. Network reachability (for downloads in step 1b)"
if have curl; then
  for u in https://api.github.com https://github.com https://afni.nimh.nih.gov https://pypi.org; do
    code="$(curl -sS -o /dev/null -w '%{http_code}' --max-time 15 "$u" 2>/dev/null)"
    case "$code" in 2*|3*) ok "reachable: $u ($code)" ;; *) warn "cannot reach $u (HTTP ${code:-000})" ;; esac
  done
else
  warn "curl missing; network check skipped"
fi

# ---- summary -----------------------------------------------------------------
section "Summary"
echo "  OK: $N_OK   WARN: $N_WARN   FAIL: $N_FAIL"
echo "  OS VERSION_ID: ${OS_VER}"
echo "  Log: $LOG"
