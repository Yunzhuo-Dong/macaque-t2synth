#!/usr/bin/env bash
# 02_setup_tools.sh — Step 1b: no-sudo tool setup for macaque preprocessing.
#
# Steps (each one is skipped if already done):
#   fsl     (optional, --only fsl) repair a $HOME/software/fsl symlink to an FSL copy in $SOFTWARE_DIR
#           FSL itself is installed with the official FSL installer.
#   ants    install ANTs release binaries to $SOFTWARE_DIR/ants-<ver> (+ symlink $SOFTWARE_DIR/ants)
#   nmt     download the symmetric NMT v2 macaque template to $TEMPLATE_DIR
#   python  pip-install missing analysis packages into $VENV_PY's venv,
#           with numpy / torch / nibabel pinned to their current versions
#   afni    OPTIONAL (only with --with-afni): user-level AFNI in $SOFTWARE_DIR/abin
#
# Never uses sudo and never edits ~/.bashrc itself; `source ganmat_env.sh` puts the
# tools on PATH. (AFNI's own installer may append to dotfiles; they are backed up first.)
#
# Usage:
#   bash 02_setup_tools.sh --dry-run         show what would be done
#   bash 02_setup_tools.sh                   fsl + ants + nmt + python
#   bash 02_setup_tools.sh --only ants       run a single step
#   bash 02_setup_tools.sh --with-afni       also attempt AFNI
#   ANTS_VERSION=x.y.z bash 02_setup_tools.sh --only ants   pin an ANTs release
set -uo pipefail

usage() { awk 'NR>1 && /^#/ {sub(/^# ?/, ""); print; next} NR>1 {exit}' "$0"; }

DRY_RUN=0; WITH_AFNI=0; ONLY=""
while [ $# -gt 0 ]; do
  case "$1" in
    --dry-run)   DRY_RUN=1 ;;
    --with-afni) WITH_AFNI=1 ;;
    --only)      ONLY="${2:?--only needs a step name}"; shift ;;
    -h|--help)   usage; exit 0 ;;
    *)           echo "unknown argument: $1"; usage; exit 2 ;;
  esac
  shift
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STAMP="$(date +%Y%m%d_%H%M%S)"
export GANMAT_ENV_QUIET=1
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/ganmat_env.sh" || { echo "research disk not mounted; aborting"; exit 1; }

mkdir -p "${PROJECT_DIR}/logs"
LOG="${PROJECT_DIR}/logs/setup_${STAMP}.log"
exec > >(tee "$LOG") 2>&1
echo "GAN-MAT macaque project — tool setup  ${STAMP}  (dry-run=${DRY_RUN})"

have()    { command -v "$1" >/dev/null 2>&1; }
section() { printf '\n== %s ==\n' "$*"; }
run()     { printf '  + %s\n' "$*"; [ "$DRY_RUN" = 1 ] || "$@"; }
fetch()   { # fetch URL DEST
  if have curl; then run curl -fL --retry 3 --connect-timeout 20 -o "$2" "$1"
  elif have wget; then run wget -O "$2" "$1"
  else echo "  need curl or wget"; return 1; fi
}
OS_VER="$( . /etc/os-release 2>/dev/null && echo "${VERSION_ID:-22.04}")"
DL_DIR="${SOFTWARE_DIR}/_downloads"

# ------------------------------------------------------------------------------
step_fsl() {
  local link="${HOME}/software/fsl" target="${SOFTWARE_DIR}/fsl"
  if [ -L "$link" ] && [ -e "$link" ]; then
    echo "  $link -> $(readlink "$link") (working, unchanged)"
  elif [ -d "$link" ]; then
    echo "  $link is a real directory; leaving it untouched"
  else
    [ -x "${target}/bin/flirt" ] || { echo "  no FSL copy at ${target}; cannot repair"; return 1; }
    [ -L "$link" ] && echo "  broken symlink: $link -> $(readlink "$link")"
    run mkdir -p "${HOME}/software"
    run ln -sfn "$target" "$link"
  fi
  [ "$DRY_RUN" = 1 ] && return 0
  local v; v="$(timeout 30 "${link}/bin/flirt" -version 2>&1 | head -1)"
  case "$v" in *[Vv]ersion*) echo "  flirt runs: $v" ;; *) echo "  flirt does not run: $v"; return 1 ;; esac
  # FSL's python scripts may hard-code an absolute interpreter path.
  local f sb interp
  for f in imglob fsl_sub; do
    [ -f "${link}/bin/$f" ] || continue
    sb="$(head -n1 "${link}/bin/$f")"; interp="${sb#\#!}"; interp="${interp%% *}"
    if [ -x "$interp" ]; then echo "  python scripts OK ($f -> $interp)"
    else echo "  WARNING: $f points to missing interpreter $interp"; return 1; fi
    break
  done
}

# ------------------------------------------------------------------------------
step_ants() {
  if [ -x "${SOFTWARE_DIR}/ants/bin/antsRegistration" ]; then
    echo "  already installed: $(readlink -f "${SOFTWARE_DIR}/ants")"; return 0
  fi
  have curl || { echo "  curl is required for this step"; return 1; }
  [ -x "$VENV_PY" ] || { echo "  VENV_PY not executable: $VENV_PY"; return 1; }
  local api="https://api.github.com/repos/ANTsX/ANTs/releases" rel meta
  if [ -n "${ANTS_VERSION:-}" ]; then
    rel="$(curl -fsSL "${api}/tags/v${ANTS_VERSION}" 2>/dev/null || curl -fsSL "${api}/tags/${ANTS_VERSION}")" \
      || { echo "  release ${ANTS_VERSION} not found"; return 1; }
  else
    rel="$(curl -fsSL "${api}/latest")" || { echo "  cannot query the GitHub API"; return 1; }
  fi
  meta="$(printf '%s' "$rel" | "$VENV_PY" -c '
import json, sys, datetime as dt
rel = json.load(sys.stdin)
assets = [a for a in rel.get("assets", []) if a["name"].endswith(".zip") and "X64" in a["name"]]
pick = None
for v in (sys.argv[1], "22.04", "20.04"):
    pick = next((a for a in assets if "ubuntu-" + v in a["name"]), None)
    if pick:
        break
if not pick:
    sys.stderr.write("  no Ubuntu X64 zip among assets:\n")
    for a in rel.get("assets", []):
        sys.stderr.write("    " + a["name"] + "\n")
    sys.exit(1)
pub = rel.get("published_at") or ""
age = -1
if pub:
    age = (dt.datetime.now(dt.timezone.utc) - dt.datetime.fromisoformat(pub.replace("Z", "+00:00"))).days
print(rel.get("tag_name", "?"), pub or "?", age, pick["name"], pick["browser_download_url"])
' "$OS_VER")" || return 1
  local tag published age asset url
  read -r tag published age asset url <<< "$meta"
  echo "  release ${tag}, published ${published} (${age} days ago): ${asset}"
  if [ "$age" -ge 0 ] && [ "$age" -lt 14 ]; then
    echo "  NOTE: this release is under 14 days old; to be conservative, pin an older one with ANTS_VERSION=x.y.z"
  fi

  run mkdir -p "$DL_DIR"
  local zip="${DL_DIR}/${asset}" top
  [ -f "$zip" ] || fetch "$url" "$zip" || return 1
  [ "$DRY_RUN" = 1 ] && return 0

  if have unzip; then
    top="$(unzip -Z1 "$zip" | head -n1 | cut -d/ -f1)"
    [ -d "${SOFTWARE_DIR}/${top}" ] || run unzip -q "$zip" -d "$SOFTWARE_DIR" || return 1
  else
    # Python fallback; restores the permission bits stored in the zip.
    top="$("$VENV_PY" - "$zip" "$SOFTWARE_DIR" <<'PY'
import sys, os, zipfile
zpath, dest = sys.argv[1], sys.argv[2]
with zipfile.ZipFile(zpath) as z:
    top = z.namelist()[0].split("/")[0]
    if not os.path.exists(os.path.join(dest, top)):
        for info in z.infolist():
            z.extract(info, dest)
            mode = (info.external_attr >> 16) & 0o777
            if mode:
                os.chmod(os.path.join(dest, info.filename), mode)
print(top)
PY
)" || return 1
  fi
  run ln -sfn "$top" "${SOFTWARE_DIR}/ants"     # relative link: survives mount-point changes
  "${SOFTWARE_DIR}/ants/bin/antsRegistration" --version 2>&1 | head -n2 | sed 's/^/  /' || return 1
  [ -x "${SOFTWARE_DIR}/ants/bin/N4BiasFieldCorrection" ] || { echo "  N4BiasFieldCorrection missing"; return 1; }
}

# ------------------------------------------------------------------------------
step_nmt() {
  if [ -n "${NMT_DIR:-}" ]; then echo "  already present: $NMT_DIR"; return 0; fi
  have curl || { echo "  curl is required for this step"; return 1; }
  run mkdir -p "$TEMPLATE_DIR" "$DL_DIR"
  local base="https://afni.nimh.nih.gov/pub/dist/atlases/macaque/nmt" ver url tgz top
  for ver in 2.1 2.0; do
    url="${base}/NMT_v${ver}_sym.tgz"
    curl -fsIL --max-time 30 "$url" >/dev/null 2>&1 || continue
    echo "  using NMT v${ver} (symmetric): $url"
    tgz="${DL_DIR}/NMT_v${ver}_sym.tgz"
    [ -f "$tgz" ] || fetch "$url" "$tgz" || return 1
    [ "$DRY_RUN" = 1 ] && return 0
    run tar -xzf "$tgz" -C "$TEMPLATE_DIR" || return 1
    top="$(tar -tzf "$tgz" 2>/dev/null | head -n1 | cut -d/ -f1)"
    echo "  extracted to ${TEMPLATE_DIR}/${top}; NIfTI files:"
    find "${TEMPLATE_DIR}/${top}" -maxdepth 2 -name '*.nii.gz' | sort | head -n 25 | sed 's/^/    /'
    return 0
  done
  echo "  could not reach the NMT download URLs"; return 1
}

# ------------------------------------------------------------------------------
step_python() {
  [ -x "$VENV_PY" ] || { echo "  VENV_PY not executable: $VENV_PY"; return 1; }
  local missing cons
  missing="$("$VENV_PY" - <<'PY'
import importlib.util
want = {"scipy": "scipy", "skimage": "scikit-image", "SimpleITK": "SimpleITK",
        "matplotlib": "matplotlib", "pandas": "pandas", "nibabel": "nibabel"}
print(" ".join(pip for mod, pip in want.items() if importlib.util.find_spec(mod) is None))
PY
)"
  if [ -z "$missing" ]; then echo "  all analysis packages present"; return 0; fi
  echo "  missing: $missing"
  cons="${PROJECT_DIR}/logs/pip_constraints_${STAMP}.txt"
  "$VENV_PY" -m pip freeze 2>/dev/null | grep -iE '^(numpy|torch|nibabel)==' > "$cons"
  echo "  pinned to current versions (pip must not change these):"
  sed 's/^/    /' "$cons"
  # shellcheck disable=SC2086
  run "$VENV_PY" -m pip install -c "$cons" $missing
}

# ------------------------------------------------------------------------------
step_afni() {
  local abin="${SOFTWARE_DIR}/abin" pkg f
  if [ -x "${abin}/3dinfo" ]; then
    echo "  already installed: $("${abin}/3dinfo" -ver 2>/dev/null | head -1)"
  else
    if ! have tcsh; then
      echo "  tcsh is not installed; AFNI's installer and @animal_warper need it."
      echo "  Ask the admin for the Ubuntu ${OS_VER} AFNI system packages (tcsh, libgsl, libmotif,"
      echo "  libglw1-mesa, xvfb, ...) listed in AFNI's install docs, then rerun with --with-afni."
      return 1
    fi
    case "$OS_VER" in 24.*) pkg=linux_ubuntu_24_64 ;; *) pkg=linux_ubuntu_16_64 ;; esac
    for f in "${HOME}/.bashrc" "${HOME}/.cshrc" "${HOME}/.tcshrc"; do
      [ -f "$f" ] && run cp -p "$f" "${f}.bak_afni_${STAMP}"
    done
    run mkdir -p "$DL_DIR"
    fetch "https://afni.nimh.nih.gov/pub/dist/bin/misc/@update.afni.binaries" "${DL_DIR}/@update.afni.binaries" || return 1
    ( cd "$DL_DIR" && run tcsh @update.afni.binaries -package "$pkg" -bindir "$abin" ) || return 1
  fi
  # Ubuntu 22.04 ships libgsl.so.27; AFNI's linux_ubuntu_16_64 build expects libgsl.so.19.
  if ! /sbin/ldconfig -p 2>/dev/null | grep -q 'libgsl\.so\.19'; then
    local g; g="$(ls /usr/lib/x86_64-linux-gnu/libgsl.so.2* 2>/dev/null | sort -V | tail -n1)"
    if [ -n "$g" ]; then
      run mkdir -p "${SOFTWARE_DIR}/afni_libs"
      run ln -sfn "$g" "${SOFTWARE_DIR}/afni_libs/libgsl.so.19"
    else
      echo "  WARNING: no system libgsl; 3dQwarp/@animal_warper will likely fail"
    fi
  fi
  [ "$DRY_RUN" = 1 ] && return 0
  export LD_LIBRARY_PATH="${SOFTWARE_DIR}/afni_libs${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
  local p miss=0
  for p in 3dinfo 3dQwarp 3dAllineate 3dUnifize; do
    if ldd "${abin}/${p}" 2>/dev/null | grep -q 'not found'; then
      echo "  $p has missing libraries:"; ldd "${abin}/${p}" | grep 'not found' | sed 's/^/    /'; miss=1
    fi
  done
  if [ -f "${abin}/@animal_warper" ]; then echo "  @animal_warper present"; else echo "  @animal_warper not found"; miss=1; fi
  [ "$miss" = 0 ] && echo "  AFNI binaries link OK (full check: afni_system_check.py -check_all)"
  return "$miss"
}

# ------------------------------------------------------------------------------
STEPS=(ants nmt python)
[ "$WITH_AFNI" = 1 ] && STEPS+=(afni)
[ -n "$ONLY" ] && STEPS=("$ONLY")

declare -A RESULT
ANY_FAIL=0
for s in "${STEPS[@]}"; do
  section "$s"
  if declare -F "step_$s" >/dev/null; then
    if "step_$s"; then RESULT[$s]=OK; else RESULT[$s]=FAILED; ANY_FAIL=1; fi
  else
    echo "  unknown step: $s"; RESULT[$s]=UNKNOWN; ANY_FAIL=1
  fi
done

section "Summary"
for s in "${STEPS[@]}"; do printf '  %-8s %s\n' "$s" "${RESULT[$s]}"; done
echo "  Log: $LOG"
echo "  Next: bash ${SCRIPT_DIR}/01_audit.sh   (re-audit with the new tools)"
exit "$ANY_FAIL"
