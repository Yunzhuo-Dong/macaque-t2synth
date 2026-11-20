#!/usr/bin/env python
"""10_myelin.py — T1w/T2w myelin analysis of the final model, on the cortical surface.

Compares three T1w/T2w maps per subject, each from the subject's N4 T1w and:
  real        the acquired T2w, corrected with the T1w's own N4 field (04b_shared_bias_target.py),
              so the ratio equals raw T1w / raw T2w as in HCP-style myelin mapping
  synthetic   the fine-tuned model of the fold in which the subject was a test subject
              (trained on that shared-field target; --model-tag)
  LUT         T1w-only control: mean real T2w per T1w-intensity bin, fitted on the fold's
              training subjects

Pipeline (GAN-MAT analysis, Park et al. 2024, adapted to the macaque):
  prepare   Yerkes19 v1.2 32k_fs_LR white / pial surfaces -> NMT v2.1 (ANTs SyN between the
            Yerkes19 cortical ribbon and the NMT cortical mask; an existing registration at
            <out-root>/yerkes19/nmt2yerkes_ or --reg-prefix is reused) -> each subject's native space through its NMT registration
  subject   ratio = minmax(N4 T1w) / minmax(T2w) clamped to 0-100 (GAN-MAT preprocessing.sh);
            16 equivolumetric surfaces between pial and white, ends dropped -> 14 depths;
            trilinear sampling at every vertex
  group     * vertex maps: depth-averaged ratio / subject's cortical mean, 2 mm FWHM on the
              midthickness surface, group mean (HCP-style surface myelin map)
            * parcels: CHARM level 5 split by hemisphere; parcel profile = mean of vertices
              after removing those > 3 scaled MAD (build_mpc.py); moments (mean, SD, skewness,
              kurtosis), MPC (residual correlation given the mean profile, Fisher z) and the
              first diffusion-map gradient (normalised angle, sparsity 0.9, alpha 0.5)
            * per subject: r of every feature with the same subject's real feature, spin test
              (1000 rotations of parcel centroids on the Yerkes19 sphere); individual
              deviations r(x - m, real - m), m = mean real features of the other folds'
              subjects; paired Wilcoxon synthetic vs LUT
Usage:
    source ganmat_env.sh
    Y=/path/to/fsaverage_LR32k          # MacaqueYerkes19_v1.2.*.32k_fs_LR files
    "$VENV_PY" 10_myelin.py --prepare --yerkes-dir "$Y"
    "$VENV_PY" 10_myelin.py --list-subs | xargs -P 4 -I{} "$VENV_PY" 10_myelin.py --yerkes-dir "$Y" --subs {}
    "$VENV_PY" 10_myelin.py --group --yerkes-dir "$Y"
"""
import argparse
import glob
import importlib.util
import json
import os
import re
import subprocess
import sys
import time
import warnings

import nibabel as nib
import numpy as np
import pandas as pd
from matplotlib.collections import PolyCollection
from matplotlib.colors import LinearSegmentedColormap
from scipy import ndimage, sparse, special, stats
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("e2", os.path.join(HERE, "05_e2_zeroshot.py"))
E2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(E2)
plt = E2.P.plt
warnings.filterwarnings("ignore", "Mean of empty slice", RuntimeWarning)
PREFIX = "MacaqueYerkes19_v1.2"
SYNTH = "synthT2w_fast_human_matched_native.nii.gz"
METHODS = ["real", "synthetic", "LUT"]
LABELS = {"real": "real T2w", "synthetic": "synthetic T2w (fine-tuned)", "LUT": "T1w only (LUT)"}
FEATURES = ["mean", "sd", "skew", "kurt", "G1"]
HCP_CMAP = LinearSegmentedColormap.from_list("hcp_like", [
    (0.00, "#000000"), (0.12, "#3a1a46"), (0.30, "#8a87c6"), (0.45, "#9aa0c8"),
    (0.55, "#3ccf4f"), (0.68, "#c8e632"), (0.78, "#ffe040"), (0.88, "#ff9a1a"), (1.00, "#ff2a1a")])
VIEWS = {"lh_lateral": ("lh", np.array([-1.0, 0, 0])), "rh_lateral": ("rh", np.array([1.0, 0, 0])),
         "lh_medial": ("lh", np.array([1.0, 0, 0])), "rh_medial": ("rh", np.array([-1.0, 0, 0]))}


def parse_args():
    env = os.environ.get
    proj = env("PROJECT_DIR", ".")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--yerkes-dir", help="folder with the MacaqueYerkes19_v1.2.*.32k_fs_LR files")
    p.add_argument("--prepare", action="store_true", help="template surfaces -> NMT space (once)")
    p.add_argument("--subs", nargs="*")
    p.add_argument("--list-subs", action="store_true")
    p.add_argument("--group", action="store_true")
    p.add_argument("--prep-root", default=os.path.join(proj, "derivatives", "prep"))
    p.add_argument("--e3-root", default=os.path.join(proj, "derivatives", "e3"))
    p.add_argument("--out-root", default=os.path.join(proj, "derivatives", "myelin"))
    p.add_argument("--nmt-dir", default=env("NMT_DIR"))
    p.add_argument("--model-tag", default="official_long200_sharedF")
    p.add_argument("--splits")
    p.add_argument("--reg-prefix", help="existing NMT->Yerkes19 registration "
                   "(default: <out-root>/yerkes19/nmt2yerkes_ if present, else computed)")
    p.add_argument("--n-surfs", type=int, default=14)
    p.add_argument("--charm-level", type=int, default=5)
    p.add_argument("--min-vertices", type=int, default=30)
    p.add_argument("--n-spins", type=int, default=1000)
    p.add_argument("--fwhm-mm", type=float, default=2.0, help="smoothing of the vertex maps")
    p.add_argument("--wall-rings", type=int, default=2, help="vertex rings next to the medial wall left out")
    p.add_argument("--threads", default=env("ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS", "8"))
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def wdir(a):
    d = os.path.join(a.out_root, "final")
    os.makedirs(d, exist_ok=True)
    return d


def splits_of(a):
    path = a.splits or (sorted(glob.glob(os.path.join(a.e3_root, "splits_*.json"))) or [None])[0]
    if not path:
        sys.exit(f"no split file in {a.e3_root}; did you `source ganmat_env.sh`?")
    return json.load(open(path))


def run(cmd, log):
    with open(log, "a") as fh:
        fh.write("\n$ " + " ".join(map(str, cmd)) + "\n")
        fh.flush()
        r = subprocess.run([str(c) for c in cmd], stdout=fh, stderr=subprocess.STDOUT)
    if r.returncode:
        sys.exit(f"command failed, see {log}")


# ===================================================================== surfaces
def yfile(d, kind, hemi=None):
    name = f"{PREFIX}.{hemi}.{kind}.32k_fs_LR.surf.gii" if hemi else f"{PREFIX}.{kind}.32k_fs_LR.dscalar.nii"
    path = os.path.join(d, name)
    if not os.path.exists(path):
        sys.exit(f"missing {path}")
    return path


def read_surf(path):
    g = nib.load(path)
    return (np.asarray(g.agg_data("NIFTI_INTENT_POINTSET"), float),
            np.asarray(g.agg_data("NIFTI_INTENT_TRIANGLE"), int))


def cortex_mask(ydir, n_vert=32492):
    """Cortical (non-medial-wall) vertices from the brain models of a Yerkes19 dscalar file."""
    c = nib.load(yfile(ydir, "MyelinMap_BC"))
    out = {}
    for name, _, bm in c.header.get_axis(1).iter_structures():
        h = "lh" if "LEFT" in name else "rh" if "RIGHT" in name else None
        if h:
            m = np.zeros(n_vert, bool)
            m[bm.vertex] = True
            out[h] = m
    return out


def inside_surface(Xv, F, shape, k=8):
    """Closed mesh (voxel-index coordinates) -> filled 'inside' mask; padded to the mesh's
    bounding box so a surface reaching beyond the grid still closes."""
    lo = np.minimum(np.floor(Xv.min(0)).astype(int) - 3, 0)
    hi = np.maximum(np.ceil(Xv.max(0)).astype(int) + 4, np.array(shape))
    a_, b_, c_ = Xv[F[:, 0]] - lo, Xv[F[:, 1]] - lo, Xv[F[:, 2]] - lo
    shell = np.zeros(tuple(hi - lo), bool)
    for i in range(k + 1):
        for j in range(k + 1 - i):
            ijk = np.rint(a_ + (i / k) * (b_ - a_) + (j / k) * (c_ - a_)).astype(int)
            shell[tuple(ijk.T)] = True
    filled = ndimage.binary_erosion(ndimage.binary_fill_holes(ndimage.binary_dilation(shell)))
    return filled[tuple(slice(-l_, -l_ + s_) for l_, s_ in zip(lo, shape))]


def points(X_ras, transforms, tmp_prefix, log):
    """RAS points through an ANTs transform chain (LPS on disk)."""
    lps = X_ras * np.array([-1, -1, 1])
    fin, fout = tmp_prefix + "in.csv", tmp_prefix + "out.csv"
    pd.DataFrame({"x": lps[:, 0], "y": lps[:, 1], "z": lps[:, 2], "t": 0.0}).to_csv(fin, index=False)
    cmd = ["antsApplyTransformsToPoints", "-d", 3, "-i", fin, "-o", fout]
    for t in transforms:
        cmd += ["-t", t]
    run(cmd, log)
    out = pd.read_csv(fout)[["x", "y", "z"]].values * np.array([-1, -1, 1])
    os.remove(fin)
    os.remove(fout)
    return out


def nmt_file(nmt_dir, suffix):
    files = glob.glob(os.path.join(nmt_dir, "**", "*.nii.gz"), recursive=True)
    for pat in (rf"^NMT_v[\d.]+_sym_05mm{suffix}\.nii\.gz$", rf"^NMT_v[\d.]+_sym{suffix}\.nii\.gz$"):
        hit = sorted(f for f in files if re.match(pat, os.path.basename(f)))
        if hit:
            return hit[0]
    sys.exit(f"NMT file *{suffix}.nii.gz not found under {nmt_dir}")


def prepare(a):
    out = os.path.join(wdir(a), "yerkes_surfaces_in_NMT.npz")
    log = os.path.join(wdir(a), "prepare.log")
    surf = {h: (read_surf(yfile(a.yerkes_dir, "white", H))[0], read_surf(yfile(a.yerkes_dir, "pial", H))[0],
                read_surf(yfile(a.yerkes_dir, "white", H))[1]) for h, H in (("lh", "L"), ("rh", "R"))}
    reg = a.reg_prefix or os.path.join(a.out_root, "yerkes19", "nmt2yerkes_")
    if not os.path.exists(reg + "1Warp.nii.gz"):
        reg = os.path.join(wdir(a), "nmt2yerkes_")
    if not os.path.exists(reg + "1Warp.nii.gz") or a.force:
        # Yerkes19 ribbon (inside pial, outside white) at 0.5 mm, smoothed; NMT cortical mask likewise
        allx = np.vstack([s[1] for s in surf.values()])
        lo = allx.min(0) - 3.0
        shape = tuple(np.ceil((allx.max(0) + 3.0 - lo) / 0.5).astype(int))
        aff = np.diag([0.5, 0.5, 0.5, 1.0])
        aff[:3, 3] = lo
        rib = np.zeros(shape, np.float32)
        for w, p, F in surf.values():
            rib += inside_surface((p - lo) / 0.5, F, shape) & ~inside_surface((w - lo) / 0.5, F, shape)
        fixed, moving = os.path.join(wdir(a), "yerkes19_ribbon.nii.gz"), os.path.join(wdir(a), "nmt_ribbon.nii.gz")
        nib.save(nib.Nifti1Image(ndimage.gaussian_filter(rib, 0.7), aff), fixed)
        g = nib.load(nmt_file(a.nmt_dir, "_GM_cortical_mask"))
        nib.save(nib.Nifti1Image(ndimage.gaussian_filter(np.asarray(g.dataobj, np.float32), 0.7), g.affine), moving)
        os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = str(a.threads)
        lin = ["-c", "[1000x500x250,1e-7,10]", "-f", "4x2x1", "-s", "2x1x0vox"]
        print("registering NMT cortical ribbon -> Yerkes19 ribbon (a few minutes) ...", flush=True)
        run(["antsRegistration", "-d", 3, "--float", 1, "-o", f"[{reg},{reg}Warped.nii.gz]", "-n", "Linear",
             "-r", f"[{fixed},{moving},1]",
             "-t", "Rigid[0.1]", "-m", f"MeanSquares[{fixed},{moving},1]", *lin,
             "-t", "Affine[0.1]", "-m", f"MeanSquares[{fixed},{moving},1]", *lin,
             "-t", "SyN[0.1,3,0]", "-m", f"MeanSquares[{fixed},{moving},1]",
             "-c", "[100x70x30,1e-7,10]", "-f", "4x2x1", "-s", "2x1x0vox"], log)
    res = {}
    for h, (w, p, F) in surf.items():
        both = points(np.vstack([w, p]), [reg + "1Warp.nii.gz", reg + "0GenericAffine.mat"],
                      os.path.join(wdir(a), f"_{h}_"), log)
        res[f"{h}_white"], res[f"{h}_pial"], res[f"{h}_F"] = both[:len(w)], both[len(w):], F
    np.savez_compressed(out, **res)
    print(f"Yerkes19 surfaces in NMT space (registration {reg}) -> {out}")


# ===================================================================== subject
def vertex_areas(X, F):
    A = 0.5 * np.linalg.norm(np.cross(X[F[:, 1]] - X[F[:, 0]], X[F[:, 2]] - X[F[:, 0]]), axis=1)
    return np.bincount(F.ravel(), np.repeat(A / 3.0, 3), len(X))


def equivolumetric(white, pial, F, n):
    """Waehnert / Wagstyl equivolumetric depths (GAN-MAT generate_equivolumetric_surfaces.py):
    n interior surfaces, pial -> white."""
    aw, ap = vertex_areas(white, F), vertex_areas(pial, F)
    out = []
    for k in range(1, n + 1):
        alpha = k / (n + 1)
        with np.errstate(divide="ignore", invalid="ignore"):
            beta = 1 - (1 / (ap - aw) * (-aw + np.sqrt((1 - alpha) * ap ** 2 + alpha * aw ** 2)))
        beta = np.where(np.isfinite(beta), beta, alpha)
        out.append(pial + (white - pial) * np.clip(beta, 0, 1)[:, None])
    return np.stack(out)


def ganmat_ratio(t1, t2, brain):
    n1 = np.where(brain, t1, 0) / max(float(t1[brain].max()), 1e-12)
    n2 = np.where(brain, t2, 0) / max(float(t2[brain].max()), 1e-12)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.clip(n1 / n2, 0, 100)
    return np.where(brain & np.isfinite(r), r, 0).astype(np.float32)


def get_lut(fold, train, a):
    path = os.path.join(a.out_root, "_lut", f"fold{fold}_lut_sharedF.npy")
    if os.path.exists(path) and not a.force:
        return np.load(path)
    lut = E2.fit_lut(a.prep_root, train, "sharedF")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path[:-4]}.{os.getpid()}.tmp.npy"
    np.save(tmp, lut)
    os.replace(tmp, path)
    return lut


def run_subject(sub, a, sp, surf_nmt):
    out = os.path.join(wdir(a), sub)
    os.makedirs(out, exist_ok=True)
    res_path = os.path.join(out, "profiles.npz")
    if os.path.exists(res_path) and not a.force:
        print(f"[{sub}] done already")
        return
    t0 = time.time()
    fold = next(i for i, f in enumerate(sp["folds"]) if sub in f["test"])
    prep = os.path.join(a.prep_root, sub)
    synth = os.path.join(a.e3_root, a.model_tag, f"fold{fold}", "eval", sub, SYNTH)
    for f in (synth, os.path.join(prep, "native", "T2w_brain_sharedF.nii.gz")):
        if not os.path.exists(f):
            sys.exit(f"[{sub}] missing {f}")
    mimg, brain, sl, t2real, t1 = E2.load_native(a.prep_root, sub, "sharedF")
    T2 = {"real": t2real,
          "synthetic": (np.asarray(nib.load(synth).dataobj, np.float32)[sl] + 1.0) / 2.0,
          "LUT": E2.lut_apply(get_lut(fold, sp["folds"][fold]["train"], a), E2.norm01(t1, brain)).astype(np.float32)}
    ratio = {k: ganmat_ratio(t1, v, brain) for k, v in T2.items()}
    aff = mimg.affine.copy()
    aff[:3, 3] = (mimg.affine @ np.r_[[s.start for s in sl], 1])[:3]
    inv = np.linalg.inv(aff)
    xfm = os.path.join(prep, "xfm", "nmt2nat_")
    chain = [f"[{xfm}0GenericAffine.mat,1]", xfm + "1InverseWarp.nii.gz"]  # NMT points -> native
    log = os.path.join(out, "subject.log")
    res = {}
    for h in ("lh", "rh"):
        w = points(surf_nmt[f"{h}_white"], chain, os.path.join(out, f"_{h}w_"), log)
        p = points(surf_nmt[f"{h}_pial"], chain, os.path.join(out, f"_{h}p_"), log)
        S = equivolumetric(w, p, surf_nmt[f"{h}_F"], a.n_surfs)
        vox = (inv[:3, :3] @ S.reshape(-1, 3).T) + inv[:3, 3:4]
        for k, r in ratio.items():
            res[f"{h}_{k}"] = ndimage.map_coordinates(r, vox, order=1, mode="constant").reshape(S.shape[:2]).astype(np.float32)
        res[f"{h}_thickness"] = np.linalg.norm(p - w, axis=1).astype(np.float32)
    np.savez_compressed(res_path, **res)
    print(f"[{sub}] fold {fold}: {a.n_surfs} depths x {res['lh_real'].shape[1]} vertices ({time.time() - t0:.0f} s)",
          flush=True)


# ===================================================================== features
def moments(P):
    """P: (depths, parcels)."""
    return {"mean": P.mean(0), "sd": P.std(0), "skew": stats.skew(P, axis=0), "kurt": stats.kurtosis(P, axis=0)}


def mpc(P):
    R = np.corrcoef(np.vstack([P.T, P.mean(1)]))
    r, rm = R[:-1, :-1], R[:-1, -1]
    den = np.sqrt(np.clip((1 - rm[:, None] ** 2) * (1 - rm[None, :] ** 2), 1e-12, None))
    z = np.arctanh(np.clip((r - rm[:, None] * rm[None, :]) / den, -0.999999, 0.999999))
    np.fill_diagonal(z, 0)
    return np.nan_to_num(z)


def gradient1(M, sparsity=0.9, alpha=0.5):
    A = M.copy()
    np.fill_diagonal(A, 0)
    A = np.where(A >= np.percentile(A, sparsity * 100, axis=1, keepdims=True), A, 0)
    nrm = np.linalg.norm(A, axis=1, keepdims=True)
    nrm[nrm == 0] = 1
    K = 1 - np.arccos(np.clip((A / nrm) @ (A / nrm).T, -1, 1)) / np.pi
    d = K.sum(1)
    L = K / np.outer(d ** alpha, d ** alpha)
    d2 = L.sum(1)
    w, v = np.linalg.eigh(L / np.sqrt(np.outer(d2, d2)))
    o = np.argsort(w)[::-1]
    w, v = w[o], v[:, o]
    return (v / v[:, [0]])[:, 1] * (w[1] / (1 - w[1]))


def features(P, g1_ref=None):
    f = {k: np.asarray(v, float) for k, v in moments(P).items()}
    Z = mpc(P)
    g = gradient1(Z)
    if g1_ref is not None and np.corrcoef(g, g1_ref)[0, 1] < 0:
        g = -g
    f["G1"] = g
    return f, Z


def parcellate(prof, labs, L):
    c = -1 / (np.sqrt(2) * special.erfcinv(1.5))
    out = np.full((prof.shape[0], len(labs)), np.nan)
    for i, l_ in enumerate(labs):
        d = prof[:, L == l_]
        d = d[:, np.all(np.isfinite(d), 0)]
        if d.shape[1]:
            m = d.mean(0)
            med = np.median(m)
            out[:, i] = d[:, ~(m > med + 3 * c * np.median(np.abs(m - med)))].mean(1)
    return out


def spin_perms(sphere, labs, L, hemi_of, n, seed=0):
    cen = np.array([sphere[h][L[h] == l_].mean(0) for l_, h in zip(labs, hemi_of)])
    cen /= np.linalg.norm(cen, axis=1, keepdims=True)
    R = Rotation.random(n, random_state=np.random.default_rng(seed)).as_matrix()
    refl = np.diag([-1.0, 1, 1])
    perms = np.zeros((n, len(labs)), int)
    for h in ("lh", "rh"):
        idx = np.where(hemi_of == h)[0]
        tree = cKDTree(cen[idx])
        for s in range(n):
            Rs = R[s] if h == "lh" else refl @ R[s] @ refl
            perms[s, idx] = idx[tree.query(cen[idx] @ Rs.T)[1]]
    return perms


def r_spin(x, y, perms):
    ok = np.isfinite(x) & np.isfinite(y)
    r = np.corrcoef(x[ok], y[ok])[0, 1]
    xp = x[perms][:, ok]
    xp = (xp - xp.mean(1, keepdims=True)) / np.maximum(xp.std(1, keepdims=True), 1e-12)
    yz = (y[ok] - y[ok].mean()) / max(y[ok].std(), 1e-12)
    rn = (xp @ yz) / ok.sum()
    return float(r), float((1 + np.sum(np.abs(rn) >= abs(r))) / (1 + len(perms)))


# ===================================================================== figures
def render(ax, X, F, vals, d, vmin, vmax, cmap=HCP_CMAP):
    up = np.array([0.0, 0.0, 1.0])
    right = np.cross(-d, up)
    s2 = np.stack([X @ right, X @ up], 1)
    tri = X[F]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
    shade = 0.35 + 0.65 * np.abs(n @ d)
    with np.errstate(invalid="ignore"):
        fv = np.nanmean(vals[F], axis=1)
    rgb = np.where(np.isfinite(fv)[:, None], cmap(np.clip((np.nan_to_num(fv) - vmin) / (vmax - vmin), 0, 1))[:, :3],
                   np.array([0.78, 0.78, 0.78]))
    o = np.argsort(tri.mean(1) @ d)
    ax.add_collection(PolyCollection(s2[F[o]], facecolors=np.clip(rgb[o] * shade[o, None], 0, 1),
                                     edgecolors="none", antialiased=False))
    ax.set_xlim(s2[:, 0].min(), s2[:, 0].max())
    ax.set_ylim(s2[:, 1].min(), s2[:, 1].max())
    ax.set_aspect("equal")
    ax.axis("off")


def surface_grid(rows, disp, title, path, bar_label, cmap=HCP_CMAP):
    """rows: [(label, {hemi: values}, vmin, vmax)]"""
    fig, axes = plt.subplots(len(rows), 4, figsize=(15, 3.1 * len(rows) + 0.6), squeeze=False)
    for i, (lab, vals, lo, hi) in enumerate(rows):
        for c, view in enumerate(VIEWS):
            h, d = VIEWS[view]
            render(axes[i, c], disp[h][0], disp[h][1], vals[h], d, lo, hi, cmap)
            if i == 0:
                axes[i, c].set_title(view.replace("_", " "))
        axes[i, 0].text(-0.08, 0.5, lab, transform=axes[i, 0].transAxes, rotation=90, ha="center", va="center", fontsize=9)
    fig.suptitle(title, fontsize=12)
    fig.subplots_adjust(left=0.05, right=0.99, top=1 - 0.5 / (3.1 * len(rows) + 0.6) - 0.02, bottom=0.10,
                        wspace=0.02, hspace=0.05)
    cax = fig.add_axes([0.35, 0.065, 0.3, 0.012])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(rows[0][2], rows[0][3]), cmap=cmap), cax=cax,
                      orientation="horizontal")
    cb.set_label(bar_label, fontsize=9)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def hcp_panel(vals, disp, lo, hi, title, path):
    fig = plt.figure(figsize=(9, 7.6))
    for i, view in enumerate(VIEWS):
        h, d = VIEWS[view]
        render(fig.add_axes([0.02 + 0.48 * (i % 2), 0.5 - 0.39 * (i // 2), 0.47, 0.38]), disp[h][0], disp[h][1],
               vals[h], d, lo, hi)
    fig.suptitle(title, fontsize=13)
    cax = fig.add_axes([0.25, 0.06, 0.5, 0.025])
    cb = fig.colorbar(plt.cm.ScalarMappable(cmap=HCP_CMAP), cax=cax, orientation="horizontal", ticks=[0, 1])
    cb.ax.set_xticklabels(["Low", "High"])
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ===================================================================== group
def run_group(a, sp):
    wd = wdir(a)
    subs = [s for s in sp["subjects"] if os.path.exists(os.path.join(wd, s, "profiles.npz"))]
    if len(subs) < 3:
        sys.exit("need at least three finished subjects")
    data = {s: dict(np.load(os.path.join(wd, s, "profiles.npz"))) for s in subs}
    ctx = cortex_mask(a.yerkes_dir)
    hemis = ("lh", "rh")
    geo = {h: read_surf(yfile(a.yerkes_dir, "midthickness", h[0].upper())) for h in hemis}
    for h in hemis:  # drop --wall-rings vertex rings next to the medial wall (white/pial meet there)
        F_ = geo[h][1]
        for _ in range(a.wall_rings):
            bad = np.zeros(len(ctx[h]), bool)
            off = ~ctx[h][F_]
            bad[F_[off.any(1)].ravel()] = True
            ctx[h] &= ~bad
    disp = {h: read_surf(yfile(a.yerkes_dir, "very_inflated", h[0].upper())) for h in hemis}
    sphere = {h: read_surf(yfile(a.yerkes_dir, "sphere", h[0].upper()))[0] for h in hemis}
    fold_of = {s: next(i for i, f in enumerate(sp["folds"]) if s in f["test"]) for s in subs}
    cat = lambda d: np.r_[d["lh"], d["rh"]]  # noqa: E731

    # ---------------- vertex maps (HCP-style)
    def smooth(v, X, F):
        n = len(X)
        A = sparse.coo_matrix((np.ones(6 * len(F)), (np.r_[F[:, 0], F[:, 1], F[:, 2], F[:, 1], F[:, 2], F[:, 0]],
                                                     np.r_[F[:, 1], F[:, 2], F[:, 0], F[:, 0], F[:, 1], F[:, 2]])),
                              shape=(n, n)).tocsr()
        A.data[:] = 1.0
        e = np.linalg.norm(X[F[:, 0]] - X[F[:, 1]], axis=1)
        it = int(round((a.fwhm_mm / 2.3548) ** 2 / (0.43 * float(np.mean(e ** 2)))))
        v = v.copy()
        for _ in range(it):
            ok = np.isfinite(v)
            num = A @ np.where(ok, v, 0) + np.where(ok, v, 0)
            den = A @ ok.astype(float) + ok
            v = np.where(ok, num / np.maximum(den, 1e-9), v)
        return v

    vmap = {}
    for m in METHODS:
        per = []
        for s in subs:
            v = {h: np.where(ctx[h], data[s][f"{h}_{m}"].mean(0), np.nan) for h in hemis}
            mu = np.nanmean(cat(v))
            per.append({h: smooth(v[h] / mu, *geo[h]) for h in hemis})
        vmap[m] = per
    gmap = {m: {h: np.nanmean([x[h] for x in vmap[m]], 0) for h in hemis} for m in METHODS}

    def vr(x, y):
        ok = np.isfinite(x) & np.isfinite(y)
        return float(np.corrcoef(x[ok], y[ok])[0, 1])

    # ---------------- parcels
    surf_nmt = dict(np.load(os.path.join(wd, "yerkes_surfaces_in_NMT.npz")))
    charm = sorted(glob.glob(os.path.join(a.nmt_dir, "**", "CHARM_in_NMT_v*_sym*.nii.gz"), recursive=True))
    if not charm:
        sys.exit(f"no CHARM atlas under {a.nmt_dir}")
    cimg = nib.load(([c for c in charm if "_05mm" in c] or charm)[0])
    lab = np.asarray(cimg.dataobj)
    if lab.ndim > 3:
        lab = lab.reshape(lab.shape[:3] + (-1,))[..., a.charm_level - 1]
    lab = np.rint(lab).astype(np.int32)
    lab = lab[tuple(ndimage.distance_transform_edt(lab == 0, return_indices=True)[1])]
    ci = np.linalg.inv(cimg.affine)
    Lh, labs, hemi_of = {}, [], []
    for hi, h in enumerate(hemis):
        mid = 0.5 * (surf_nmt[f"{h}_white"] + surf_nmt[f"{h}_pial"])
        ijk = np.clip(np.rint((ci[:3, :3] @ mid.T).T + ci[:3, 3]).astype(int), 0, np.array(lab.shape) - 1)
        Lh[h] = np.where(ctx[h], lab[tuple(ijk.T)] + 10000 * (hi + 1), 0)
        u, cnt = np.unique(Lh[h][Lh[h] > 0], return_counts=True)
        labs += list(u[cnt >= a.min_vertices])
        hemi_of += [h] * int((cnt >= a.min_vertices).sum())
    labs, hemi_of = np.array(labs), np.array(hemi_of)
    perms = spin_perms(sphere, labs, Lh, hemi_of, a.n_spins)
    print(f"{len(subs)} subjects, {len(labs)} parcels (CHARM level {a.charm_level}, split by hemisphere), "
          f"{a.n_spins} spins", flush=True)

    def parcels(d, m):
        return np.concatenate([parcellate(np.where(ctx[h][None], d[f"{h}_{m}"], np.nan), labs[hemi_of == h], Lh[h])
                               for h in hemis], 1)

    P = {m: {s: parcels(data[s], m) for s in subs} for m in METHODS}
    _, Zg = features(np.nanmean([P["real"][s] for s in subs], 0))
    g1_ref = gradient1(Zg)
    F, Z = {}, {}
    for m in METHODS:
        F[m], Z[m] = {}, {}
        for s in subs:
            F[m][s], Z[m][s] = features(P[m][s], g1_ref)
    # other folds' mean real features (reference for individual deviations)
    Fo, Zo = {}, {}
    for s in subs:
        oth = [o for o in subs if fold_of[o] != fold_of[s]]
        Fo[s] = {k: np.mean([F["real"][o][k] for o in oth], 0) for k in FEATURES}
        Zo[s] = np.mean([Z["real"][o] for o in oth], 0)
    iu = np.triu_indices(len(labs), 1)

    rows = []
    for m in ("synthetic", "LUT"):
        for i, s in enumerate(subs):
            row = {"subject": s, "fold": fold_of[s], "method": m,
                   "vertex_r": vr(cat(vmap[m][i]), cat(vmap["real"][i])),
                   "MPC_r": float(np.corrcoef(Z[m][s][iu], Z["real"][s][iu])[0, 1]),
                   "MPC_dev_r": float(np.corrcoef((Z[m][s] - Zo[s])[iu], (Z["real"][s] - Zo[s])[iu])[0, 1])}
            for k in FEATURES:
                row[f"{k}_r"], row[f"{k}_p_spin"] = r_spin(F[m][s][k], F["real"][s][k], perms)
                row[f"{k}_dev_r"], row[f"{k}_dev_p_spin"] = r_spin(F[m][s][k] - Fo[s][k], F["real"][s][k] - Fo[s][k], perms)
            rows.append(row)
    ind = pd.DataFrame(rows)
    ind.to_csv(os.path.join(wd, "myelin_subject_metrics.csv"), index=False, float_format="%.4f")

    cols = ["vertex_r"] + [f"{k}_r" for k in FEATURES] + ["MPC_r"] + [f"{k}_dev_r" for k in FEATURES] + ["MPC_dev_r"]
    summ, tests = [], []
    for c in cols:
        x = ind[ind.method == "synthetic"].set_index("subject")[c]
        y = ind[ind.method == "LUT"].set_index("subject")[c].loc[x.index]
        sig = c.replace("_r", "_p_spin")
        row = {"metric": c, "synthetic": f"{x.mean():.3f} ± {x.std():.3f}", "LUT": f"{y.mean():.3f} ± {y.std():.3f}",
               "synthetic_better": f"{int((x > y).sum())}/{len(x)}", "wilcoxon_p": stats.wilcoxon(x, y).pvalue}
        if sig in ind:
            for m in ("synthetic", "LUT"):
                row[f"{m}_p_spin<0.05"] = f"{(ind[ind.method == m][sig] < 0.05).mean():.0%}"
        summ.append(row)
        tests.append({"metric": c, "synthetic_mean": x.mean(), "synthetic_sd": x.std(), "LUT_mean": y.mean(),
                      "LUT_sd": y.std(), "n_synthetic_better": int((x > y).sum()), "n": len(x),
                      "wilcoxon_p": row["wilcoxon_p"]})
    summ = pd.DataFrame(summ)
    pd.DataFrame(tests).to_csv(os.path.join(wd, "myelin_summary.csv"), index=False, float_format="%.4g")

    gF = {m: {k: np.mean([F[m][s][k] for s in subs], 0) for k in FEATURES} for m in METHODS}
    gZ = {m: np.mean([Z[m][s] for s in subs], 0) for m in METHODS}
    grp = []
    for m in ("synthetic", "LUT"):
        row = {"method": m, "vertex_r": vr(cat(gmap[m]), cat(gmap["real"])),
               "MPC_r": float(np.corrcoef(gZ[m][iu], gZ["real"][iu])[0, 1])}
        for k in FEATURES:
            row[f"{k}_r"], row[f"{k}_p_spin"] = r_spin(gF[m][k], gF["real"][k], perms)
        grp.append(row)
    grp = pd.DataFrame(grp)
    grp.to_csv(os.path.join(wd, "myelin_group_metrics.csv"), index=False, float_format="%.4f")
    G = grp.set_index("method")

    # ---------------- figures
    lo, hi = np.nanpercentile(cat(gmap["real"]), [2, 98])
    surface_grid([(LABELS[m] + ("" if m == "real" else f"\nr = {G.loc[m, 'vertex_r']:.2f}"), gmap[m], lo, hi)
                  for m in METHODS],
                 disp, f"T1w/T2w myelin map, group mean of {len(subs)} macaques (vertex level, {a.fwhm_mm:g} mm FWHM)",
                 os.path.join(wd, "myelin_surface_maps.png"), "T1w/T2w / subject cortical mean")
    for m in ("real", "synthetic"):
        hcp_panel(gmap[m], disp, lo, hi, f"T1w/T2w myelin map, {LABELS[m]}\n{len(subs)} macaques",
                  os.path.join(wd, f"myelin_hcp_style_{m}.png"))

    def paint(vals):
        out = {}
        for h in hemis:
            v = np.full(len(Lh[h]), np.nan)
            for i in np.where(hemi_of == h)[0]:
                v[Lh[h] == labs[i]] = vals[i]
            out[h] = v
        return out

    for k, title, cm in (("mean", "intracortical mean T1w/T2w per parcel", HCP_CMAP),
                         ("G1", "microstructural profile covariance, gradient 1", plt.cm.viridis)):
        rows_ = []
        for m in METHODS:
            l_, h_ = np.nanpercentile(gF[m][k], [2, 98])
            extra = "" if m == "real" else f"\nr = {G.loc[m, f'{k}_r']:.2f}, p_spin = {G.loc[m, f'{k}_p_spin']:.3f}"
            rows_.append((LABELS[m] + extra, paint(gF[m][k]), l_, h_))
        surface_grid(rows_, disp, f"{title}, group mean of {len(subs)} macaques (CHARM parcels; each row its own range)",
                     os.path.join(wd, f"myelin_parcel_{k}.png"), "low  →  high", cm)
    order = np.argsort(g1_ref)
    vmax = np.percentile(np.abs(gZ["real"][iu]), 98)
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.6))
    for ax, m in zip(axes, METHODS):
        mm = gZ[m]
        ax.imshow(mm[np.ix_(order, order)], cmap=plt.cm.RdBu_r, vmin=-vmax, vmax=vmax)
        ax.set_title(LABELS[m] + ("" if m == "real" else f"\nr = {G.loc[m, 'MPC_r']:.2f}"), fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(f"group-mean microstructural profile covariance ({len(labs)} parcels, ordered by real gradient 1)")
    fig.tight_layout()
    fig.savefig(os.path.join(wd, "myelin_mpc.png"), dpi=130)
    plt.close(fig)
    fig, axes = plt.subplots(2, 7, figsize=(17, 7.5), sharey="row")
    for row_i, kind in enumerate(("r", "dev_r")):
        keys = (["vertex"] if kind == "r" else []) + FEATURES + ["MPC"]
        for j, k in enumerate(keys):
            ax = axes[row_i, j]
            vals = [ind[ind.method == m][f"{k}_{kind}"].values for m in ("synthetic", "LUT")]
            ax.boxplot(vals, widths=0.55, showfliers=False)
            for q, x in enumerate(vals):
                ax.plot(np.full(len(x), q + 1) + np.random.default_rng(q).uniform(-0.12, 0.12, len(x)), x, ".",
                        ms=5, alpha=0.7, color=("tab:blue", "tab:orange")[q])
            for s in subs:  # paired lines
                yy = [ind[(ind.method == m) & (ind.subject == s)][f"{k}_{kind}"].values[0] for m in ("synthetic", "LUT")]
                ax.plot([1, 2], yy, color="0.75", lw=0.5, zorder=0)
            ax.set_xticks([1, 2])
            ax.set_xticklabels(["synthetic", "LUT"])
            ax.axhline(0, color="k", lw=0.5)
            ax.set_title(k)
        if kind == "dev_r":
            axes[row_i, 6].axis("off")
        axes[row_i, 0].set_ylabel("r with the subject's real-T2w feature" if kind == "r"
                                  else "r of deviations from the\nother subjects' mean")
    fig.suptitle(f"individual level, {len(subs)} macaques (each dot one subject; lines pair the two methods)")
    fig.tight_layout()
    fig.savefig(os.path.join(wd, "myelin_individual.png"), dpi=130)
    plt.close(fig)

    with pd.option_context("display.width", 250, "display.max_columns", 20):
        print("\n== individual level: synthetic vs LUT (r with the same subject's real-T2w feature) ==")
        print(summ.to_string(index=False, float_format=lambda x: f"{x:.2g}"))
        print("\n== group-mean features vs group-mean real ==")
        print(grp.round(3).to_string(index=False))
    print(f"\ntables and figures in {wd}")


def main():
    a = parse_args()
    sp = splits_of(a)
    if a.list_subs:
        print("\n".join(sp["subjects"]))
        return
    if not a.yerkes_dir:
        sys.exit("--yerkes-dir is required")
    if a.prepare:
        prepare(a)
        return
    if a.group:
        run_group(a, sp)
        return
    path = os.path.join(wdir(a), "yerkes_surfaces_in_NMT.npz")
    if not os.path.exists(path):
        sys.exit("run with --prepare first")
    surf_nmt = dict(np.load(path))
    for sub in a.subs or sp["subjects"]:
        run_subject(sub, a, sp, surf_nmt)


if __name__ == "__main__":
    main()
