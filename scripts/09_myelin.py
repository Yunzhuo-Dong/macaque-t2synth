#!/usr/bin/env python
"""09_myelin.py — downstream T1w/T2w (myelin-proxy) analysis, macaque, native volume space.

For every subject, T1w/T2w ratio maps are built from four T2w versions:
  real      the acquired T2w (reference)
  E3        synthetic T2w of the fine-tuned model of the fold in which the subject was a TEST subject
  LUT       T2w predicted from T1w intensity alone (lookup table fitted on that fold's TRAINING
            subjects): its ratio carries no information beyond T1w, so E3 has to beat it
  zeroshot  synthetic T2w of the untouched human GAN-MAT model (if evaluated)

Macaque-specific choices (instead of the human surface pipeline of the original paper):
  * cortex  = NMT v2 cortical GM mask warped to the subject through its SyN registration
  * parcels = CHARM (Cortical Hierarchy Atlas of the Rhesus Macaque), level --charm-level
  * depth   = relative cortical depth from distance maps (0 = pial side, 1 = white matter),
              in --n-depth bins (macaque cortex is ~2 mm = 6-7 voxels at 0.3 mm, so the
              paper's 14 surfaces are not resolvable)
  * scale   = every T2w version on a zero-preserving scale (real: /P99.9; synthetic: (y+1)/2,
              which inverts the training target 2*T2/P99.9 - 1; LUT fitted on that scale), so
              ratio maps differ only by a global factor; each ratio map is divided by its
              cortical median.

Comparisons with the real-T2w ratio (per subject, and on the group average):
  voxel r; voxel r given T1w = information beyond T1w: since log(T1/T2) = log T1 - log T2 and
  log T1 is shared by all ratio maps, it is the partial correlation of log T2w (synthetic vs
  real) given T1w intensity, binned exactly like the LUT (so the LUT scores 0); relative MAE,
  parcel-mean r, depth-profile moments (mean, SD, skewness, kurtosis) r across parcels,
  MPC similarity (partial correlation of depth profiles, Fisher z) and its first gradient
  (diffusion map embedding).

Usage:
    source ganmat_env.sh
    # per subject (CPU; run several in parallel)
    "$VENV_PY" 09_myelin.py --subs sub-032125 sub-032126
    # all subjects, 4 at a time, then the group analysis
    "$VENV_PY" 09_myelin.py --list-subs | xargs -P 4 -I{} "$VENV_PY" 09_myelin.py --subs {}
    "$VENV_PY" 09_myelin.py --group
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

import nibabel as nib
import numpy as np
import pandas as pd
from scipy import ndimage, stats

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("e2mod", os.path.join(HERE, "05_e2_zeroshot.py"))
E2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(E2)
P = E2.P
METHODS = ["E3", "LUT", "zeroshot"]
SYNTH = "synthT2w_fast_human_matched_native.nii.gz"


def parse_args():
    proj = os.environ.get("PROJECT_DIR", ".")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--subs", nargs="*")
    p.add_argument("--list-subs", action="store_true", help="print the subjects of the split file and exit")
    p.add_argument("--group", action="store_true", help="aggregate finished subjects")
    p.add_argument("--prep-root", default=os.path.join(proj, "derivatives", "prep"))
    p.add_argument("--e3-root", default=os.path.join(proj, "derivatives", "e3"))
    p.add_argument("--out-root", default=os.path.join(proj, "derivatives", "myelin"))
    p.add_argument("--nmt-dir", default=os.environ.get("NMT_DIR"))
    p.add_argument("--model-tag", default="official_long200")
    p.add_argument("--zeroshot-tag", default="zeroshot")
    p.add_argument("--splits", help="default: the single splits_*.json in --e3-root")
    p.add_argument("--charm-level", type=int, default=5)
    p.add_argument("--n-depth", type=int, default=8)
    p.add_argument("--min-parcel-vox", type=int, default=200)
    p.add_argument("--eps", type=float, default=0.02, help="floor of the T2w denominator (zero-preserving units)")
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def find_nmt_file(nmt_dir, pattern):
    files = glob.glob(os.path.join(nmt_dir, "**", "*.nii.gz"), recursive=True)
    hits = sorted(f for f in files if re.match(pattern, os.path.basename(f)))
    full = [f for f in hits if "_05mm" not in os.path.basename(f)]
    if not (full or hits):
        sys.exit(f"NMT file matching {pattern} not found under {nmt_dir}")
    return (full or hits)[0]


def warp_to_native(src, ref, dst, xfm_prefix, interp):
    subprocess.run(["antsApplyTransforms", "-d", "3", "-i", src, "-r", ref, "-o", dst, "-n", interp,
                    "-t", xfm_prefix + "1Warp.nii.gz", "-t", xfm_prefix + "0GenericAffine.mat"],
                   check=True, capture_output=True)


def zp(x, m):
    """Zero-preserving scale: divide by the 99.9th percentile inside the mask."""
    return x / max(float(np.percentile(x[m], 99.9)), 1e-12)


def get_lut(fold, train, a):
    path = os.path.join(a.out_root, "_lut", f"fold{fold}_lut.npy")
    if os.path.exists(path) and not a.force:
        return np.load(path)
    pairs = []
    for s in train:
        _, b, _, t2, t1 = E2.load_native(a.prep_root, s)
        pairs.append((E2.norm01(t1, b)[b], zp(t2, b)[b]))
    lut = E2.lut_from_pairs(pairs)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path[:-4]}.{os.getpid()}.tmp.npy"  # atomic: parallel subjects of one fold may race
    np.save(tmp, lut)
    os.replace(tmp, path)
    return lut


def mpc(profiles):
    """Partial correlation between parcel depth profiles given the mean profile, Fisher z."""
    R = np.corrcoef(np.vstack([profiles, profiles.mean(0)]))
    r, rm = R[:-1, :-1], R[:-1, -1]
    den = np.sqrt(np.clip((1 - rm[:, None] ** 2) * (1 - rm[None, :] ** 2), 1e-12, None))
    pr = np.clip((r - rm[:, None] * rm[None, :]) / den, -0.999999, 0.999999)
    z = np.arctanh(pr)
    np.fill_diagonal(z, 0)
    return np.nan_to_num(z)


def gradient1(M, sparsity=0.9, alpha=0.5):
    """First diffusion-map gradient (BrainSpace-style: row-wise sparsity, normalised angle)."""
    A = M.copy()
    np.fill_diagonal(A, 0)
    thr = np.percentile(A, sparsity * 100, axis=1, keepdims=True)
    A = np.where(A >= thr, A, 0)
    nrm = np.linalg.norm(A, axis=1, keepdims=True)
    nrm[nrm == 0] = 1
    cos = np.clip((A / nrm) @ (A / nrm).T, -1, 1)
    K = 1 - np.arccos(cos) / np.pi
    d = K.sum(1)
    L = K / np.outer(d ** alpha, d ** alpha)
    d2 = L.sum(1)
    w, v = np.linalg.eigh(L / np.sqrt(np.outer(d2, d2)))
    order = np.argsort(w)[::-1]
    w, v = w[order], v[:, order]
    psi = v / v[:, [0]]
    return psi[:, 1] * (w[1] / (1 - w[1]))


def corr(x, y):
    ok = np.isfinite(x) & np.isfinite(y)
    return float(np.corrcoef(x[ok], y[ok])[0, 1]) if ok.sum() > 2 else float("nan")


def moments(prof):
    return {"mean": prof.mean(1), "sd": prof.std(1), "skew": stats.skew(prof, axis=1),
            "kurt": stats.kurtosis(prof, axis=1)}


# ===================================================================== subject
def run_subject(sub, a, sp, nmt):
    fold = next(i for i, f in enumerate(sp["folds"]) if sub in f["test"])
    train = sp["folds"][fold]["train"]
    out = os.path.join(a.out_root, sub)
    os.makedirs(out, exist_ok=True)
    res_path = os.path.join(out, "myelin_subject.npz")
    if os.path.exists(res_path) and not a.force:
        print(f"[{sub}] done already")
        return
    t0 = time.time()
    prep = os.path.join(a.prep_root, sub)
    man = json.load(open(os.path.join(prep, "manifest.json")))
    t1_native = man["inputs"]["T1w"]
    xfm = os.path.join(prep, "xfm", "nmt2nat_")

    # 1) NMT cortex mask, segmentation and CHARM parcels -> native
    gm_f, seg_f, ch_f = (os.path.join(out, f) for f in ("cortex_prob.nii.gz", "nmtseg.nii.gz", "charm.nii.gz"))
    if a.force or not os.path.exists(gm_f):
        warp_to_native(nmt["gm"], t1_native, gm_f, xfm, "Linear")
    if a.force or not os.path.exists(seg_f):
        warp_to_native(nmt["seg"], t1_native, seg_f, xfm, "NearestNeighbor")
    if a.force or not os.path.exists(ch_f):
        cimg = nib.load(nmt["charm"])
        cdat = np.asarray(cimg.dataobj)
        if cdat.ndim > 3:  # one volume per hierarchy level
            cdat = cdat.reshape(cdat.shape[:3] + (-1,))[..., a.charm_level - 1]
        lvl = os.path.join(out, "_charm_level_nmt.nii.gz")
        nib.save(nib.Nifti1Image(cdat.astype(np.int32), cimg.affine), lvl)
        warp_to_native(lvl, t1_native, ch_f, xfm, "NearestNeighbor")
        os.remove(lvl)

    # 2) native data, cropped to the brain bounding box
    mimg, brain, sl, t2, t1 = E2.load_native(a.prep_root, sub)
    crop = lambda f, dt=np.float32: np.asarray(nib.load(f).dataobj, dtype=dt)[sl]  # noqa: E731
    seg, charm = crop(seg_f, np.int16), crop(ch_f, np.int32)
    wm = brain & (seg == 4)
    outside = (~brain) | (seg == 1)
    cortex = brain & (crop(gm_f) > 0.5) & ~wm & (charm > 0)
    zooms = tuple(float(z) for z in mimg.header.get_zooms()[:3])

    # 3) relative cortical depth (0 = pial side, 1 = white matter)
    d_out = ndimage.distance_transform_edt(~outside, sampling=zooms)
    d_wm = ndimage.distance_transform_edt(~wm, sampling=zooms)
    depth = d_out / np.maximum(d_out + d_wm, 1e-6)
    dbin = np.clip((depth * a.n_depth).astype(int), 0, a.n_depth - 1)

    # 4) T2w versions on one zero-preserving scale, and ratio maps
    t1s = zp(t1, brain)
    T2 = {"real": zp(t2, brain)}
    synth = {"E3": os.path.join(a.e3_root, a.model_tag, f"fold{fold}", "eval", sub, SYNTH),
             "zeroshot": os.path.join(a.e3_root, a.zeroshot_tag, f"fold{fold}", "eval", sub, SYNTH)}
    for k, f in synth.items():
        if os.path.exists(f):
            T2[k] = (crop(f) + 1.0) / 2.0  # inverts the target scaling 2*T2/P99.9 - 1
        elif k == "E3":
            sys.exit(f"[{sub}] missing {f}: run 08_eval_cv.py for {a.model_tag} first")
    T2["LUT"] = E2.lut_apply(get_lut(fold, train, a), E2.norm01(t1, brain)).astype(np.float32)
    ratio = {}
    for k, v in T2.items():
        r = np.where(cortex, t1s / np.maximum(v, a.eps), 0).astype(np.float32)
        ratio[k] = r / float(np.median(r[cortex]))

    # 5) parcels x depth profiles
    labs, cnt = np.unique(charm[cortex], return_counts=True)
    labs = labs[cnt >= a.min_parcel_vox]
    prof = {k: np.full((len(labs), a.n_depth), np.nan) for k in ratio}
    pmean = {k: np.full(len(labs), np.nan) for k in ratio}
    lab_idx = charm[cortex]
    bins_c = dbin[cortex]
    for i, lab in enumerate(labs):
        sel = lab_idx == lab
        for k, r in ratio.items():
            rv = r[cortex][sel]
            pmean[k][i] = rv.mean()
            for b in range(a.n_depth):
                bb = bins_c[sel] == b
                if bb.sum() >= 10:
                    prof[k][i, b] = rv[bb].mean()
    keep = np.all(np.isfinite(prof["real"]), axis=1)

    # 6) per-subject metrics
    rows = []
    tb = E2.norm01(t1, brain)[cortex]  # the LUT's own T1w binning -> LUT residual is exactly 0
    logt2 = {k: np.log(np.maximum(v[cortex], a.eps)) for k, v in T2.items()}
    rr = ratio["real"][cortex]
    mpc_real = mpc(prof["real"][keep])
    mom_real = moments(prof["real"][keep])
    for k in [m for m in METHODS if m in ratio]:
        rk = ratio[k][cortex]
        dk, dr = E2.given_t1_residual(logt2[k], tb), E2.given_t1_residual(logt2["real"], tb)
        mk = mpc(prof[k][keep])
        iu = np.triu_indices(keep.sum(), 1)
        row = {"subject": sub, "fold": fold, "method": k, "n_cortex_vox": int(cortex.sum()),
               "n_parcels": int(keep.sum()),
               "voxel_r": corr(rk, rr),
               "voxel_r_given_T1": float(np.corrcoef(dk, dr)[0, 1]) if dk.std() > 1e-9 else 0.0,
               "voxel_rel_MAE": float(np.mean(np.abs(rk - rr)) / np.mean(rr)),
               "parcel_r": corr(pmean[k][keep], pmean["real"][keep]),
               "MPC_r": corr(mk[iu], mpc_real[iu])}
        mom_k = moments(prof[k][keep])
        for name in mom_real:
            row[f"moment_{name}_r"] = corr(mom_k[name], mom_real[name])
        rows.append(row)
    pd.DataFrame(rows).to_csv(os.path.join(out, "myelin_subject_metrics.csv"), index=False, float_format="%.5f")
    np.savez_compressed(res_path, labels=labs[keep], **{f"prof_{k}": prof[k][keep] for k in prof},
                        **{f"pmean_{k}": pmean[k][keep] for k in pmean})

    # 7) QC figure (native, cropped)
    aff = mimg.affine.copy()
    aff[:3, 3] = (mimg.affine @ np.r_[[s.start for s in sl], 1])[:3]
    can = lambda arr: P.canon(nib.Nifti1Image(np.asarray(arr, np.float32), aff))  # noqa: E731
    lo, hi = np.percentile(rr, [2, 98])
    c = P.centroid(can(cortex.astype(np.uint8)) > 0)
    qrows = [{"img": can(t1s * brain), "label": "T1w", "contour": can(cortex.astype(np.uint8))},
             {"img": can(np.where(cortex, depth, 0)), "label": "cortical depth (0 pial, 1 WM)", "vmin": 0, "vmax": 1,
              "cmap": "viridis"}]
    for k in ["real"] + [m for m in METHODS if m in ratio]:
        qrows.append({"img": can(ratio[k]), "label": f"T1w/T2w, {k} T2w", "vmin": lo, "vmax": hi, "cmap": "magma"})
    P.qc_figure(os.path.join(out, "qc_myelin.png"), qrows, c,
                f"{sub}: T1w/T2w in cortex (each map / its cortical median), fold {fold}")
    print(f"[{sub}] {int(cortex.sum())} cortex voxels, {int(keep.sum())} parcels, "
          + ", ".join(f"{r['method']} voxel r {r['voxel_r']:.3f} given-T1 {r['voxel_r_given_T1']:.3f} "
                      f"parcel r {r['parcel_r']:.3f}" for r in rows) + f" ({time.time() - t0:.0f} s)", flush=True)


# ======================================================================= group
def run_group(a):
    files = sorted(glob.glob(os.path.join(a.out_root, "sub-*", "myelin_subject.npz")))
    if not files:
        sys.exit("no finished subjects")
    subj = pd.concat([pd.read_csv(os.path.join(os.path.dirname(f), "myelin_subject_metrics.csv")) for f in files])
    subj.to_csv(os.path.join(a.out_root, "myelin_subject_metrics_all.csv"), index=False, float_format="%.5f")
    data = [dict(np.load(f)) for f in files]
    common = sorted(set.intersection(*[set(d["labels"].tolist()) for d in data]))
    methods = ["real"] + [m for m in METHODS if all(f"prof_{m}" in d for d in data)]

    def stack(key):
        out = []
        for d in data:
            idx = {l: i for i, l in enumerate(d["labels"].tolist())}
            out.append(d[key][[idx[l] for l in common]])
        return np.stack(out)
    prof = {m: stack(f"prof_{m}") for m in methods}        # subjects x parcels x depth
    pmean = {m: stack(f"pmean_{m}").mean(0) for m in methods}
    mpcs = {m: np.mean([mpc(p) for p in prof[m]], axis=0) for m in methods}
    grads = {m: gradient1(mpcs[m]) for m in methods}
    mom = {m: moments(prof[m].mean(0)) for m in methods}
    iu = np.triu_indices(len(common), 1)
    grp = []
    for m in methods[1:]:
        g = corr(grads[m], grads["real"])
        row = {"method": m, "n_subjects": len(data), "n_parcels": len(common),
               "group_parcel_r": corr(pmean[m], pmean["real"]),
               "group_MPC_r": corr(mpcs[m][iu], mpcs["real"][iu]),
               "group_gradient1_r": abs(g)}
        for name in mom["real"]:
            row[f"group_moment_{name}_r"] = corr(mom[m][name], mom["real"][name])
        grp.append(row)
    grp = pd.DataFrame(grp)
    grp.to_csv(os.path.join(a.out_root, "myelin_group_metrics.csv"), index=False, float_format="%.4f")

    metrics = ["voxel_r", "voxel_r_given_T1", "voxel_rel_MAE", "parcel_r", "MPC_r",
               "moment_mean_r", "moment_sd_r", "moment_skew_r", "moment_kurt_r"]
    summ = subj.groupby("method")[metrics].agg(["mean", "std"])
    pairs = []
    from scipy.stats import wilcoxon
    for b in [m for m in ("LUT", "zeroshot") if m in subj["method"].unique()]:
        A = subj[subj.method == "E3"].set_index("subject")
        B = subj[subj.method == b].set_index("subject")
        c = A.index.intersection(B.index)
        for met in metrics:
            x, y = A.loc[c, met].values, B.loc[c, met].values
            better = (x < y) if met == "voxel_rel_MAE" else (x > y)
            p = wilcoxon(x, y).pvalue if len(c) >= 5 and np.any(x != y) else float("nan")
            pairs.append({"A": "E3", "B": b, "metric": met, "n": len(c), "A_mean": x.mean(), "B_mean": y.mean(),
                          "A_better_in": f"{int(better.sum())}/{len(c)}", "wilcoxon_p": p})
    pairs = pd.DataFrame(pairs)
    pairs.to_csv(os.path.join(a.out_root, "myelin_paired.csv"), index=False, float_format="%.4g")

    # figure: group parcel map, MPC matrices, gradient
    plt = P.plt
    show = [m for m in ("E3", "LUT") if m in methods]
    fig, ax = plt.subplots(2, 1 + len(show), figsize=(5 * (1 + len(show)), 9.5))
    ax[0, 0].imshow(mpcs["real"], cmap="RdBu_r", vmin=-1, vmax=1)
    ax[0, 0].set_title("group MPC, real T2w")
    ax[1, 0].axis("off")
    for j, m in enumerate(show, start=1):
        ax[0, j].imshow(mpcs[m], cmap="RdBu_r", vmin=-1, vmax=1)
        ax[0, j].set_title(f"group MPC, {m}  (r = {corr(mpcs[m][iu], mpcs['real'][iu]):.2f})")
        g = grads[m] * np.sign(corr(grads[m], grads["real"]) or 1)
        ax[1, j].scatter(grads["real"], g, s=10, alpha=0.7, label="gradient 1")
        ax[1, j].scatter(pmean["real"], pmean[m], s=10, alpha=0.7, label="parcel T1w/T2w")
        ax[1, j].set_xlabel("real T2w")
        ax[1, j].set_ylabel(m)
        ax[1, j].legend(fontsize=8)
        ax[1, j].set_title(f"{m}: parcel r {corr(pmean[m], pmean['real']):.2f}, "
                           f"gradient |r| {abs(corr(grads[m], grads['real'])):.2f}")
    fig.tight_layout()
    fig.savefig(os.path.join(a.out_root, "myelin_group.png"), dpi=100)
    plt.close(fig)

    with pd.option_context("display.width", 250, "display.max_columns", 40, "display.precision", 3):
        print("\n== per subject (mean, sd) ==")
        print(summ.to_string())
        print("\n== group average ==")
        print(grp.to_string(index=False))
        print("\n== paired (E3 vs controls) ==")
        print(pairs.to_string(index=False))
    print(f"\ntables and figure in {a.out_root}")


def main():
    a = parse_args()
    splits = a.splits or (sorted(glob.glob(os.path.join(a.e3_root, "splits_*.json"))) or [None])[0]
    if not splits:
        sys.exit(f"no split file in {a.e3_root}; did you `source ganmat_env.sh`?")
    sp = json.load(open(splits))
    if a.list_subs:
        print("\n".join(sp["subjects"]))
        return
    if a.group:
        run_group(a)
        return
    nmt = {"gm": find_nmt_file(a.nmt_dir, r"^NMT_v[\d.]+_sym(_05mm)?_GM_cortical_mask\.nii\.gz$"),
           "seg": P.find_nmt(a.nmt_dir)["seg"],
           "charm": find_nmt_file(a.nmt_dir, r"^CHARM_in_NMT_v[\d.]+_sym(_05mm)?\.nii\.gz$")}
    for sub in a.subs or sp["subjects"]:
        run_subject(sub, a, sp, nmt)


if __name__ == "__main__":
    main()
