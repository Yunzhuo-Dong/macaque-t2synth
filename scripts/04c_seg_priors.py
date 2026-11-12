#!/usr/bin/env python
"""04c_seg_priors.py — FAST guided by subject-specific NMT tissue priors.

Why: with a tight macaque brain mask there is little CSF, so plain FAST's darkest class
drifts into grey matter and much of the cortical ribbon ends up labelled WM.
Here FAST stays the segmentation tool (as in GAN-MAT's pipeline), but its classes are
anchored by the NMT v2 segmentation warped into the grid through the subject's SyN
registration (-a identity, -A priors, -P priors used throughout).

Writes a SECOND set of inputs next to the plain-FAST ones (nothing is overwritten):
  grid/NMTseg_syn_grid.nii.gz        NMT labels via SyN (reused if 04b made it)
  grid/NMTseg3_syn_grid.nii.gz       3-class version (1 CSF, 2 GM = L2+L3, 3 WM)
  grid/priors/prior_{csf,gm,wm}.nii.gz
  grid/priors/T1w_grid_pveseg.nii.gz FAST + priors labels
  grid/T1w_grid_pveseg_priors.npy    GAN-MAT input channels (same construction)
  qc/qc_05_seg_priors.png            plain FAST vs FAST+priors vs NMT-SyN
and adds stage "07b_seg_priors" to manifest.json.

Usage:
    source ganmat_env.sh
    "$VENV_PY" 04c_seg_priors.py --subs sub-032128 sub-032130
"""
import argparse
import importlib.util
import json
import os
import subprocess
import time

import nibabel as nib
import numpy as np
from scipy import ndimage

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("prep04", os.path.join(HERE, "04_preprocess.py"))
P = importlib.util.module_from_spec(spec)
spec.loader.exec_module(P)

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--subs", nargs="+", required=True)
ap.add_argument("--prep-root", default=os.path.join(os.environ.get("PROJECT_DIR", "."), "derivatives", "prep"))
ap.add_argument("--prior-sigma-mm", type=float, default=0.6, help="Gaussian smoothing of the warped labels")
ap.add_argument("--prior-floor", type=float, default=0.05, help="minimum prior probability per class")
ap.add_argument("--force", action="store_true")
a = ap.parse_args()


def dice(x, y):
    return round(float(2 * (x & y).sum() / max(1, x.sum() + y.sum())), 3)


def channel_stats(ch):
    fr, zbg, ztis = [], [], []
    for k in range(3):  # exactly dataset.py's z-score
        x = np.asarray(ch[..., k], dtype=np.float64)
        z = (x - x.mean()) / x.std()
        fr.append(round(float((x != 0).mean()), 4))
        zbg.append(round(float(z[x == 0].mean()), 3))
        ztis.append(round(float(z[x != 0].mean()), 3))
    return fr, zbg, ztis


for sub in a.subs:
    t0 = time.time()
    root = os.path.join(a.prep_root, sub)
    x, g = (lambda f: os.path.join(root, "xfm", f)), (lambda f: os.path.join(root, "grid", f))
    pdir = g("priors")
    os.makedirs(pdir, exist_ok=True)
    mpath = os.path.join(root, "manifest.json")
    man = json.load(open(mpath))
    gimg = nib.load(g("grid_ref.nii.gz"))
    print(f"[{sub}]", flush=True)

    # 1) NMT segmentation in the grid via SyN (grid <- native <- NMT)
    syn = g("NMTseg_syn_grid.nii.gz")
    if a.force or not os.path.exists(syn):
        subprocess.run(["antsApplyTransforms", "-d", "3", "-i", man["nmt"]["seg"], "-r", g("grid_ref.nii.gz"),
                        "-t", f"[{x('brainaff_0GenericAffine.mat')},1]", "-t", x("nmt2nat_1Warp.nii.gz"),
                        "-t", x("nmt2nat_0GenericAffine.mat"), "-n", "NearestNeighbor", "-o", syn],
                       check=True, capture_output=True)
    nmt = np.asarray(nib.load(syn).dataobj).astype(int)
    brain = np.asarray(nib.load(g("brain_mask_grid.nii.gz")).dataobj) > 0
    n3 = np.select([nmt == 1, np.isin(nmt, [2, 3]), nmt == 4], [1, 2, 3], 0).astype(np.int16)
    P.save_like(n3 * brain, gimg, g("NMTseg3_syn_grid.nii.gz"), np.int16)

    # 2) soft priors: smoothed one-hot labels, floored, renormalised inside the mask.
    #    Voxels without an NMT tissue label (0 inside the mask, 5 = vessels) get flat priors.
    sig = a.prior_sigma_mm / float(gimg.header.get_zooms()[0])
    onehot = np.stack([(n3 == k).astype(np.float32) for k in (1, 2, 3)], -1)
    onehot[(n3 == 0) & brain] = 1.0 / 3
    pri = np.stack([ndimage.gaussian_filter(onehot[..., k], sig) for k in range(3)], -1)
    pri = np.maximum(pri, a.prior_floor)
    pri /= pri.sum(-1, keepdims=True)
    pri *= brain[..., None]
    names = ("prior_csf", "prior_gm", "prior_wm")
    for k, nm in enumerate(names):
        P.save_like(pri[..., k], gimg, os.path.join(pdir, f"{nm}.nii.gz"))
    del onehot, pri
    ident = os.path.join(pdir, "identity.mat")
    with open(ident, "w") as fh:
        fh.write("1 0 0 0\n0 1 0 0\n0 0 1 0\n0 0 0 1\n")

    # 3) FAST with priors throughout (labels 1/2/3 follow the prior order CSF/GM/WM)
    prefix = os.path.join(pdir, "T1w_grid")
    seg_path = prefix + "_pveseg.nii.gz"
    if a.force or not os.path.exists(seg_path):
        cmd = ["fast", "-t", "1", "-n", "3", "-N", "-a", ident, "-A",
               *[os.path.join(pdir, f"{nm}.nii.gz") for nm in names], "-P", "-o", prefix, g("T1w_grid.nii.gz")]
        print("    $ " + " ".join(cmd), flush=True)
        with open(os.path.join(root, "logs", "07b_seg_priors.log"), "a") as fh:
            subprocess.run(cmd, check=True, stdout=fh, stderr=subprocess.STDOUT)
    seg_pri = np.asarray(nib.load(seg_path).dataobj).astype(np.int16)
    seg_fast = np.asarray(nib.load(g("T1w_grid_pveseg.nii.gz")).dataobj).astype(np.int16)
    t1 = np.asarray(nib.load(g("T1w_grid.nii.gz")).dataobj, dtype=np.float32)

    # 4) channels, same construction as GAN-MAT preprocessing.py
    npy = g("T1w_grid_pveseg_priors.npy")
    if a.force or not os.path.exists(npy):
        ch = np.zeros(t1.shape + (3,), np.float32)
        for k in (1, 2, 3):
            sel = seg_pri == k
            ch[..., k - 1][sel] = t1[sel]
        np.save(npy, ch)
    ch = np.load(npy, mmap_mode="r")
    fr, zbg, ztis = channel_stats(ch)

    rec = {"prior_sigma_mm": a.prior_sigma_mm, "prior_floor": a.prior_floor,
           "fast_priors_label_mean_T1": [round(float(t1[seg_pri == k].mean()), 2) for k in (1, 2, 3)],
           "channel_fractions": fr, "zscore_background": zbg, "zscore_tissue_mean": ztis}
    for tag, seg in (("fast_plain", seg_fast), ("fast_priors", seg_pri)):
        sh = [round(float((seg[brain] == k).mean()), 3) for k in (1, 2, 3)]
        rec[f"{tag}_brain_shares_csf_gm_wm"] = sh
        rec[f"{tag}_dice_vs_NMT_SyN"] = {nm: dice((seg == k) & brain, (n3 == k) & brain)
                                         for nm, k in (("CSF", 1), ("GM", 2), ("WM", 3))}
    rec["NMT_SyN_brain_shares_csf_gm_wm"] = [round(float((n3[brain] == k).mean()), 3) for k in (1, 2, 3)]
    m = rec["fast_priors_label_mean_T1"]
    rec["label_order_ok"] = bool(m[0] < m[1] < m[2])
    rec.update(status="done", seconds=round(time.time() - t0, 1), finished=time.strftime("%Y-%m-%d %H:%M:%S"))
    man["stages"]["07b_seg_priors"] = rec
    tmp = mpath + ".tmp"
    json.dump(man, open(tmp, "w"), indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    os.replace(tmp, mpath)

    # 5) QC
    c = P.centroid(P.canon(g("brain_mask_grid.nii.gz"), np.uint8) > 0)
    t1c = P.canon(g("T1w_grid.nii.gz"))
    lab = lambda arr: P.canon(nib.Nifti1Image(arr.astype(np.int16), gimg.affine), np.int16)  # noqa: E731
    P.qc_figure(os.path.join(root, "qc", "qc_05_seg_priors.png"),
                [{"img": t1c, "label": "plain FAST", "labels": lab(seg_fast)},
                 {"img": t1c, "label": "FAST + NMT priors", "labels": lab(seg_pri)},
                 {"img": t1c, "label": "NMT via SyN (reference)", "labels": lab(n3)}],
                c, f"{sub}: blue CSF, green GM, yellow WM")

    print(f"  brain shares CSF/GM/WM  plain FAST {rec['fast_plain_brain_shares_csf_gm_wm']}  "
          f"FAST+priors {rec['fast_priors_brain_shares_csf_gm_wm']}  NMT-SyN {rec['NMT_SyN_brain_shares_csf_gm_wm']}")
    print(f"  Dice vs NMT-SyN         plain FAST {rec['fast_plain_dice_vs_NMT_SyN']}")
    print(f"                          FAST+priors {rec['fast_priors_dice_vs_NMT_SyN']}")
    print(f"  label order ok: {rec['label_order_ok']}   channel fractions {fr}   ({rec['seconds']} s)")
