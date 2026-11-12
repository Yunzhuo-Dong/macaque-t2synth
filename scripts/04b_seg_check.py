#!/usr/bin/env python
"""04b_seg_check.py — is FAST's GM/WM split plausible for macaque?

Warps the NMT v2 segmentation into the GAN-MAT grid through the subject's SyN
registration (much closer than the affine-only comparison in the manifest), then
reports tissue shares and which NMT labels FAST assigns to CSF/GM/WM.
Read-only for prep outputs; writes grid/NMTseg_syn_grid.nii.gz and qc/qc_04_seg_check.png.

Usage:
    source ganmat_env.sh
    "$VENV_PY" 04b_seg_check.py --subs sub-032128 sub-032130
"""
import argparse
import importlib.util
import os
import subprocess

import nibabel as nib
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("prep04", os.path.join(HERE, "04_preprocess.py"))
P = importlib.util.module_from_spec(spec)
spec.loader.exec_module(P)

ap = argparse.ArgumentParser()
ap.add_argument("--subs", nargs="+", required=True)
ap.add_argument("--prep-root", default=os.path.join(os.environ.get("PROJECT_DIR", "."), "derivatives", "prep"))
a = ap.parse_args()

for sub in a.subs:
    root = os.path.join(a.prep_root, sub)
    x, g = (lambda f: os.path.join(root, "xfm", f)), (lambda f: os.path.join(root, "grid", f))
    man = P.json.load(open(os.path.join(root, "manifest.json")))
    out = g("NMTseg_syn_grid.nii.gz")
    # grid <- native <- NMT : listed last = applied first to the moving image
    subprocess.run(["antsApplyTransforms", "-d", "3", "-i", man["nmt"]["seg"], "-r", g("grid_ref.nii.gz"),
                    "-t", f"[{x('brainaff_0GenericAffine.mat')},1]", "-t", x("nmt2nat_1Warp.nii.gz"),
                    "-t", x("nmt2nat_0GenericAffine.mat"), "-n", "NearestNeighbor", "-o", out],
                   check=True, capture_output=True)
    nmt = np.asarray(nib.load(out).dataobj).astype(int)
    fast = np.asarray(nib.load(g("T1w_grid_pveseg.nii.gz")).dataobj).astype(int)
    brain = np.asarray(nib.load(g("brain_mask_grid.nii.gz")).dataobj) > 0

    print(f"\n===== {sub}  (inside grid brain mask, {brain.sum()} voxels)")
    fl = fast[brain]
    nl = nmt[brain]
    print("FAST shares  CSF/GM/WM :", "  ".join(f"{(fl == k).mean():.3f}" for k in (1, 2, 3)))
    labs = [k for k in range(0, 6) if (nl == k).any()]
    print("NMT(SyN) shares        :", "  ".join(f"L{k}:{(nl == k).mean():.3f}" for k in labs))
    print("row = NMT label, columns = share assigned by FAST to  CSF    GM     WM   (unlabelled)")
    for k in labs:
        sel = nl == k
        r = [(fl[sel] == j).mean() for j in (1, 2, 3, 0)]
        print(f"   L{k} (n={sel.sum():>8}):                           " + "  ".join(f"{v:.3f}" for v in r))
    # 3-class view of NMT: 1 CSF, 2+3 GM, 4 WM (label 3 meaning to be confirmed visually)
    n3 = np.select([nmt == 1, np.isin(nmt, [2, 3]), nmt == 4], [1, 2, 3], 0)
    for name, k in (("CSF", 1), ("GM", 2), ("WM", 3)):
        A, B = (fast == k) & brain, (n3 == k) & brain
        print(f"Dice FAST vs NMT-SyN {name}: {2 * (A & B).sum() / max(1, A.sum() + B.sum()):.3f}")

    t1 = P.canon(g("T1w_grid.nii.gz"))
    c = P.centroid(P.canon(g("brain_mask_grid.nii.gz"), np.uint8) > 0)
    gimg = nib.load(g("grid_ref.nii.gz"))
    P.qc_figure(os.path.join(root, "qc", "qc_04_seg_check.png"),
                [{"img": t1, "label": "grid T1w"},
                 {"img": t1, "label": "FAST (blue CSF, green GM, yellow WM)", "labels": P.canon(g("T1w_grid_pveseg.nii.gz"), np.int16)},
                 {"img": t1, "label": "NMT via SyN (L2+L3 shown as GM)",
                  "labels": P.canon(nib.Nifti1Image(n3.astype(np.int16), gimg.affine), np.int16)}],
                c, f"{sub}: FAST vs SyN-warped NMT segmentation")
    print("figure:", os.path.join(root, "qc", "qc_04_seg_check.png"))
