#!/usr/bin/env python
"""04d_nmt_channels.py — GAN-MAT input channels from the SyN-warped NMT segmentation.

Two label mappings (intensities always come from the subject's own T1w):
  nmt_l3gm : L1 -> CSF, L2 + L3 -> GM, L4 -> WM   (anatomical: deep GM counted as GM)
  nmt_l3wm : L1 -> CSF, L2 -> GM, L3 + L4 -> WM   (deep GM as WM, as FAST tends to do on human T1)
Brain voxels without an NMT tissue label (0, or 5 = vessels) keep their plain-FAST label.

Also prints, per NMT label, how plain FAST labelled those voxels: this separates
"FAST calls deep GM WM" from "FAST calls cortex WM".

Needs grid/NMTseg_syn_grid.nii.gz from 04b_seg_check.py or 04c_seg_priors.py.
Writes grid/T1w_grid_nmtL3gm.npy, grid/T1w_grid_nmtL3wm.npy and manifest stage "07c_nmt_channels".

Usage:
    source ganmat_env.sh
    "$VENV_PY" 04d_nmt_channels.py --subs sub-032128 sub-032130
"""
import argparse
import json
import os
import time

import nibabel as nib
import numpy as np

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--subs", nargs="+", required=True)
ap.add_argument("--prep-root", default=os.path.join(os.environ.get("PROJECT_DIR", "."), "derivatives", "prep"))
a = ap.parse_args()

MAPS = {"nmt_l3gm": {1: 1, 2: 2, 3: 2, 4: 3}, "nmt_l3wm": {1: 1, 2: 2, 3: 3, 4: 3}}
FILES = {"nmt_l3gm": "T1w_grid_nmtL3gm.npy", "nmt_l3wm": "T1w_grid_nmtL3wm.npy"}

for sub in a.subs:
    t0 = time.time()
    root = os.path.join(a.prep_root, sub)
    g = lambda f: os.path.join(root, "grid", f)  # noqa: E731
    if not os.path.exists(g("NMTseg_syn_grid.nii.gz")):
        raise SystemExit(f"{sub}: grid/NMTseg_syn_grid.nii.gz missing; run 04c_seg_priors.py first")
    nmt = np.asarray(nib.load(g("NMTseg_syn_grid.nii.gz")).dataobj).astype(np.int16)
    fast = np.asarray(nib.load(g("T1w_grid_pveseg.nii.gz")).dataobj).astype(np.int16)
    brain = np.asarray(nib.load(g("brain_mask_grid.nii.gz")).dataobj) > 0
    t1 = np.asarray(nib.load(g("T1w_grid.nii.gz")).dataobj, dtype=np.float32)

    print(f"\n===== {sub}")
    print("plain FAST label of voxels with each NMT label (row shares)      CSF    GM     WM")
    conf = {}
    for k, name in ((1, "CSF"), (2, "cortical GM?"), (3, "deep GM?"), (4, "WM"), (5, "vessels"), (0, "unlabelled")):
        sel = brain & (nmt == k)
        if not sel.any():
            continue
        r = [round(float((fast[sel] == j).mean()), 3) for j in (1, 2, 3)]
        conf[f"L{k}"] = {"share_of_brain": round(float(sel.sum() / brain.sum()), 4), "fast_csf_gm_wm": r}
        print(f"  L{k} {name:13s} ({sel.sum() / brain.sum():5.1%} of brain)                           "
              + "  ".join(f"{v:.3f}" for v in r))
    mean_t1 = {f"L{k}": round(float(t1[brain & (nmt == k)].mean()), 1) for k in (1, 2, 3, 4) if (brain & (nmt == k)).any()}
    print("  mean T1w per NMT label:", mean_t1)

    rec = {"fast_label_by_nmt_label": conf, "mean_T1w_by_nmt_label": mean_t1}
    for tag, mp in MAPS.items():
        lab = np.zeros(nmt.shape, np.int16)
        for src, dst in mp.items():
            lab[nmt == src] = dst
        fill = brain & (lab == 0)
        lab[fill] = fast[fill]
        lab[~brain] = 0
        ch = np.zeros(t1.shape + (3,), np.float32)
        for k in (1, 2, 3):
            sel = lab == k
            ch[..., k - 1][sel] = t1[sel]
        np.save(g(FILES[tag]), ch)
        fr = [round(float((lab == k).mean()), 4) for k in (1, 2, 3)]
        sh = [round(float((lab[brain] == k).mean()), 3) for k in (1, 2, 3)]
        rec[tag] = {"channel_fractions": fr, "brain_shares_csf_gm_wm": sh,
                    "filled_from_fast_share": round(float(fill.sum() / brain.sum()), 4)}
        print(f"  {tag}: brain shares CSF/GM/WM {sh}  channel fractions {fr}  "
              f"(filled from FAST: {fill.sum() / brain.sum():.1%})")
        del ch
    rec.update(status="done", seconds=round(time.time() - t0, 1), finished=time.strftime("%Y-%m-%d %H:%M:%S"))
    mpath = os.path.join(root, "manifest.json")
    man = json.load(open(mpath))
    man["stages"]["07c_nmt_channels"] = rec
    json.dump(man, open(mpath + ".tmp", "w"), indent=2)
    os.replace(mpath + ".tmp", mpath)
