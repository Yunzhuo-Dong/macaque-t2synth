#!/usr/bin/env python
"""04b_shared_bias_target.py — T2w training target in the same bias frame as the T1w input.

04_preprocess.py corrects T1w and T2w with two separate N4 fields (F1, F2). The T1w/T2w ratio
of those images is T1raw/T2raw * F2/F1, and the smooth F2/F1 term turns out to remove most of the
cortical myelin pattern (real_n4 vs real_hcp r = 0.14 in 13_myelin_hcp_style.py). HCP, whose
T1w_restore / T2w_restore images GAN-MAT was trained on, corrects BOTH images with one shared
field, so their ratio is unaffected.

Here the T2w is corrected with the T1w's own N4 field F1 (exactly T1raw / T1w_brain inside the
brain mask), so that  T1w_brain / T2w_brain_sharedF = T1raw / T2raw.  Only the T1w is needed to
form F1, so training and testing stay consistent; the network input does not change.

Writes per subject:
  native/T2w_brain_sharedF.nii.gz         raw T2w / F1, brain AND T2w FOV
  grid/T2w_grid_sharedF.nii.gz            same in the model grid
  grid/T2w_grid_target_sharedF.npy/.nii.gz  [-1, 1] target, scaled exactly like step 08 of 04
  sharedF_info.json                        field statistics
Usage:
    source ganmat_env.sh
    ls -d "$PROJECT_DIR"/derivatives/prep/sub-* | xargs -n1 basename | xargs -P 4 -I{} "$VENV_PY" 04b_shared_bias_target.py --subs {}
"""
import argparse
import json
import os
import subprocess
import sys

import nibabel as nib
import numpy as np
from scipy import ndimage


def parse_args():
    proj = os.environ.get("PROJECT_DIR", ".")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--subs", nargs="+", required=True)
    p.add_argument("--prep-root", default=os.path.join(proj, "derivatives", "prep"))
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def load(path, dtype=np.float32):
    img = nib.load(path)
    return img, np.asarray(img.dataobj, dtype=dtype)


def save_like(arr, ref_img, path, dtype=np.float32):
    out = nib.Nifti1Image(np.asarray(arr, dtype), ref_img.affine, ref_img.header)
    out.set_data_dtype(dtype)
    nib.save(out, path)


def run_subject(sub, a):
    d = os.path.join(a.prep_root, sub)
    n, g = (lambda f: os.path.join(d, "native", f)), (lambda f: os.path.join(d, "grid", f))
    out_npy = g("T2w_grid_target_sharedF.npy")
    if os.path.exists(out_npy) and not a.force:
        print(f"[{sub}] done already")
        return
    man = json.load(open(os.path.join(d, "manifest.json")))
    mimg, mask = load(n("brain_mask.nii.gz"), np.uint8)
    mask = mask > 0
    _, fov = load(n("T2w_fov.nii.gz"), np.uint8)
    _, t1raw = load(man["inputs"]["T1w"])
    _, t1n4 = load(n("T1w_brain.nii.gz"))
    _, t2raw = load(n("T2w_on_T1w.nii.gz"))
    _, t2n4 = load(n("T2w_brain.nii.gz"))
    if t1raw.shape != mask.shape:
        sys.exit(f"[{sub}] raw T1w {t1raw.shape} is not on the native grid {mask.shape}")

    # F1 = T1raw / T1w_brain (N4 output = input / field, so this is the field itself)
    ok = mask & (t1n4 > 0) & (t1raw > 0)
    f1 = np.zeros(mask.shape, np.float32)
    f1[ok] = t1raw[ok] / t1n4[ok]
    med = float(np.median(f1[ok]))
    bad = ok & ((f1 < 0.2 * med) | (f1 > 5 * med))          # clipped / degenerate voxels
    good = ok & ~bad
    lo, hi = np.argwhere(mask).min(0), np.argwhere(mask).max(0) + 1
    bb = tuple(slice(int(l_), int(h_)) for l_, h_ in zip(lo, hi))  # mask bounding box keeps memory small
    _, idx = ndimage.distance_transform_edt(~good[bb], return_indices=True)
    f1[bb] = f1[bb][tuple(idx)]                                # fill degenerate voxels from the nearest good one
    valid = mask & (fov > 0)
    t2s = np.where(valid, np.clip(t2raw, 0, None) / np.maximum(f1, 1e-6), 0).astype(np.float32)
    save_like(t2s, mimg, n("T2w_brain_sharedF.nii.gz"))

    # into the model grid exactly like step 06 of 04_preprocess.py
    gref = g("grid_ref.nii.gz")
    aff = os.path.join(d, "xfm", "brainaff_0GenericAffine.mat")
    tmp = g("_t2_sharedF_tmp.nii.gz")
    subprocess.run(["antsApplyTransforms", "-d", "3", "-i", n("T2w_brain_sharedF.nii.gz"), "-r", gref,
                    "-t", f"[{aff},1]", "-n", "Linear", "-o", tmp], check=True, capture_output=True)
    gimg = nib.load(gref)
    _, mg = load(g("brain_mask_grid.nii.gz"), np.uint8)
    _, fg = load(g("T2w_fov_grid.nii.gz"), np.uint8)
    _, vg = load(g("valid_grid.nii.gz"), np.uint8)
    t2g = np.clip(load(tmp)[1], 0, None) * (mg > 0) * (fg > 0)
    os.remove(tmp)
    save_like(t2g, gimg, g("T2w_grid_sharedF.nii.gz"))
    v = vg > 0
    p999 = float(np.percentile(t2g[v], 99.9))
    tgt = np.full(t2g.shape, -1.0, np.float32)
    tgt[v] = np.clip(2.0 * t2g[v] / p999 - 1.0, -1.0, 1.0)
    save_like(tgt, gimg, g("T2w_grid_target_sharedF.nii.gz"))
    np.save(out_npy, tgt)

    # how different is the new target from the old one? (the smooth F2/F1 term)
    f2 = np.where(valid & (t2n4 > 0), np.clip(t2raw, 0, None) / np.maximum(t2n4, 1e-6), np.nan)
    ratio = f2 / f1
    q = np.nanpercentile(ratio[valid], [5, 50, 95])
    p5, p95 = np.percentile(f1[valid], [5, 95])
    info = {"F1_p95_over_p5": round(float(p95 / p5), 3), "F2_over_F1_p5_p50_p95": [round(float(x), 3) for x in q],
            "F2_over_F1_p95_over_p5": round(float(q[2] / q[0]), 3), "bad_field_voxels": int(bad.sum()),
            "target_scale_p99_9": round(p999, 3)}
    json.dump(info, open(os.path.join(d, "sharedF_info.json"), "w"), indent=2)
    print(f"[{sub}] {info}", flush=True)


def main():
    a = parse_args()
    for s in a.subs:
        run_subject(s, a)


if __name__ == "__main__":
    main()
