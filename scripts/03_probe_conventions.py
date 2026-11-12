#!/usr/bin/env python
"""03_probe_conventions.py — Step 2 probes. Read-only; prints a report.

A) NMT v2.1 brain size vs GAN-MAT's human input grid -> candidate macaque voxel sizes
B) value distribution of existing GAN-MAT outputs (output_T2w.npy) -> infers the
   tanh target-scaling convention the generator was trained with

Usage:
    source ganmat_env.sh
    "$VENV_PY" 03_probe_conventions.py | tee "$PROJECT_DIR/logs/probe_$(date +%Y%m%d_%H%M%S).log"
"""
import glob
import os

import nibabel as nib
import numpy as np

# Human reference, measured from GAN-MAT's template/MNI152_T1_0.8mm_brain.nii.gz
# placed into 256^3 exactly as functions/preprocessing.py does (LAS, 0.8 mm).
HUMAN_BRAIN_VOX = 4_052_687
HUMAN_BBOX_VOX = np.array([183, 237, 194])
HUMAN_FRAC = HUMAN_BRAIN_VOX / 256**3


def section(title):
    print(f"\n== {title} ==")


# ---------------------------------------------------------------------------- A
section("A. NMT brain vs GAN-MAT human input grid")
nmt_dir = os.environ.get("NMT_DIR", "")
masks = sorted(p for p in glob.glob(os.path.join(nmt_dir, "**", "*_brainmask.nii.gz"), recursive=True)
               if "05mm" not in p)
if not masks:
    print(f"  no full-resolution NMT brainmask found under '{nmt_dir}'")
else:
    mpath = masks[0]
    img = nib.load(mpath)
    m = np.asarray(img.dataobj) > 0
    z = np.array(img.header.get_zooms()[:3], dtype=float)
    idx = np.argwhere(m)
    ext_mm = (idx.max(0) - idx.min(0) + 1) * z
    vol_mm3 = m.sum() * np.prod(z)
    print(f"  mask   : {mpath}")
    print(f"  voxel  : {z} mm   axcodes {''.join(nib.aff2axcodes(img.affine))}")
    print(f"  brain  : {vol_mm3 / 1000:.1f} mL   bbox {np.round(ext_mm, 1)} mm")
    print(f"  human  : {HUMAN_BRAIN_VOX} voxels = {HUMAN_FRAC:.3f} of 256^3, bbox {HUMAN_BBOX_VOX} voxels")

    s_vol = (vol_mm3 / HUMAN_BRAIN_VOX) ** (1 / 3)
    s_fit = ext_mm.max() / 240.0  # keep ~8 voxels margin on each side
    print(f"  isotropic voxel matching human brain voxel COUNT : {s_vol:.3f} mm")
    print(f"  per-axis voxel matching human BBOX (anisotropic) : {np.round(ext_mm / HUMAN_BBOX_VOX, 3)} mm")
    print(f"  smallest isotropic voxel that fits 256 + margin  : {s_fit:.3f} mm")
    print("  candidates:")
    for s in sorted({round(s_vol, 3), round(s_fit, 3), 0.30, 0.33, 0.35}):
        ev = np.round(ext_mm / s).astype(int)
        frac = vol_mm3 / s**3 / 256**3
        print(f"    s={s:.3f} mm  bbox {ev} vox  fits256={bool((ev <= 248).all())}  "
              f"brain fraction {frac:.3f} (human {HUMAN_FRAC:.3f})")

    segs = glob.glob(os.path.join(os.path.dirname(mpath), "*_segmentation.nii.gz"))
    if segs:
        sd = np.asarray(nib.load(segs[0]).dataobj).astype(int)
        labs, cnt = np.unique(sd[sd > 0], return_counts=True)
        print(f"  {os.path.basename(segs[0])} label shares: "
              + ", ".join(f"{l}:{c / cnt.sum():.3f}" for l, c in zip(labs, cnt)))

# ---------------------------------------------------------------------------- B
section("B. Existing GAN-MAT outputs (target-scaling convention)")
root = os.environ.get("GANMAT_ROOT", "")
outs = sorted(glob.glob(os.path.join(root, "**", "output_T2w.npy"), recursive=True))
if not outs:
    print(f"  no output_T2w.npy found under '{root}'")
for p in outs:
    o = np.load(p).astype(np.float32)
    c = 8
    corners = np.concatenate([o[:c, :c, :c].ravel(), o[-c:, -c:, -c:].ravel(),
                              o[:c, -c:, :c].ravel(), o[-c:, :c, -c:].ravel()])
    q = np.percentile(o, [0, 1, 5, 25, 50, 75, 95, 99, 100])
    print(f"  {os.path.relpath(p, root)}   shape {o.shape}")
    print(f"    percentiles 0/1/5/25/50/75/95/99/100 : {np.round(q, 3)}")
    print(f"    corner background mean {corners.mean():.3f} (sd {corners.std():.3f});  "
          f"share < -0.95: {(o < -0.95).mean():.3f};  share > 0.95: {(o > 0.95).mean():.3f}")

    inp = os.path.join(os.path.dirname(p), "T1w_MNI_pveseg.npy")
    if os.path.exists(inp):
        x = np.load(inp)
        brain = (x != 0).any(-1)
        ch = [(x[..., k] != 0).mean() for k in range(x.shape[-1])]
        print(f"    input brain fraction {brain.mean():.3f} (human {HUMAN_FRAC:.3f});  "
              f"channel fractions {np.round(ch, 4)} (human approx [0.040 0.102 0.100])")
        print(f"    output mean inside input brain {o[brain].mean():.3f}, outside {o[~brain].mean():.3f}")
        del x

print("\nReading B: a background close to -1 means the generator was trained on targets scaled to\n"
      "[-1, 1] with background at -1; fine-tuning targets must then follow the same convention.")
