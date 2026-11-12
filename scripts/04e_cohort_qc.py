#!/usr/bin/env python
"""04e_cohort_qc.py — one-page overview of preprocessing for the whole cohort.

1. Table of every subject's manifest (stages done, auto checks, brain volume, T2 motion,
   grid margin, channel fractions, target mean), with robust outlier flags
   (|value - cohort median| > 3 x MAD) -> <prep-root>/cohort_qc.csv
2. Montage pages, 8 subjects each: native T1w with the brain mask (coronal, sagittal) and
   the grid T1w with FAST labels (coronal)          -> <prep-root>/cohort_qc_montage_<n>.png

Usage:
    source ganmat_env.sh
    "$VENV_PY" 04e_cohort_qc.py                    # all subjects under derivatives/prep
    "$VENV_PY" 04e_cohort_qc.py --no-montage       # table only (seconds)
"""
import argparse
import glob
import importlib.util
import json
import os
import re

import nibabel as nib
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("prep04", os.path.join(HERE, "04_preprocess.py"))
P = importlib.util.module_from_spec(spec)
spec.loader.exec_module(P)
plt = P.plt

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--prep-root", default=os.path.join(os.environ.get("PROJECT_DIR", "."), "derivatives", "prep"))
ap.add_argument("--subs", nargs="*")
ap.add_argument("--no-montage", action="store_true")
ap.add_argument("--per-page", type=int, default=8)
a = ap.parse_args()

subs = a.subs or sorted(os.path.basename(os.path.dirname(m))
                        for m in glob.glob(os.path.join(a.prep_root, "sub-*", "manifest.json")))
rows = []
for s in subs:
    mp = os.path.join(a.prep_root, s, "manifest.json")
    if not os.path.exists(mp):
        rows.append({"subject": s, "stages_done": 0, "error": "no manifest"})
        continue
    m = json.load(open(mp))
    st = m.get("stages", {})
    g = lambda k, f, d=None: st.get(k, {}).get(f, d)  # noqa: E731
    disp = g("05_t2", "t2_rigid_brain_displacement_mm", {}) or {}
    ch = g("07_fast", "channel_fractions", [np.nan] * 3)
    rows.append({
        "subject": s,
        "stages_done": sum(1 for k in st if re.match(r"^0[1-9]_", k) and st[k].get("status") == "done"),
        "error": m.get("error", ""),
        "auto_checks_passed": m.get("auto_checks_passed"),
        "visual_qc": str(m.get("visual_qc", ""))[:12],
        "brain_ml": g("04_mask", "brain_volume_ml"),
        "t2_motion_p95_mm": disp.get("p95"),
        "t2_rigid_applied": g("05_t2", "t2_rigid_applied"),
        "t2_fov_cov": g("05_t2", "t2_fov_coverage_of_brain"),
        "grid_margin_vox": g("06_grid", "grid_min_margin_vox"),
        "grid_brain_frac": g("06_grid", "grid_brain_fraction"),
        "ch_csf": ch[0], "ch_gm": ch[1], "ch_wm": ch[2],
        "target_brain_mean": g("08_target", "target_brain_mean"),
        "minutes_total": round(sum(v.get("seconds", 0) for v in st.values()) / 60, 1),
    })
df = pd.DataFrame(rows)

flags = {s: [] for s in df["subject"]}
for col in ("brain_ml", "grid_brain_frac", "ch_csf", "ch_gm", "ch_wm", "target_brain_mean", "t2_motion_p95_mm"):
    if col not in df or df[col].notna().sum() < 4:
        continue
    v = df[col].astype(float)
    med, mad = v.median(), (v - v.median()).abs().median() * 1.4826
    if mad > 0:
        for s, x in zip(df["subject"], v):
            if pd.notna(x) and abs(x - med) > 3 * mad:
                flags[s].append(col)
for _, r in df.iterrows():
    if r.get("stages_done", 0) < 9 or r.get("error"):
        flags[r["subject"]].insert(0, "INCOMPLETE")
    if r.get("auto_checks_passed") is False:
        flags[r["subject"]].insert(0, "AUTO_CHECK_FAIL")
    if r.get("t2_rigid_applied"):
        flags[r["subject"]].append("t2_motion_corrected")
df["flags"] = [",".join(flags[s]) for s in df["subject"]]

csv = os.path.join(a.prep_root, "cohort_qc.csv")
df.to_csv(csv, index=False, float_format="%.4f")
with pd.option_context("display.width", 250, "display.max_columns", 30, "display.precision", 3):
    print(df.drop(columns=["error"]).to_string(index=False))
done = df[df["stages_done"] >= 9]
print(f"\n{len(done)}/{len(df)} subjects complete; {sum(1 for f in df['flags'] if f)} flagged. table: {csv}")

if not a.no_montage:
    ok = [s for s in done["subject"]]
    for page, i in enumerate(range(0, len(ok), a.per_page), start=1):
        chunk = ok[i:i + a.per_page]
        fig, axes = plt.subplots(len(chunk), 3, figsize=(11, 3.3 * len(chunk)), squeeze=False)
        for r, s in enumerate(chunk):
            root = os.path.join(a.prep_root, s)
            head = P.canon(os.path.join(root, "native", "T1w_n4head.nii.gz"))
            mask = P.canon(os.path.join(root, "native", "brain_mask.nii.gz"), np.uint8) > 0
            c = P.centroid(mask)
            vmax = float(np.percentile(head[mask], 99.5))
            for k, (name, sl, ml) in enumerate([("coronal", head[:, c[1], :], mask[:, c[1], :]),
                                                ("sagittal", head[c[0], :, :], mask[c[0], :, :])]):
                img, msk = (sl.T[:, ::-1], ml.T[:, ::-1]) if name == "coronal" else (sl.T, ml.T)
                ax = axes[r, k]
                ax.imshow(img, origin="lower", cmap="gray", vmin=0, vmax=vmax)
                ax.contour(msk.astype(float), levels=[0.5], colors="r", linewidths=0.5)
                ax.set_title(f"{s} | mask, {name}", fontsize=8)
                ax.axis("off")
            del head, mask
            gt1 = P.canon(os.path.join(root, "grid", "T1w_grid.nii.gz"))
            seg = P.canon(os.path.join(root, "grid", "T1w_grid_pveseg.nii.gz"), np.int16)
            gc = P.centroid(seg > 0)
            ax = axes[r, 2]
            ax.imshow(gt1[:, gc[1], :].T[:, ::-1], origin="lower", cmap="gray")
            lab = seg[:, gc[1], :].T[:, ::-1]
            ax.imshow(np.ma.masked_where(lab == 0, lab), origin="lower", alpha=0.4, vmin=1, vmax=3,
                      cmap=P.ListedColormap(["#4c9be8", "#7bd36b", "#f2c14e"]))
            fl = flags.get(s) or []
            ax.set_title(f"{s} | grid + FAST" + (f"  [{', '.join(fl)}]" if fl else ""), fontsize=8,
                         color="r" if fl else "k")
            ax.axis("off")
        fig.suptitle(f"cohort QC page {page}: red = brain mask; blue CSF, green GM, yellow WM (radiological)",
                     fontsize=10)
        fig.tight_layout(rect=(0, 0, 1, 0.98))
        out = os.path.join(a.prep_root, f"cohort_qc_montage_{page}.png")
        fig.savefig(out, dpi=100)
        plt.close(fig)
        print("montage:", out)
