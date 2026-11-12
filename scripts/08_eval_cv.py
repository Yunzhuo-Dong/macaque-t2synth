#!/usr/bin/env python
"""08_eval_cv.py — native-space evaluation of every fold, then one cohort-level result table.

For each experiment folder under <e3-root> (e.g. official_..._cos, scratch_..._cos) and each fold
with a ckpt_best.pth, runs 05_e2_zeroshot.py on that fold's TEST subjects, with the lookup-table
baseline fitted on the same fold's TRAINING subjects. With --zeroshot it also evaluates the
untouched human GAN-MAT model on every fold (folder "zeroshot"), so all methods share subjects,
masks and baselines. Finished evaluations are skipped (use --force to redo).

Aggregation (every test subject appears exactly once per method):
  cv_results_long.csv  every subject x method x mask row
  cv_summary.csv       mean / sd per method and mask
  cv_paired.csv        paired comparisons (per-subject wins + Wilcoxon signed-rank p):
                       each model vs E0_T1w_LUT, and official vs scratch when both exist

Usage:
    source ganmat_env.sh && export CUDA_VISIBLE_DEVICES=0
    "$VENV_PY" 08_eval_cv.py --zeroshot                  # evaluate everything that is trained
    "$VENV_PY" 08_eval_cv.py --aggregate-only            # just rebuild the tables
"""
import argparse
import glob
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
METRICS = ["MAE", "PSNR", "pearson_r", "SSIM", "beyond_T1_r", "GM_WM_contrast"]
HIGHER_BETTER = {"MAE": False, "PSNR": True, "pearson_r": True, "SSIM": True, "beyond_T1_r": True}

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--e3-root", default=os.path.join(os.environ.get("PROJECT_DIR", "."), "derivatives", "e3"))
ap.add_argument("--prep-root", default=os.path.join(os.environ.get("PROJECT_DIR", "."), "derivatives", "prep"))
ap.add_argument("--splits", help="split file (default: the single splits_*.json in --e3-root)")
ap.add_argument("--tags", nargs="*", help="experiment folders to evaluate (default: all with fold checkpoints)")
ap.add_argument("--zeroshot", action="store_true", help="also evaluate the official model zero-shot per fold")
ap.add_argument("--aggregate-only", action="store_true")
ap.add_argument("--force", action="store_true")
ap.add_argument("--eval-args", default="", help="extra arguments passed to 05_e2_zeroshot.py")
a = ap.parse_args()

splits = a.splits or (sorted(glob.glob(os.path.join(a.e3_root, "splits_*.json"))) or [None])[0]
if not splits or not os.path.exists(splits):
    sys.exit(f"no split file found in {a.e3_root}")
sp = json.load(open(splits))
print(f"splits: {splits}  ({len(sp['folds'])} folds, {len(sp['subjects'])} subjects)")

tags = a.tags or sorted({os.path.basename(os.path.dirname(os.path.dirname(c)))
                         for c in glob.glob(os.path.join(a.e3_root, "*", "fold*", "ckpt_best.pth"))})


def run_eval(out, label, fold, ckpt=None, seg="fast", norm="human_matched"):
    csv = os.path.join(out, f"{label.lower()}_metrics.csv")
    f = sp["folds"][fold]
    if os.path.exists(csv) and not a.force:
        done = set(pd.read_csv(csv)["subject"])
        if set(f["test"]) <= done:
            return "exists"
        print(f"(incomplete: missing {sorted(set(f['test']) - done)}, re-running)", end=" ")
    cmd = [sys.executable, os.path.join(HERE, "05_e2_zeroshot.py"), "--subs", *f["test"],
           "--fit-subs", *f["train"], "--label", label, "--segs", seg, "--variants", norm,
           "--prep-root", a.prep_root, "--out-root", out, *a.eval_args.split()]
    if ckpt:
        cmd += ["--checkpoint", ckpt]
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "eval.log"), "w") as fh:
        r = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT)
    return "ok" if r.returncode == 0 and os.path.exists(csv) else f"FAILED (see {out}/eval.log)"


if not a.aggregate_only:
    jobs = []
    for tag in tags:
        for fold in range(len(sp["folds"])):
            ck = os.path.join(a.e3_root, tag, f"fold{fold}", "ckpt_best.pth")
            if not os.path.exists(ck):
                print(f"  {tag} fold{fold}: no checkpoint, skipped")
                continue
            cfg = json.load(open(os.path.join(os.path.dirname(ck), "config.json")))
            if cfg.get("split", {}).get("test") != sp["folds"][fold]["test"]:
                sys.exit(f"{tag} fold{fold} was trained on a different split than {splits}")
            jobs.append((tag, fold, ck, cfg.get("seg", "fast"), cfg.get("norm", "human_matched")))
    if a.zeroshot:
        jobs += [("zeroshot", f, None, "fast", "human_matched") for f in range(len(sp["folds"]))]
    for i, (tag, fold, ck, seg, norm) in enumerate(jobs, 1):
        out = os.path.join(a.e3_root, tag, f"fold{fold}", "eval")
        print(f"[{i}/{len(jobs)}] {tag} fold{fold} ...", end=" ", flush=True)
        print(run_eval(out, "E2" if ck is None else "E3", fold, ck, seg, norm), flush=True)

# ------------------------------------------------------------------- aggregate
rows = []
for csv in sorted(glob.glob(os.path.join(a.e3_root, "*", "fold*", "eval", "e[23]_metrics.csv"))):
    tag = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(csv))))
    if a.tags and tag not in set(a.tags) | {"zeroshot"}:  # keep pilot runs out of the tables
        continue
    fold = int(os.path.basename(os.path.dirname(os.path.dirname(csv)))[4:])
    d = pd.read_csv(csv)
    d["fold"] = fold
    model = d["method"].str.match(r"^E[23]_")
    d.loc[model, "method"] = tag  # model rows are named after their experiment
    rows.append(d)
if not rows:
    sys.exit("no evaluation results found")
df = pd.concat(rows, ignore_index=True)
df = df.drop_duplicates(subset=["subject", "method", "mask"])  # baselines appear once per experiment
df.to_csv(os.path.join(a.e3_root, "cv_results_long.csv"), index=False, float_format="%.5f")

summ = []
for (meth, mask), g in df.groupby(["method", "mask"]):
    r = {"method": meth, "mask": mask, "n_subjects": g["subject"].nunique()}
    for m in METRICS + (["T1_explainable_R2"] if "T1_explainable_R2" in g else []):
        if m in g and g[m].notna().any():
            r[f"{m}_mean"], r[f"{m}_sd"] = g[m].mean(), g[m].std()
    summ.append(r)
summ = pd.DataFrame(summ)
summ.to_csv(os.path.join(a.e3_root, "cv_summary.csv"), index=False, float_format="%.4f")


def paired(m1, m2, mask):
    a1 = df[(df["method"] == m1) & (df["mask"] == mask)].set_index("subject")
    a2 = df[(df["method"] == m2) & (df["mask"] == mask)].set_index("subject")
    common = a1.index.intersection(a2.index)
    out = []
    for m, hb in HIGHER_BETTER.items():
        if m not in a1 or a1.loc[common, m].isna().all() or a2.loc[common, m].isna().all():
            continue
        x, y = a1.loc[common, m].values, a2.loc[common, m].values
        wins = int(((x > y) if hb else (x < y)).sum())
        try:
            from scipy.stats import wilcoxon
            p = float(wilcoxon(x, y).pvalue) if len(common) >= 5 and np.any(x != y) else float("nan")
        except Exception:
            p = float("nan")
        out.append({"A": m1, "B": m2, "mask": mask, "metric": m, "n": len(common),
                    "A_mean": x.mean(), "B_mean": y.mean(), "A_better_in": f"{wins}/{len(common)}",
                    "wilcoxon_p": p})
    return out


models = [m for m in df["method"].unique() if not m.startswith("E0_") and not m.startswith("real_")]
pairs = []
for mask in ("brain", "core"):
    for m in models:
        pairs += paired(m, "E0_T1w_LUT", mask)
    off = [m for m in models if m.startswith("official")]
    scr = [m for m in models if m.startswith("scratch")]
    for o in off:
        for s in scr:
            pairs += paired(o, s, mask)
    for o in off:
        if "zeroshot" in models:
            pairs += paired(o, "zeroshot", mask)
pairs = pd.DataFrame(pairs)
pairs.to_csv(os.path.join(a.e3_root, "cv_paired.csv"), index=False, float_format="%.4g")

with pd.option_context("display.width", 250, "display.max_columns", 40, "display.precision", 3):
    for mask in ("brain", "core"):
        s = summ[summ["mask"] == mask]
        cols = ["method", "n_subjects"] + [c for c in s.columns if c.endswith("_mean")]
        print(f"\n== {mask} mask: mean over test subjects ==")
        print(s[cols].to_string(index=False))
    if len(pairs):
        print("\n== paired comparisons (A vs B; Wilcoxon needs n >= 5) ==")
        print(pairs.to_string(index=False))
print(f"\ntables: {a.e3_root}/cv_results_long.csv, cv_summary.csv, cv_paired.csv")
