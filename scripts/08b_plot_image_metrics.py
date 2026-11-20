#!/usr/bin/env python
"""08b_plot_image_metrics.py — per-subject image metrics of the final comparison (results figure).

Reads cv_results_long.csv written by 08_eval_cv.py and plots Pearson r, SSIM, MAE and beyond-T1 r
for whole brain and core, one dot per test subject. Methods that are not in the table are skipped.

Usage:
    "$VENV_PY" 08b_plot_image_metrics.py --out ../docs/figures/results_image_metrics.png
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

METHODS = [("official_long200_sharedF", "fine-tuned", "tab:red"),
           ("E0_T1w_LUT@sharedF", "T1w LUT", "tab:blue"),
           ("E0_T1w_inverted@sharedF", "inverted\nT1w", "tab:gray")]
METRICS = [("pearson_r", "Pearson r ↑"), ("SSIM", "SSIM ↑"), ("MAE", "MAE ↓"), ("beyond_T1_r", "beyond-T1 r ↑")]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default=os.path.join(os.environ.get("PROJECT_DIR", "."), "derivatives", "e3",
                                                  "cv_results_long.csv"))
    ap.add_argument("--out", default="results_image_metrics.png")
    a = ap.parse_args()
    df = pd.read_csv(a.csv)
    meths = [m for m in METHODS if m[0] in set(df["method"])]
    n = df[df["method"] == meths[0][0]]["subject"].nunique()
    fig, axes = plt.subplots(2, len(METRICS), figsize=(4.2 * len(METRICS), 8.2))
    for i, (mask, mlab) in enumerate((("brain", "whole brain"), ("core", "core (1 mm eroded)"))):
        for j, (metric, title) in enumerate(METRICS):
            ax = axes[i, j]
            vals = [df[(df["method"] == m) & (df["mask"] == mask)][metric].dropna().values for m, _, _ in meths]
            bp = ax.boxplot(vals, widths=0.55, showfliers=False, patch_artist=True)
            for k, (patch, (_, _, col)) in enumerate(zip(bp["boxes"], meths)):
                patch.set_facecolor(col)
                patch.set_alpha(0.3)
                x = vals[k]
                ax.plot(np.full(len(x), k + 1) + np.random.default_rng(k).uniform(-0.15, 0.15, len(x)), x, ".",
                        color=col, ms=6)
            ax.set_xticks(range(1, len(meths) + 1))
            ax.set_xticklabels([m[1] for m in meths], fontsize=8)
            ax.set_title(f"{title} — {mlab}", fontsize=10)
            ax.grid(axis="y", alpha=0.3)
    fig.suptitle(f"Synthetic T2w vs real T2w (shared-field target), native space, {n} macaques "
                 "(5-fold CV, each subject tested once)")
    fig.tight_layout()
    fig.savefig(a.out, dpi=120)
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
