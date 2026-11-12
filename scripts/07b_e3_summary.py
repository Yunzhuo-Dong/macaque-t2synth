#!/usr/bin/env python
"""07b_e3_summary.py — overview of all fine-tuning runs (read-only).

Scans <e3-root>/<experiment>/<fold or overfit_*>/ for val_log.csv and train_log.csv and writes
  * a table: validation MAE before fine-tuning (step 0), best (and at which epoch), last,
    relative improvement, and whether the best came at the very end (= probably undertrained)
    -> printed and saved as <e3-root>/e3_summary.csv
  * a figure: validation MAE per epoch (one line per run) and the brain L1 training loss
    -> <e3-root>/e3_curves.png

MAE is in target units ([-1, 1] scale) inside brain AND T2w FOV, in the 256^3 grid. It is a
quick progress measure; the final comparison is the native-space evaluation with 05_e2_zeroshot.py.

Usage:
    source ganmat_env.sh
    "$VENV_PY" 07b_e3_summary.py                    # all runs
    "$VENV_PY" 07b_e3_summary.py --filter bs8       # only experiments whose name contains 'bs8'
"""
import argparse
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--e3-root", default=os.path.join(os.environ.get("PROJECT_DIR", "."), "derivatives", "e3"))
ap.add_argument("--filter", default="", help="substring of the experiment name")
a = ap.parse_args()

runs = []
for vpath in sorted(glob.glob(os.path.join(a.e3_root, "*", "*", "val_log.csv"))):
    run_dir = os.path.dirname(vpath)
    exp, part = os.path.basename(os.path.dirname(run_dir)), os.path.basename(run_dir)
    if a.filter not in exp:
        continue
    v = pd.read_csv(vpath)
    v = v[pd.to_numeric(v["step"], errors="coerce").notna()].astype({"step": int})
    v = v.drop_duplicates("step", keep="last").sort_values("step")
    t = pd.read_csv(os.path.join(run_dir, "train_log.csv")) if os.path.exists(os.path.join(run_dir, "train_log.csv")) else None
    cfg = json.load(open(os.path.join(run_dir, "config.json"))) if os.path.exists(os.path.join(run_dir, "config.json")) else {}
    n_train = max(1, len(cfg.get("split", {}).get("train", [])))
    per_step = cfg.get("batch_size", 1) * cfg.get("world_size", 1)
    v["epoch"] = v["step"] * per_step / n_train
    runs.append({"exp": exp, "part": part, "dir": run_dir, "val": v, "train": t, "cfg": cfg})

if not runs:
    raise SystemExit(f"no runs found under {a.e3_root}" + (f" matching '{a.filter}'" if a.filter else ""))

rows = []
for r in runs:
    v, cfg = r["val"], r["cfg"]
    start = v.loc[v["step"] == 0, "val_mae_brain"]
    start = float(start.iloc[0]) if len(start) else float("nan")
    ib = v["val_mae_brain"].idxmin()
    best, best_ep, last = float(v.loc[ib, "val_mae_brain"]), float(v.loc[ib, "epoch"]), float(v["val_mae_brain"].iloc[-1])
    planned = cfg.get("steps_derived") or cfg.get("steps")
    done_steps = int(v["step"].max())
    rows.append({"experiment": r["exp"], "run": r["part"], "init": cfg.get("init"), "lr": cfg.get("lr"),
                 "batch": per_step if (per_step := cfg.get("batch_size", 1) * cfg.get("world_size", 1)) else 1,
                 "epochs_done": round(float(v["epoch"].max()), 1),
                 "finished": bool(planned and done_steps >= planned),
                 "val_mae_start": round(start, 4), "val_mae_best": round(best, 4), "best_at_epoch": round(best_ep, 1),
                 "val_mae_last": round(last, 4),
                 "improvement_%": round(100 * (start - best) / start, 1) if start == start else None,
                 "best_is_last": bool(ib == v.index[-1])})
df = pd.DataFrame(rows)
df.to_csv(os.path.join(a.e3_root, "e3_summary.csv"), index=False)
with pd.option_context("display.width", 250, "display.max_columns", 20):
    print(df.to_string(index=False))
if df["best_is_last"].any():
    print("\nbest_is_last = True: validation error was still falling at the end -> more epochs (or a higher lr) may help.")
if (~df["finished"]).any():
    print("finished = False: run still in progress or stopped early.")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for r in runs:
    lab = f"{r['exp']} | {r['part']}"
    axes[0].plot(r["val"]["epoch"], r["val"]["val_mae_brain"], marker="o", ms=3, label=lab)
    t = r["train"]
    if t is not None and len(t):
        x = t["epoch"] if "epoch" in t else t["step"] * (r["cfg"].get("batch_size", 1) * r["cfg"].get("world_size", 1)) / max(1, len(r["cfg"].get("split", {}).get("train", [])))
        axes[1].plot(x, t["l1_brain"].rolling(max(1, len(t) // 30), min_periods=1).mean(), label=lab)
axes[0].set(xlabel="epoch", ylabel="validation MAE (brain, grid, [-1,1] units)", title="validation (epoch 0 = zero-shot)")
axes[1].set(xlabel="epoch", ylabel="training L1 in brain (smoothed)", title="training loss")
for ax in axes:
    ax.grid(alpha=0.3)
axes[0].legend(fontsize=7)
fig.tight_layout()
out = os.path.join(a.e3_root, "e3_curves.png")
fig.savefig(out, dpi=110)
print(f"\ntable : {os.path.join(a.e3_root, 'e3_summary.csv')}\nfigure: {out}")
