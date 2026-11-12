#!/usr/bin/env python
"""05_e2_zeroshot.py — Experiment E2: human-pretrained GAN-MAT, zero-shot, on the
macaque-specific inputs produced by 04_preprocess.py.

Per subject:
  0  input channels from several segmentations (see SEGS): plain FAST, FAST + NMT priors,
     and SyN-warped NMT labels with deep GM counted as GM or as WM
  1  input normalisation, two variants
       official      : dataset.py's per-channel z-score over the whole 256^3 volume
       human_matched : the same z-score, computed as if each tissue occupied the
                       fraction of the grid it occupies in the human reference
                       (removes the macaque/human tissue-proportion offset)
  2  generator forward pass (same model, checkpoint and call as GAN-MAT's t2.py)
  3  synthetic T2w: grid -> native T1w space (inverse of the 12-DOF brain affine)
  4  evaluation in native space against the real T2w, inside brain mask AND T2w FOV
     ("brain"), and inside that mask eroded by ~1 mm ("core")
  5  baselines on the same masks: normalised T1w (E0), inverted T1w (E0-inv) and, with
     --fit-subs, the best voxel-wise function of T1w intensity learned on those (training)
     subjects (E0_T1w_LUT: a 256-bin lookup table, monotonic or not)
  6  "beyond-T1" correlation = partial correlation of synthetic and real T2w given T1w intensity:
     within the evaluated subject and mask, the conditional mean given T1w intensity (256 bins) is
     removed from both images and the residuals are correlated. Only this part can add
     information to a T1w/T2w ratio; any image that is a function of T1w alone gets 0 by
     construction. The reference row reports T1_explainable_R2: the share of real-T2w variance
     explained by T1w intensity in that subject and mask.

Metric conventions (every image is normalised independently inside the mask):
  x' = clip((x - P1) / (P99 - P1), 0, 1);  MAE, RMSE, PSNR (data range 1) and 3D SSIM
  (7^3 window, averaged over mask voxels) use x'.  Pearson r uses the raw values.
  Tissue means use the SyN-warped NMT labels (independent of the input segmentation;
  falls back to plain-FAST labels if 04c has not run). A T2w-like image has CSF > GM > WM.

Usage:
    source ganmat_env.sh
    export CUDA_VISIBLE_DEVICES=0          # pick a free GPU
    "$VENV_PY" 05_e2_zeroshot.py --subs sub-032128 sub-032130
"""
import argparse
import importlib.util
import json
import os
import subprocess
import sys
import time

import nibabel as nib
import numpy as np
import pandas as pd
from scipy import ndimage
from skimage.metrics import structural_similarity

HERE = os.path.dirname(os.path.abspath(__file__))
HUMAN_FRACTIONS = np.array([0.0404, 0.1091, 0.0753])  # CSF, GM, WM share of 256^3 (human sub-01)
VARIANTS = ("official", "human_matched")
SEGS = {"fast": "T1w_grid_pveseg.npy",             # plain FAST (04_preprocess.py)
        "priors": "T1w_grid_pveseg_priors.npy",    # FAST + NMT priors (04c_seg_priors.py)
        "nmt_l3gm": "T1w_grid_nmtL3gm.npy",        # NMT via SyN, deep GM as GM (04d_nmt_channels.py)
        "nmt_l3wm": "T1w_grid_nmtL3wm.npy"}        # NMT via SyN, deep GM as WM (04d_nmt_channels.py)
DEFAULT_SEGS = ["fast", "nmt_l3gm", "nmt_l3wm"]


def load_prep_module():
    """Reuse I/O and QC helpers from 04_preprocess.py (same folder or ../step3_prep)."""
    for cand in (os.path.join(HERE, "04_preprocess.py"), os.path.join(HERE, "..", "step3_prep", "04_preprocess.py")):
        if os.path.exists(cand):
            spec = importlib.util.spec_from_file_location("prep04", cand)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod
    sys.exit("04_preprocess.py not found next to this script")


P = load_prep_module()


def parse_args():
    env = os.environ.get
    proj = env("PROJECT_DIR", ".")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--subs", nargs="+", required=True)
    p.add_argument("--prep-root", default=os.path.join(proj, "derivatives", "prep"))
    p.add_argument("--out-root", default=os.path.join(proj, "derivatives", "e2"))
    p.add_argument("--ganmat-root", default=env("GANMAT_ROOT"))
    p.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=VARIANTS)
    p.add_argument("--segs", nargs="+", default=DEFAULT_SEGS, choices=list(SEGS))
    p.add_argument("--device", default="cuda", help="cuda / cpu (CUDA_VISIBLE_DEVICES picks the GPU)")
    p.add_argument("--checkpoint", help="generator checkpoint (default: official GAN-MAT model.pth); "
                   "use with the --variants/--segs the model was fine-tuned with")
    p.add_argument("--label", default="E2", help="prefix of the method names in the results (e.g. E3)")
    p.add_argument("--fit-subs", nargs="+", help="subjects (never the evaluated ones) to learn the T1w->T2w "
                   "lookup-table baseline (E0_T1w_LUT) from, e.g. the fold's training subjects")
    p.add_argument("--erode-mm", type=float, default=1.0, help="erosion for the 'core' mask")
    p.add_argument("--allow-unreviewed", action="store_true",
                   help="run even if visual_qc in the prep manifest is not PASS")
    p.add_argument("--mock-model", action="store_true",
                   help="PIPELINE TEST ONLY: replace the generator by a fixed numpy function")
    p.add_argument("--force", action="store_true")
    return p.parse_args()


# ================================================================= normalisation
def normalise(ch, variant):
    """ch: (X, Y, Z, 3) channels as built by 04_preprocess.py. Returns float32 + stats."""
    out = np.empty(ch.shape, np.float32)
    stats = []
    for k in range(3):
        x = np.asarray(ch[..., k], dtype=np.float64)
        tissue = x[x != 0]
        if tissue.size < 1000:
            raise ValueError(f"channel {k} has only {tissue.size} non-zero voxels; check FAST output")
        f = tissue.size / x.size
        if variant == "official":  # exactly functions/model/dataset.py
            mean, std = x.mean(), x.std()
        else:  # z-score as if the tissue occupied the human fraction fh of the grid
            fh = HUMAN_FRACTIONS[k]
            mu, var_t = tissue.mean(), tissue.var()
            mean = fh * mu
            std = np.sqrt(fh * (var_t + mu**2) - mean**2)
        z = (x - mean) / std
        out[..., k] = z
        stats.append({"fraction": round(f, 4), "background": round(float(-mean / std), 3),
                      "tissue_mean": round(float(z[x != 0].mean()), 3)})
    return out, stats


def human_reference_stats():
    f = HUMAN_FRACTIONS
    return {"background": np.round(-np.sqrt(f / (1 - f)), 3).tolist(),
            "tissue_mean_if_uniform": np.round(np.sqrt((1 - f) / f), 3).tolist()}


# ===================================================================== generator
class Generator:
    def __init__(self, a):
        self.mock = a.mock_model
        if self.mock:
            print("!! --mock-model: numpy stand-in, results are meaningless (pipeline test only)")
            return
        import torch
        self.torch = torch
        sys.path.insert(0, os.path.join(a.ganmat_root, "functions", "model"))
        from model import Pix2Pix_3D  # official GAN-MAT architecture
        self.device = torch.device(a.device if (a.device == "cpu" or torch.cuda.is_available()) else "cpu")
        ckpt_path = a.checkpoint or os.path.join(a.ganmat_root, "functions", "model", "model.pth")
        try:
            ckpt = torch.load(ckpt_path, map_location=self.device)
        except Exception:  # official / own checkpoint (trusted) saved with non-tensor objects
            ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        nker = int(ckpt.get("meta", {}).get("nker", 64))
        self.net = Pix2Pix_3D(in_channels=3, out_channels=1, nker=nker).to(self.device)
        print(f"checkpoint: {ckpt_path}")
        self.net.load_state_dict(ckpt["netG"])
        self.net.eval()  # as t2.py: BatchNorm running stats, dropout off
        name = torch.cuda.get_device_name(self.device) if self.device.type == "cuda" else "cpu"
        print(f"generator loaded on {name}")

    def __call__(self, x):
        """x: (X, Y, Z, 3) normalised -> (X, Y, Z) tanh output."""
        if self.mock:
            return np.tanh(0.6 * x[..., 0] + 0.2 * x[..., 1] - 0.4 * x[..., 2] - 0.3).astype(np.float32)
        t = self.torch
        with t.no_grad():
            inp = t.from_numpy(np.ascontiguousarray(x.transpose(3, 0, 1, 2))[None]).float().to(self.device)
            y = self.net(inp)[0, 0].float().cpu().numpy()
        if self.device.type == "cuda":
            t.cuda.empty_cache()
        return y


# ======================================================================= metrics
def bbox_slices(mask, pad=4):
    idx = np.argwhere(mask)
    lo = np.maximum(idx.min(0) - pad, 0)
    hi = np.minimum(idx.max(0) + pad + 1, mask.shape)
    return tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))


def norm01(x, m):
    lo, hi = np.percentile(x[m], [1, 99])
    return np.clip((x - lo) / max(hi - lo, 1e-12), 0.0, 1.0)


LUT_BINS = 256


def lut_from_pairs(pairs):
    """Mean normalised T2w per bin of normalised T1w, pooled over (t1, t2) voxel arrays."""
    s, c = np.zeros(LUT_BINS), np.zeros(LUT_BINS)
    for t, r in pairs:
        idx = np.clip(np.rint(t * (LUT_BINS - 1)).astype(int), 0, LUT_BINS - 1)
        s += np.bincount(idx, weights=r, minlength=LUT_BINS)
        c += np.bincount(idx, minlength=LUT_BINS)
    centres = np.arange(LUT_BINS) / (LUT_BINS - 1)
    ok = c > 0
    return np.interp(centres, centres[ok], s[ok] / c[ok])


def lut_apply(lut, t):
    return lut[np.clip(np.rint(t * (LUT_BINS - 1)).astype(int), 0, LUT_BINS - 1)]


def given_t1_residual(x, t):
    """x minus its conditional mean given T1w intensity t (both voxel vectors, t in [0, 1])."""
    idx = np.clip(np.rint(t * (LUT_BINS - 1)).astype(int), 0, LUT_BINS - 1)
    cnt = np.maximum(np.bincount(idx, minlength=LUT_BINS), 1)
    return x - (np.bincount(idx, weights=x, minlength=LUT_BINS) / cnt)[idx]


def evaluate(pred, ref, m, labels, with_image_metrics=True, t1=None):
    """t1: the subject's T1w (same crop), for the partial correlation given T1w intensity."""
    p = norm01(pred, m)
    row = {"n_voxels": int(m.sum())}
    if t1 is not None and with_image_metrics:
        tn = norm01(t1, m)[m]
        dp, dr = given_t1_residual(p[m], tn), given_t1_residual(norm01(ref, m)[m], tn)
        row["beyond_T1_r"] = float(np.corrcoef(dp, dr)[0, 1]) if dp.std() > 1e-6 else 0.0
    if with_image_metrics:
        r = norm01(ref, m)
        d = p[m] - r[m]
        rmse = float(np.sqrt(np.mean(d**2)))
        _, smap = structural_similarity(p, r, data_range=1.0, win_size=7, full=True)
        row.update(MAE=float(np.mean(np.abs(d))), RMSE=rmse, PSNR=float(20 * np.log10(1.0 / max(rmse, 1e-12))),
                   pearson_r=float(np.corrcoef(pred[m], ref[m])[0, 1]), SSIM=float(smap[m].mean()))
    means = {}
    for k, name in ((1, "CSF"), (2, "GM"), (3, "WM")):
        sel = m & (labels == k)
        means[name] = float(p[sel].mean()) if sel.any() else float("nan")
    row.update({f"mean_{k}": v for k, v in means.items()})
    row["t2_order_CSF>GM>WM"] = bool(means["CSF"] > means["GM"] > means["WM"])
    row["GM_WM_contrast"] = float((means["GM"] - means["WM"]) / max(means["GM"] + means["WM"], 1e-12))
    return row


# ======================================================================= per subject
def load_native(prep_root, sub):
    n = lambda f: os.path.join(prep_root, sub, "native", f)  # noqa: E731
    mimg = nib.load(n("brain_mask.nii.gz"))
    brain = (np.asarray(mimg.dataobj) > 0) & (np.asarray(nib.load(n("T2w_fov.nii.gz")).dataobj) > 0)
    sl = bbox_slices(brain)
    crop = lambda path: np.asarray(nib.load(path).dataobj, dtype=np.float32)[sl]  # noqa: E731
    return mimg, brain[sl], sl, crop(n("T2w_brain.nii.gz")), crop(n("T1w_brain.nii.gz"))


def fit_lut(prep_root, subs):
    pairs = []
    for s in subs:
        _, b, _, r, t = load_native(prep_root, s)
        pairs.append((norm01(t, b)[b], norm01(r, b)[b]))
        print(f"  LUT: + {s} ({b.sum()} voxels)", flush=True)
    return lut_from_pairs(pairs)


def run_subject(sub, a, gen, lut=None):
    prep = os.path.join(a.prep_root, sub)
    man = json.load(open(os.path.join(prep, "manifest.json")))
    if not man.get("auto_checks_passed"):
        raise RuntimeError(f"{sub}: preprocessing auto-checks did not pass: {man.get('auto_checks')}")
    vqc = str(man.get("visual_qc", ""))
    if not vqc.upper().startswith("PASS"):
        msg = f"{sub}: visual_qc is '{vqc[:40]}' (not PASS)"
        if not a.allow_unreviewed:
            raise RuntimeError(msg + "; review qc/*.png first or pass --allow-unreviewed")
        print("!! " + msg + " — continuing because of --allow-unreviewed")

    out = os.path.join(a.out_root, sub)
    os.makedirs(out, exist_ok=True)
    g = lambda f: os.path.join(prep, "grid", f)  # noqa: E731
    n = lambda f: os.path.join(prep, "native", f)  # noqa: E731
    t1_native = man["inputs"]["T1w"]
    aff = os.path.join(prep, "xfm", "brainaff_0GenericAffine.mat")
    gref = nib.load(g("grid_ref.nii.gz"))
    info = {"subject": sub, "visual_qc": vqc, "variants": {}, "human_reference": human_reference_stats()}

    conds = []
    for seg in a.segs:
        if not os.path.exists(g(SEGS[seg])):
            print(f"!! {sub}: {SEGS[seg]} missing, skipping seg '{seg}' (run 04c_seg_priors.py for 'priors')")
            continue
        ch = np.load(g(SEGS[seg]))
        for v in a.variants:
            tag = f"{seg}_{v}"
            conds.append(tag)
            native = os.path.join(out, f"synthT2w_{tag}_native.nii.gz")
            if os.path.exists(native) and not a.force:
                continue
            x, stats = normalise(ch, v)
            t0 = time.time()
            y = gen(x)
            P.save_like(y, gref, os.path.join(out, f"synthT2w_{tag}_grid.nii.gz"))
            subprocess.run(["antsApplyTransforms", "-d", "3", "-i", os.path.join(out, f"synthT2w_{tag}_grid.nii.gz"),
                            "-r", t1_native, "-t", aff, "-n", "Linear", "--float", "1",
                            "-o", native], check=True, capture_output=True)
            info["variants"][tag] = {"input_stats": stats, "seconds": round(time.time() - t0, 1),
                                     "output_percentiles_1_50_99": np.round(np.percentile(y, [1, 50, 99]), 3).tolist()}
            print(f"  {sub} {tag:20s} input bg {[s['background'] for s in stats]}  "
                  f"tissue {[s['tissue_mean'] for s in stats]}")
        del ch

    # evaluation labels: SyN-warped NMT (independent of the input segmentation) if available
    src = g("NMTseg3_syn_grid.nii.gz")
    if not os.path.exists(src):
        src = g("T1w_grid_pveseg.nii.gz")
        print(f"!! {sub}: NMTseg3_syn_grid.nii.gz missing; tissue means use plain-FAST labels (circular for 'fast')")
    info["eval_labels"] = src
    labels_native = os.path.join(out, "eval_labels_native.nii.gz")
    if not os.path.exists(labels_native) or a.force:
        subprocess.run(["antsApplyTransforms", "-d", "3", "-i", src, "-r", t1_native,
                        "-t", aff, "-n", "NearestNeighbor", "-o", labels_native], check=True, capture_output=True)

    # ---- evaluation in native space, cropped to the brain bounding box
    mimg, brain, sl, ref, t1 = load_native(a.prep_root, sub)
    vox = float(min(mimg.header.get_zooms()[:3]))
    core = ndimage.binary_erosion(brain, iterations=max(1, int(round(a.erode_mm / vox))))
    crop = lambda path: np.asarray(nib.load(path).dataobj, dtype=np.float32)[sl]  # noqa: E731
    labels = np.asarray(nib.load(labels_native).dataobj)[sl].astype(np.int16)
    preds = {f"{a.label}_{c}": crop(os.path.join(out, f"synthT2w_{c}_native.nii.gz")) for c in conds}
    preds["E0_T1w"] = t1
    preds["E0_T1w_inverted"] = np.where(brain, t1.max() - t1, 0).astype(np.float32)
    base = None
    if lut is not None:
        t1n = norm01(t1, brain)
        base = lut_apply(lut, t1n).astype(np.float32)
        preds["E0_T1w_LUT"] = np.where(brain, base, 0).astype(np.float32)

    rows = []
    for mname, m in (("brain", brain), ("core", core)):
        refrow = {"subject": sub, "method": "real_T2w (reference)", "mask": mname,
                  **evaluate(ref, ref, m, labels, with_image_metrics=False)}
        rn = norm01(ref, m)[m]  # how much of this subject's T2w is a function of its own T1w at all
        refrow["T1_explainable_R2"] = float(1 - np.var(given_t1_residual(rn, norm01(t1, m)[m])) / np.var(rn))
        rows.append(refrow)
        for meth, pred in preds.items():
            row = evaluate(pred, ref, m, labels, t1=t1)
            rows.append({"subject": sub, "method": meth, "mask": mname, **row})

    # ---- QC figure (native space, RAS-canonical display)
    canon = lambda arr: P.canon(nib.Nifti1Image(arr, mimg.affine))  # noqa: E731
    qrows = [{"img": canon(norm01(t1, brain) * brain), "label": "T1w (input)", "vmin": 0, "vmax": 1},
             {"img": canon(norm01(ref, brain) * brain), "label": "real T2w", "vmin": 0, "vmax": 1}]
    for cnd in conds:
        qrows.append({"img": canon(norm01(preds[f"{a.label}_{cnd}"], brain) * brain),
                      "label": f"synthetic T2w, {cnd}", "vmin": 0, "vmax": 1})
    for cnd in conds[:1]:  # one error map keeps the figure readable
        if True:
            err = np.abs(norm01(preds[f"{a.label}_{cnd}"], brain) - norm01(ref, brain)) * brain
            qrows.append({"img": canon(err), "label": f"|synthetic({cnd}) - real|", "vmin": 0, "vmax": 0.5,
                          "cmap": "magma"})
    c = P.centroid(canon(brain.astype(np.uint8)) > 0)
    P.qc_figure(os.path.join(out, f"qc_{a.label.lower()}_native.png"), qrows, c,
                f"{sub}: {a.label}, native space, each image normalised to its P1-P99 in the mask")

    with open(os.path.join(out, "e2_info.json"), "w") as fh:
        json.dump(info, fh, indent=2)
    return rows


def main():
    a = parse_args()
    if not a.ganmat_root and not a.mock_model:
        sys.exit("missing --ganmat-root / GANMAT_ROOT; did you `source ganmat_env.sh`?")
    os.makedirs(a.out_root, exist_ok=True)
    gen = Generator(a)
    lut = None
    if a.fit_subs:
        leak = sorted(set(a.fit_subs) & set(a.subs))
        if leak:
            sys.exit(f"--fit-subs overlaps the evaluated subjects {leak}: the baseline would see its test data")
        print(f"fitting the T1w->T2w lookup-table baseline on {len(a.fit_subs)} subjects")
        lut = fit_lut(a.prep_root, a.fit_subs)
    rows, failed = [], []
    for sub in a.subs:
        print(f"[{sub}]", flush=True)
        try:
            rows += run_subject(sub, a, gen, lut)
        except Exception as e:
            print(f"[{sub}] FAILED: {type(e).__name__}: {e}", file=sys.stderr)
            failed.append(sub)
    if rows:
        df = pd.DataFrame(rows)
        tag = "_MOCK" if a.mock_model else ""
        csv = os.path.join(a.out_root, f"{a.label.lower()}_metrics{tag}.csv")
        df.to_csv(csv, index=False, float_format="%.5f")
        cols = ["subject", "method", "MAE", "RMSE", "PSNR", "pearson_r", "SSIM",
                "t2_order_CSF>GM>WM", "GM_WM_contrast", "beyond_T1_r", "T1_explainable_R2"]
        with pd.option_context("display.width", 200, "display.max_columns", 20, "display.precision", 4):
            print("\n== brain mask (brain AND T2w FOV), native space ==")
            print(df[df["mask"] == "brain"][cols].to_string(index=False))
        print(f"\nall rows (incl. 'core' mask and tissue means): {csv}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
