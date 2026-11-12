#!/usr/bin/env python
"""04_preprocess.py — Step 3: macaque-specific preprocessing for GAN-MAT.

Per subject ("native" = the subject's own T1w voxel grid):
  01  header check (is the T2w grid an exact sub-grid of the T1w grid?) + raw orientation QC
  02  N4 on the whole-head T1w, plus a 0.6 mm copy for registration
  03  NMT v2 head -> native T1w registration (rigid, affine, SyN)
  04  brain mask = NMT brainmask warped to native; second N4 inside the mask
  05  T2w resampled onto the T1w grid; rigid T2w->T1w check; N4 inside the mask
  06  native brain -> NMT affine (12 DOF, brain only; mirrors GAN-MAT's native->MNI step);
      model grid = 256^3, 0.30 mm, LAS, centred on the NMT brain
  07  FSL FAST -N in the grid -> CSF/GM/WM channels  (GAN-MAT input, .npy)
  08  T2w target in the grid, scaled to [-1, 1] with background -1  (.npy)
  09  QC figures + automatic checks in manifest.json

Each stage is skipped when its outputs already exist (--force redoes everything).
Nothing outside <out-root>/<sub>/ is written.

Usage:
    source ganmat_env.sh
    "$VENV_PY" 04_preprocess.py --sub sub-032128
"""
import argparse
import glob
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import time

import nibabel as nib
import numpy as np
from scipy import ndimage

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402

# Reference statistics from the human GAN-MAT example (pilot, simplified pipeline),
# measured by 03_probe_conventions.py. One subject only: a rough reference.
HUMAN_REF = {"source": "GAN-MAT example sub-01, human, pilot pipeline (n=1)",
             "brain_fraction": 0.225, "channel_fractions": [0.0404, 0.1091, 0.0753],
             "model_output_brain_mean": -0.427}
NMT_TISSUE_MAP = {"CSF": [1], "GM": [2, 3], "WM": [4]}  # NMT v2 segmentation labels (5 = vessels)


# ============================================================================ utils
def parse_args():
    env = os.environ.get
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sub", required=True, help="subject id, e.g. sub-032128")
    p.add_argument("--t1", help="T1w path (default: found under --data-dir)")
    p.add_argument("--t2", help="T2w path (default: found under --data-dir)")
    p.add_argument("--data-dir", default=env("DATA_DIR"))
    p.add_argument("--nmt-dir", default=env("NMT_DIR"))
    p.add_argument("--out-root", default=os.path.join(env("PROJECT_DIR", "."), "derivatives", "prep"))
    p.add_argument("--grid-size", type=int, default=256)
    p.add_argument("--grid-spacing", type=float, default=0.30)
    p.add_argument("--reg-spacing", type=float, default=0.6, help="spacing of the registration copies (mm)")
    p.add_argument("--init", choices=["com", "antsai"], default="com",
                   help="NMT->native initialisation: centre of mass, or antsAI rotation search")
    p.add_argument("--t2-rigid", choices=["auto", "always", "never"], default="auto")
    p.add_argument("--force", action="store_true")
    p.add_argument("--stop-after", type=int, default=9, help="stop after this stage number")
    return p.parse_args()


def find_inputs(data_dir, sub):
    """Unique T2w for the subject, and the T1w in the same directory (the archive also
    contains hard-linked T1w copies elsewhere, which are skipped)."""
    t2s = [h for h in glob.glob(os.path.join(data_dir, "**", f"{sub}*_T2w.nii*"), recursive=True)]
    seen, uniq = set(), []
    for h in sorted(t2s):
        st = os.stat(h)
        if (st.st_dev, st.st_ino) not in seen:
            seen.add((st.st_dev, st.st_ino))
            uniq.append(h)
    if len(uniq) != 1:
        raise FileNotFoundError(f"expected exactly one T2w for {sub} under {data_dir}, found: {uniq}")
    t2 = uniq[0]
    t1s = sorted(glob.glob(os.path.join(os.path.dirname(t2), f"{sub}*_T1w.nii*")))
    if len(t1s) != 1:
        raise FileNotFoundError(f"expected exactly one T1w next to {t2}, found: {t1s}")
    return t1s[0], t2


def find_nmt(nmt_dir):
    files = glob.glob(os.path.join(nmt_dir, "**", "*.nii.gz"), recursive=True)

    def pick(suffix, prefer_05mm):
        pat = re.compile(r"^NMT_v[\d.]+_sym(_05mm)?" + suffix + r"\.nii\.gz$")
        hits = sorted(f for f in files if pat.match(os.path.basename(f)))
        lo = [f for f in hits if "_05mm" in os.path.basename(f)]
        hi = [f for f in hits if "_05mm" not in os.path.basename(f)]
        order = lo + hi if prefer_05mm else hi + lo
        if not order:
            raise FileNotFoundError(f"NMT file '*{suffix}.nii.gz' not found under {nmt_dir}")
        return order[0]

    return {"head_reg": pick("", True), "ss_reg": pick("_SS", True), "mask_reg": pick("_brainmask", True),
            "mask": pick("_brainmask", False), "seg": pick("_segmentation", False)}


def save_like(data, ref_img, path, dtype=np.float32):
    hdr = ref_img.header.copy()
    hdr.set_data_dtype(dtype)
    img = nib.Nifti1Image(np.asarray(data).astype(dtype), ref_img.affine, hdr)
    img.header["scl_slope"], img.header["scl_inter"] = 1, 0
    nib.save(img, path)


def load(path, dtype=np.float32):
    img = nib.load(path)
    return img, np.asarray(img.dataobj, dtype=dtype)


def largest_component_filled(mask):
    lab, n = ndimage.label(mask)
    if n == 0:
        return mask
    sizes = ndimage.sum(mask, lab, index=np.arange(1, n + 1))
    out = lab == (int(np.argmax(sizes)) + 1)
    return ndimage.binary_fill_holes(out)


def read_itk_affine(path):
    """ITK/ANTs .mat -> (A 3x3, t 3, centre 3), all in LPS physical space."""
    from scipy.io import loadmat
    d = loadmat(path)
    key = next(k for k in d if k.startswith("AffineTransform"))
    par = np.asarray(d[key]).ravel()
    return par[:9].reshape(3, 3), par[9:12], np.asarray(d["fixed"]).ravel()


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


# =============================================================================== QC
def canon(img_or_path, dtype=np.float32):
    img = nib.load(img_or_path) if isinstance(img_or_path, str) else img_or_path
    return np.asarray(nib.as_closest_canonical(img).dataobj, dtype=dtype)


def planes(vol, c):
    return [("sagittal", vol[c[0], :, :]), ("coronal", vol[:, c[1], :]), ("axial", vol[:, :, c[2]])]


LETTERS = {"sagittal": ("P", "A", "I", "S"), "coronal": ("R", "L", "I", "S"), "axial": ("R", "L", "P", "A")}


def qc_figure(path, rows, centre, title):
    """rows: dicts with img, label, and optional contour (binary), edge (img, level),
    labels (int volume overlay), vmin/vmax/cmap. Arrays are RAS-canonical.
    Display is radiological for coronal/axial (subject's R on the image's left)."""
    fig, axes = plt.subplots(len(rows), 3, figsize=(13, 4.2 * len(rows)), squeeze=False)
    lab_cmap = ListedColormap(["#4c9be8", "#7bd36b", "#f2c14e"])
    for r, row in enumerate(rows):
        vol = row["img"]
        pos = vol[vol > 0]
        vmax = row.get("vmax", float(np.percentile(pos, 99.5)) if pos.size else 1.0)
        vmin = row.get("vmin", 0.0)
        for k, (name, sl) in enumerate(planes(vol, centre)):
            ax = axes[r, k]
            show = sl.T if name == "sagittal" else sl.T[:, ::-1]  # radiological for cor/ax
            ax.imshow(show, origin="lower", cmap=row.get("cmap", "gray"), vmin=vmin, vmax=vmax,
                      interpolation="nearest")
            flip = (lambda a: a) if name == "sagittal" else (lambda a: a[:, ::-1])
            if row.get("labels") is not None:
                lab = flip(planes(row["labels"], centre)[k][1].T)
                ax.imshow(np.ma.masked_where(lab == 0, lab), origin="lower", cmap=lab_cmap, vmin=1, vmax=3,
                          alpha=0.4, interpolation="nearest")
            if row.get("contour") is not None:
                con = flip(planes(row["contour"], centre)[k][1].T.astype(float))
                if con.max() > 0:
                    ax.contour(con, levels=[0.5], colors="r", linewidths=0.6)
            if row.get("edge") is not None:
                evol, level = row["edge"]
                e = flip(ndimage.gaussian_filter(planes(evol, centre)[k][1].T, 1.0))
                if e.max() > level:
                    ax.contour(e, levels=[level], colors="#00e5ff", linewidths=0.5)
            lo_x, hi_x, lo_y, hi_y = LETTERS[name]
            for (x, y, t) in ((0.01, 0.5, lo_x), (0.97, 0.5, hi_x), (0.5, 0.01, lo_y), (0.5, 0.95, hi_y)):
                ax.text(x, y, t, transform=ax.transAxes, color="y", fontsize=9, ha="center", va="center")
            ax.set_title(f"{row['label']} | {name}", fontsize=9)
            ax.axis("off")
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.35 / len(rows) / 4.2 * 4.2 / 3))
    fig.savefig(path, dpi=110)
    plt.close(fig)


def centroid(mask):
    idx = np.argwhere(mask)
    return [int(v) for v in np.round(idx.mean(0))] if idx.size else [s // 2 for s in mask.shape]


# ========================================================================= pipeline
class Prep:
    def __init__(self, a):
        self.a = a
        self.root = os.path.join(a.out_root, a.sub)
        for d in ("native", "xfm", "grid", "qc", "logs"):
            os.makedirs(os.path.join(self.root, d), exist_ok=True)
        self.t1, self.t2 = (a.t1, a.t2) if (a.t1 and a.t2) else find_inputs(a.data_dir, a.sub)
        self.nmt = find_nmt(a.nmt_dir)
        self.mpath = self.p("manifest.json")
        self.m = json.load(open(self.mpath)) if os.path.exists(self.mpath) else {}
        self.m.setdefault("stages", {})
        self.m.update(subject=a.sub, inputs={"T1w": self.t1, "T2w": self.t2}, nmt=self.nmt,
                      params={k: v for k, v in vars(a).items() if k not in ("t1", "t2")},
                      human_reference=HUMAN_REF)
        self.threads = os.environ.get("ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS", "8")
        n, x, g = (lambda f: self.p("native", f)), (lambda f: self.p("xfm", f)), (lambda f: self.p("grid", f))
        self.f = dict(
            t1_n4head=n("T1w_n4head.nii.gz"), t1_n4head_reg=n("T1w_n4head_reg.nii.gz"),
            mask=n("brain_mask.nii.gz"), t1_brain=n("T1w_brain.nii.gz"), t1_brain_reg=n("T1w_brain_reg.nii.gz"),
            t2_on_t1=n("T2w_on_T1w.nii.gz"), t2_fov=n("T2w_fov.nii.gz"), t2_brain=n("T2w_brain.nii.gz"),
            nmt2nat=x("nmt2nat_"), t2rig=x("t2rigid_"), brainaff=x("brainaff_"),
            grid_ref=g("grid_ref.nii.gz"), t1_grid=g("T1w_grid.nii.gz"), mask_grid=g("brain_mask_grid.nii.gz"),
            t2_grid=g("T2w_grid.nii.gz"), fov_grid=g("T2w_fov_grid.nii.gz"), valid_grid=g("valid_grid.nii.gz"),
            nmtseg_grid=g("NMTseg_grid.nii.gz"), pveseg=g("T1w_grid_pveseg.nii.gz"),
            input_npy=g("T1w_grid_pveseg.npy"), target_npy=g("T2w_grid_target.npy"),
            target_nii=g("T2w_grid_target.nii.gz"))

    # ------------------------------------------------------------------ helpers
    def p(self, *parts):
        return os.path.join(self.root, *parts)

    def save_manifest(self):
        tmp = self.mpath + ".tmp"
        with open(tmp, "w") as fh:
            json.dump(self.m, fh, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
        os.replace(tmp, self.mpath)

    def done(self, *paths):
        return (not self.a.force) and all(os.path.exists(x) for x in paths)

    def run(self, cmd, stage):
        cmd = [str(c) for c in cmd]
        log = self.p("logs", f"{stage}.log")
        print("    $ " + " ".join(cmd), flush=True)
        env = dict(os.environ, ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=self.threads)
        with open(log, "a") as fh:
            fh.write("\n$ " + " ".join(cmd) + "\n")
            fh.flush()
            r = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env)
        if r.returncode != 0:
            raise RuntimeError(f"command failed (exit {r.returncode}); see {log}")

    def downsample(self, src, dst, stage):
        s = self.a.reg_spacing
        self.run(["ResampleImageBySpacing", 3, src, dst, s, s, s, 1], stage)

    def apply(self, src, ref, dst, transforms, interp, stage):
        cmd = ["antsApplyTransforms", "-d", 3, "-i", src, "-r", ref, "-o", dst, "-n", interp, "--float", 1]
        for t in transforms:
            cmd += ["-t", t]
        self.run(cmd, stage)

    def stage(self, num, name, fn):
        key = f"{num:02d}_{name}"
        print(f"[{self.a.sub}] stage {key}", flush=True)
        t0 = time.time()
        info = fn(key) or {}
        rec = self.m["stages"].get(key, {})
        rec.update(info)
        rec.update(status="done", seconds=round(time.time() - t0, 1),
                   finished=time.strftime("%Y-%m-%d %H:%M:%S"))
        self.m["stages"][key] = rec
        self.save_manifest()

    # ------------------------------------------------------------------- stages
    def s01_header(self, key):
        t1i, t2i = nib.load(self.t1), nib.load(self.t2)
        M = np.linalg.inv(t1i.affine) @ t2i.affine  # T2w voxel -> T1w voxel
        R, tr = M[:3, :3], M[:3, 3]
        subgrid = bool(np.allclose(R, np.eye(3), atol=1e-3) and np.allclose(tr, np.round(tr), atol=1e-2))
        off = np.round(tr).astype(int)
        inside = bool(subgrid and (off >= 0).all() and (off + np.array(t2i.shape[:3]) <= np.array(t1i.shape[:3])).all())
        qc = self.p("qc", "qc_00_raw_orientation.png")
        if not self.done(qc):
            vol = canon(t1i)
            qc_figure(qc, [{"img": vol, "label": "raw T1w"}], [s // 2 for s in vol.shape],
                      f"{self.a.sub}: raw T1w, check anatomical orientation letters")
        return {"t1_shape": t1i.shape, "t2_shape": t2i.shape,
                "t1_zooms": t1i.header.get_zooms()[:3], "t2_zooms": t2i.header.get_zooms()[:3],
                "axcodes": "".join(nib.aff2axcodes(t1i.affine)),
                "t2_voxel_to_t1_voxel": M.round(5), "t2_is_exact_subgrid": subgrid,
                "t2_subgrid_offset": off if subgrid else None, "t2_subgrid_inside_t1": inside,
                "input_sha256": {"T1w": sha256(self.t1), "T2w": sha256(self.t2)}}

    def s02_n4_head(self, key):
        f = self.f
        if not self.done(f["t1_n4head"]):
            self.run(["N4BiasFieldCorrection", "-d", 3, "-i", self.t1, "-o", f["t1_n4head"],
                      "-s", 4, "-b", "[50]", "-c", "[50x50x50x50,1e-7]"], key)
        if not self.done(f["t1_n4head_reg"]):
            self.downsample(f["t1_n4head"], f["t1_n4head_reg"], key)

    def s03_nmt_to_native(self, key):
        f, pre = self.f, self.f["nmt2nat"]
        if self.done(pre + "0GenericAffine.mat", pre + "1Warp.nii.gz"):
            return
        fixed, moving = f["t1_n4head_reg"], self.nmt["head_reg"]
        # dilated NMT brain mask on the moving grid, to focus SyN on the brain
        mdil = pre + "nmt_brainmask_dil.nii.gz"
        self.apply(self.nmt["mask_reg"], moving, mdil, [], "NearestNeighbor", key)
        rad = max(1, int(round(3.0 / float(nib.load(moving).header.get_zooms()[0]))))
        self.run(["ImageMath", 3, mdil, "MD", mdil, rad], key)
        if self.a.init == "antsai":
            self.run(["antsAI", "-d", 3, "-v", 1, "-m", f"Mattes[{fixed},{moving},32,Regular,0.25]",
                      "-t", "Rigid[0.1]", "-s", "[20,0.12]", "-g", "[5,0x0x0]", "-p", 0, "-c", 10,
                      "-o", pre + "init.mat"], key)
            init = pre + "init.mat"
        else:
            init = f"[{fixed},{moving},1]"
        lin = ["-c", "[1000x500x250,1e-6,10]", "-f", "4x2x1", "-s", "2x1x0vox"]
        self.run(["antsRegistration", "-d", 3, "-v", 1, "--float", 1,
                  "-o", f"[{pre},{pre}Warped.nii.gz]", "-n", "Linear", "-u", 0, "-w", "[0.005,0.995]",
                  "-r", init,
                  "-t", "Rigid[0.1]", "-m", f"MI[{fixed},{moving},1,32,Regular,0.25]", *lin, "-x", "[NULL,NULL]",
                  "-t", "Affine[0.1]", "-m", f"MI[{fixed},{moving},1,32,Regular,0.25]", *lin, "-x", "[NULL,NULL]",
                  "-t", "SyN[0.1,3,0]", "-m", f"CC[{fixed},{moving},1,3]",
                  "-c", "[100x70x30,1e-6,10]", "-f", "4x2x1", "-s", "2x1x0vox", "-x", f"[NULL,{mdil}]"], key)

    def s04_mask(self, key):
        f, pre = self.f, self.f["nmt2nat"]
        if not self.done(f["mask"]):
            prob = self.p("native", "brain_mask_prob.nii.gz")
            self.apply(self.nmt["mask"], self.t1, prob, [pre + "1Warp.nii.gz", pre + "0GenericAffine.mat"],
                       "Linear", key)
            pimg, pv = load(prob)
            mask = largest_component_filled(pv > 0.5)
            save_like(mask, pimg, f["mask"], np.uint8)
            os.remove(prob)
        if not self.done(f["t1_brain"]):
            tmp = self.p("native", "T1w_n4brain_tmp.nii.gz")
            self.run(["N4BiasFieldCorrection", "-d", 3, "-i", self.t1, "-x", f["mask"], "-o", tmp,
                      "-s", 4, "-b", "[30]", "-c", "[50x50x50x50,1e-7]"], key)
            timg, tv = load(tmp)
            _, mv = load(f["mask"], np.uint8)
            save_like(np.clip(tv, 0, None) * (mv > 0), timg, f["t1_brain"])
            os.remove(tmp)
        if not self.done(f["t1_brain_reg"]):
            self.downsample(f["t1_brain"], f["t1_brain_reg"], key)
        mimg, mv = load(f["mask"], np.uint8)
        vol_ml = float(mv.sum() * np.prod(mimg.header.get_zooms()[:3]) / 1000)
        return {"brain_volume_ml": round(vol_ml, 2)}

    def s05_t2(self, key):
        f = self.f
        info = {}
        t2_transforms = []
        applied_flag = self.p("xfm", "t2rigid_APPLIED")
        if self.a.force and os.path.exists(applied_flag):
            os.remove(applied_flag)
        if not self.done(f["t2_on_t1"], f["t2_fov"]):
            self._t2_to_t1(t2_transforms, key)
        if not self.done(f["t2_brain"]):
            self._t2_n4(key)
        rigid_mat = f["t2rig"] + "0GenericAffine.mat"
        if not os.path.exists(rigid_mat) or self.a.force:
            # Native resolution: at the coarser registration spacing, estimation noise alone
            # can exceed half a native voxel and would trigger needless re-interpolation.
            self.run(["antsRegistration", "-d", 3, "-v", 1, "--float", 1, "-o", f"[{f['t2rig']}]",
                      "-n", "Linear", "-w", "[0.005,0.995]",
                      "-t", "Rigid[0.1]", "-m", f"MI[{f['t1_brain']},{f['t2_brain']},1,32,Regular,0.1]",
                      "-c", "[200x100x50,1e-7,10]", "-f", "4x2x1", "-s", "2x1x0vox"], key)
        A, t, c = read_itk_affine(rigid_mat)
        mimg, mv = load(f["mask"], np.uint8)
        idx = np.argwhere(mv > 0)[::53]  # subsample of brain voxels
        ras = (mimg.affine @ np.c_[idx, np.ones(len(idx))].T).T[:, :3]
        lps = ras * np.array([-1, -1, 1])  # ITK transforms live in LPS
        disp = np.linalg.norm((lps - c) @ (A - np.eye(3)).T + t, axis=1)
        angle = math.degrees(math.acos(float(np.clip((np.trace(A) - 1) / 2, -1, 1))))
        half_vox = 0.5 * float(min(mimg.header.get_zooms()[:3]))
        p95 = float(np.percentile(disp, 95))
        use = self.a.t2_rigid == "always" or (self.a.t2_rigid == "auto" and p95 > half_vox)
        info.update(t2_rigid_brain_displacement_mm={"mean": round(float(disp.mean()), 4), "p95": round(p95, 4),
                                                    "max": round(float(disp.max()), 4)},
                    t2_rigid_rotation_deg=round(angle, 4), t2_rigid_translation_mm=np.round(t, 4),
                    t2_rigid_threshold_p95_mm=half_vox, t2_rigid_applied=bool(use))
        if use and not os.path.exists(applied_flag):
            print("    T2w->T1w rigid correction exceeds half a voxel: re-resampling T2w with it", flush=True)
            self._t2_to_t1([rigid_mat], key)
            self._t2_n4(key)
            open(applied_flag, "w").write(rigid_mat + "\n")
        _, fov = load(f["t2_fov"], np.uint8)
        info["t2_fov_coverage_of_brain"] = round(float((fov[mv > 0] > 0).mean()), 5)
        return info

    def _t2_to_t1(self, transforms, key):
        f = self.f
        t2i = nib.load(self.t2)
        ones = self.p("native", "T2w_fov_src.nii.gz")
        save_like(np.ones(t2i.shape[:3], np.uint8), t2i, ones, np.uint8)
        self.apply(self.t2, self.t1, f["t2_on_t1"], transforms, "Linear", key)
        self.apply(ones, self.t1, f["t2_fov"], transforms, "NearestNeighbor", key)
        os.remove(ones)

    def _t2_n4(self, key):
        f = self.f
        _, mv = load(f["mask"], np.uint8)
        fimg, fov = load(f["t2_fov"], np.uint8)
        n4mask = self.p("native", "T2w_n4mask_tmp.nii.gz")
        save_like((mv > 0) & (fov > 0), fimg, n4mask, np.uint8)
        tmp = self.p("native", "T2w_n4_tmp.nii.gz")
        self.run(["N4BiasFieldCorrection", "-d", 3, "-i", f["t2_on_t1"], "-x", n4mask, "-o", tmp,
                  "-s", 4, "-b", "[30]", "-c", "[50x50x50x50,1e-7]"], key)
        timg, tv = load(tmp)
        save_like(np.clip(tv, 0, None) * (mv > 0) * (fov > 0), timg, f["t2_brain"])
        os.remove(tmp)
        os.remove(n4mask)

    def s06_grid(self, key):
        f, a = self.f, self.a
        pre = f["brainaff"]
        aff = pre + "0GenericAffine.mat"
        if not self.done(aff):
            lin = ["-c", "[1000x500x250,1e-6,10]", "-f", "4x2x1", "-s", "2x1x0vox"]
            self.run(["antsRegistration", "-d", 3, "-v", 1, "--float", 1, "-o", f"[{pre},{pre}Warped.nii.gz]",
                      "-n", "Linear", "-u", 0, "-w", "[0.005,0.995]", "-r", f["nmt2nat"] + "0GenericAffine.mat",
                      "-t", "Rigid[0.1]", "-m", f"MI[{f['t1_brain_reg']},{self.nmt['ss_reg']},1,32,Regular,0.25]", *lin,
                      "-t", "Affine[0.1]", "-m", f"MI[{f['t1_brain_reg']},{self.nmt['ss_reg']},1,32,Regular,0.25]", *lin],
                     key)
        if not self.done(f["grid_ref"]):
            mimg = nib.load(self.nmt["mask"])
            idx = np.argwhere(np.asarray(mimg.dataobj) > 0)
            cw = mimg.affine @ np.r_[(idx.min(0) + idx.max(0)) / 2.0, 1.0]
            N, s = a.grid_size, a.grid_spacing
            A = np.diag([-s, s, s, 1.0])  # LAS, like GAN-MAT's human MNI arrays
            A[:3, 3] = cw[:3] - A[:3, :3] @ np.full(3, (N - 1) / 2.0)
            ref = nib.Nifti1Image(np.zeros((N, N, N), np.uint8), A)
            ref.header.set_qform(A, code=1)
            ref.header.set_sform(A, code=1)
            nib.save(ref, f["grid_ref"])
        inv = [f"[{aff},1]"]
        outs = [f["t1_grid"], f["mask_grid"], f["t2_grid"], f["fov_grid"], f["nmtseg_grid"], f["valid_grid"]]
        if not self.done(*outs):
            tmp = {k: self.p("grid", f"_{k}.nii.gz") for k in ("t1", "mask", "t2", "fov")}
            self.apply(f["t1_brain"], f["grid_ref"], tmp["t1"], inv, "Linear", key)
            self.apply(f["mask"], f["grid_ref"], tmp["mask"], inv, "Linear", key)
            self.apply(f["t2_brain"], f["grid_ref"], tmp["t2"], inv, "Linear", key)
            self.apply(f["t2_fov"], f["grid_ref"], tmp["fov"], inv, "NearestNeighbor", key)
            self.apply(self.nmt["seg"], f["grid_ref"], f["nmtseg_grid"], [], "NearestNeighbor", key)
            gimg = nib.load(f["grid_ref"])
            m = largest_component_filled(load(tmp["mask"])[1] > 0.5)
            fov = load(tmp["fov"])[1] > 0.5
            save_like(m, gimg, f["mask_grid"], np.uint8)
            save_like(m & fov, gimg, f["valid_grid"], np.uint8)
            save_like(fov, gimg, f["fov_grid"], np.uint8)
            save_like(np.clip(load(tmp["t1"])[1], 0, None) * m, gimg, f["t1_grid"])
            save_like(np.clip(load(tmp["t2"])[1], 0, None) * m * fov, gimg, f["t2_grid"])
            for t in tmp.values():
                os.remove(t)
        _, m = load(f["mask_grid"], np.uint8)
        idx = np.argwhere(m > 0)
        lo, hi = idx.min(0), idx.max(0)
        margin = int(min(lo.min(), (a.grid_size - 1 - hi).min()))
        return {"grid_affine": nib.load(f["grid_ref"]).affine.round(4),
                "grid_brain_bbox_vox": (hi - lo + 1), "grid_min_margin_vox": margin,
                "grid_brain_fraction": round(float(m.mean()), 4),
                "to_native_cmd": f"antsApplyTransforms -d 3 -i <grid_image> -r {self.t1} -t {aff} -n Linear -o <out>"}

    def s07_fast(self, key):
        f = self.f
        if not self.done(f["pveseg"]):
            prefix = f["t1_grid"].replace(".nii.gz", "")
            self.run(["fast", "-t", 1, "-n", 3, "-N", "-o", prefix, f["t1_grid"]], key)
            for extra in glob.glob(prefix + "_pve_*.nii.gz") + glob.glob(prefix + "_mixeltype.nii.gz"):
                os.remove(extra)
        _, t1 = load(f["t1_grid"])
        _, seg = load(f["pveseg"], np.int16)
        _, nmt = load(f["nmtseg_grid"], np.int16)
        means = [float(t1[seg == k].mean()) if (seg == k).any() else float("nan") for k in (1, 2, 3)]
        order_ok = bool(means[0] < means[1] < means[2])
        if not self.done(f["input_npy"]):
            ch = np.zeros(t1.shape + (3,), np.float32)
            for k in (1, 2, 3):  # same construction as GAN-MAT functions/preprocessing.py
                sel = seg == k
                ch[..., k - 1][sel] = t1[sel]
            np.save(f["input_npy"], ch)
        ch = np.load(f["input_npy"], mmap_mode="r")
        fr, zbg, ztis = [], [], []
        for k in range(3):  # exactly dataset.py's per-channel z-score over the whole volume
            x = np.asarray(ch[..., k], dtype=np.float64)
            z = (x - x.mean()) / x.std()
            fr.append(round(float((x != 0).mean()), 4))
            zbg.append(round(float(z[x == 0].mean()), 3))
            ztis.append(round(float(z[x != 0].mean()), 3))
        dice = {}
        for i, (name, labs) in enumerate(NMT_TISSUE_MAP.items(), start=1):
            a_, b_ = seg == i, np.isin(nmt, labs)
            dice[name] = round(float(2 * (a_ & b_).sum() / max(1, a_.sum() + b_.sum())), 3)
        return {"fast_label_mean_T1": [round(v, 2) for v in means], "fast_label_order_ok": order_ok,
                "channel_fractions": fr, "zscore_background": zbg, "zscore_tissue_mean": ztis,
                "dice_vs_affine_NMT_segmentation": dice}

    def s08_target(self, key):
        f = self.f
        gimg, t2 = load(f["t2_grid"])
        _, valid = load(f["valid_grid"], np.uint8)
        v = valid > 0
        p999 = float(np.percentile(t2[v], 99.9))
        tgt = np.full(t2.shape, -1.0, np.float32)
        tgt[v] = np.clip(2.0 * t2[v] / p999 - 1.0, -1.0, 1.0)
        if not self.done(f["target_npy"]):
            np.save(f["target_npy"], tgt)
            save_like(tgt, gimg, f["target_nii"])
        return {"target_scale_p99_9": round(p999, 3),
                "target_brain_mean": round(float(tgt[v].mean()), 4),
                "target_brain_percentiles_5_50_95": np.round(np.percentile(tgt[v], [5, 50, 95]), 4),
                "target_clipped_fraction": round(float((tgt[v] >= 1.0).mean()), 5)}

    def s09_qc(self, key):
        f, sub = self.f, self.a.sub
        mask = canon(f["mask"], np.uint8) > 0
        c = centroid(mask)
        head = canon(f["t1_n4head"])
        qc_figure(self.p("qc", "qc_01_brain_mask.png"),
                  [{"img": head, "label": "T1w N4 + brain mask", "contour": mask}], c,
                  f"{sub}: NMT-derived brain mask on native T1w (red)")
        t1b, t2b = canon(f["t1_brain"]), canon(f["t2_brain"])
        level = float(np.percentile(t1b[mask], 60))
        qc_figure(self.p("qc", "qc_02_t2_alignment.png"),
                  [{"img": t1b, "label": "T1w brain", "edge": (t1b, level)},
                   {"img": t2b, "label": "T2w brain (on T1w grid)", "edge": (t1b, level)}], c,
                  f"{sub}: cyan = T1w GM/WM boundary, drawn on both modalities")
        del head, t1b, t2b, mask
        gm = canon(f["mask_grid"], np.uint8) > 0
        cg = centroid(gm)
        g_t1, seg = canon(f["t1_grid"]), canon(f["pveseg"], np.int16)
        ch = np.load(f["input_npy"], mmap_mode="r")
        gimg = nib.load(f["grid_ref"])
        ch_c = [canon(nib.Nifti1Image(np.asarray(ch[..., k]), gimg.affine)) for k in range(3)]
        tgt = canon(f["target_nii"])
        qc_figure(self.p("qc", "qc_03_grid_inputs_target.png"),
                  [{"img": g_t1, "label": "grid T1w + FAST (blue CSF, green GM, yellow WM)", "labels": seg},
                   {"img": ch_c[0], "label": "channel 0 = CSF"},
                   {"img": ch_c[1], "label": "channel 1 = GM"},
                   {"img": ch_c[2], "label": "channel 2 = WM"},
                   {"img": tgt, "label": "T2w target [-1, 1]", "vmin": -1.0, "vmax": 1.0}], cg,
                  f"{sub}: GAN-MAT grid ({self.a.grid_size}^3, {self.a.grid_spacing} mm)")
        st = self.m["stages"]
        g = lambda s, k: st.get(s, {}).get(k)  # noqa: E731
        checks = {
            "brain_volume_60_to_130_ml": bool(60 <= (g("04_mask", "brain_volume_ml") or 0) <= 130),
            "t2_fov_covers_brain_99.5pct": bool((g("05_t2", "t2_fov_coverage_of_brain") or 0) >= 0.995),
            "grid_margin_at_least_2_vox": bool((g("06_grid", "grid_min_margin_vox") or -1) >= 2),
            "fast_label_order_csf_gm_wm": bool(g("07_fast", "fast_label_order_ok")),
        }
        self.m["auto_checks"] = checks
        self.m["auto_checks_passed"] = all(checks.values())
        self.m.setdefault("visual_qc", "PENDING: inspect qc/*.png, then set to PASS or FAIL with a note")
        return {"figures": sorted(glob.glob(self.p("qc", "*.png")))}

    def run_all(self):
        stages = [(1, "header", self.s01_header), (2, "n4_head", self.s02_n4_head),
                  (3, "nmt_to_native", self.s03_nmt_to_native), (4, "mask", self.s04_mask),
                  (5, "t2", self.s05_t2), (6, "grid", self.s06_grid), (7, "fast", self.s07_fast),
                  (8, "target", self.s08_target), (9, "qc", self.s09_qc)]
        for num, name, fn in stages:
            if num > self.a.stop_after:
                break
            self.stage(num, name, fn)
        print(f"[{self.a.sub}] finished. manifest: {self.mpath}", flush=True)


def main():
    a = parse_args()
    if not a.nmt_dir:
        sys.exit("missing --nmt-dir / NMT_DIR; did you `source ganmat_env.sh`?")
    if not (a.t1 and a.t2) and not a.data_dir:
        sys.exit("missing --data-dir / DATA_DIR (or pass both --t1 and --t2)")
    prep = Prep(a)
    try:
        prep.run_all()
    except Exception as e:
        prep.m["error"] = f"{type(e).__name__}: {e}"
        prep.save_manifest()
        print(f"[{a.sub}] FAILED: {e}", file=sys.stderr, flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
