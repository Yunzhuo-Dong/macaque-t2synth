#!/usr/bin/env python
"""07_finetune.py — E3: fine-tune GAN-MAT's generator on macaque data (reconstruction loss).

Inputs per subject come from 04_preprocess.py (grid channels + [-1, 1] T2w target + valid mask),
normalised as in 05_e2_zeroshot.py (default: plain FAST channels, human-matched z-score, the
best zero-shot condition in E2).

Design decisions (see project notes):
  * BatchNorm FROZEN by default (running statistics from the human training data; the 1x1x1
    bottleneck cannot be trained with batch size 1). BN affine weights stay trainable.
  * Loss = mean |pred - target| inside brain AND T2w FOV  +  bg_weight * mean |...| outside.
  * Whole-volume augmentation only (the 8-level U-Net needs 256^3): left-right flip and integer
    shifts of up to --shift voxels, filled with each channel's background value.
  * Subject-level k-fold splits, written once to <out-root>/splits_k<K>_seed<S>.json and reused
    by every experiment, so all conditions see identical train / val / test subjects.
  * --init scratch trains the same architecture from random weights: the control that shows
    whether human pre-training helps.

Multi-GPU (DistributedDataParallel): launch with torchrun; each GPU takes a different subject per
step, so one optimiser step sees --nproc_per_node volumes (effective batch = number of GPUs).
Results are then not directly comparable with single-GPU runs: use one setup for all experiments.
    torchrun --standalone --nproc_per_node 2 07_finetune.py --fold 0

Usage:
    source ganmat_env.sh && export CUDA_VISIBLE_DEVICES=1
    # 1) sanity check: overfit one subject (should drive the error far below zero-shot)
    "$VENV_PY" 07_finetune.py --overfit sub-032128 --steps 300 --val-every 50
    # 2) one fold of the real experiment
    "$VENV_PY" 07_finetune.py --fold 0
    # 3) then evaluate that fold's test subjects in native space (command printed at the end)
"""
import argparse
import contextlib
import csv
import glob
import importlib.util
import json
import math
import os
import sys
import time

import nibabel as nib
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def load_module(name, fname):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, fname))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


E2 = load_module("e2mod", "05_e2_zeroshot.py")  # normalise(), SEGS, VARIANTS; E2.P = 04 helpers
P = E2.P


def parse_args():
    env = os.environ.get
    proj = env("PROJECT_DIR", ".")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_argument_group("data / split")
    g.add_argument("--prep-root", default=os.path.join(proj, "derivatives", "prep"))
    g.add_argument("--out-root", default=os.path.join(proj, "derivatives", "e3"))
    g.add_argument("--fold", type=int, default=0)
    g.add_argument("--k", type=int, default=5)
    g.add_argument("--n-val", type=int, default=2, help="validation subjects taken from the training part")
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--overfit", metavar="SUB", help="train and validate on this one subject (sanity check)")
    g.add_argument("--seg", choices=list(E2.SEGS), default="fast")
    g.add_argument("--norm", choices=list(E2.VARIANTS), default="human_matched")
    g.add_argument("--target", choices=["n4", "sharedF"], default="n4",
                   help="T2w target: separately N4-corrected (04_preprocess.py) or corrected with the T1w's "
                        "own N4 field (04b_shared_bias_target.py; keeps the T1w/T2w myelin contrast)")
    g.add_argument("--allow-unreviewed", action="store_true", help="include subjects whose visual_qc is not PASS")
    m = p.add_argument_group("model")
    m.add_argument("--ganmat-root", default=env("GANMAT_ROOT"))
    m.add_argument("--init", choices=["official", "scratch"], default="official")
    m.add_argument("--train-layers", choices=["all", "decoder", "last2"], default="all")
    m.add_argument("--bn", choices=["frozen", "train"], help="default: frozen (official), train (scratch)")
    m.add_argument("--dropout", choices=["on", "off"], default="on")
    m.add_argument("--nker", type=int, default=64, help=argparse.SUPPRESS)  # pipeline tests only
    t = p.add_argument_group("training")
    t.add_argument("--lr", type=float, default=1e-5)
    t.add_argument("--lr-schedule", choices=["cosine", "constant"], default="cosine",
                   help="cosine: decay from --lr to 5%% of it over the run (smooth end, last ~= best)")
    t.add_argument("--steps", type=int, default=4000, help="optimiser steps (ignored when --epochs is given)")
    t.add_argument("--epochs", type=float, help="passes over the training subjects; sets --steps")
    t.add_argument("--batch-size", type=int, default=1,
                   help="volumes per optimiser step per GPU, by gradient accumulation (memory stays at one "
                        "volume; exactly equivalent to a real batch while BatchNorm is frozen)")
    t.add_argument("--val-every", type=int, help="optimiser steps between validations "
                   "(default: once per epoch with --epochs, else 200)")
    t.add_argument("--bg-weight", type=float, default=0.1)
    t.add_argument("--shift", type=int, default=4)
    t.add_argument("--no-flip", action="store_true")
    t.add_argument("--amp", choices=["bf16", "off"], default="bf16")
    t.add_argument("--grad-ckpt", action="store_true", help="activation checkpointing (less memory, ~30%% slower)")
    t.add_argument("--device", default="cuda")
    t.add_argument("--tag", help="experiment name (default built from the settings)")
    t.add_argument("--resume", action="store_true")
    t.add_argument("--ckpt-every", type=int, default=200,
                   help="optimiser steps between resume checkpoints (~2.6 GB each, with Adam state); "
                        "also written at the end. The best model is saved whenever validation improves.")
    a = p.parse_args()
    a.bn = a.bn or ("frozen" if a.init == "official" else "train")
    a.tag = a.tag or (f"{a.init}_{a.seg}_{a.norm}_{a.train_layers}_lr{a.lr:g}"
                      + (f"_bs{a.batch_size}" if a.batch_size > 1 else "")
                      + ("_cos" if a.lr_schedule == "cosine" else "")
                      + ("" if a.target == "n4" else f"_{a.target}"))
    return a


# ======================================================================== subjects
def eligible(prep_root, allow_unreviewed):
    subs, skipped = [], {}
    for mp in sorted(glob.glob(os.path.join(prep_root, "sub-*", "manifest.json"))):
        s = os.path.basename(os.path.dirname(mp))
        m = json.load(open(mp))
        if not m.get("auto_checks_passed"):
            skipped[s] = "auto checks failed or incomplete"
        elif not allow_unreviewed and not str(m.get("visual_qc", "")).upper().startswith("PASS"):
            skipped[s] = "visual_qc not PASS"
        else:
            subs.append(s)
    return subs, skipped


def get_splits(subs, a):
    path = os.path.join(a.out_root, f"splits_k{a.k}_seed{a.seed}.json")
    if os.path.exists(path):
        sp = json.load(open(path))
        if sorted(sp["subjects"]) != sorted(subs):
            sys.exit(f"eligible subjects differ from {path}.\n  in file: {sp['subjects']}\n  now:     {sorted(subs)}\n"
                     "Delete the file deliberately if the cohort really changed (all folds must then be rerun).")
        return sp, path
    rng = np.random.default_rng(a.seed)
    order = [str(x) for x in rng.permutation(sorted(subs))]
    folds = [order[i::a.k] for i in range(a.k)]
    sp = {"subjects": sorted(subs), "k": a.k, "seed": a.seed, "n_val": a.n_val, "folds": []}
    for f in range(a.k):
        rest = [s for s in order if s not in folds[f]]
        sp["folds"].append({"test": sorted(folds[f]), "val": sorted(rest[:a.n_val]), "train": sorted(rest[a.n_val:])})
    os.makedirs(a.out_root, exist_ok=True)
    tmp = f"{path}.{os.getpid()}.tmp"  # atomic: parallel jobs never read a half-written file
    json.dump(sp, open(tmp, "w"), indent=2)
    os.replace(tmp, path)
    return sp, path


def load_subject(s, a):
    g = lambda f: os.path.join(a.prep_root, s, "grid", f)  # noqa: E731
    x, _ = E2.normalise(np.load(g(E2.SEGS[a.seg])), a.norm)
    x = np.ascontiguousarray(x.transpose(3, 0, 1, 2))
    tname = "T2w_grid_target.npy" if a.target == "n4" else f"T2w_grid_target_{a.target}.npy"
    if not os.path.exists(g(tname)):
        sys.exit(f"{s}: {tname} missing; run 04b_shared_bias_target.py first")
    y = np.load(g(tname)).astype(np.float32)[None]
    v = np.asarray(nib.load(g("valid_grid.nii.gz")).dataobj).astype(np.uint8)[None]
    return {"x": x, "y": y, "v": v, "bg": x[:, 0, 0, 0].copy()}  # corner voxel = background


def shift_fill(arr, s, fill):
    out = np.empty_like(arr)
    out[...] = fill
    dst, src = [slice(None)], [slice(None)]
    for k, n in zip(s, arr.shape[1:]):
        dst.append(slice(k, None) if k >= 0 else slice(0, n + k))
        src.append(slice(0, n - k) if k >= 0 else slice(-k, None))
    out[tuple(dst)] = arr[tuple(src)]
    return out


def augment(d, rng, a):
    x, y, v = d["x"], d["y"], d["v"]
    if not a.no_flip and rng.random() < 0.5:  # axis 1 = left-right in the LAS grid
        x, y, v = x[:, ::-1], y[:, ::-1], v[:, ::-1]
    if a.shift:
        s = rng.integers(-a.shift, a.shift + 1, 3)
        x = shift_fill(x, s, d["bg"][:, None, None, None])
        y = shift_fill(y, s, -1.0)
        v = shift_fill(v, s, 0)
    return np.ascontiguousarray(x), np.ascontiguousarray(y), np.ascontiguousarray(v)


# =========================================================================== model
def build(a, torch, device):
    sys.path.insert(0, os.path.join(a.ganmat_root, "functions", "model"))
    from model import Pix2Pix_3D
    net = Pix2Pix_3D(in_channels=3, out_channels=1, nker=a.nker)
    if a.init == "official":
        path = os.path.join(a.ganmat_root, "functions", "model", "model.pth")
        try:
            ck = torch.load(path, map_location="cpu")
        except Exception:  # official checkpoint (trusted) with non-tensor objects
            ck = torch.load(path, map_location="cpu", weights_only=False)
        net.load_state_dict(ck["netG"])
    else:  # pix2pix initialisation
        for mod in net.modules():
            if isinstance(mod, (torch.nn.Conv3d, torch.nn.ConvTranspose3d)):
                torch.nn.init.normal_(mod.weight, 0.0, 0.02)
                if mod.bias is not None:
                    torch.nn.init.zeros_(mod.bias)
            elif isinstance(mod, torch.nn.BatchNorm3d):
                torch.nn.init.normal_(mod.weight, 1.0, 0.02)
                torch.nn.init.zeros_(mod.bias)
    for n, prm in net.named_parameters():
        top = n.split(".")[0]
        prm.requires_grad = (a.train_layers == "all" or (a.train_layers == "decoder" and top.startswith("dec"))
                             or (a.train_layers == "last2" and top in ("dec7", "dec8")))
    wrapped = []
    if a.grad_ckpt:
        from torch.utils.checkpoint import checkpoint

        class Ckpt(torch.nn.Module):
            def __init__(self, m):
                super().__init__()
                self.m = m

            def forward(self, *xs):
                return checkpoint(self.m, *xs, use_reentrant=False)

        for name in [f"enc{i}" for i in range(1, 9)] + [f"dec{i}" for i in range(1, 9)]:
            if hasattr(net, name):
                setattr(net, name, Ckpt(getattr(net, name)))
                wrapped.append(name)
    return net.to(device), set(wrapped)


def set_train_modes(net, a, torch):
    net.train()
    for name, mod in net.named_modules():
        if isinstance(mod, torch.nn.modules.batchnorm._BatchNorm) and (a.bn == "frozen" or name.startswith("enc8")):
            mod.eval()  # enc8 sees a 1x1x1 map: batch statistics are undefined with batch size 1
        if isinstance(mod, torch.nn.Dropout3d) and a.dropout == "off":
            mod.eval()


def clean_state(net, wrapped):
    out = {}
    for k, v in net.state_dict().items():
        parts = k.split(".")
        if parts[0] in wrapped and len(parts) > 2 and parts[1] == "m":
            k = ".".join([parts[0]] + parts[2:])
        out[k] = v.detach().cpu()
    return out


# ========================================================================= helpers
def to_t(arr, torch, device):
    return torch.from_numpy(arr).unsqueeze(0).to(device, non_blocking=True)


def predict(net, d, torch, device, amp):
    net.eval()
    with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp):
        out = net(to_t(d["x"], torch, device))
    return out.float()[0, 0].cpu().numpy()


def preview(path, d, pred, title):
    c = int(np.argwhere(d["v"][0]).mean(0)[1])  # coronal slice through the brain centre
    panels = [(d["x"].sum(0)[:, c, :], "input (sum of channels)", None),
              (d["y"][0][:, c, :], "target T2w", (-1, 1)), (pred[:, c, :], "prediction", (-1, 1)),
              (np.abs(pred - d["y"][0])[:, c, :] * d["v"][0][:, c, :], "|error| in mask", (0, 0.5))]
    fig, axes = P.plt.subplots(1, 4, figsize=(15, 4))
    for ax, (img, lab, rng) in zip(axes, panels):
        kw = {"vmin": rng[0], "vmax": rng[1]} if rng else {}
        ax.imshow(img.T, origin="lower", cmap="magma" if "error" in lab else "gray", **kw)
        ax.set_title(lab, fontsize=9)
        ax.axis("off")
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=90)
    P.plt.close(fig)


# ============================================================================ main
def main():
    a = parse_args()
    import torch
    import torch.distributed as dist
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank, local_rank = int(os.environ.get("RANK", "0")), int(os.environ.get("LOCAL_RANK", "0"))
    ddp = world > 1
    use_cuda = a.device != "cpu" and torch.cuda.is_available()
    if ddp:
        if use_cuda:
            torch.cuda.set_device(local_rank)
        dist.init_process_group("nccl" if use_cuda else "gloo")
    device = torch.device(f"cuda:{local_rank}" if use_cuda else "cpu")
    main_proc = rank == 0
    log = print if main_proc else (lambda *x, **k: None)
    torch.manual_seed(a.seed)                                  # identical initial weights on every rank
    rng = np.random.default_rng(a.seed + 1000 * a.fold)        # identical subject order on every rank
    arng = np.random.default_rng(a.seed + 1000 * a.fold + 17 * rank + 1)  # per-rank augmentation
    amp = a.amp == "bf16" and device.type == "cuda"

    if a.overfit:
        split = {"train": [a.overfit], "val": [a.overfit], "test": []}
        out = os.path.join(a.out_root, a.tag, f"overfit_{a.overfit}")
        a.shift, a.no_flip, a.dropout = 0, True, "off"  # pure capacity check
    else:
        subs, skipped = eligible(a.prep_root, a.allow_unreviewed)
        for s, why in skipped.items():
            log(f"  excluded {s}: {why}")
        if len(subs) < a.k + a.n_val + 1:
            sys.exit(f"only {len(subs)} eligible subjects for k={a.k}, n_val={a.n_val}")
        if ddp and not main_proc:
            dist.barrier()  # let rank 0 write the split file first
        sp, sp_path = get_splits(subs, a)
        if ddp and main_proc:
            dist.barrier()
        split = sp["folds"][a.fold]
        out = os.path.join(a.out_root, a.tag, f"fold{a.fold}")
        log(f"splits: {sp_path}")
    os.makedirs(out, exist_ok=True)
    per_step = a.batch_size * world  # volumes per optimiser step
    steps_per_epoch = max(1, math.ceil(len(split["train"]) / per_step))
    if a.epochs:
        a.steps = max(1, math.ceil(a.epochs * len(split["train"]) / per_step))
    if a.val_every is None:
        a.val_every = steps_per_epoch if a.epochs else 200
    log(f"schedule: {a.steps} optimiser steps x {per_step} volumes = "
        f"{a.steps * per_step / len(split['train']):.1f} epochs over {len(split['train'])} training subjects; "
        f"validation every {a.val_every} steps")
    log(f"output: {out}\ntrain {split['train']}\nval   {split['val']}\ntest  {split['test']}")

    t0 = time.time()
    data = {}
    for r in range(world):  # ranks load one after another: avoids simultaneous memory / disk peaks
        if r == rank:
            data = {s: load_subject(s, a) for s in sorted(set(split["train"]) | set(split["val"]))}
        if ddp:
            dist.barrier()
    log(f"loaded {len(data)} subjects in {time.time() - t0:.0f} s")

    net, wrapped = build(a, torch, device)
    params = [prm for prm in net.parameters() if prm.requires_grad]
    opt = torch.optim.Adam(params, lr=a.lr, betas=(0.5, 0.999))
    n_tr = sum(prm.numel() for prm in params)
    log(f"device {device} x {world} | amp {amp} | init {a.init} | bn {a.bn} | dropout {a.dropout} | "
        f"trainable {n_tr / 1e6:.1f}M params | grad-ckpt {bool(wrapped)} | effective batch {per_step} "
        f"({world} GPU x {a.batch_size} accumulated)")

    meta = {k: v for k, v in vars(a).items()}
    meta.update(split=split, nker=a.nker, world_size=world, steps_derived=a.steps, val_every_derived=a.val_every)
    if main_proc:
        json.dump(meta, open(os.path.join(out, "config.json"), "w"), indent=2)
    start, best, best_step = 1, float("inf"), 0
    last_ck = os.path.join(out, "ckpt_last.pth")
    if a.resume and os.path.exists(last_ck):
        ck = torch.load(last_ck, map_location="cpu", weights_only=False)
        net_sd = {}
        for k, v in ck["netG"].items():  # map clean keys back onto wrapped modules
            parts = k.split(".")
            net_sd[".".join([parts[0], "m"] + parts[1:]) if parts[0] in wrapped else k] = v
        net.load_state_dict(net_sd)
        opt.load_state_dict(ck["opt"])
        start, best, best_step = ck["step"] + 1, ck["best"], ck["best_step"]
        log(f"resumed from step {ck['step']} (best val {best:.4f} at {best_step})")

    core = net  # unwrapped module: validation, checkpoints, mode switches
    if ddp:
        from torch.nn.parallel import DistributedDataParallel as DDP
        net = DDP(net, device_ids=[local_rank] if use_cuda else None)
    tlog = open(os.path.join(out, "train_log.csv"), "a", newline="") if main_proc else open(os.devnull, "w")
    vlog = open(os.path.join(out, "val_log.csv"), "a", newline="") if main_proc else open(os.devnull, "w")
    tw, vw = csv.writer(tlog), csv.writer(vlog)
    if start == 1 and main_proc:
        tw.writerow(["step", "epoch", "lr", "loss", "l1_brain", "l1_bg", "sec", "max_mem_gib"])
        vw.writerow(["step", "val_mae_brain", *split["val"]])

    def validate(step):
        maes = []
        for s in split["val"]:
            pred = predict(core, data[s], torch, device, amp)
            v = data[s]["v"][0] > 0
            maes.append(float(np.abs(pred - data[s]["y"][0])[v].mean()))
            if s == split["val"][0]:
                preview(os.path.join(out, f"preview_step{step:05d}.png"), data[s], pred,
                        f"{a.tag} | {s} | step {step} | val MAE {np.mean(maes):.4f}")
        set_train_modes(core, a, torch)
        return float(np.mean(maes)), maes

    if start == 1 and main_proc:
        v0, per = validate(0)
        vw.writerow([0, f"{v0:.5f}", *[f"{x:.5f}" for x in per]])
        vlog.flush()
        log(f"step     0 | val MAE (before fine-tuning) {v0:.4f}")

    if ddp:
        dist.barrier()
    set_train_modes(core, a, torch)
    queue = []
    for step in range(1, a.steps + 1):
        picks = []  # batch_size subjects per rank, from a subject order shared by all ranks
        for _ in range(per_step):
            if not queue:
                queue = list(rng.permutation(split["train"]))
            picks.append(queue.pop())
        if step < start:  # resuming: fast-forward the shared order without training
            continue
        mine = picks[rank * a.batch_size:(rank + 1) * a.batch_size]
        if a.lr_schedule == "cosine":  # stateless, so resuming needs no scheduler state
            frac = (step - 1) / max(1, a.steps - 1)
            for grp in opt.param_groups:
                grp["lr"] = a.lr * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * frac)))
        ts = time.time()
        opt.zero_grad(set_to_none=True)
        stats = torch.zeros(3, device=device)
        for i, sub in enumerate(mine):  # gradient accumulation, one volume in memory at a time
            d = data[sub]
            xa, ya, va = augment(d, arng, a) if (a.shift or not a.no_flip) else (d["x"], d["y"], d["v"])
            x, y, v = to_t(xa, torch, device), to_t(ya, torch, device), to_t(va, torch, device) > 0
            sync = contextlib.nullcontext() if (not ddp or i == len(mine) - 1) else net.no_sync()
            with sync:
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp):
                    pred = net(x)
                err = (pred.float() - y).abs()
                l_brain, l_bg = err[v].mean(), err[~v].mean()
                loss = l_brain + a.bg_weight * l_bg
                (loss / len(mine)).backward()
            stats += torch.stack([loss.detach(), l_brain.detach(), l_bg.detach()]) / len(mine)
            del x, y, v, pred, err
        opt.step()
        if device.type == "cuda":
            torch.cuda.synchronize()
        mem = torch.cuda.max_memory_allocated() / 2**30 if device.type == "cuda" else 0.0
        if ddp:  # log the mean over ranks
            dist.all_reduce(stats)
            stats /= world
        loss, l_brain, l_bg = stats
        tw.writerow([step, f"{step * per_step / len(split['train']):.2f}", f"{opt.param_groups[0]['lr']:.3g}",
                     f"{loss.item():.5f}",
                     f"{l_brain.item():.5f}", f"{l_bg.item():.5f}", f"{time.time() - ts:.2f}", f"{mem:.2f}"])
        if step == start or step % (1 if a.steps <= 200 else 20) == 0:
            tlog.flush()
            log(f"step {step:5d} | epoch {step * per_step / len(split['train']):6.2f} | loss {loss.item():.4f} "
                f"(brain {l_brain.item():.4f}, bg {l_bg.item():.4f}) | {time.time() - ts:.2f} s/step "
                f"| max GPU mem {mem:.1f} GiB", flush=True)
        if (step % a.val_every == 0 or step == a.steps) and not main_proc:
            dist.barrier()  # wait while rank 0 validates and saves
            continue
        if step % a.val_every == 0 or step == a.steps:
            vm, per = validate(step)
            vw.writerow([step, f"{vm:.5f}", *[f"{x:.5f}" for x in per]])
            vlog.flush()
            tag = ""
            if vm < best:
                best, best_step = vm, step
                torch.save({"netG": clean_state(core, wrapped), "meta": meta, "step": step, "val_mae": vm},
                           os.path.join(out, "ckpt_best.pth"))
                tag = "  <- best"
            if step % a.ckpt_every == 0 or step == a.steps:
                torch.save({"netG": clean_state(core, wrapped), "opt": opt.state_dict(), "step": step,
                            "best": best, "best_step": best_step, "meta": meta}, last_ck)
            print(f"step {step:5d} | epoch {step * per_step / len(split['train']):6.2f} | val MAE {vm:.4f}{tag}",
                  flush=True)
            if ddp:
                dist.barrier()
    tlog.close()
    vlog.close()
    if ddp:
        dist.destroy_process_group()
    if not main_proc:
        return
    json.dump({"best_val_mae": best, "best_step": best_step, "split": split, "tag": a.tag},
              open(os.path.join(out, "summary.json"), "w"), indent=2)
    print(f"\nbest val MAE {best:.4f} at step {best_step}; checkpoint {os.path.join(out, 'ckpt_best.pth')}")
    if split["test"]:
        print("evaluate the test subjects in native space with:\n"
              f'  "$VENV_PY" 05_e2_zeroshot.py --subs {" ".join(split["test"])} --checkpoint '
              f'{os.path.join(out, "ckpt_best.pth")} --label E3 --segs {a.seg} --variants {a.norm} '
              f'--fit-subs {" ".join(split["train"])} --target {a.target} --out-root {os.path.join(out, "eval")}')


if __name__ == "__main__":
    main()
