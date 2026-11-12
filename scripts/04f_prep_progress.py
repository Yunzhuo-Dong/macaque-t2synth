#!/usr/bin/env python
"""04f_prep_progress.py — progress of the batch preprocessing (read-only).

For each subject: done / running (current stage, minutes in it) / FAILED / pending,
plus an ETA from the subjects already finished, and current CPU load and free memory.

Usage:
    source ganmat_env.sh
    "$VENV_PY" 04f_prep_progress.py                      # subjects found in the full data set
    "$VENV_PY" 04f_prep_progress.py --watch 60           # refresh every 60 s (Ctrl-c to stop)
    "$VENV_PY" 04f_prep_progress.py --subs sub-032125 sub-032126
"""
import argparse
import glob
import json
import os
import re
import subprocess
import time

ws, proj = os.environ.get("WORKSPACE", ""), os.environ.get("PROJECT_DIR", ".")
ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--subs", nargs="*")
ap.add_argument("--data-dir", default=os.path.join(ws, "data", "PRIME-DE_UCDavis_full"))
ap.add_argument("--prep-root", default=os.path.join(proj, "derivatives", "prep"))
ap.add_argument("--log-dir", default=os.path.join(proj, "logs"))
ap.add_argument("--parallel", type=int, default=3, help="jobs running at once (for the ETA)")
ap.add_argument("--watch", type=int, default=0, help="refresh interval in seconds")
a = ap.parse_args()

STAGES = ["01_header", "02_n4_head", "03_nmt_to_native", "04_mask", "05_t2",
          "06_grid", "07_fast", "08_target", "09_qc"]


def subjects():
    if a.subs:
        return sorted(a.subs)
    found = {re.search(r"(sub-[A-Za-z0-9]+)", os.path.basename(f)).group(1)
             for f in glob.glob(os.path.join(a.data_dir, "**", "sub-*_T2w.nii*"), recursive=True)}
    found |= {os.path.basename(d) for d in glob.glob(os.path.join(a.prep_root, "sub-*"))}
    return sorted(found)


def running_subjects():
    try:
        out = subprocess.run(["pgrep", "-af", "04_preprocess.py"], capture_output=True, text=True).stdout
    except FileNotFoundError:
        return set()
    return set(re.findall(r"--sub\s+(sub-[A-Za-z0-9]+)", out))


def status(sub, running):
    root = os.path.join(a.prep_root, sub)
    man = {}
    if os.path.exists(os.path.join(root, "manifest.json")):
        try:
            man = json.load(open(os.path.join(root, "manifest.json")))
        except json.JSONDecodeError:  # being rewritten right now
            man = {}
    st = man.get("stages", {})
    done = [s for s in STAGES if st.get(s, {}).get("status") == "done"]
    minutes = sum(st[s].get("seconds", 0) for s in done) / 60
    logs = sorted(glob.glob(os.path.join(root, "logs", "*.log")), key=os.path.getmtime)
    last_write = (time.time() - os.path.getmtime(logs[-1])) / 60 if logs else None
    if sub in running:
        cur = STAGES[len(done)] if len(done) < len(STAGES) else "finishing"
        since = (time.time() - os.path.getmtime(os.path.join(root, "manifest.json"))) / 60 \
            if os.path.exists(os.path.join(root, "manifest.json")) else 0
        note = f"in {cur} for {since:.0f} min" + (f", last log write {last_write:.0f} min ago" if logs else "")
        return "running", len(done), minutes + since, note
    if len(done) == len(STAGES):
        chk = man.get("auto_checks_passed")
        return "done", len(done), minutes, "auto checks " + ("PASS" if chk else "FAIL")
    if man.get("error"):
        return "FAILED", len(done), minutes, man["error"][:90]
    if done or glob.glob(os.path.join(a.log_dir, f"prep_{sub}_*.log")):
        return "stopped", len(done), minutes, "not running and not finished (killed or failed early?)"
    return "pending", 0, 0, ""


def report():
    run = running_subjects()
    rows = [(s, *status(s, run)) for s in subjects()]
    print(time.strftime("%Y-%m-%d %H:%M:%S"),
          f"| load avg {os.getloadavg()[0]:.1f} / {os.cpu_count()} cpus", end="")
    try:
        mem = {l.split(":")[0]: int(l.split()[1]) for l in open("/proc/meminfo")}
        print(f" | memory available {mem['MemAvailable'] / 2**20:.0f} GiB")
    except OSError:
        print()
    print(f"{'subject':12s} {'status':8s} {'stages':>6s} {'min':>6s}  note")
    for s, stt, nd, mins, note in rows:
        print(f"{s:12s} {stt:8s} {nd:>4d}/9 {mins:6.1f}  {note}")
    c = {k: sum(1 for r in rows if r[1] == k) for k in ("done", "running", "FAILED", "stopped", "pending")}
    print("  ".join(f"{k}: {v}" for k, v in c.items()), end="")
    finished = sorted(r[3] for r in rows if r[1] == "done" and r[3] > 0)
    if finished:
        med = finished[len(finished) // 2]
        left = c["pending"] * med + sum(max(med - r[3], 0) for r in rows if r[1] == "running")
        print(f"  | median {med:.0f} min/subject, ETA ~{left / max(1, a.parallel) / 60:.1f} h")
    else:
        print("  | ETA after the first subject finishes")


if a.watch:
    try:
        while True:
            os.system("clear")
            report()
            time.sleep(a.watch)
    except KeyboardInterrupt:
        pass
else:
    report()
