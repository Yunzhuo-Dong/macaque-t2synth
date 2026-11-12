#!/usr/bin/env python
"""06_fetch_ucdavis.py — what PRIME-DE UC Davis data do we have, and get the rest.

Modes (each includes the ones above it):
  (default)   LOCAL inventory, read-only: paired T1w+T2w subjects under the data folders
  --remote    + list what the public fcp-indi S3 bucket holds for the site (read-only)
  --download  + fetch the missing anatomical T1w/T2w (+ .json sidecars):
                  1. per-file from S3 if the site is stored unpacked there, otherwise
                  2. the site tarball from S3 (~27 GB, resumable), extracting only anat T1w/T2w
  --tarball F + extract anat T1w/T2w from a tarball you downloaded yourself
                (e.g. from the NITRC portal), no network needed

Downloads go to --dest (default $WORKSPACE/data/PRIME-DE_UCDavis_full), NOT into DATA_DIR,
so the verified example copy stays untouched and 04_preprocess.py never sees duplicate
subjects. Preprocess the full set with:  04_preprocess.py --data-dir <dest> ...

Data licence: UC Davis PRIME-DE collection, CC-BY-NC-SA (non-commercial; cite Milham et al.,
2018, Neuron, and acknowledge the UC Davis contributors as the PRIME-DE site page requires).

Usage:
    source ganmat_env.sh
    "$VENV_PY" 06_fetch_ucdavis.py                       # inventory only
    "$VENV_PY" 06_fetch_ucdavis.py --remote              # + what S3 has
    "$VENV_PY" 06_fetch_ucdavis.py --download            # + download missing (run in tmux)
    "$VENV_PY" 06_fetch_ucdavis.py --tarball ~/Downloads/ucdavis.tar.gz
"""
import argparse
import glob
import os
import re
import sys
import tarfile
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

NS = "{http://s3.amazonaws.com/doc/2006-03-01/}"
ANAT = re.compile(r"_(T1w|T2w)\.(nii\.gz|nii|json)$")
PRIME_PREFIX = "data/Projects/INDI/PRIME/"


# ===================================================================== inventory
def inventory(roots):
    seen, subs, tarballs = set(), {}, []
    for root in roots:
        if not root or not os.path.isdir(root):
            continue
        for f in glob.glob(os.path.join(root, "**", "*_T[12]w.nii*"), recursive=True):
            st = os.stat(f)
            if (st.st_dev, st.st_ino) in seen:  # hard-linked duplicates
                continue
            seen.add((st.st_dev, st.st_ino))
            m = re.search(r"(sub-[A-Za-z0-9]+)", os.path.basename(f))
            mod = "T1w" if "_T1w." in f else "T2w"
            if m:
                subs.setdefault(m.group(1), {"T1w": [], "T2w": []})[mod].append(f)
        for ext in ("*.tar.gz", "*.tgz", "*.tar"):
            tarballs += glob.glob(os.path.join(root, "**", ext), recursive=True)
    return subs, sorted(set(tarballs))


def print_inventory(title, roots, subs, tarballs):
    print(f"\n== {title}")
    for r in roots:
        print(f"   {r}  ({'exists' if r and os.path.isdir(r) else 'missing'})")
    paired = sorted(s for s, d in subs.items() if d["T1w"] and d["T2w"])
    t1only = sorted(s for s, d in subs.items() if d["T1w"] and not d["T2w"])
    t2only = sorted(s for s, d in subs.items() if d["T2w"] and not d["T1w"])
    print(f"   subjects with T1w AND T2w : {len(paired)}  {' '.join(paired)}")
    print(f"   subjects with T1w only    : {len(t1only)}  {' '.join(t1only)}")
    if t2only:
        print(f"   subjects with T2w only    : {len(t2only)}  {' '.join(t2only)}")
    multi = [s for s in paired if len(subs[s]["T1w"]) > 1 or len(subs[s]["T2w"]) > 1]
    if multi:
        print(f"   NOTE several runs/sessions for: {' '.join(multi)}")
    for t in tarballs:
        print(f"   tarball: {t}  ({os.path.getsize(t) / 2**30:.1f} GiB)")
    return paired


# ======================================================================== S3
def s3_list(base, prefix, delimiter=None, max_keys=None):
    keys, prefixes, token = [], [], None
    while True:
        q = {"list-type": "2", "prefix": prefix}
        if delimiter:
            q["delimiter"] = delimiter
        if token:
            q["continuation-token"] = token
        with urllib.request.urlopen(f"{base}/?{urllib.parse.urlencode(q)}", timeout=60) as r:
            root = ET.fromstring(r.read())
        keys += [(c.findtext(NS + "Key"), int(c.findtext(NS + "Size")))
                 for c in root.findall(NS + "Contents")]
        prefixes += [p.findtext(NS + "Prefix") for p in root.findall(NS + "CommonPrefixes")]
        if root.findtext(NS + "IsTruncated") != "true" or (max_keys and len(keys) >= max_keys):
            return keys, prefixes
        token = root.findtext(NS + "NextContinuationToken")


def remote_survey(base, site):
    print(f"\n== remote: {base}/{PRIME_PREFIX}")
    keys, prefixes = s3_list(base, PRIME_PREFIX, delimiter="/")
    hits = [k for k in keys if site in k[0].lower()] + [(p, None) for p in prefixes if site in p.lower()]
    print(f"   {len(prefixes)} folders, {len(keys)} files at top level; matching '{site}':")
    for k, size in hits:
        print(f"     {k}" + (f"  ({size / 2**30:.1f} GiB)" if size else "  (folder)"))
    anat = []
    for p in [p for p in prefixes if site in p.lower()]:
        sub_keys, _ = s3_list(base, p)
        anat += [(k, s) for k, s in sub_keys if ANAT.search(k)]
    if anat:
        n = len({re.search(r"(sub-[A-Za-z0-9]+)", k).group(1) for k, _ in anat if "sub-" in k})
        print(f"   unpacked anat files on S3: {len(anat)} files, {n} subjects, "
              f"{sum(s for _, s in anat) / 2**30:.2f} GiB")
    tarball = next((k for k, s in keys if k.lower().endswith(f"/{site}.tar.gz")), None)
    tar_size = next((s for k, s in keys if k == tarball), None)
    return anat, tarball, tar_size


# =================================================================== download
def download(url, dest, expected=None, retries=8):
    if os.path.exists(dest) and (expected is None or os.path.getsize(dest) == expected):
        print(f"   have {dest}")
        return
    part = dest + ".part"
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    for attempt in range(1, retries + 1):
        have = os.path.getsize(part) if os.path.exists(part) else 0
        req = urllib.request.Request(url, headers={"Range": f"bytes={have}-"} if have else {})
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                if have and r.status != 206:  # server ignored the range: restart
                    have = 0
                total = expected or (have + int(r.headers.get("Content-Length", 0)))
                mode, done, t0, last = ("ab" if have else "wb"), have, time.time(), 0.0
                with open(part, mode) as fh:
                    while True:
                        chunk = r.read(8 << 20)
                        if not chunk:
                            break
                        fh.write(chunk)
                        done += len(chunk)
                        if total and time.time() - last > 30:
                            last = time.time()
                            rate = (done - have) / max(1e-6, last - t0) / 2**20
                            print(f"   {os.path.basename(dest)}: {done / 2**30:.2f}/{total / 2**30:.2f} GiB "
                                  f"({100 * done / total:.1f}%, {rate:.1f} MiB/s)", flush=True)
            if expected and os.path.getsize(part) != expected:
                raise IOError(f"size {os.path.getsize(part)} != expected {expected}")
            os.replace(part, dest)
            return
        except Exception as e:  # network hiccup: resume
            print(f"   attempt {attempt} failed ({type(e).__name__}: {e}); resuming in 20 s", flush=True)
            time.sleep(20)
    raise SystemExit(f"download failed after {retries} attempts: {url}")


def extract_anat(tarpath, dest):
    os.makedirs(dest, exist_ok=True)
    n = 0
    print(f"   extracting anat T1w/T2w from {tarpath} (streams the whole archive) ...", flush=True)
    with tarfile.open(tarpath, "r:*") as tf:
        for m in tf:
            if not (m.isfile() and ANAT.search(m.name)):
                continue
            if m.name.startswith("/") or ".." in m.name.split("/"):
                print(f"   skipping unsafe path {m.name}")
                continue
            out = os.path.join(dest, m.name)
            if os.path.exists(out) and os.path.getsize(out) == m.size:
                continue
            try:
                tf.extract(m, dest, filter="data")
            except TypeError:  # Python without extraction filters
                tf.extract(m, dest)
            n += 1
    print(f"   extracted {n} files")


# ======================================================================= main
def main():
    ws = os.environ.get("WORKSPACE", "")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default=os.environ.get("DATA_DIR"), help="existing (example) data")
    ap.add_argument("--dest", default=os.path.join(ws, "data", "PRIME-DE_UCDavis_full") if ws else None)
    ap.add_argument("--site", default="ucdavis")
    ap.add_argument("--s3-base", default="https://fcp-indi.s3.amazonaws.com")
    ap.add_argument("--remote", action="store_true")
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--tarball", help="extract from this local tarball instead of downloading")
    a = ap.parse_args()
    if not a.dest:
        sys.exit("no --dest and WORKSPACE unset; did you `source ganmat_env.sh`?")

    subs, tars = inventory([a.data_dir])
    print_inventory("local: DATA_DIR", [a.data_dir], subs, tars)
    subs_d, tars_d = inventory([a.dest])
    have = set(print_inventory("local: download destination", [a.dest], subs_d, tars_d))

    if a.tarball:
        extract_anat(a.tarball, a.dest)
    elif a.remote or a.download:
        try:
            anat, tarball, tar_size = remote_survey(a.s3_base, a.site)
        except Exception as e:
            sys.exit(f"\nS3 not reachable ({type(e).__name__}: {e}).\n"
                     "Download the 'University of California, Davis (27GB)' archive from the NITRC portal\n"
                     "(https://fcon_1000.projects.nitrc.org/indi/PRIMEdownloads.html, needs a NITRC account\n"
                     "added to the 1000 Functional Connectomes project), then rerun with --tarball <file>.")
        if a.download:
            if anat:
                todo = [(k, s) for k, s in anat
                        if not any(f"{x}_" in os.path.basename(k) for x in have)]
                print(f"\n== downloading {len(todo)} anat files ({sum(s for _, s in todo) / 2**30:.2f} GiB)")
                for k, s in todo:
                    rel = k[len(PRIME_PREFIX):]
                    download(f"{a.s3_base}/{urllib.parse.quote(k)}", os.path.join(a.dest, rel), s)
            elif tarball:
                local = os.path.join(a.dest, "_downloads", os.path.basename(tarball))
                print(f"\n== downloading {tarball} ({tar_size / 2**30:.1f} GiB, resumable)")
                download(f"{a.s3_base}/{urllib.parse.quote(tarball)}", local, tar_size)
                extract_anat(local, a.dest)
            else:
                sys.exit("\nNothing for this site on S3; use the NITRC portal and --tarball.")
    if a.tarball or a.download:
        subs_d, _ = inventory([a.dest])
        paired = print_inventory("result: download destination", [a.dest], subs_d, [])
        if paired:
            print(f"\nnext:  bash 04_run_prep.sh --data-dir {a.dest} {' '.join(paired[:3])} ...")


if __name__ == "__main__":
    main()
