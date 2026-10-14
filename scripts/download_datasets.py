#!/usr/bin/env python
"""Dataset acquisition helper. Only official/permitted channels. Never bypasses access controls or licenses.

  python scripts/download_datasets.py list
  python scripts/download_datasets.py instructions faceforensics|dfdc|celebdf
  python scripts/download_datasets.py kaggle-dataset  xhlulu/140k-real-and-fake-faces --dest data/raw/140k --expected-gb 4
  python scripts/download_datasets.py kaggle-competition deepfake-detection-challenge --file train_sample_videos.zip --dest data/raw/dfdc_sample
  python scripts/download_datasets.py url https://... --sha256 <hex> --dest data/raw/x

Kaggle credentials come from KAGGLE_USERNAME/KAGGLE_KEY or ~/.kaggle/kaggle.json (never printed or stored here).
"""
import argparse, hashlib, os, shutil, subprocess, sys, urllib.request, zipfile
from pathlib import Path

CATALOG = {
    "dfdc": ("DeepFake Detection Challenge (Meta/Facebook AI)", "https://ai.meta.com/datasets/dfdc/",
             "Requires accepting the DFDC terms (Meta page / Kaggle competition rules). Preview set ~5k videos (arXiv:1910.08854). "
             "Kaggle's train_sample_videos.zip is the usual small starting point (size NOT verified in this repo - check before downloading)."),
    "faceforensics": ("FaceForensics++ (TUM)", "https://github.com/ondyari/FaceForensics",
                      "Requires a request form; the download script link is e-mailed after acceptance. Data under the FaceForensics Terms of Use."),
    "celebdf": ("Celeb-DF v2", "https://github.com/yuezunli/celeb-deepfakeforensics",
                "Access is by request form on the official repository (terms not re-verified here - read them on the official page)."),
    "140k-faces": ("140k Real and Fake Faces (Kaggle: xhlulu)", "https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces",
                   "Kaggle account + API token. Real = FFHQ photos, fake = StyleGAN faces (GAN faces, NOT face-swap deepfakes). "
                   "Check the dataset page for the license before any non-research use."),
}


def free_gb(path: Path) -> float:
    path.mkdir(parents=True, exist_ok=True)
    return shutil.disk_usage(path).free / 1e9


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(1 << 20):
            h.update(b)
    return h.hexdigest()


def need_kaggle() -> str:
    exe = shutil.which("kaggle")
    has_cred = (os.getenv("KAGGLE_USERNAME") and os.getenv("KAGGLE_KEY")) or (Path.home() / ".kaggle" / "kaggle.json").exists()
    if not exe:
        sys.exit("BLOCKER: the `kaggle` CLI is not installed. Run: pip install kaggle")
    if not has_cred:
        sys.exit("BLOCKER: Kaggle credentials missing. Create an API token at https://www.kaggle.com/settings and either set "
                 "KAGGLE_USERNAME/KAGGLE_KEY or place kaggle.json in ~/.kaggle/ (chmod 600). Nothing was downloaded.")
    return exe


def guard_space(dest: Path, expected_gb: float):
    free = free_gb(dest)
    if expected_gb and free < expected_gb * 2.2:   # download + extraction
        sys.exit(f"BLOCKER: only {free:.1f} GB free at {dest}; need about {expected_gb * 2.2:.1f} GB for {expected_gb} GB of data.")


def run(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode:
        sys.exit("Download failed (details below; credentials are never echoed):\n" + (r.stderr or r.stdout)[-1500:] +
                 "\nCommon causes: competition rules not accepted on kaggle.com, wrong slug, expired token, no network.")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    sub.add_parser("instructions").add_argument("name", choices=CATALOG)
    k = sub.add_parser("kaggle-dataset"); k.add_argument("slug"); k.add_argument("--dest", required=True); k.add_argument("--expected-gb", type=float, default=0)
    c = sub.add_parser("kaggle-competition"); c.add_argument("competition"); c.add_argument("--file"); c.add_argument("--dest", required=True); c.add_argument("--expected-gb", type=float, default=0)
    u = sub.add_parser("url"); u.add_argument("url"); u.add_argument("--dest", required=True); u.add_argument("--sha256", required=True)
    a = ap.parse_args()

    if a.cmd == "list":
        for key, (name, url, note) in CATALOG.items():
            print(f"[{key}] {name}\n   {url}\n   {note}\n")
        return
    if a.cmd == "instructions":
        name, url, note = CATALOG[a.name]
        print(f"{name}\nOfficial page: {url}\n{note}\nAfter you have legitimate access, place the files under data/raw/<name>/ and run "
              f"scripts/inspect_dataset.py.")
        return
    dest = Path(a.dest); dest.mkdir(parents=True, exist_ok=True)
    marker = dest / ".complete"
    if marker.exists():
        print(f"Already downloaded ({marker}). Delete the marker to force a re-download."); return
    if a.cmd in ("kaggle-dataset", "kaggle-competition"):
        exe = need_kaggle(); guard_space(dest, a.expected_gb)
        if a.cmd == "kaggle-dataset":
            run([exe, "datasets", "download", "-d", a.slug, "-p", str(dest), "--unzip"])
        else:
            cmd = [exe, "competitions", "download", "-c", a.competition, "-p", str(dest)]
            if a.file: cmd += ["-f", a.file]
            run(cmd)
            for z in dest.glob("*.zip"):
                with zipfile.ZipFile(z) as zf: zf.extractall(dest)
    else:  # url, resumable with HTTP Range, integrity-checked
        if not a.url.startswith("https://"): sys.exit("Only https URLs are accepted.")
        out = dest / Path(a.url.split("?")[0]).name; part = out.with_suffix(out.suffix + ".part")
        pos = part.stat().st_size if part.exists() else 0
        req = urllib.request.Request(a.url, headers={"Range": f"bytes={pos}-"} if pos else {})
        with urllib.request.urlopen(req, timeout=60) as r, open(part, "ab" if pos and r.status == 206 else "wb") as f:
            while b := r.read(1 << 20): f.write(b)
        if sha256(part) != a.sha256.lower():
            sys.exit("INTEGRITY FAILURE: SHA256 mismatch; the partial file was kept at " + str(part) + " - delete it and retry.")
        part.rename(out)
    marker.write_text("ok")
    print("Download complete ->", dest)


if __name__ == "__main__":
    main()
