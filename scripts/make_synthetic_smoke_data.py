#!/usr/bin/env python
"""Generate a tiny SYNTHETIC dataset (obvious patterns, not faces) to smoke-test the pipeline plumbing offline.
Models trained on it are meaningless; they are marked smoke_test=true and refused by the API."""
import argparse
from pathlib import Path

import cv2
import numpy as np


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="data/synthetic_smoke"); ap.add_argument("--n", type=int, default=60)
    a = ap.parse_args(); out = Path(a.out); rng = np.random.default_rng(0)
    for lab in ("real", "fake"):
        (out / "images" / lab).mkdir(parents=True, exist_ok=True); (out / "videos" / lab).mkdir(parents=True, exist_ok=True)
    for i in range(a.n):
        for lab in ("real", "fake"):
            base = np.tile(np.linspace(40, 200, 96, dtype=np.uint8)[None, :, None], (96, 1, 3)).copy()
            base = (base + rng.normal(0, 6, base.shape)).clip(0, 255).astype(np.uint8)
            if lab == "fake":
                base[::4, ::4] = 255                       # grid artefact stands in for a "fake" signature
            cv2.imwrite(str(out / "images" / lab / f"{i:04d}.png"), base)
    reals = [f"{i:03d}" for i in range(12)]
    fakes = [f"{i:03d}_{(i ^ 1):03d}" for i in range(12)]              # FaceForensics++-style swap PAIRS (000_001, 001_000, ...)
    for lab, names in (("real", reals), ("fake", fakes)):
        for n in names:
            w = cv2.VideoWriter(str(out / "videos" / lab / f"{n}.avi"), cv2.VideoWriter_fourcc(*"MJPG"), 10, (128, 128))
            for t in range(24):
                f = np.full((128, 128, 3), 80, np.uint8); cv2.circle(f, (30 + t * 3, 64), 22, (210, 190, 170), -1)
                if lab == "fake": f[::5, ::5] = 255
                w.write(f)
            w.release()
    print("Synthetic smoke data written to", out)


if __name__ == "__main__":
    main()
