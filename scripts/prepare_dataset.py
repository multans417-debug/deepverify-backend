#!/usr/bin/env python
"""Prepare image and/or video data (validation, dedup, leakage-safe split, face-crop extraction, reports)."""
import _bootstrap  # noqa: F401
import argparse, json, logging
from pathlib import Path

from deepverify.data.prepare import prepare_images, prepare_videos
from deepverify.preprocessing.video import VideoConfig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=["image", "video"], required=True)
    ap.add_argument("--root", required=True, help="raw dataset folder")
    ap.add_argument("--out", required=True, help="processed output folder (manifests, frames, reports)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-per-class", type=int, help="images: cap per class (smoke/limited compute)")
    ap.add_argument("--max-videos", type=int, help="videos: cap total videos (balanced per class)")
    ap.add_argument("--near-dup", action="store_true", help="images: also drop identical-dHash near duplicates")
    ap.add_argument("--num-frames", type=int, default=16); ap.add_argument("--min-valid-frames", type=int, default=8)
    ap.add_argument("--crop-size", type=int, default=224)
    ap.add_argument("--face-detector", choices=["haar", "center"], default="haar")
    ap.add_argument("--smoke", action="store_true", help="allow the non-detector 'center' crop (synthetic smoke tests only)")
    ap.add_argument("--force", action="store_true", help="re-extract frames even if cached")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    if a.face_detector == "center" and not a.smoke:
        ap.error("--face-detector center is only allowed with --smoke (it is not a real face detector)")
    out = Path(a.out)
    if a.kind == "image":
        rep = prepare_images(Path(a.root), out, seed=a.seed, max_per_class=a.max_per_class, near_dup=a.near_dup)
    else:
        v = VideoConfig(num_frames=a.num_frames, min_valid_frames=a.min_valid_frames, input_size=a.crop_size,
                        face_detector=a.face_detector, timeout_seconds=120, max_seconds=3600)
        rep = prepare_videos(Path(a.root), out, vcfg=v, seed=a.seed, max_videos=a.max_videos, force=a.force)
    print(json.dumps(rep, indent=2))
    if not rep["overall"]["REAL"] or not rep["overall"]["FAKE"]:
        raise SystemExit("ERROR: one class is empty after preparation - check labels/exclusions.csv")


if __name__ == "__main__":
    main()
