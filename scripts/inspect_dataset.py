#!/usr/bin/env python
"""Inspect a raw dataset folder: label discovery, extensions, ambiguity, source groups, decode checks, size."""
import _bootstrap  # noqa: F401
import argparse, collections, json
from pathlib import Path

import cv2

from deepverify.data.labels import IMAGE_EXT, VIDEO_EXT, discover
from deepverify.data.splits import assign_groups
from deepverify.preprocessing.image import load_rgb
from deepverify.utils.errors import DeepVerifyError


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root"); ap.add_argument("--kind", choices=["image", "video"], required=True)
    ap.add_argument("--check", type=int, default=50, help="decode-check this many random files")
    a = ap.parse_args()
    root = Path(a.root)
    items, excluded = discover(root, IMAGE_EXT if a.kind == "image" else VIDEO_EXT, use_metadata=a.kind == "video")
    out = {"root": str(root), "labelled": len(items), "excluded_ambiguous": len(excluded),
           "by_label": dict(collections.Counter(i["label"] for i in items)),
           "by_extension": dict(collections.Counter(Path(i["path"]).suffix.lower() for i in items)),
           "label_sources": dict(collections.Counter(i["label_source"] for i in items)),
           "total_gb": round(sum(Path(i["path"]).stat().st_size for i in items) / 1e9, 3)}
    if a.kind == "video":
        g = assign_groups(items); out["source_groups"] = len(set(g.values()))
    import random
    bad = 0
    for it in random.Random(0).sample(items, min(a.check, len(items))):
        try:
            if a.kind == "image": load_rgb(it["path"])
            else:
                cap = cv2.VideoCapture(it["path"]); ok = cap.isOpened() and cap.grab(); cap.release()
                if not ok: raise DeepVerifyError("cannot decode")
        except DeepVerifyError: bad += 1
    out["decode_check"] = {"sampled": min(a.check, len(items)), "failed": bad}
    if excluded: out["excluded_examples"] = [e["path"] for e in excluded[:5]]
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
