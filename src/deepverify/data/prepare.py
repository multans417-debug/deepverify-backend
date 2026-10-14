"""Reproducible dataset preparation (images and videos). No torch dependency."""
from __future__ import annotations

import hashlib
import json
import logging
import random
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from deepverify.data.labels import IMAGE_EXT, VIDEO_EXT, discover
from deepverify.data.manifest import LABEL_TO_INT, class_counts, write_manifest
from deepverify.data.splits import assign_groups, assign_splits, largest_group_share
from deepverify.preprocessing.image import load_rgb
from deepverify.preprocessing.video import VideoConfig, extract_face_crops
from deepverify.preprocessing.face import make_detector
from deepverify.utils.errors import DeepVerifyError

log = logging.getLogger("deepverify.prepare")


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(1 << 20):
            h.update(b)
    return h.hexdigest()


def _dhash(rgb: np.ndarray) -> str:
    g = cv2.cvtColor(cv2.resize(rgb, (9, 8), interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2GRAY)
    return "".join("1" if v else "0" for v in (g[:, 1:] > g[:, :-1]).flatten())


def _write_reports(out: Path, rows: list, excluded: list, extra: dict) -> dict:
    report = {"overall": class_counts(rows),
              "by_split": {s: class_counts([r for r in rows if r["split"] == s]) for s in ("train", "val", "test")},
              "excluded_count": len(excluded), **extra}
    (out / "class_distribution.json").write_text(json.dumps(report, indent=2))
    write_manifest(out / "exclusions.csv", excluded, ["path", "reason"])
    return report


def prepare_images(root: Path, out: Path, *, seed=42, ratios=(0.7, 0.15, 0.15), max_per_class=None,
                   near_dup=False, max_pixels=40_000_000) -> dict:
    """Image manifest: path,label,split,group,sha256. Honors an existing train/valid/test directory layout."""
    out.mkdir(parents=True, exist_ok=True)
    items, excluded = discover(Path(root), IMAGE_EXT, use_metadata=False)
    rng = random.Random(seed)
    if max_per_class:  # cap BEFORE hashing/decoding so smoke runs are cheap; seeded for reproducibility
        capped = []
        for lab in ("REAL", "FAKE"):
            sub = [i for i in items if i["label"] == lab]
            rng.shuffle(sub)
            capped += sub[:max_per_class]
        items = sorted(capped, key=lambda i: i["path"])
    # drop corrupt + exact duplicates (priority test > val > train so a duplicate never leaks into train)
    prio = {"test": 0, "val": 1, "train": 2, None: 3}
    items.sort(key=lambda i: (prio[i["preset_split"]], i["path"]))
    seen_sha, seen_dh, kept = {}, {}, []
    for it in items:
        try:
            rgb = load_rgb(it["path"], max_pixels)
        except DeepVerifyError as exc:
            excluded.append({"path": it["path"], "reason": f"corrupt: {exc.message}"})
            continue
        sha = _sha256(it["path"])
        if sha in seen_sha:
            excluded.append({"path": it["path"], "reason": f"exact_duplicate_of:{seen_sha[sha]}"})
            continue
        seen_sha[sha] = it["path"]
        if near_dup:
            dh = _dhash(rgb)
            if dh in seen_dh:
                excluded.append({"path": it["path"], "reason": f"near_duplicate_of:{seen_dh[dh]}"})
                continue
            seen_dh[dh] = it["path"]
        it["sha256"] = sha
        kept.append(it)
    preset = all(i["preset_split"] for i in kept) and bool(kept)
    if preset:
        for i in kept:
            i["group"] = i["sha256"][:16]
        split_of = lambda i: i["preset_split"]  # noqa: E731
        note = "Used the dataset's own train/valid/test directory split."
    else:
        for i in kept:
            i["group"] = i["sha256"][:16]  # no identity metadata -> each image is its own group (documented limitation)
        gl = {i["group"]: {i["label"]} for i in kept}
        sp = assign_splits(gl, ratios, seed)
        split_of = lambda i: sp[i["group"]]  # noqa: E731
        note = "Random seeded split by image; no source-identity metadata available (leakage between near-identical shots of one person is possible)."
    rows = [{"path": i["path"], "label": LABEL_TO_INT[i["label"]], "split": split_of(i), "group": i["group"],
             "sha256": i["sha256"]} for i in kept]
    write_manifest(out / "image_manifest.csv", rows, ["path", "label", "split", "group", "sha256"])
    (out / "image_prep_config.json").write_text(json.dumps(
        {"seed": seed, "ratios": ratios, "max_per_class": max_per_class, "near_dup": near_dup, "root": str(root)}, indent=2))
    return _write_reports(out, rows, excluded, {"split_note": note, "kind": "image"})


def prepare_videos(root: Path, out: Path, *, vcfg: VideoConfig, seed=42, ratios=(0.7, 0.15, 0.15),
                   max_videos=None, force=False) -> dict:
    """Split by source group FIRST, then extract face crops (PNG) per video. Resumable: finished videos are skipped.

    Outputs: video_manifest.csv (one row per video), frames_manifest.csv (one row per frame, group=video's group),
    class_distribution.json, exclusions.csv, video_prep_config.json.
    """
    out.mkdir(parents=True, exist_ok=True)
    items, excluded = discover(Path(root), VIDEO_EXT, use_metadata=True)
    if max_videos:
        rng = random.Random(seed)
        per = max(1, max_videos // 2)
        capped = []
        for lab in ("REAL", "FAKE"):
            sub = [i for i in items if i["label"] == lab]
            rng.shuffle(sub)
            capped += sub[:per]
        items = sorted(capped, key=lambda i: i["path"])
    groups, grouping_mode, grouping_warning = assign_groups(items), "union", None
    share = largest_group_share(groups)
    if share > 0.4 and len(items) >= 10:
        grouping_warning = (f"Strict identity grouping merged {share:.0%} of videos into one group (cannot split). Fell back to "
                            "first-identity grouping: the same person's face may appear across splits. Prefer the dataset's official "
                            "train/test list if it has one.")
        log.warning(grouping_warning)
        groups, grouping_mode = assign_groups(items, "first"), "first_identity"
    for k, it in enumerate(items):
        it["group"] = groups[k]
    sp = assign_splits({g: {i["label"] for i in items if i["group"] == g} for g in set(groups.values())}, ratios, seed)
    det = make_detector(vcfg.face_detector)
    frames_root = out / "frames"
    vrows, frows = [], []
    for n, it in enumerate(items):
        vid = hashlib.sha1(it["path"].encode()).hexdigest()[:12]
        d = frames_root / vid
        done = d / "done.json"
        try:
            if done.exists() and not force:
                stats = json.loads(done.read_text())
            else:
                crops, stats = extract_face_crops(it["path"], vcfg, det)
                d.mkdir(parents=True, exist_ok=True)
                for k, c in enumerate(crops):  # PNG = lossless, so train == serve pixels exactly
                    cv2.imwrite(str(d / f"{k:03d}.png"), c[:, :, ::-1])
                done.write_text(json.dumps(stats))
        except DeepVerifyError as exc:
            excluded.append({"path": it["path"], "reason": f"{exc.code}: {exc.message}"})
            continue
        except Exception as exc:  # unexpected decoder failures are recorded, never silently skipped
            excluded.append({"path": it["path"], "reason": f"error:{type(exc).__name__}"})
            continue
        split, label = sp[it["group"]], LABEL_TO_INT[it["label"]]
        vrows.append({"video_id": vid, "source_path": it["path"], "label": label, "split": split,
                      "group": it["group"], "frames_dir": str(d), **{k: stats[k] for k in ("frames_sampled", "faces_found", "frames_used")}})
        for k in range(stats["frames_used"]):
            frows.append({"path": str(d / f"{k:03d}.png"), "label": label, "split": split, "group": it["group"],
                          "video_id": vid, "frame_index": k})
        if (n + 1) % 25 == 0:
            log.info("prepared %d/%d videos", n + 1, len(items))
    write_manifest(out / "video_manifest.csv", vrows, ["video_id", "source_path", "label", "split", "group", "frames_dir",
                                                       "frames_sampled", "faces_found", "frames_used"])
    write_manifest(out / "frames_manifest.csv", frows, ["path", "label", "split", "group", "video_id", "frame_index"])
    (out / "video_prep_config.json").write_text(json.dumps(
        {"seed": seed, "ratios": ratios, "max_videos": max_videos, "video_config": vcfg.to_dict(), "root": str(root)}, indent=2))
    leak = _check_group_leakage(vrows)
    return _write_reports(out, vrows, excluded, {
        "kind": "video", "frames_total": len(frows), "group_leakage_check": leak,
        "grouping_mode": grouping_mode, "grouping_warning": grouping_warning,
        "split_note": "Split by source group (identity/original-video union-find) before frame extraction."})


def _check_group_leakage(rows: list) -> dict:
    by_group: dict = {}
    for r in rows:
        by_group.setdefault(r["group"], set()).add(r["split"])
    bad = [g for g, s in by_group.items() if len(s) > 1]
    return {"groups": len(by_group), "groups_in_multiple_splits": len(bad), "passed": not bad}
