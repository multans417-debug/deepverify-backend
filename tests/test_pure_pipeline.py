"""Tests that need only numpy/opencv/pillow/sklearn (no torch, no fastapi). All data here is SYNTHETIC."""
import json
import time
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

from deepverify.data.labels import discover, label_from_path
from deepverify.data.manifest import read_manifest
from deepverify.data.prepare import prepare_images, prepare_videos
from deepverify.data.splits import assign_groups, assign_splits, identity_tokens
from deepverify.evaluation.metrics import compute_metrics, fit_temperature, select_threshold, sigmoid
from deepverify.preprocessing.augment import apply_params, draw_params
from deepverify.preprocessing.image import PreprocessConfig, load_rgb, preprocess_rgb
from deepverify.preprocessing.video import VideoConfig, extract_face_crops, sample_indices
from deepverify.utils.errors import InvalidMedia, NoFaceFound, VideoTooLong
from deepverify.utils.ratelimit import SlidingWindowLimiter
from deepverify.utils.sniff import sniff_image_format, sniff_video_container


def _img(path: Path, seed: int, size=64):
    arr = np.random.default_rng(seed).integers(0, 255, (size, size, 3), dtype=np.uint8)
    Image.fromarray(arr).save(path)


def _video(path: Path, frames=40, fps=10, size=(160, 120)):
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, size)
    for i in range(frames):
        f = np.full((size[1], size[0], 3), 90, np.uint8)
        cv2.circle(f, (20 + i * 2, 60), 25, (200, 180, 160), -1)
        w.write(f)
    w.release()


def test_sniffing():
    assert sniff_image_format(b"\xff\xd8\xff\xe0" + b"0" * 20) == "jpeg"
    assert sniff_image_format(b"\x89PNG\r\n\x1a\n" + b"0" * 8) == "png"
    assert sniff_image_format(b"RIFF\x00\x00\x00\x00WEBPVP8 ") == "webp"
    assert sniff_image_format(b"MZ\x90\x00 not an image") is None
    assert sniff_video_container(b"\x00\x00\x00\x18ftypmp42" + b"0" * 8) == "mp4"
    assert sniff_video_container(b"\x00\x00\x00\x14ftypqt  " + b"0" * 8) == "mov"
    assert sniff_video_container(b"RIFF\x00\x00\x00\x00AVI LIST") == "avi"
    assert sniff_video_container(b"\x1a\x45\xdf\xa3" + b"0" * 12) == "mkv"
    assert sniff_video_container(b"just text") is None


def test_rate_limiter():
    t = [0.0]
    lim = SlidingWindowLimiter(2, 60, clock=lambda: t[0])
    assert lim.allow("a") and lim.allow("a") and not lim.allow("a") and lim.allow("b")
    t[0] = 61
    assert lim.allow("a")


def test_identity_groups_prevent_leakage():
    items = [{"stem": s, "original": None} for s in ["000", "001", "000_001", "002", "003_002", "id0_0001", "id1_id0_0002"]]
    g = assign_groups(items)
    assert g[0] == g[1] == g[2]            # FF++: 000, 001 and fake 000_001 tied together
    assert g[3] == g[4]                    # 002 and fake 003_002
    assert g[5] == g[6]                    # Celeb-DF identities
    assert g[0] != g[3]


def test_dfdc_original_links_fake_to_real():
    items = [{"stem": "fake1", "original": "real1.mp4"}, {"stem": "real1", "original": None}, {"stem": "real2", "original": None}]
    g = assign_groups(items)
    assert g[0] == g[1] != g[2]


def test_splits_are_deterministic_and_disjoint():
    gl = {f"g{i}": {"REAL" if i % 2 else "FAKE"} for i in range(40)}
    a, b = assign_splits(gl, seed=1), assign_splits(gl, seed=1)
    assert a == b and set(a.values()) == {"train", "val", "test"}
    assert assign_splits(gl, seed=2) != a


def test_label_discovery_never_guesses(tmp_path):
    (tmp_path / "real").mkdir(); (tmp_path / "fake").mkdir(); (tmp_path / "misc").mkdir(); (tmp_path / "real" / "fake").mkdir()
    _img(tmp_path / "real" / "a.png", 1); _img(tmp_path / "fake" / "b.png", 2)
    _img(tmp_path / "misc" / "c.png", 3); _img(tmp_path / "real" / "fake" / "d.png", 4)
    items, excluded = discover(tmp_path, {".png"}, use_metadata=False)
    assert {Path(i["path"]).name: i["label"] for i in items} == {"a.png": "REAL", "b.png": "FAKE"}
    assert {Path(e["path"]).name for e in excluded} == {"c.png", "d.png"}  # ambiguous -> excluded, not guessed


def test_prepare_images_handles_corrupt_and_duplicates(tmp_path):
    root = tmp_path / "raw"
    for lab in ("real", "fake"):
        (root / lab).mkdir(parents=True)
        for i in range(12):
            _img(root / lab / f"{i}.png", i + (100 if lab == "fake" else 0))
    (root / "real" / "broken.png").write_bytes(b"not an image at all")
    (root / "fake" / "dup.png").write_bytes((root / "fake" / "0.png").read_bytes())
    rep = prepare_images(root, tmp_path / "out", seed=7)
    rows = read_manifest(tmp_path / "out" / "image_manifest.csv")
    assert len(rows) == 24 and rep["excluded_count"] == 2
    assert len({r["sha256"] for r in rows}) == len(rows)
    assert {r["split"] for r in rows} == {"train", "val", "test"}
    reasons = (tmp_path / "out" / "exclusions.csv").read_text()
    assert "corrupt" in reasons and "exact_duplicate" in reasons


def test_video_extraction_and_prepare_no_leakage(tmp_path):
    cfg = VideoConfig(num_frames=8, min_valid_frames=4, input_size=64, face_detector="center")
    v = tmp_path / "v.avi"; _video(v)
    crops, stats = extract_face_crops(v, cfg)
    assert len(crops) == 8 and crops[0].shape == (64, 64, 3) and stats["frames_used"] == 8
    # deterministic: same input -> identical pixels (train/serve parity)
    crops2, _ = extract_face_crops(v, cfg)
    assert all(np.array_equal(a, b) for a, b in zip(crops, crops2))
    root = tmp_path / "vids"
    for lab, names in (("real", ["000", "001", "002", "003", "004", "005"]), ("fake", ["000_001", "002_003", "004_005", "005_000"])):
        (root / lab).mkdir(parents=True)
        for n in names:
            _video(root / lab / f"{n}.avi")
    (root / "fake" / "corrupt.avi").write_bytes(b"garbage")
    rep = prepare_videos(root, tmp_path / "out", vcfg=cfg, seed=3)
    assert rep["group_leakage_check"]["passed"] and rep["excluded_count"] >= 1
    vm = read_manifest(tmp_path / "out" / "video_manifest.csv")
    fm = read_manifest(tmp_path / "out" / "frames_manifest.csv")
    assert all(len(list(Path(r["frames_dir"]).glob("*.png"))) == 8 for r in vm)
    vid_split = {r["video_id"]: r["split"] for r in vm}
    assert all(vid_split[f["video_id"]] == f["split"] for f in fm)  # frames inherit their video's split


def test_video_errors(tmp_path):
    cfg = VideoConfig(num_frames=8, min_valid_frames=4, input_size=64, face_detector="haar")
    v = tmp_path / "noface.avi"; _video(v)
    with pytest.raises(NoFaceFound):        # a plain circle is not a face -> explicit error, no guess
        extract_face_crops(v, cfg)
    bad = tmp_path / "bad.avi"; bad.write_bytes(b"\x00" * 100)
    with pytest.raises(InvalidMedia):
        extract_face_crops(bad, cfg)
    long_cfg = VideoConfig(face_detector="center", max_seconds=1.0)
    with pytest.raises(VideoTooLong):
        extract_face_crops(v, long_cfg)


def test_image_preprocess_and_augment():
    cfg = PreprocessConfig(input_size=32)
    rgb = np.random.default_rng(0).integers(0, 255, (50, 70, 3), dtype=np.uint8)
    x = preprocess_rgb(rgb, cfg)
    assert x.shape == (3, 32, 32) and x.dtype == np.float32
    aug = apply_params(rgb, draw_params(np.random.default_rng(1)))
    assert aug.shape == rgb.shape and aug.dtype == np.uint8
    assert PreprocessConfig.from_dict(cfg.to_dict()) == cfg
    assert list(sample_indices(5, 16)) == [0, 1, 2, 3, 4]


def test_load_rgb_rejects_garbage(tmp_path):
    f = tmp_path / "x.png"; f.write_bytes(b"nope")
    with pytest.raises(InvalidMedia):
        load_rgb(f)
    big = tmp_path / "big.png"; _img(big, 1, 200)
    with pytest.raises(InvalidMedia):
        load_rgb(big, max_pixels=1000)


def test_metrics_threshold_and_temperature():
    rng = np.random.default_rng(0)
    n = 5000
    y = np.r_[np.zeros(n), np.ones(n)].astype(int)
    x = np.r_[rng.normal(-1.5, 1.2, n), rng.normal(1.5, 1.2, n)]
    z = x * 3.0                                    # deliberately over-confident logits
    s = sigmoid(z)
    m = compute_metrics(y, s, 0.5)
    assert 0.85 < m["accuracy"] <= 1 and m["roc_auc"] > 0.9
    cm = m["confusion_matrix"]["matrix"]
    assert cm[0][0] + cm[0][1] == n and cm[1][0] + cm[1][1] == n
    T = fit_temperature(z, y)
    # equal-variance Gaussians: ideal logit = 2*mu*x/sigma^2, i.e. ideal T = 3*sigma^2/(2*mu) = 1.44
    assert abs(T - 1.44) < 0.15
    assert compute_metrics(y, sigmoid(z / T), 0.5)["ece_10_bins"] < m["ece_10_bins"]
    t, method = select_threshold(y, s)
    assert 0 < t < 1 and "F1" in method
    one = compute_metrics(np.zeros(5), np.full(5, 0.2), 0.5)
    assert one["roc_auc"] is None and "warning" in one


def test_giant_component_falls_back_and_is_reported():
    from deepverify.data.splits import largest_group_share
    chain = [{"stem": f"{i:03d}", "original": None} for i in range(12)] + \
            [{"stem": f"{i:03d}_{(i + 1) % 12:03d}", "original": None} for i in range(12)]
    assert largest_group_share(assign_groups(chain)) == 1.0          # one chain -> single group, unsplittable
    first = assign_groups(chain, "first")
    assert len(set(first.values())) == 12


def test_prepare_videos_reports_grouping_fallback(tmp_path):
    cfg = VideoConfig(num_frames=4, min_valid_frames=2, input_size=32, face_detector="center")
    root = tmp_path / "vids"
    for lab, names in (("real", [f"{i:03d}" for i in range(10)]), ("fake", [f"{i:03d}_{(i + 1) % 10:03d}" for i in range(10)])):
        (root / lab).mkdir(parents=True)
        for n in names:
            _video(root / lab / f"{n}.avi", frames=12)
    rep = prepare_videos(root, tmp_path / "out", vcfg=cfg, seed=1)
    assert rep["grouping_mode"] == "first_identity" and rep["grouping_warning"]
    assert rep["by_split"]["val"]["REAL"] + rep["by_split"]["val"]["FAKE"] > 0   # splits actually non-empty
