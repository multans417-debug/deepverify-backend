"""PLUMBING tests: a RANDOMLY INITIALISED model is written to a temp dir and served through the real inference code.
They prove the code path (preprocess -> model -> response, loader integrity checks, error handling) works.
They say NOTHING about detection quality. Requires torch, torchvision, fastapi, httpx, scikit-learn, scipy."""
import dataclasses
import io

import cv2
import numpy as np
import pytest

for m in ("torch", "torchvision", "fastapi", "httpx", "sklearn", "scipy"):
    pytest.importorskip(m)
from fastapi.testclient import TestClient  # noqa: E402
from PIL import Image  # noqa: E402

from deepverify.config import Settings  # noqa: E402
from deepverify.main import create_app  # noqa: E402
from deepverify.models import image_model, video_model  # noqa: E402
from deepverify.preprocessing.image import PreprocessConfig  # noqa: E402
from deepverify.preprocessing.video import VideoConfig  # noqa: E402
from deepverify.training.artifacts import save_artifacts  # noqa: E402

CAL = {"temperature": 1.0, "threshold": 0.5, "threshold_method": "plumbing"}


def write_models(art):
    pre = PreprocessConfig(input_size=64)
    save_artifacts("image", art, state_dict=image_model.ImageDeepfakeNet().state_dict(), architecture=image_model.ARCH,
                   model_config={"dropout": 0.3}, preprocess=pre.to_dict(), video=None, cal=CAL, training_info={}, metrics={},
                   smoke=True, notes=["plumbing"])
    v = VideoConfig(num_frames=4, min_valid_frames=2, input_size=64, face_detector="center")
    vpre = PreprocessConfig(input_size=64, use_face_crop=True, face_detector="center")
    save_artifacts("video", art, state_dict=video_model.VideoDeepfakeNet(hidden=16).state_dict(), architecture=video_model.ARCH,
                   model_config={"hidden": 16, "dropout": 0.3}, preprocess=vpre.to_dict(), video=v.to_dict(), cal=CAL,
                   training_info={}, metrics={}, smoke=True, notes=["plumbing"])


def settings(tmp_path, **over):
    (tmp_path / "uploads").mkdir(exist_ok=True)
    base = Settings(artifact_dir=tmp_path / "art", cors_origins=("http://localhost:5173",), required_models=("image", "video"),
                    max_image_bytes=2_000_000, max_video_bytes=5_000_000, max_image_pixels=4_000_000, max_video_seconds=60,
                    extract_timeout_seconds=30, max_concurrent_inferences=1, queue_timeout_seconds=5, rate_limit_per_minute=1000,
                    trust_proxy_headers=False, torch_threads=1, allow_smoke_models=True, tmp_dir=tmp_path / "uploads")
    return dataclasses.replace(base, **over)


def png(seed=0):
    arr = np.random.default_rng(seed).integers(0, 255, (80, 80, 3), dtype=np.uint8)
    b = io.BytesIO(); Image.fromarray(arr).save(b, "PNG"); return b.getvalue()


def video_bytes(tmp_path):
    p = tmp_path / "v.avi"
    w = cv2.VideoWriter(str(p), cv2.VideoWriter_fourcc(*"MJPG"), 10, (96, 96))
    for i in range(20):
        f = np.full((96, 96, 3), 90, np.uint8); cv2.circle(f, (20 + 2 * i, 48), 15, (200, 180, 160), -1); w.write(f)
    w.release(); return p.read_bytes()


@pytest.mark.plumbing
def test_real_code_path_image_and_video(tmp_path):
    write_models(tmp_path / "art")
    with TestClient(create_app(settings(tmp_path))) as c:
        assert c.get("/ready").status_code == 200
        info = c.get("/api/v1/models").json()["models"]
        assert info["image"]["training_status"] == "smoke-test only" and info["video"]["frames_per_video"] == 4
        r1 = c.post("/api/v1/detect/image", files={"file": ("x.png", png(1), "image/png")})
        r2 = c.post("/api/v1/detect/image", files={"file": ("x.png", png(1), "image/png")})
        assert r1.status_code == 200 and r1.json()["media_type"] == "image"
        assert 0.0 <= r1.json()["fake_probability"] <= 1.0 and r1.json()["prediction"] in ("REAL", "FAKE")
        assert r1.json()["fake_probability"] == r2.json()["fake_probability"]           # deterministic inference
        v = c.post("/api/v1/detect/video", files={"file": ("v.avi", video_bytes(tmp_path), "video/x-msvideo")})
        assert v.status_code == 200 and v.json()["details"]["frames_used"] == 4
    assert list((tmp_path / "uploads").iterdir()) == []


@pytest.mark.plumbing
def test_corrupt_media_with_valid_magic_bytes(tmp_path):
    write_models(tmp_path / "art")
    with TestClient(create_app(settings(tmp_path))) as c:
        bad_img = c.post("/api/v1/detect/image", files={"file": ("a.png", b"\x89PNG\r\n\x1a\n" + b"garbage" * 20, "image/png")})
        bad_vid = c.post("/api/v1/detect/video", files={"file": ("a.mp4", b"\x00\x00\x00\x18ftypmp42" + b"junk" * 50, "video/mp4")})
        assert bad_img.status_code == 422 and bad_img.json()["error"]["code"] == "invalid_or_corrupt_media"
        assert bad_vid.status_code == 422
    assert list((tmp_path / "uploads").iterdir()) == []


@pytest.mark.plumbing
def test_loader_refuses_smoke_models_and_tampered_weights(tmp_path):
    write_models(tmp_path / "art")
    with TestClient(create_app(settings(tmp_path, allow_smoke_models=False))) as c:
        assert c.get("/ready").status_code == 503
        assert "smoke" in c.get("/api/v1/models").json()["models"]["image"]["error"].lower()
    (tmp_path / "art" / "image_model.pth").write_bytes(b"tampered")
    with TestClient(create_app(settings(tmp_path))) as c:
        assert "SHA256" in c.get("/api/v1/models").json()["models"]["image"]["error"]


@pytest.mark.real_model
def test_genuine_inference_with_trained_artifacts():
    """Runs only when real trained artifacts exist in ./artifacts (produced by the training pipeline)."""
    import os
    from pathlib import Path
    art = Path(os.getenv("ARTIFACT_DIR", "artifacts"))
    if not (art / "image_model.pth").exists():
        pytest.skip("no trained image artifacts in ./artifacts yet")
    s = dataclasses.replace(Settings.from_env(), artifact_dir=art)
    with TestClient(create_app(s)) as c:
        sample = os.getenv("DEEPVERIFY_TEST_IMAGE")
        data = open(sample, "rb").read() if sample else png(3)
        r = c.post("/api/v1/detect/image", files={"file": ("t.png", data, "image/png")})
        assert r.status_code in (200, 422), r.text      # 422 only if the model needs a face and the sample has none
        if r.status_code == 200:
            assert r.json()["prediction"] in ("REAL", "FAKE") and 0 <= r.json()["fake_probability"] <= 1
        vid = os.getenv("DEEPVERIFY_TEST_VIDEO")
        if (art / "video_model.pth").exists() and vid:
            rv = c.post("/api/v1/detect/video", files={"file": ("t.mp4", open(vid, "rb").read(), "video/mp4")})
            assert rv.status_code == 200 and rv.json()["media_type"] == "video", rv.text
