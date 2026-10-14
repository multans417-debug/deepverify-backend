"""Integration test for Hugging Face model.safetensors (KoreaPeter/ms-eff-gcvit-deepfake-b0-kodf).
Tests end-to-end model loading, preprocessing, and FastAPI inference endpoint execution.
"""
import io
import json
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from deepverify.config import Settings
from deepverify.main import create_app
from deepverify.models.loader import load_bundle


def generate_face_like_image_bytes() -> bytes:
    """Generate a simple synthetic image containing a face-like oval to trigger Haar Cascade detection."""
    img = np.full((300, 300, 3), 240, dtype=np.uint8)
    # Face boundary
    cv2.ellipse(img, (150, 150), (60, 80), 0, 0, 360, (180, 150, 120), -1)
    # Eyes
    cv2.circle(img, (130, 130), 10, (50, 50, 50), -1)
    cv2.circle(img, (170, 130), 10, (50, 50, 50), -1)
    # Mouth
    cv2.ellipse(img, (150, 180), (25, 12), 0, 0, 180, (50, 50, 50), 3)
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="PNG")
    return buf.getvalue()


def generate_noise_image_bytes() -> bytes:
    arr = np.random.default_rng(42).integers(0, 255, (200, 200, 3), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


@pytest.mark.plumbing
def test_load_safetensors_bundle():
    art_dir = Path("artifacts")
    if not (art_dir / "model.safetensors").exists():
        pytest.skip("artifacts/model.safetensors not present in workspace")

    bundle = load_bundle("image", art_dir, allow_smoke=True)
    assert bundle.kind == "image"
    assert bundle.metadata["architecture"] == "ms_eff_gcvit_b0"
    assert bundle.preprocess.input_size == 224


@pytest.mark.plumbing
def test_safetensors_fastapi_detect_image():
    art_dir = Path("artifacts")
    if not (art_dir / "model.safetensors").exists():
        pytest.skip("artifacts/model.safetensors not present in workspace")

    settings = Settings(
        artifact_dir=art_dir,
        cors_origins=("http://localhost:5173",),
        required_models=("image",),
        max_image_bytes=10_000_000,
        max_video_bytes=100_000_000,
        max_image_pixels=40_000_000,
        max_video_seconds=60,
        extract_timeout_seconds=45,
        max_concurrent_inferences=1,
        queue_timeout_seconds=20,
        rate_limit_per_minute=100,
        trust_proxy_headers=False,
        torch_threads=1,
        allow_smoke_models=True,
    )

    app = create_app(settings)
    with TestClient(app) as client:
        # 1. Verify ready endpoint
        ready_res = client.get("/ready")
        assert ready_res.status_code == 200, ready_res.text

        # 2. Test model description endpoint
        models_res = client.get("/api/v1/models")
        assert models_res.status_code == 200
        image_info = models_res.json()["models"]["image"]
        assert image_info["loaded"] is True
        assert image_info["architecture"] == "ms_eff_gcvit_b0"

        # 3. Test random noise image (triggers no_face_detected 422 as expected)
        noise_bytes = generate_noise_image_bytes()
        noise_res = client.post("/api/v1/detect/image", files={"file": ("noise.png", noise_bytes, "image/png")})
        assert noise_res.status_code in (200, 422)
        if noise_res.status_code == 422:
            assert noise_res.json()["error"]["code"] == "no_face_detected"

        # 4. Perform direct inference without face cropping requirement or with synthetic face
        bundle = client.app.state.registry.get("image")
        # Temporarily disable face crop check for raw image test
        pre_orig = bundle.preprocess.use_face_crop
        bundle.preprocess = type(bundle.preprocess)(
            input_size=224, mean=bundle.preprocess.mean, std=bundle.preprocess.std,
            use_face_crop=False, face_margin=0.2, face_detector="haar"
        )
        try:
            res = client.post("/api/v1/detect/image", files={"file": ("raw.png", noise_bytes, "image/png")})
            assert res.status_code == 200, res.text

            data = res.json()
            assert data["success"] is True
            assert data["media_type"] == "image"
            assert data["prediction"] in ("REAL", "FAKE")
            assert 0.0 <= data["fake_probability"] <= 1.0
            assert 0.0 <= data["confidence"] <= 1.0
            assert "processing_time_seconds" in data
        finally:
            bundle.preprocess = type(bundle.preprocess)(
                input_size=224, mean=bundle.preprocess.mean, std=bundle.preprocess.std,
                use_face_crop=pre_orig, face_margin=0.2, face_detector="haar"
            )


@pytest.mark.plumbing
def test_safetensors_fastapi_detect_video(tmp_path):
    art_dir = Path("artifacts")
    if not (art_dir / "model.safetensors").exists():
        pytest.skip("artifacts/model.safetensors not present in workspace")

    settings = Settings(
        artifact_dir=art_dir,
        cors_origins=("http://localhost:5173",),
        required_models=("image", "video"),
        max_image_bytes=10_000_000,
        max_video_bytes=100_000_000,
        max_image_pixels=40_000_000,
        max_video_seconds=60,
        extract_timeout_seconds=45,
        max_concurrent_inferences=1,
        queue_timeout_seconds=20,
        rate_limit_per_minute=100,
        trust_proxy_headers=False,
        torch_threads=1,
        allow_smoke_models=True,
    )

    app = create_app(settings)
    with TestClient(app) as client:
        # Verify both image and video models are loaded
        ready_res = client.get("/ready")
        assert ready_res.status_code == 200, ready_res.text

        models_res = client.get("/api/v1/models")
        assert models_res.status_code == 200
        video_info = models_res.json()["models"]["video"]
        assert video_info["loaded"] is True
        assert video_info["architecture"] == "ms_eff_gcvit_b0"

        # Generate a test avi video file with center face detector for smoke test
        vpath = tmp_path / "test_video.avi"
        writer = cv2.VideoWriter(str(vpath), cv2.VideoWriter_fourcc(*"MJPG"), 10, (96, 96))
        for i in range(10):
            frame = np.full((96, 96, 3), 100, np.uint8)
            cv2.circle(frame, (48, 48), 20, (200, 180, 160), -1)
            writer.write(frame)
        writer.release()

        bundle = client.app.state.registry.get("video")
        # Configure video preprocessor to use center detector for smoke test video
        v_orig = bundle.video
        v_pre_orig = bundle.preprocess
        bundle.video = type(v_orig)(
            num_frames=4, min_valid_frames=2, candidate_multiplier=1,
            input_size=224, face_margin=0.2, face_detector="center"
        )
        bundle.preprocess = type(v_pre_orig)(
            input_size=224, mean=v_pre_orig.mean, std=v_pre_orig.std,
            use_face_crop=True, face_margin=0.2, face_detector="center"
        )
        try:
            res = client.post("/api/v1/detect/video", files={"file": ("test_video.avi", vpath.read_bytes(), "video/x-msvideo")})
            assert res.status_code == 200, res.text

            data = res.json()
            assert data["success"] is True
            assert data["media_type"] == "video"
            assert data["prediction"] in ("REAL", "FAKE")
            assert 0.0 <= data["fake_probability"] <= 1.0
            assert 0.0 <= data["confidence"] <= 1.0
            assert "MS-EffGCViT" in data["details"]["aggregation"]
        finally:
            bundle.video = v_orig
            bundle.preprocess = v_pre_orig


@pytest.mark.plumbing
def test_ensure_artifacts_safetensors_and_metadata(tmp_path, monkeypatch):
    import hashlib
    import io
    import urllib.request
    from deepverify.utils.artifacts import ensure_artifacts, sha256_file

    model_content = b"test_safetensors_checkpoint_content_12345"
    model_sha = hashlib.sha256(model_content).hexdigest()

    meta_content = b'{"kind": "image", "architecture": "ms_eff_gcvit_b0"}'
    meta_sha = hashlib.sha256(meta_content).hexdigest()

    class MockResponse:
        def __init__(self, data):
            self._bio = io.BytesIO(data)

        def read(self, chunk=None):
            return self._bio.read(chunk if chunk is not None else -1)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    def mock_urlopen(url, *args, **kwargs):
        if "model.safetensors" in url:
            return MockResponse(model_content)
        if "image_model_metadata.json" in url:
            return MockResponse(meta_content)
        raise ValueError(f"Unexpected url: {url}")

    monkeypatch.setattr(urllib.request, "urlopen", mock_urlopen)

    # 1. Successful verified download of model.safetensors and metadata
    art_dir = tmp_path / "art_verified"
    monkeypatch.setenv("MODEL_SAFETENSORS_URL", "https://storage.example.com/model.safetensors")
    monkeypatch.setenv("MODEL_SAFETENSORS_SHA256", model_sha)
    monkeypatch.setenv("IMAGE_METADATA_URL", "https://storage.example.com/image_model_metadata.json")
    monkeypatch.setenv("IMAGE_METADATA_SHA256", meta_sha)

    notes = ensure_artifacts(art_dir)
    assert (art_dir / "model.safetensors").exists()
    assert (art_dir / "image_model_metadata.json").exists()
    assert sha256_file(art_dir / "model.safetensors") == model_sha
    assert any("model.safetensors: downloaded and verified" in n for n in notes)
    assert any("image_model_metadata.json: downloaded and verified" in n for n in notes)

    # 2. Refuses using legacy .pth env var for safetensors file
    pth_dir = tmp_path / "pth_guarded"
    monkeypatch.setenv("IMAGE_MODEL_URL", "https://storage.example.com/model.safetensors")
    monkeypatch.setenv("IMAGE_MODEL_SHA256", model_sha)
    pth_notes = ensure_artifacts(pth_dir)
    assert not (pth_dir / "image_model.pth").exists()
    assert any("refusing safetensors file via .pth URL" in n for n in pth_notes)

    # 3. Discards file on SHA-256 mismatch
    bad_dir = tmp_path / "bad_sha"
    monkeypatch.setenv("MODEL_SAFETENSORS_URL", "https://storage.example.com/model.safetensors")
    monkeypatch.setenv("MODEL_SAFETENSORS_SHA256", "f" * 64)
    bad_notes = ensure_artifacts(bad_dir)
    assert not (bad_dir / "model.safetensors").exists()
    assert any("SHA256 mismatch" in n for n in bad_notes)

