"""API tests. Marked tests labelled `mocked` replace the model with a stub and NEVER count as inference evidence.
Requires fastapi + httpx + torch (pip install -r requirements-train.txt); skipped otherwise."""
import dataclasses
import io
from types import SimpleNamespace

import pytest

pytest.importorskip("fastapi"); pytest.importorskip("httpx"); pytest.importorskip("torch")
from fastapi.testclient import TestClient  # noqa: E402
from PIL import Image  # noqa: E402

from deepverify.config import Settings  # noqa: E402
from deepverify.main import create_app  # noqa: E402
from deepverify.services import inference  # noqa: E402
from deepverify.services.registry import ModelRegistry  # noqa: E402


def make_settings(tmp_path, **over):
    up = tmp_path / "uploads"; up.mkdir(exist_ok=True)
    base = Settings(artifact_dir=tmp_path / "art", cors_origins=("http://localhost:5173",), required_models=("image", "video"),
                    max_image_bytes=200_000, max_video_bytes=500_000, max_image_pixels=4_000_000, max_video_seconds=60,
                    extract_timeout_seconds=30, max_concurrent_inferences=1, queue_timeout_seconds=5, rate_limit_per_minute=1000,
                    trust_proxy_headers=False, torch_threads=1, allow_smoke_models=False, tmp_dir=up)
    return dataclasses.replace(base, **over)


def png_bytes(size=32):
    b = io.BytesIO(); Image.new("RGB", (size, size), (120, 80, 200)).save(b, "PNG"); return b.getvalue()


def stub_registry(settings):
    reg = ModelRegistry(settings)
    bundle = SimpleNamespace(version="mock-0", threshold=0.5, calibrated=False, temperature=1.0, smoke_test=False,
                             metadata={}, preprocess=None, video=None)
    reg.bundles["image"] = reg.bundles["video"] = bundle
    return reg


def fake_result(media):
    return {"success": True, "media_type": media, "prediction": "REAL", "fake_probability": 0.2, "confidence": 0.8,
            "score_is_calibrated": False, "decision_threshold": 0.5, "model_version": "mock-0",
            "processing_time_seconds": 0.0, "details": {}}


def test_health_ok_and_ready_503_without_models(tmp_path):
    s = make_settings(tmp_path)
    with TestClient(create_app(s)) as c:                      # real start-up: artifacts dir is empty
        assert c.get("/health").json()["status"] == "ok"
        r = c.get("/ready")
        assert r.status_code == 503 and r.json()["ready"] is False
        assert all(m["loaded"] is False for m in c.get("/api/v1/models").json()["models"].values())
        d = c.post("/api/v1/detect/image", files={"file": ("a.png", png_bytes(), "image/png")})
        assert d.status_code == 503 and d.json()["error"]["code"] == "model_unavailable"   # clear error, never a fake prediction
        assert c.get("/docs").status_code == 200


@pytest.mark.mocked
def test_invalid_type_oversize_and_schema(tmp_path, monkeypatch):
    s = make_settings(tmp_path)
    monkeypatch.setattr(inference, "detect_image", lambda p, b, st: fake_result("image"))
    with TestClient(create_app(s, stub_registry(s), load_models=False)) as c:
        bad = c.post("/api/v1/detect/image", files={"file": ("evil.png", b"MZ not an image" * 10, "image/png")})
        assert bad.status_code == 415 and bad.json()["success"] is False
        big = c.post("/api/v1/detect/image", files={"file": ("big.png", png_bytes() + b"\0" * 400_000, "image/png")})
        assert big.status_code == 413
        ok = c.post("/api/v1/detect/image", files={"file": ("ok.png", png_bytes(), "image/png")})
        body = ok.json()
        assert ok.status_code == 200 and {"prediction", "fake_probability", "confidence", "model_version", "warning",
                                          "processing_time_seconds", "score_is_calibrated"} <= set(body)
        assert c.post("/api/v1/detect/image").status_code == 422     # missing field
    assert list((tmp_path / "uploads").iterdir()) == []              # temp files cleaned up


@pytest.mark.mocked
def test_temp_cleanup_when_inference_raises(tmp_path, monkeypatch):
    s = make_settings(tmp_path)

    def boom(p, b, st):
        raise RuntimeError("model exploded")
    monkeypatch.setattr(inference, "detect_image", boom)
    with TestClient(create_app(s, stub_registry(s), load_models=False), raise_server_exceptions=False) as c:
        r = c.post("/api/v1/detect/image", files={"file": ("a.png", png_bytes(), "image/png")})
        assert r.status_code == 500 and "exploded" not in r.text        # internals are not leaked
    assert list((tmp_path / "uploads").iterdir()) == []


@pytest.mark.mocked
def test_rate_limit(tmp_path, monkeypatch):
    s = make_settings(tmp_path, rate_limit_per_minute=2)
    monkeypatch.setattr(inference, "detect_image", lambda p, b, st: fake_result("image"))
    with TestClient(create_app(s, stub_registry(s), load_models=False)) as c:
        codes = [c.post("/api/v1/detect/image", files={"file": ("a.png", png_bytes(), "image/png")}).status_code for _ in range(3)]
    assert codes == [200, 200, 429]


def test_cors_allows_only_configured_origin(tmp_path):
    s = make_settings(tmp_path, cors_origins=("https://app.example.com",))
    with TestClient(create_app(s, load_models=False)) as c:
        ok = c.options("/api/v1/detect/image", headers={"Origin": "https://app.example.com", "Access-Control-Request-Method": "POST"})
        no = c.options("/api/v1/detect/image", headers={"Origin": "https://evil.example", "Access-Control-Request-Method": "POST"})
    assert ok.headers.get("access-control-allow-origin") == "https://app.example.com"
    assert "access-control-allow-origin" not in no.headers


def test_settings_from_env(monkeypatch):
    monkeypatch.setenv("CORS_ORIGINS", "https://a.example, https://b.example")
    monkeypatch.setenv("MAX_VIDEO_BYTES", "12345")
    monkeypatch.setenv("TRUST_PROXY_HEADERS", "true")
    s = Settings.from_env()
    assert s.cors_origins == ("https://a.example", "https://b.example") and s.max_video_bytes == 12345 and s.trust_proxy_headers
