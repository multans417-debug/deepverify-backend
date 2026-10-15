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


def test_cors_deployed_frontend_and_preflights(tmp_path):
    frontend_origin = "https://deepverify-frontend-zu2d.onrender.com"
    s = make_settings(tmp_path, cors_origins=(frontend_origin, "http://localhost:5173"))
    with TestClient(create_app(s, load_models=False)) as c:
        for path, method in [
            ("/ready", "GET"),
            ("/health", "GET"),
            ("/api/v1/models", "GET"),
            ("/api/v1/detect/image", "POST"),
            ("/api/v1/detect/video", "POST"),
        ]:
            res = c.options(
                path,
                headers={
                    "Origin": frontend_origin,
                    "Access-Control-Request-Method": method,
                    "Access-Control-Request-Headers": "Content-Type, Accept, X-Requested-With",
                },
            )
            assert res.status_code == 200, f"Preflight failed for {path} with status {res.status_code}"
            assert res.headers.get("access-control-allow-origin") == frontend_origin
            assert "GET" in res.headers.get("access-control-allow-methods", "")
            assert "POST" in res.headers.get("access-control-allow-methods", "")

        ready_res = c.get("/ready", headers={"Origin": frontend_origin})
        assert ready_res.status_code == 503
        assert ready_res.headers.get("access-control-allow-origin") == frontend_origin

        health_res = c.get("/health", headers={"Origin": frontend_origin})
        assert health_res.status_code == 200
        assert health_res.headers.get("access-control-allow-origin") == frontend_origin

        err_res = c.post("/api/v1/detect/image", headers={"Origin": frontend_origin})
        assert err_res.status_code == 422
        assert err_res.headers.get("access-control-allow-origin") == frontend_origin

        unallowed = c.get("/ready", headers={"Origin": "https://untrusted-site.com"})
        assert "access-control-allow-origin" not in unallowed.headers


def test_cors_default_origins_and_trailing_slash_normalization(monkeypatch):
    monkeypatch.delenv("CORS_ORIGINS", raising=False)
    s = Settings.from_env()
    assert "https://deepverify-frontend-zu2d.onrender.com" in s.cors_origins

    monkeypatch.setenv("CORS_ORIGINS", "https://deepverify-frontend-zu2d.onrender.com/, http://localhost:5173/")
    s2 = Settings.from_env()
    assert s2.cors_origins == ("https://deepverify-frontend-zu2d.onrender.com", "http://localhost:5173")


def test_settings_from_env(monkeypatch):
    monkeypatch.setenv("CORS_ORIGINS", "https://a.example, https://b.example")
    monkeypatch.setenv("MAX_VIDEO_BYTES", "12345")
    monkeypatch.setenv("TRUST_PROXY_HEADERS", "true")
    s = Settings.from_env()
    assert s.cors_origins == ("https://a.example", "https://b.example") and s.max_video_bytes == 12345 and s.trust_proxy_headers
