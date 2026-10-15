"""Tests for memory optimizations: model sharing, parameter freezing, thread settings,
and video inference chunking equivalence.
"""
from pathlib import Path
import numpy as np
import pytest
import torch

from deepverify.config import Settings
from deepverify.models.loader import load_bundle
from deepverify.preprocessing.image import PreprocessConfig, preprocess_rgb
from deepverify.services.inference import _aggregate_frame_scores, detect_video
from deepverify.services.registry import ModelRegistry


def test_default_thread_settings(monkeypatch):
    monkeypatch.delenv("TORCH_THREADS", raising=False)
    monkeypatch.delenv("TORCH_INTEROP_THREADS", raising=False)
    s = Settings.from_env()
    assert s.torch_threads == 1
    assert s.torch_interop_threads == 1

    monkeypatch.setenv("TORCH_THREADS", "4")
    monkeypatch.setenv("TORCH_INTEROP_THREADS", "2")
    s2 = Settings.from_env()
    assert s2.torch_threads == 4
    assert s2.torch_interop_threads == 2


@pytest.mark.plumbing
def test_model_sharing_when_arch_and_sha_match():
    art_dir = Path("artifacts")
    if not (art_dir / "model.safetensors").exists():
        pytest.skip("artifacts/model.safetensors not present in workspace")

    settings = Settings(
        artifact_dir=art_dir,
        cors_origins=("http://localhost:5173",),
        required_models=("image", "video"),
        max_image_bytes=2_000_000,
        max_video_bytes=5_000_000,
        max_image_pixels=4_000_000,
        max_video_seconds=60,
        extract_timeout_seconds=30,
        max_concurrent_inferences=1,
        queue_timeout_seconds=5,
        rate_limit_per_minute=1000,
        trust_proxy_headers=False,
        torch_threads=1,
        allow_smoke_models=False,
    )
    reg = ModelRegistry(settings)
    reg.load_all()

    image_bundle = reg.bundles["image"]
    video_bundle = reg.bundles["video"]

    # 1. Single model instance shared between image and video pipelines
    assert image_bundle.model is video_bundle.model

    # 2. Separate metadata and configurations preserved
    assert image_bundle.kind == "image"
    assert video_bundle.kind == "video"
    assert image_bundle.video is None
    assert video_bundle.video is not None
    assert video_bundle.video.num_frames == 16
    assert image_bundle.preprocess.input_size == 224
    assert video_bundle.preprocess.input_size == 224

    # 3. Parameters are frozen (requires_grad = False)
    for p in image_bundle.model.parameters():
        assert p.requires_grad is False


def test_model_cache_separation_on_different_keys(tmp_path):
    cache = {}
    m1 = torch.nn.Linear(2, 2)
    m2 = torch.nn.Linear(2, 2)
    cache[("arch_a", "sha_1")] = m1
    cache[("arch_b", "sha_2")] = m2

    assert cache[("arch_a", "sha_1")] is m1
    assert cache[("arch_b", "sha_2")] is m2
    assert cache[("arch_a", "sha_1")] is not cache[("arch_b", "sha_2")]


@pytest.mark.plumbing
def test_video_chunked_inference_equivalence_and_order():
    art_dir = Path("artifacts")
    if not (art_dir / "model.safetensors").exists():
        pytest.skip("artifacts/model.safetensors not present in workspace")

    bundle = load_bundle("video", art_dir, allow_smoke=True)
    bundle.model.eval()

    # Create 16 distinct synthetic face crops
    rng = np.random.default_rng(1234)
    crops = [rng.integers(0, 255, (224, 224, 3), dtype=np.uint8) for _ in range(16)]

    # 1. Unchunked inference (single batch of 16)
    with torch.inference_mode():
        x_full = torch.from_numpy(np.stack([preprocess_rgb(c, bundle.preprocess) for c in crops]))
        logits_full = bundle.model(x_full)
        probs_full = torch.sigmoid(logits_full / bundle.temperature).cpu().numpy()

    # 2. Chunked inference with chunk_size = 4
    with torch.inference_mode():
        chunks_4 = []
        for i in range(0, len(crops), 4):
            x_c = torch.from_numpy(np.stack([preprocess_rgb(c, bundle.preprocess) for c in crops[i : i + 4]]))
            chunks_4.append(bundle.model(x_c))
        logits_chunked_4 = torch.cat(chunks_4, dim=0)
        probs_chunked_4 = torch.sigmoid(logits_chunked_4 / bundle.temperature).cpu().numpy()

    # 3. Chunked inference with chunk_size = 2
    with torch.inference_mode():
        chunks_2 = []
        for i in range(0, len(crops), 2):
            x_c = torch.from_numpy(np.stack([preprocess_rgb(c, bundle.preprocess) for c in crops[i : i + 2]]))
            chunks_2.append(bundle.model(x_c))
        logits_chunked_2 = torch.cat(chunks_2, dim=0)
        probs_chunked_2 = torch.sigmoid(logits_chunked_2 / bundle.temperature).cpu().numpy()

    # Assert logits and probabilities match to numerical precision
    diff_4 = (logits_full - logits_chunked_4).abs().max().item()
    diff_2 = (logits_full - logits_chunked_2).abs().max().item()
    assert diff_4 < 1e-5, f"Logits diff with chunk_size=4 too large: {diff_4}"
    assert diff_2 < 1e-5, f"Logits diff with chunk_size=2 too large: {diff_2}"

    # Assert aggregations match identically
    agg_full = _aggregate_frame_scores(probs_full, agg_mode="conf")
    agg_chunked_4 = _aggregate_frame_scores(probs_chunked_4, agg_mode="conf")
    agg_chunked_2 = _aggregate_frame_scores(probs_chunked_2, agg_mode="conf")
    assert abs(agg_full - agg_chunked_4) < 1e-6
    assert abs(agg_full - agg_chunked_2) < 1e-6

    # Verify frame order preservation: check each individual frame output matches
    for idx in range(16):
        with torch.inference_mode():
            x_single = torch.from_numpy(preprocess_rgb(crops[idx], bundle.preprocess)).unsqueeze(0)
            logit_single = bundle.model(x_single)[0]
        assert abs(logits_chunked_4[idx].item() - logit_single.item()) < 1e-5
