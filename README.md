# DeepVerify backend

FastAPI service that classifies uploaded images/videos as REAL or FAKE with models **you train** (PyTorch, CPU inference),
plus the full data -> training -> evaluation pipeline.

## Honest status
| Stage | State |
|---|---|
| Datasets downloaded | **No** - the build sandbox had no network and public deepfake sets need your access approval (`docs/dataset_research.md`) |
| Models trained | **No** - no GPU/torch in the sandbox. `artifacts/` is empty, so the API answers 503 instead of inventing results |
| Code | Data pipeline, training, evaluation, API, tests, Docker, Render config written. Data/preprocessing/metrics/upload logic **executed and tested** (21 tests); torch/FastAPI parts **syntax-checked only** until you run `pytest` (`docs/environment_audit.md`) |
| Deployed | **No** |

## Quick start
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-train.txt
pytest -q                                   # plumbing tests use a random model: they test code paths, not accuracy
python scripts/make_synthetic_smoke_data.py
python scripts/run_training_pipeline.py --smoke --image-root data/synthetic_smoke/images --video-root data/synthetic_smoke/videos
```
Real training: `docs/colab_training.md`. Serve: `cp .env.example .env && uvicorn deepverify.main:app --reload --app-dir src`.

## API
| Endpoint | Purpose |
|---|---|
| `GET /health` | liveness |
| `GET /ready` | 200 only if required models are loaded, else 503 with reasons |
| `GET /api/v1/models` | loaded models, training status, thresholds, calibration |
| `POST /api/v1/detect/image` | multipart `file` (JPEG/PNG/WEBP) |
| `POST /api/v1/detect/video` | multipart `file` (MP4/MOV/AVI/MKV) |
| `GET /docs` | OpenAPI UI |

Response: `prediction`, `fake_probability`, `confidence`, `score_is_calibrated`, `decision_threshold`, `model_version`, `processing_time_seconds`, `details`, `warning`.
**Score semantics:** `fake_probability = sigmoid(logit / T)`, T fitted by temperature scaling on the validation split, decision threshold chosen on validation to maximise F1.
It is *not* a guarantee of correctness; calibration holds only for data like the training set (`ece_10_bins` in `artifacts/*_test_metrics.json`). `confidence` is the score of the predicted class.
Errors: `{success:false,error:{code,message}}` - 413 too large, 415 wrong type, 422 corrupt/no face/too long, 429 rate limit, 503 model missing/busy, 504 extraction timeout.

## Models
- **Image:** ResNet18 (ImageNet init, early layers frozen) fine-tuned on labelled REAL/FAKE images; single logit.
- **Video:** per-frame ResNet18 features of time-ordered face crops -> GRU -> one video-level logit (temporal model, not frame voting).
- Preprocessing code is shared by training and serving; settings are stored in `*_model_metadata.json` and used at inference.
- Models from `--smoke` runs carry `smoke_test: true` and are refused unless `ALLOW_SMOKE_MODELS=true`.

## Security & privacy
Uploads: streamed with size caps (also enforced in an ASGI middleware), magic-byte type check + full decode, client file names ignored (fixed name in a private temp dir deleted in `finally`),
pixel/duration/extraction-time limits, weights loaded with `weights_only=True` and SHA-256 verified, CORS allow-list, per-IP rate limit, bounded inference concurrency, no media or file names logged, no media stored.
Limits: the rate limiter is per process (use an edge limiter for multi-instance); media transits Render's infrastructure; Haar face detection misses hard faces; predictions can be wrong - never treat them as proof about a person.

## Layout
`src/deepverify/{api,schemas,services,models,preprocessing,data,training,evaluation,utils}`, `scripts/`, `tests/`, `docs/`, `Dockerfile`, `render.yaml`, `.env.example`.
Resume any stage: re-run `scripts/run_training_pipeline.py` with the same arguments.
