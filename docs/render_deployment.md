# Deploying the backend on Render

Nothing was deployed or verified on Render during the build. Follow these steps and check `/ready`.

1. Push this folder to GitHub (`.gitignore` already excludes models, data and secrets).
2. **Make the artifacts available at runtime** (Render's container filesystem is ephemeral). Pick one:
   - **Hosted files via Private Hugging Face Repo (recommended, config in `render.yaml`):**
     1. Create a private model repository on Hugging Face (e.g. `https://huggingface.co/<your-username>/deepverify-models`).
     2. Upload the 3 files to your repository:
        - `model.safetensors`
        - `image_model_metadata.json`
        - `video_model_metadata.json`
     3. Generate a Hugging Face user access token with **Read** permissions at `https://huggingface.co/settings/tokens`.
     4. Set the following environment variables in the Render Dashboard (never commit real tokens):
        - `HF_TOKEN`: your Hugging Face read access token (`hf_...`)
        - `MODEL_SAFETENSORS_URL`: `https://huggingface.co/<org>/<repo>/resolve/main/model.safetensors`
        - `MODEL_SAFETENSORS_SHA256`: `3ec590c32f94f4297615c4d915e07d4bec4ebd4a7e0171f1e7a1b603a7f9b255`
        - `IMAGE_METADATA_URL`: `https://huggingface.co/<org>/<repo>/resolve/main/image_model_metadata.json`
        - `IMAGE_METADATA_SHA256`: `9a5f6f7693bae9fad8f67e9436bb2a30c3bab0f42b443987f93a0d900086df02`
        - `VIDEO_METADATA_URL`: `https://huggingface.co/<org>/<repo>/resolve/main/video_model_metadata.json`
        - `VIDEO_METADATA_SHA256`: `b17221979fc29a124b250ace950daef74d325a92641fff5e06fbe8d540952ecb`
     At start-up, the service downloads any missing files into `ARTIFACT_DIR` using Bearer authorization. The downloader ensures the `Authorization` header is stripped across CDN redirects (e.g., `cdn-lfs.huggingface.co` or S3) to prevent credential leakage. It rejects any non-HTTPS URLs, unexpected HTML/login pages, or files whose SHA-256 differs. Both image and video pipelines share this single checkpoint.
   - **Persistent disk:** attach a disk (a paid-instance feature - check current Render docs), mount it, set `ARTIFACT_DIR` to the mount path, and copy the files once.
3. Render -> New -> Blueprint -> select the repo (uses `render.yaml`, Docker runtime). Set `CORS_ORIGINS` to your frontend origin exactly
   (e.g. `https://deepverify-frontend.onrender.com`, comma-separate several; no wildcard).
4. Memory: torch + MS-EffGCViT (GCViT-B0) + decoding video needs memory; start with `standard`, watch metrics, adjust.
   CPU inference only; keep `MAX_CONCURRENT_INFERENCES=1` and one uvicorn worker.
5. Verify: `GET /health` -> 200; `GET /ready` -> 200 (503 lists exactly which model failed and why); `GET /api/v1/models`; then a real upload.
6. Free instances sleep and cold-start slowly (model download + load) - use a paid instance for a usable demo.

Frontend: call `POST {BACKEND}/api/v1/detect/image|video` with multipart field `file`; errors are `{success:false, error:{code,message}}`.
Note: your earlier DeepGuard plan used `POST /api/detect`; this backend exposes `/api/v1/detect/image` and `/api/v1/detect/video` as specified.
