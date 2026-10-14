# Deploying the backend on Render

Nothing was deployed or verified on Render during the build. Follow these steps and check `/ready`.

1. Push this folder to GitHub (`.gitignore` already excludes models, data and secrets).
2. **Make the artifacts available at runtime** (Render's container filesystem is ephemeral). Pick one:
   - **Hosted files (default, config in `render.yaml`):** upload the 3 artifact files (`model.safetensors`, `image_model_metadata.json`,
     `video_model_metadata.json`) to private/public HTTPS storage you control (e.g. AWS S3 presigned URLs, Cloudflare R2, GitHub Release asset, Hugging Face model repository).
     Compute `sha256sum` of each and set the environment variables in the Render dashboard:
     - `MODEL_SAFETENSORS_URL` and `MODEL_SAFETENSORS_SHA256`
     - `IMAGE_METADATA_URL` and `IMAGE_METADATA_SHA256`
     - `VIDEO_METADATA_URL` and `VIDEO_METADATA_SHA256`
     At start-up, the service downloads any missing files into `ARTIFACT_DIR` preserving the `.safetensors` filename, **refuses non-HTTPS URLs or any file whose SHA-256 differs**, and the loader validates the checkpoint hash against the metadata before serving inference. Both image and video pipelines share this single checkpoint.
   - **Persistent disk:** attach a disk (a paid-instance feature - check current Render docs), mount it, set `ARTIFACT_DIR` to the mount path, and copy the files once.
3. Render -> New -> Blueprint -> select the repo (uses `render.yaml`, Docker runtime). Set `CORS_ORIGINS` to your frontend origin exactly
   (e.g. `https://deepverify-frontend.onrender.com`, comma-separate several; no wildcard).
4. Memory: torch + MS-EffGCViT (GCViT-B0) + decoding video needs memory; start with `standard`, watch metrics, adjust.
   CPU inference only; keep `MAX_CONCURRENT_INFERENCES=1` and one uvicorn worker.
5. Verify: `GET /health` -> 200; `GET /ready` -> 200 (503 lists exactly which model failed and why); `GET /api/v1/models`; then a real upload.
6. Free instances sleep and cold-start slowly (model download + load) - use a paid instance for a usable demo.

Frontend: call `POST {BACKEND}/api/v1/detect/image|video` with multipart field `file`; errors are `{success:false, error:{code,message}}`.
Note: your earlier DeepGuard plan used `POST /api/detect`; this backend exposes `/api/v1/detect/image` and `/api/v1/detect/video` as specified.
