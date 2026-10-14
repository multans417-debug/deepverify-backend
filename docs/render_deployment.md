# Deploying the backend on Render

Nothing was deployed or verified on Render during the build. Follow these steps and check `/ready`.

1. Push this folder to GitHub (`.gitignore` already excludes models, data and secrets).
2. **Make the artifacts available at runtime** (Render's container filesystem is ephemeral). Pick one:
   - **Hosted files (default, config in `render.yaml`):** upload the 4 files (`image_model.pth`, `image_model_metadata.json`, `video_model.pth`,
     `video_model_metadata.json`) to private storage that serves HTTPS URLs you control (e.g. a private release asset / signed URLs / a model repo).
     Compute `sha256sum` of each and set `IMAGE_MODEL_URL`/`IMAGE_MODEL_SHA256`, ... in the Render dashboard. At start-up the service downloads
     missing files, **refuses any file whose SHA-256 differs**, and the loader additionally checks the weights hash against the metadata.
   - **Persistent disk:** attach a disk (a paid-instance feature - check current Render docs), mount it, set `ARTIFACT_DIR` to the mount path, copy the files once.
3. Render -> New -> Blueprint -> select the repo (uses `render.yaml`, Docker runtime). Set `CORS_ORIGINS` to your frontend origin exactly
   (e.g. `https://deepverify-frontend.onrender.com`, comma-separate several; no wildcard).
4. Memory: torch + ResNet18 + decoding video needs substantially more RAM than the smallest plans; start with `standard`, watch the metrics, adjust.
   CPU inference only; keep `MAX_CONCURRENT_INFERENCES=1` and one uvicorn worker.
5. Verify: `GET /health` -> 200; `GET /ready` -> 200 (503 lists exactly which model failed and why); `GET /api/v1/models`; then a real upload.
6. Free instances sleep and cold-start slowly (model download + load) - use a paid instance for a usable demo.

Frontend: call `POST {BACKEND}/api/v1/detect/image|video` with multipart field `file`; errors are `{success:false, error:{code,message}}`.
Note: your earlier DeepGuard plan used `POST /api/detect`; this backend exposes `/api/v1/detect/image` and `/api/v1/detect/video` as specified.
