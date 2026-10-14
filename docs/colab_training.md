# Train on Google Colab (GPU) - exact commands

Runtime -> Change runtime type -> GPU. Training must not run inside the Render web service.

```bash
!git clone <YOUR_REPO_URL> deepverify-backend && cd deepverify-backend
%cd deepverify-backend
!pip install -r requirements-train.txt
!python -m pytest -q            # real pytest run of every test (plumbing tests use a random model)
```

### 1. Prove the pipeline works (2 minutes, synthetic data, produces a model the API refuses)
```bash
!python scripts/make_synthetic_smoke_data.py
!python scripts/run_training_pipeline.py --smoke --image-root data/synthetic_smoke/images --video-root data/synthetic_smoke/videos --force
```

### 2. Get real data (you must accept each dataset's terms yourself)
```python
from google.colab import files; files.upload()      # upload kaggle.json (never commit it)
```
```bash
!mkdir -p ~/.kaggle && mv kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
# accept the DFDC competition rules on kaggle.com first, then:
!python scripts/download_datasets.py kaggle-competition deepfake-detection-challenge --file train_sample_videos.zip --dest data/raw/dfdc_sample --expected-gb 4
!python scripts/inspect_dataset.py data/raw/dfdc_sample --kind video
# FaceForensics++: apply via the form on its GitHub page, then run the script you receive (c23, limited video count).
```

### 3. Full run (image model on the same face crops as the video model)
```bash
!python scripts/run_training_pipeline.py --video-root data/raw/dfdc_sample --image-from-frames
```
Interrupted? Run the **same command** again: finished stages are skipped and training resumes from `artifacts/checkpoints/*_last.pt`. `--force` restarts.
Outputs: `artifacts/{image,video}_model.pth`, `*_metadata.json`, `*_training_metrics.json`, `*_test_metrics.json`, `docs/training_report.md`.
**Read the report's flags** (small data, overfitting, suspiciously high AUC, leakage) before trusting any number.

### 4. Keep the artifacts
Copy `artifacts/*.pth` and `artifacts/*_metadata.json` to Drive, then host them (see `docs/render_deployment.md`). Never commit them to Git.
