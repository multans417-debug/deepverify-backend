# Environment audit (authoring sandbox)

Measured with shell commands at build time - not assumed.

| Item | Finding |
|---|---|
| OS | Linux (kernel 6.18), x86_64 |
| Python | 3.12.3 (target runtime is 3.11 via the Dockerfile; all code byte-compiles on 3.12) |
| CPU / RAM | 1 core / ~4 GB, no swap |
| Disk | ~10 GB free |
| GPU / CUDA | **None** (`nvidia-smi` not found) |
| Network | **Disabled** - `pip download` found no distributions and `curl` to github.com returned nothing |
| Installed | numpy 2.4.4, opencv 4.13, pillow 12.1.1, scikit-learn 1.8.0, mediapipe. **Not installed:** torch, torchvision, fastapi, uvicorn, pytest |
| Can do | create files, run shell, run Python with the packages above |
| Cannot do | download datasets, install packages, train (no torch/GPU), run FastAPI, deploy |

## Consequences (reported honestly)
- **No dataset was downloaded and no model was trained in the sandbox.** Every public video deepfake dataset also needs an access agreement that only you can accept.
- Verified here by execution: data discovery/labels, leakage-safe grouping, image + video preparation, face-crop extraction,
  corrupt-file handling, metrics/threshold/temperature maths, upload validation, body-size middleware, rate limiter, pipeline stage planning.
- **Written and syntax-checked but NOT executed here:** model definitions, training engine, evaluation script, FastAPI app and its tests
  (`test_api.py`, `test_model_plumbing.py`). They need `pip install -r requirements-train.txt`; run `pytest` on Colab/locally first.
- Pinned versions in `requirements*.txt` are a commonly compatible set, **not tested together in this sandbox**.
