# Dataset research

Searched during the build (sources: arXiv / official GitHub / Hugging Face cards). **Nothing was downloaded.**

| Dataset | Official source | Content / size (as verified) | Access | Automated download? |
|---|---|---|---|---|
| DFDC (Meta/Facebook AI) | Meta dataset page (https://ai.meta.com/datasets/dfdc/, earlier ai.facebook.com/datasets/dfdc); paper arXiv:1910.08854 | Preview set ~5k videos (arXiv:1910.08854); full set >100k clips from 3,426 consenting actors (DFDC paper abstract). Labels in `metadata.json` (label, original) | Must request access and accept terms of service (preview paper) | Only after you accept; Kaggle `train_sample_videos.zip` is the usual small subset (**its size was not verified here** - check the Kaggle page) |
| FaceForensics++ | https://github.com/ondyari/FaceForensics | 1000 originals x 4 manipulations (Deepfakes, Face2Face, FaceSwap, NeuralTextures) + Google/Jigsaw DeepFakeDetection (3000+ videos, 28 actors). Compression options raw/c23/c40 | Google form; the download-script link is e-mailed after approval. Data under the FaceForensics Terms of Use (code MIT) | No - needs the approved link; the script can limit the video count |
| Celeb-DF v2 | https://github.com/yuezunli/celeb-deepfakeforensics | **Not verified in this session** (recalled: ~590 real / ~5.6k fake) | Request form (read the terms on the official page) | No |
| 140k Real and Fake Faces | https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces | 70k real (FFHQ) + 70k fake (StyleGAN) faces; 100k/20k/20k train/valid/test per a mirror's card. **License and download size not verified** | Free Kaggle account + API token | Yes (`download_datasets.py kaggle-dataset`) |

## Suitability
- **Deepfake (face-swap) videos:** DFDC sample or FaceForensics++ (use c23 and cap the count ~400-1000 videos to stay near 1-5 GB).
- **140k faces are GAN-generated faces, not face-swaps.** A model trained on them detects StyleGAN artefacts and will likely
  generalise poorly to swapped faces. Use it as a second domain, not as proof of deepfake detection.
- Recommended plan: DFDC sample (or FF++ c23 subset) -> `run_training_pipeline.py --video-root ... --image-from-frames`
  (image model trained on the same face-crop domain as the video model). Optionally add 140k as a separate image model.

## Hard rules followed
No mirrors, no scraping, no credential bypass; credentials only via env/`~/.kaggle`; every downloader stops with an explicit BLOCKER message.
