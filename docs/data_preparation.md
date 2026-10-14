# Data preparation

`scripts/prepare_dataset.py --kind image|video --root RAW --out data/processed/<kind>`

**Labels** come only from a documented `metadata.json` (DFDC style) or unambiguous directory names (`real`, `original_sequences`,
`fake`, `manipulated_sequences`, `Celeb-real`, ...). A path matching both or neither is **excluded and listed in `exclusions.csv`** - never guessed.

**Validation:** every image is fully decoded (corrupt files excluded); videos are probed/decoded; no-face videos are excluded with reason `no_face_detected`.

**De-duplication (images):** exact SHA-256 duplicates removed (priority test > val > train, so a duplicate can never leak into train);
`--near-dup` also drops identical 64-bit dHash images.

**Leakage prevention (videos):** split by *source group* **before** extracting frames. Groups are connected components over identity tokens:
DFDC fake <-> `original`; Celeb-DF `id16_id0_0003` -> {id16,id0}; FaceForensics++ `000_003` -> {000,003}. If one component swallows >40% of
the videos (cannot be split) the code falls back to first-identity grouping, records `grouping_mode`/`grouping_warning` in `class_distribution.json`, and
you should prefer the dataset's official train/test list. A post-hoc check asserts no group appears in two splits. Without identity metadata each file
is its own group (documented limitation: near-copies of one source can leak). Image datasets with their own train/valid/test folders keep that split.

**Face crops:** OpenCV Haar cascade (no download, CPU) -> largest face, 30% margin, square crop. Haar misses profile/small/occluded faces; swap in a stronger
detector via `preprocessing/face.py` if recall is poor. Per video: sample 2x candidates evenly, keep frames with a face, require >= `min_valid_frames`, pick
`num_frames` evenly. Crops are stored as lossless PNG so training pixels equal serving pixels.

**Outputs:** `image_manifest.csv` / `video_manifest.csv` / `frames_manifest.csv` (frame -> video_id -> split), `class_distribution.json`, `exclusions.csv`, `*_prep_config.json`.
Disk: ~1-2 MB per video for 16 crops at 224 px (`--crop-size 160` shrinks it).
