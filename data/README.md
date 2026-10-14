# data/

Git-ignored. Layout produced by the scripts:
```
data/raw/<dataset>/...            downloaded datasets (see docs/dataset_research.md)
data/processed/image/             image_manifest.csv, class_distribution.json, exclusions.csv
data/processed/video/             video_manifest.csv, frames_manifest.csv, frames/<video_id>/NNN.png, reports
data/synthetic_smoke/             SYNTHETIC pipeline-plumbing data only (meaningless for accuracy)
```
Respect each dataset's license; never commit data, credentials or models.
