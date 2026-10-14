#!/usr/bin/env python
"""End-to-end orchestration: check -> prepare -> train -> evaluate -> report. Resumable.

Full run (Colab GPU):
  python scripts/run_training_pipeline.py --image-root data/raw/140k --video-root data/raw/dfdc_sample
Image model from video face crops (recommended for deepfake videos): add --image-from-frames instead of --image-root.
Smoke run on synthetic data (verifies plumbing only; produces a model the API refuses):
  python scripts/make_synthetic_smoke_data.py && python scripts/run_training_pipeline.py --smoke \
      --image-root data/synthetic_smoke/images --video-root data/synthetic_smoke/videos
If interrupted or a stage fails, re-run the SAME command: finished stages are skipped (state: artifacts/pipeline_state.json)
and training resumes from artifacts/checkpoints/*_last.pt. Use --force to redo everything.
"""
import _bootstrap  # noqa: F401
import argparse, json, os, shutil, subprocess, sys
from pathlib import Path

S = Path(__file__).resolve().parent


def check_resources(args):
    free = shutil.disk_usage(".").free / 1e9
    info = {"free_disk_gb": round(free, 1), "cpus": os.cpu_count()}
    try:
        import torch
        info.update(torch=torch.__version__, cuda=torch.cuda.is_available(),
                    gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
    except ImportError:
        sys.exit("BLOCKER: PyTorch is not installed. Run: pip install -r requirements-train.txt")
    print("Resources:", json.dumps(info))
    if free < 5 and not args.smoke:
        sys.exit(f"BLOCKER: only {free:.1f} GB free; a real run needs more (frames cache ~1-2 MB per video + datasets).")
    if not info["cuda"] and not args.smoke:
        print("WARNING: no GPU detected. Video training on CPU will be very slow; use Colab/Kaggle GPU or --smoke.")
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image-root"); ap.add_argument("--video-root")
    ap.add_argument("--work-dir", default="data/processed"); ap.add_argument("--artifact-dir", default=os.getenv("ARTIFACT_DIR", "artifacts"))
    ap.add_argument("--smoke", action="store_true"); ap.add_argument("--force", action="store_true")
    ap.add_argument("--image-epochs", type=int); ap.add_argument("--video-epochs", type=int)
    ap.add_argument("--image-from-frames", action="store_true",
                    help="train the image model on the face crops extracted from --video-root (matches the video domain)")
    a = ap.parse_args()
    if not (a.image_root or a.video_root): ap.error("give --image-root and/or --video-root")
    if a.image_from_frames and not a.video_root: ap.error("--image-from-frames needs --video-root")
    if a.image_from_frames and a.image_root: ap.error("use either --image-root or --image-from-frames, not both")
    art, work = Path(a.artifact_dir), Path(a.work_dir); art.mkdir(parents=True, exist_ok=True)
    state_path = art / "pipeline_state.json"
    if a.force:   # a forced rerun must not silently resume stale checkpoints
        shutil.rmtree(art / "checkpoints", ignore_errors=True); state_path.unlink(missing_ok=True)
    done = set() if not state_path.exists() else set(json.loads(state_path.read_text()))
    py = sys.executable
    check_resources(a)
    smoke_flags = ["--smoke", "--no-pretrained"] if a.smoke else []
    steps = []
    image_manifest = work / ("video/frames_manifest.csv" if a.image_from_frames else "image/image_manifest.csv")
    want_image = bool(a.image_root or a.image_from_frames)
    if a.image_root:
        steps.append(("prepare_image", [py, S / "prepare_dataset.py", "--kind", "image", "--root", a.image_root, "--out", work / "image"]
                      + (["--max-per-class", "150"] if a.smoke else [])))
        steps.append(("train_image", [py, S / "train_image.py", "--manifest", image_manifest, "--artifact-dir", art, "--resume",
                                      "--epochs", str(a.image_epochs or (2 if a.smoke else 8))]
                      + (["--batch-size", "16", "--num-workers", "0", "--input-size", "112"] if a.smoke else []) + smoke_flags))
    if a.video_root:
        steps.append(("prepare_video", [py, S / "prepare_dataset.py", "--kind", "video", "--root", a.video_root, "--out", work / "video"]
                      + (["--max-videos", "24", "--num-frames", "8", "--min-valid-frames", "4", "--crop-size", "112", "--face-detector", "center", "--smoke"] if a.smoke else [])))
        if a.image_from_frames:
            steps.append(("train_image", [py, S / "train_image.py", "--manifest", image_manifest, "--artifact-dir", art, "--resume",
                                          "--face-crop", "pipeline", "--epochs", str(a.image_epochs or (2 if a.smoke else 8))]
                          + (["--batch-size", "16", "--num-workers", "0"] if a.smoke else []) + smoke_flags))
        steps.append(("train_video", [py, S / "train_video.py", "--manifest", work / "video" / "video_manifest.csv", "--artifact-dir", art,
                                      "--resume", "--epochs", str(a.video_epochs or (2 if a.smoke else 10))]
                      + (["--batch-size", "4", "--num-workers", "0"] if a.smoke else []) + smoke_flags))
    ev = [py, S / "evaluate_models.py", "--artifact-dir", art] + (["--allow-smoke", "--num-workers", "0"] if a.smoke else [])
    if want_image: ev += ["--image-manifest", image_manifest]
    if a.video_root: ev += ["--video-manifest", work / "video" / "video_manifest.csv"]
    steps.append(("evaluate", ev))
    for name, cmd in steps:
        if name in done:
            print(f"[skip] {name} (already completed)"); continue
        print(f"\n=== {name} ===")
        r = subprocess.run([str(c) for c in cmd])
        if r.returncode:
            print(f"\nSTAGE FAILED: {name}. Completed stages are kept. Fix the cause and re-run the same command to resume.")
            sys.exit(r.returncode)
        done.add(name); state_path.write_text(json.dumps(sorted(done)))
    print("\nPipeline complete. Report: docs/training_report.md ; artifacts:", art)


if __name__ == "__main__":
    main()
