"""Video -> aligned face-crop sequence. Used by dataset preparation AND by the API (identical pipeline)."""
from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass

import cv2
import numpy as np

from deepverify.preprocessing.face import crop_square, make_detector
from deepverify.utils.errors import InvalidMedia, NoFaceFound, ProcessingTimeout, VideoTooLong


@dataclass(frozen=True)
class VideoConfig:
    num_frames: int = 16            # frames fed to the temporal model
    min_valid_frames: int = 8       # fewer detected faces than this -> NoFaceFound (never a silent guess)
    candidate_multiplier: int = 2   # sample num_frames*mult candidates, keep those with a face
    input_size: int = 224
    face_margin: float = 0.3
    face_detector: str = "haar"
    max_seconds: float = 60.0
    timeout_seconds: float = 45.0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "VideoConfig":
        return cls(**{k: d[k] for k in cls.__dataclass_fields__ if k in d})


def sample_indices(n_frames: int, k: int) -> np.ndarray:
    """k roughly evenly spaced frame indices in [0, n_frames-1] (deduplicated, sorted)."""
    return np.unique(np.linspace(0, max(n_frames - 1, 0), num=k).round().astype(int))


def probe_video(path) -> dict:
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise InvalidMedia("Video could not be opened/decoded.")
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        if n <= 0 or not math.isfinite(fps) or fps <= 0:
            raise InvalidMedia("Video metadata (frame count / fps) is unreadable.")
        return {"frame_count": n, "fps": fps, "duration_seconds": n / fps}
    finally:
        cap.release()


def extract_face_crops(path, cfg: VideoConfig, detector=None) -> tuple:
    """Return (crops, stats). crops: list of `cfg.num_frames` RGB uint8 arrays (input_size^2), time-ordered.

    Strategy: sample num_frames*candidate_multiplier evenly spaced frames, detect the largest face in each,
    keep frames with a face; if fewer than min_valid_frames -> NoFaceFound; otherwise pick num_frames of the
    valid ones evenly (repeating frames only if fewer than num_frames faces were found).
    """
    deadline = time.monotonic() + cfg.timeout_seconds
    detector = detector or make_detector(cfg.face_detector)
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise InvalidMedia("Video could not be opened/decoded.")
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        if n <= 0 or not math.isfinite(fps) or fps <= 0:
            raise InvalidMedia("Video metadata (frame count / fps) is unreadable.")
        if n / fps > cfg.max_seconds:
            raise VideoTooLong(f"Video is longer than the {cfg.max_seconds:.0f}s limit.")
        wanted = sample_indices(n, cfg.num_frames * cfg.candidate_multiplier)
        wanted_set, last = set(wanted.tolist()), int(wanted[-1])
        valid, attempted = [], 0
        for idx in range(last + 1):
            if time.monotonic() > deadline:
                raise ProcessingTimeout("Frame extraction exceeded the time limit.")
            if not cap.grab():
                break
            if idx not in wanted_set:
                continue
            ok, bgr = cap.retrieve()
            if not ok:
                continue
            attempted += 1
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            box = detector.detect_largest(rgb)
            if box is not None:
                valid.append(crop_square(rgb, box, cfg.face_margin, cfg.input_size))
    finally:
        cap.release()
    if attempted == 0:
        raise InvalidMedia("No frames could be decoded from the video.")
    if len(valid) < cfg.min_valid_frames:
        raise NoFaceFound(f"Faces were found in only {len(valid)} of {attempted} sampled frames "
                          f"(minimum {cfg.min_valid_frames}).")
    pick = np.linspace(0, len(valid) - 1, cfg.num_frames).round().astype(int)
    crops = [valid[i] for i in pick]
    return crops, {"frames_sampled": attempted, "faces_found": len(valid), "frames_used": len(crops),
                   "duration_seconds": round(n / fps, 2)}
