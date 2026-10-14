from __future__ import annotations

import csv
from pathlib import Path

LABEL_TO_INT = {"REAL": 0, "FAKE": 1}


def write_manifest(path: Path, rows: list, fields: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def read_manifest(path: Path, split: str | None = None) -> list:
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["label"] = int(r["label"])
    return [r for r in rows if split is None or r["split"] == split]


def class_counts(rows: list) -> dict:
    return {"REAL": sum(1 for r in rows if r["label"] == 0), "FAKE": sum(1 for r in rows if r["label"] == 1)}
