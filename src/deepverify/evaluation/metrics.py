"""Metrics, threshold selection and temperature calibration (numpy / scikit-learn / scipy only)."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss, confusion_matrix,
                             precision_recall_fscore_support, roc_auc_score)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(np.asarray(x, dtype=np.float64), -60, 60)))


def expected_calibration_error(y, p, bins: int = 10) -> float:
    y, p = np.asarray(y), np.asarray(p)
    edges, ece = np.linspace(0, 1, bins + 1), 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & ((p < hi) if hi < 1 else (p <= hi))
        if m.any():
            ece += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(ece)


def compute_metrics(y_true, scores, threshold: float) -> dict:
    """scores = P(FAKE). Positive class = FAKE (1). Headline precision/recall/F1 are for FAKE."""
    y, s = np.asarray(y_true).astype(int), np.asarray(scores, dtype=np.float64)
    out = {"n": int(len(y)), "threshold": float(threshold),
           "class_counts": {"REAL": int((y == 0).sum()), "FAKE": int((y == 1).sum())}}
    if len(y) == 0:
        out["error"] = "empty evaluation set"
        return out
    pred = (s >= threshold).astype(int)
    p, r, f, sup = precision_recall_fscore_support(y, pred, labels=[0, 1], zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    both = len(set(y.tolist())) == 2
    out.update({
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(p[1]), "recall": float(r[1]), "f1": float(f[1]),
        "roc_auc": float(roc_auc_score(y, s)) if both else None,
        "average_precision": float(average_precision_score(y, s)) if both else None,
        "confusion_matrix": {"labels": ["REAL", "FAKE"], "matrix": [[int(tn), int(fp)], [int(fn), int(tp)]]},
        "per_class": {"REAL": {"precision": float(p[0]), "recall": float(r[0]), "f1": float(f[0]), "support": int(sup[0])},
                      "FAKE": {"precision": float(p[1]), "recall": float(r[1]), "f1": float(f[1]), "support": int(sup[1])}},
        "brier_score": float(brier_score_loss(y, s)),
        "ece_10_bins": expected_calibration_error(y, s),
    })
    if not both:
        out["warning"] = "Only one class present: ROC-AUC undefined."
    return out


def select_threshold(y_true, scores) -> tuple:
    """Pick the threshold maximising F1 (FAKE) on VALIDATION scores. Returns (threshold, method)."""
    y, s = np.asarray(y_true).astype(int), np.asarray(scores, dtype=np.float64)
    if len(set(y.tolist())) < 2:
        return 0.5, "default_0.5 (validation set has one class)"
    cands = np.unique(np.quantile(s, np.linspace(0, 1, 501)))
    best_t, best_f, best_gap = 0.5, -1.0, 9.0
    for t in cands:
        pred = (s >= t).astype(int)
        tp, fp, fn = int(((pred == 1) & (y == 1)).sum()), int(((pred == 1) & (y == 0)).sum()), int(((pred == 0) & (y == 1)).sum())
        f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
        gap = abs(t - 0.5)
        if f1 > best_f + 1e-12 or (abs(f1 - best_f) <= 1e-12 and gap < best_gap):
            best_t, best_f, best_gap = float(t), f1, gap
    return best_t, "max F1 for class FAKE on the validation split"


def fit_temperature(logits, labels) -> float:
    """Temperature scaling (Guo et al. 2017): minimise validation NLL over a single scalar T."""
    from scipy.optimize import minimize_scalar
    z, y = np.asarray(logits, dtype=np.float64), np.asarray(labels, dtype=np.float64)

    def nll(log_t):
        p = np.clip(sigmoid(z / np.exp(log_t)), 1e-7, 1 - 1e-7)
        return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())

    res = minimize_scalar(nll, bounds=(-3.0, 3.0), method="bounded")
    return float(np.clip(np.exp(res.x), 0.05, 20.0))
