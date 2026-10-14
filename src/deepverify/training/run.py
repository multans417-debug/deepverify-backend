from __future__ import annotations


def training_warnings(history: list, val_metrics: dict, sizes: dict, detector: str) -> list:
    """Honest red flags recorded next to every trained model."""
    w = []
    if detector == "center":
        w.append("SMOKE: 'center' crop used instead of a face detector - model is not a real detector.")
    if sizes["train"] < 500 or sizes["val"] < 100:
        w.append(f"Small data (train={sizes['train']}, val={sizes['val']}): metrics are noisy and generalisation is doubtful.")
    last = history[-1]
    if last["train_acc"] - last["val_acc_at_0.5"] > 0.10:
        w.append("Possible overfitting: train accuracy exceeds validation accuracy by more than 10 points.")
    auc = val_metrics.get("roc_auc")
    if auc is not None and auc > 0.995:
        w.append("Suspiciously high validation ROC-AUC (>0.995): check for leakage / trivially separable data before trusting it.")
    vc = sizes["val_classes"]
    if min(vc.values()) == 0:
        w.append("Validation split has a single class: threshold/calibration are not meaningful.")
    ratio = max(sizes["train_classes"].values()) / max(1, min(sizes["train_classes"].values()))
    if ratio > 1.5:
        w.append(f"Class imbalance {ratio:.1f}:1 in training data (handled with pos_weight).")
    return w
