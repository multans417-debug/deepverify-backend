"""Generic binary training loop used by both image and video models (single logit, BCE-with-logits)."""
from __future__ import annotations

import copy
import logging
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from deepverify.evaluation.metrics import sigmoid
from sklearn.metrics import roc_auc_score

log = logging.getLogger("deepverify.train")


def set_train(model) -> None:
    model.train()
    for m in getattr(model, "frozen_modules", []):
        m.eval()                      # keep BatchNorm statistics of frozen layers fixed


@torch.inference_mode()
def predict_logits(model, loader, device) -> tuple:
    model.eval()
    zs, ys = [], []
    for x, y in loader:
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            z = model(x.to(device, non_blocking=True))
        zs.append(z.float().cpu().numpy()); ys.append(y.numpy())
    return np.concatenate(zs), np.concatenate(ys)


def _auc(y, z):
    return float(roc_auc_score(y, z)) if len(set(y.astype(int).tolist())) == 2 else None


def train_model(model, train_ds, val_ds, *, device, epochs, batch_size, lr, weight_decay, num_workers,
                pos_weight: float, ckpt_path, resume: bool, patience: int, seed: int) -> dict:
    model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs))
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([float(pos_weight)], device=device))
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    pin = device.type == "cuda"
    g = torch.Generator(); g.manual_seed(seed)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin,
                          drop_last=len(train_ds) > batch_size, generator=g)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin)

    history, best, start, bad, resumed_from = [], {"score": -1e9, "epoch": -1, "state": None}, 0, 0, None
    if resume and ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
        history, best, start, bad = ck["history"], ck["best"], ck["epoch"] + 1, ck["bad"]
        resumed_from = ck["epoch"]
        log.info("Resumed from epoch %d", ck["epoch"])
    t0, stopped_early = time.time(), False
    for epoch in range(start, epochs):
        train_ds.set_epoch(epoch)
        set_train(model)
        tl, tc, tn = 0.0, 0, 0
        for x, y in train_dl:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                z = model(x)
                loss = loss_fn(z.float(), y)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            tl += float(loss) * len(y); tc += int(((z > 0).float() == y).sum()); tn += len(y)
        sched.step()
        vz, vy = predict_logits(model, val_dl, device)
        vloss = float(loss_fn(torch.from_numpy(vz).float().to(device), torch.from_numpy(vy).float().to(device)))
        vauc = _auc(vy, vz)
        vacc = float(((vz > 0) == (vy > 0.5)).mean())
        rec = {"epoch": epoch, "train_loss": tl / max(tn, 1), "train_acc": tc / max(tn, 1), "val_loss": vloss,
               "val_acc_at_0.5": vacc, "val_roc_auc": vauc, "lr": sched.get_last_lr()[0]}
        history.append(rec)
        log.info("epoch %d | train_loss %.4f acc %.3f | val_loss %.4f acc %.3f auc %s", epoch, rec["train_loss"],
                 rec["train_acc"], vloss, vacc, f"{vauc:.4f}" if vauc is not None else "n/a")
        score = vauc if vauc is not None else -vloss
        if score > best["score"]:
            best = {"score": score, "epoch": epoch, "state": copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})}
            bad = 0
        else:
            bad += 1
        tmp = str(ckpt_path) + ".tmp"
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(), "history": history,
                    "best": best, "epoch": epoch, "bad": bad}, tmp)
        os.replace(tmp, ckpt_path)           # atomic: an interrupted run never leaves a corrupt checkpoint
        if bad >= patience:
            stopped_early = True
            log.info("Early stopping (no val improvement for %d epochs)", patience)
            break
    if best["state"] is None:
        raise RuntimeError("Training produced no model (zero epochs run?).")
    model.load_state_dict(best["state"])
    vz, vy = predict_logits(model, val_dl, device)       # validation logits of the BEST model (for calibration/threshold)
    return {"history": history, "best_epoch": best["epoch"], "best_state": best["state"], "val_logits": vz, "val_labels": vy,
            "train_seconds": round(time.time() - t0, 1), "stopped_early": stopped_early, "resumed_from_epoch": resumed_from}
