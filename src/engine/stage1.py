from __future__ import annotations

import copy
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F


def masked_smooth_l1(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, beta: float = 5.0) -> torch.Tensor:
    if mask.sum() == 0:
        return torch.tensor(0.0, device=pred.device)
    diff = torch.abs(pred - target)
    loss = torch.where(diff < beta, 0.5 * diff * diff / beta, diff - 0.5 * beta)
    return (loss * mask).sum() / mask.sum()


def grad_norm_clip(parameters, max_norm=1.0):
    if max_norm is not None and max_norm > 0:
        torch.nn.utils.clip_grad_norm_(parameters, max_norm)


@torch.no_grad()
def evaluate_calendar_detail(encoder_cal, sow_head, slen_head, loader, device: str, cfg: Dict):
    if len(loader.dataset) == 0:
        return {"loss": np.nan, "mae_d": np.nan, "mae_l": np.nan}

    beta = cfg["training"]["smooth_beta"]
    sow_w = cfg["training"].get("sow_loss_w", 1.0)
    slen_w = cfg["training"].get("slen_loss_w", 1.0)

    encoder_cal.eval()
    sow_head.eval()
    slen_head.eval()

    total_loss, total_d, total_l = 0.0, 0.0, 0.0
    cnt_d, cnt_l = 0.0, 0.0

    for x_cal, _yu, _m, meta_true, _idx in loader:
        x_cal, meta_true = x_cal.to(device), meta_true.to(device)
        e = encoder_cal(x_cal)
        d_pred = sow_head(e)
        l_pred = slen_head(e)
        d_true = meta_true[:, 0:1]
        l_true = meta_true[:, 1:2]
        mask_d = (~torch.isnan(d_true)).float()
        mask_l = (~torch.isnan(l_true)).float()

        d_loss = masked_smooth_l1(d_pred, d_true, mask_d, beta=beta)
        l_loss = masked_smooth_l1(l_pred, l_true, mask_l, beta=beta)
        loss = sow_w * d_loss + slen_w * l_loss

        total_loss += loss.item() * x_cal.size(0)
        total_d += (torch.abs(d_pred - d_true) * mask_d).sum().item()
        total_l += (torch.abs(l_pred - l_true) * mask_l).sum().item()
        cnt_d += mask_d.sum().item()
        cnt_l += mask_l.sum().item()

    return {
        "loss": total_loss / max(len(loader.dataset), 1),
        "mae_d": total_d / max(cnt_d, 1.0),
        "mae_l": total_l / max(cnt_l, 1.0),
    }


def train_calendar_head(encoder_cal, sow_head, slen_head, train_loader, val_loader, device: str, cfg: Dict):
    encoder_cal.to(device)
    sow_head.to(device)
    slen_head.to(device)

    params = list(encoder_cal.parameters()) + list(sow_head.parameters()) + list(slen_head.parameters())
    train_cfg = cfg["training"]
    beta = train_cfg["smooth_beta"]
    grad_clip = train_cfg["grad_clip"]
    sow_w = train_cfg.get("sow_loss_w", 1.0)
    slen_w = train_cfg.get("slen_loss_w", 1.0)

    optimizer = torch.optim.AdamW(
        params,
        lr=train_cfg["lr_cal"],
        weight_decay=train_cfg["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.6, patience=6, min_lr=1e-5
    )

    best_metric = float("inf")
    best_epoch = -1
    best_state = None

    for epoch in range(1, train_cfg["epochs_cal"] + 1):
        encoder_cal.train()
        sow_head.train()
        slen_head.train()
        total_loss = 0.0

        for x_cal, _yu, _m, meta_true, _idx in train_loader:
            x_cal, meta_true = x_cal.to(device), meta_true.to(device)
            e = encoder_cal(x_cal)
            d_pred = sow_head(e)
            l_pred = slen_head(e)
            d_true = meta_true[:, 0:1]
            l_true = meta_true[:, 1:2]
            mask_d = (~torch.isnan(d_true)).float()
            mask_l = (~torch.isnan(l_true)).float()

            d_loss = masked_smooth_l1(d_pred, d_true, mask_d, beta=beta)
            l_loss = masked_smooth_l1(l_pred, l_true, mask_l, beta=beta)
            loss = sow_w * d_loss + slen_w * l_loss

            optimizer.zero_grad()
            loss.backward()
            grad_norm_clip(params, grad_clip)
            optimizer.step()
            total_loss += loss.item() * x_cal.size(0)

        train_loss = total_loss / max(len(train_loader.dataset), 1)
        val_metrics = evaluate_calendar_detail(encoder_cal, sow_head, slen_head, val_loader, device, cfg)
        metric = val_metrics["loss"] if len(val_loader.dataset) > 0 else train_loss
        scheduler.step(metric)

        if metric < best_metric:
            best_metric = metric
            best_epoch = epoch
            best_state = {
                "encoder_cal": copy.deepcopy(encoder_cal.state_dict()),
                "sow_head": copy.deepcopy(sow_head.state_dict()),
                "slen_head": copy.deepcopy(slen_head.state_dict()),
            }

        lr_now = optimizer.param_groups[0]["lr"]
        print(
            f"[Stage1] Epoch {epoch:03d}/{train_cfg['epochs_cal']} "
            f"TrainLoss={train_loss:.4f} VAL_loss={val_metrics['loss']:.4f} "
            f"VAL_d={val_metrics['mae_d']:.3f} VAL_L={val_metrics['mae_l']:.3f} lr={lr_now:.6f}"
        )

    if best_state is not None:
        encoder_cal.load_state_dict(best_state["encoder_cal"])
        sow_head.load_state_dict(best_state["sow_head"])
        slen_head.load_state_dict(best_state["slen_head"])

    print(f"[Stage1] Best epoch = {best_epoch}, best metric = {best_metric:.4f}")
