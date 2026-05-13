from __future__ import annotations

import copy
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F

from src.engine.evaluate import ant_aggregate_batch, compute_zero_leak_val_mae


class ModelEMA:
    def __init__(self, module: torch.nn.Module, decay: float = 0.995):
        self.decay = float(decay)
        self.shadow = {k: v.detach().clone() for k, v in module.state_dict().items()}
        self.backup = None

    @torch.no_grad()
    def update(self, module: torch.nn.Module):
        for k, v in module.state_dict().items():
            self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1.0 - self.decay)

    @torch.no_grad()
    def apply_to(self, module: torch.nn.Module):
        self.backup = {k: v.detach().clone() for k, v in module.state_dict().items()}
        module.load_state_dict(self.shadow, strict=True)

    @torch.no_grad()
    def restore(self, module: torch.nn.Module):
        if self.backup is not None:
            module.load_state_dict(self.backup, strict=True)
            self.backup = None


def apply_ema_triplet(ema_enc_ant, ema_bridge, ema_student, encoder_ant, bridge_nl, student_parallel):
    if ema_enc_ant is None:
        return
    ema_enc_ant.apply_to(encoder_ant)
    ema_bridge.apply_to(bridge_nl)
    ema_student.apply_to(student_parallel)


def restore_ema_triplet(ema_enc_ant, ema_bridge, ema_student, encoder_ant, bridge_nl, student_parallel):
    if ema_enc_ant is None:
        return
    ema_enc_ant.restore(encoder_ant)
    ema_bridge.restore(bridge_nl)
    ema_student.restore(student_parallel)


def masked_smooth_l1(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, beta: float = 5.0) -> torch.Tensor:
    if mask.sum() == 0:
        return torch.tensor(0.0, device=pred.device)
    diff = torch.abs(pred - target)
    loss = torch.where(diff < beta, 0.5 * diff * diff / beta, diff - 0.5 * beta)
    return (loss * mask).sum() / mask.sum()


def order_penalty(y_unit: torch.Tensor, weight: float = 1.0) -> torch.Tensor:
    if weight <= 0:
        return torch.tensor(0.0, device=y_unit.device)
    diffs = y_unit[:, :-1] - y_unit[:, 1:]
    return weight * F.relu(diffs).mean()


def grad_norm_clip(parameters, max_norm=1.0):
    if max_norm is not None and max_norm > 0:
        torch.nn.utils.clip_grad_norm_(parameters, max_norm)


def get_student_mix_ratio(epoch: int, epochs: int, max_ratio: float = 0.75, warmup_frac: float = 0.25):
    if epochs <= 1:
        return max_ratio
    warmup_epochs = max(1, int(round(epochs * warmup_frac)))
    if epoch <= warmup_epochs:
        return 0.15
    progress = (epoch - warmup_epochs) / max(epochs - warmup_epochs, 1)
    progress = float(np.clip(progress, 0.0, 1.0))
    return 0.15 + (max_ratio - 0.15) * progress


def train_teacher_ant(encoder_ant, teacher_ar, series_list, loader, device: str, config: Dict, scaler_ant: Dict):
    train_cfg = config["training"]
    order_w = train_cfg["order_w_teacher"]
    beta = train_cfg["smooth_beta"]
    grad_clip = train_cfg["grad_clip"]

    encoder_ant.to(device)
    teacher_ar.to(device)
    params = list(encoder_ant.parameters()) + list(teacher_ar.parameters())
    optimizer = torch.optim.AdamW(params, lr=train_cfg["lr_ant"], weight_decay=train_cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(train_cfg["epochs_ant"] // 2, 1), eta_min=1e-5)

    epochs_teacher = max(1, train_cfg["epochs_ant"] // 2)
    for epoch in range(1, epochs_teacher + 1):
        encoder_ant.train(); teacher_ar.train()
        total = 0.0
        tf_ratio = 1.0 - 0.35 * ((epoch - 1) / max(epochs_teacher - 1, 1))

        for _x_cal, y_unit, mask, meta_true, idxs in loader:
            y_unit = y_unit.to(device)
            mask = mask.to(device)
            meta_true = meta_true.to(device)
            d_true = meta_true[:, 0:1].cpu().numpy().reshape(-1)
            l_true = meta_true[:, 1:2].cpu().numpy().reshape(-1)

            x_ant_np = ant_aggregate_batch(series_list, idxs, d_true, l_true, config, scaler_ant)
            x_ant = torch.tensor(x_ant_np, dtype=torch.float32, device=device)
            e = encoder_ant(x_ant)
            y_teacher = teacher_ar(e, y_teacher=y_unit, teacher_forcing_ratio=tf_ratio)

            sup = masked_smooth_l1(y_teacher, y_unit, mask, beta=beta)
            ord_loss = order_penalty(y_teacher, weight=order_w)
            loss = sup + ord_loss

            optimizer.zero_grad()
            loss.backward()
            grad_norm_clip(params, grad_clip)
            optimizer.step()
            total += loss.item() * x_ant.size(0)

        scheduler.step()
        lr_now = optimizer.param_groups[0]["lr"]
        print(f"[Stage2-Teacher] Epoch {epoch:03d}/{epochs_teacher} Loss={total / len(loader.dataset):.4f} tf_ratio={tf_ratio:.3f} lr={lr_now:.6f}")


def train_student_ant(
    encoder_cal,
    sow_head,
    slen_head,
    encoder_ant,
    teacher_ar,
    bridge_nl,
    student_parallel,
    train_series_list,
    val_series_list,
    train_loader,
    val_loader,
    device: str,
    config: Dict,
    scaler_ant: Dict,
):
    train_cfg = config["training"]
    ranges = config["ranges"]
    beta = train_cfg["smooth_beta"]
    grad_clip = train_cfg["grad_clip"]

    encoder_cal.to(device); sow_head.to(device); slen_head.to(device)
    encoder_ant.to(device); teacher_ar.to(device); bridge_nl.to(device); student_parallel.to(device)

    encoder_cal.eval(); sow_head.eval(); slen_head.eval()
    for p in encoder_cal.parameters():
        p.requires_grad = False
    for p in sow_head.parameters():
        p.requires_grad = False
    for p in slen_head.parameters():
        p.requires_grad = False
    for p in teacher_ar.parameters():
        p.requires_grad = False

    params = list(encoder_ant.parameters()) + list(bridge_nl.parameters()) + list(student_parallel.parameters())
    optimizer = torch.optim.AdamW(params, lr=train_cfg["lr_ant"], weight_decay=train_cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.65, patience=5, min_lr=1e-5)

    use_ema = train_cfg.get("use_ema", False)
    ema_decay = train_cfg.get("ema_decay", 0.995)
    if use_ema:
        ema_enc_ant = ModelEMA(encoder_ant, decay=ema_decay)
        ema_bridge = ModelEMA(bridge_nl, decay=ema_decay)
        ema_student = ModelEMA(student_parallel, decay=ema_decay)
    else:
        ema_enc_ant = ema_bridge = ema_student = None

    best_zero_leak_val = float("inf")
    best_epoch = -1
    best_state = None

    for epoch in range(1, train_cfg["epochs_ant"] + 1):
        encoder_ant.train(); bridge_nl.train(); student_parallel.train()
        total = 0.0
        mix_ratio = get_student_mix_ratio(
            epoch,
            train_cfg["epochs_ant"],
            max_ratio=train_cfg["student_pred_mix_max"],
            warmup_frac=train_cfg["student_pred_mix_warmup"],
        )

        for x_cal, y_unit, mask, meta_true, idxs in train_loader:
            x_cal = x_cal.to(device)
            y_unit = y_unit.to(device)
            mask = mask.to(device)
            meta_true = meta_true.to(device)

            with torch.no_grad():
                e_cal = encoder_cal(x_cal)
                d_pred_cur = sow_head(e_cal).squeeze(-1).cpu().numpy()
                l_pred_cur = slen_head(e_cal).squeeze(-1).cpu().numpy()

            d_true = meta_true[:, 0:1].cpu().numpy().reshape(-1)
            l_true = meta_true[:, 1:2].cpu().numpy().reshape(-1)

            d_used = (1.0 - mix_ratio) * d_true + mix_ratio * d_pred_cur
            l_used = (1.0 - mix_ratio) * l_true + mix_ratio * l_pred_cur
            if train_cfg["dl_jitter_std"] > 0:
                d_used += np.random.normal(0.0, train_cfg["dl_jitter_std"], size=d_used.shape)
                l_used += np.random.normal(0.0, train_cfg["dl_jitter_std"], size=l_used.shape)
            d_used = np.clip(d_used, ranges["sow_min"], ranges["sow_max"])
            l_used = np.clip(l_used, ranges["slen_min"], ranges["slen_max"])

            x_ant_np = ant_aggregate_batch(train_series_list, idxs, d_used, l_used, config, scaler_ant)
            x_ant = torch.tensor(x_ant_np, dtype=torch.float32, device=device)
            e = encoder_ant(x_ant)

            with torch.no_grad():
                y_teacher = teacher_ar(e, y_teacher=y_unit, teacher_forcing_ratio=0.85)
            y_soft = bridge_nl(y_teacher)
            y_student = student_parallel(e)

            sup = masked_smooth_l1(y_student, y_unit, mask, beta=beta)
            kd = masked_smooth_l1(y_student, y_soft, torch.ones_like(mask), beta=beta)
            ord_loss = order_penalty(y_student, weight=train_cfg["order_w_student"])
            loss = train_cfg["lambda_sup"] * sup + train_cfg["lambda_kd"] * kd + ord_loss

            optimizer.zero_grad()
            loss.backward()
            grad_norm_clip(params, grad_clip)
            optimizer.step()
            if ema_enc_ant is not None:
                ema_enc_ant.update(encoder_ant)
                ema_bridge.update(bridge_nl)
                ema_student.update(student_parallel)
            total += loss.item() * x_ant.size(0)

        apply_ema_triplet(ema_enc_ant, ema_bridge, ema_student, encoder_ant, bridge_nl, student_parallel)
        zero_leak_val = compute_zero_leak_val_mae(
            encoder_cal,
            sow_head,
            slen_head,
            encoder_ant,
            student_parallel,
            val_series_list,
            val_loader,
            device,
            config,
            scaler_ant,
        )
        restore_ema_triplet(ema_enc_ant, ema_bridge, ema_student, encoder_ant, bridge_nl, student_parallel)
        scheduler.step(zero_leak_val if np.isfinite(zero_leak_val) else (total / max(len(train_loader.dataset), 1)))
        lr_now = optimizer.param_groups[0]["lr"]

        print(
            f"[Stage2-Student] Epoch {epoch:03d}/{train_cfg['epochs_ant']} Loss={total / len(train_loader.dataset):.4f} "
            f"mix={mix_ratio:.3f} ZeroLeak_VAL_MAE={zero_leak_val:.4f} lr={lr_now:.6f}"
        )

        if zero_leak_val < best_zero_leak_val:
            best_zero_leak_val = zero_leak_val
            best_epoch = epoch
            best_state = {
                "encoder_ant": copy.deepcopy((ema_enc_ant.shadow if ema_enc_ant is not None else encoder_ant.state_dict())),
                "bridge_nl": copy.deepcopy((ema_bridge.shadow if ema_bridge is not None else bridge_nl.state_dict())),
                "student_parallel": copy.deepcopy((ema_student.shadow if ema_student is not None else student_parallel.state_dict())),
            }

    if best_state is not None:
        encoder_ant.load_state_dict(best_state["encoder_ant"])
        bridge_nl.load_state_dict(best_state["bridge_nl"])
        student_parallel.load_state_dict(best_state["student_parallel"])

    print(f"[Stage2-Student] Best zero-leak VAL epoch = {best_epoch}, MAE = {best_zero_leak_val:.4f}")
