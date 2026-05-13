from __future__ import annotations

import math
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.data.preprocessing import build_ant_matrix_for_sample
from src.data.dataset import collate_fn


def fit_ant_scaler(series_list, meta_array, config: Dict):
    mats_all = []
    for i in range(len(series_list)):
        ss = series_list[i]
        d_true = float(meta_array[i][3])
        l_true = float(meta_array[i][4])
        mats_all.append(build_ant_matrix_for_sample(ss, d_true, l_true, config))
    arr = np.vstack(mats_all).astype(float)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    feat_mean = np.nanmean(arr, axis=0)
    feat_std = np.nanstd(arr, axis=0)
    feat_mean = np.where(np.isfinite(feat_mean), feat_mean, 0.0)
    feat_std = np.where(np.isfinite(feat_std) & (feat_std > 0), feat_std, 1.0)
    return {"mean": feat_mean, "std": feat_std}


def ant_aggregate_batch(series_list, idxs: torch.Tensor, d_used: np.ndarray, l_used: np.ndarray, config: Dict, scaler_ant: Dict) -> np.ndarray:
    x_ant = []
    for ii, idx in enumerate(idxs.cpu().numpy().tolist()):
        ss = series_list[idx]
        xi = build_ant_matrix_for_sample(ss, float(d_used[ii]), float(l_used[ii]), config)
        if scaler_ant is not None and scaler_ant["mean"] is not None:
            xi = (xi - scaler_ant["mean"]) / scaler_ant["std"]
            xi = np.nan_to_num(xi, nan=0.0, posinf=0.0, neginf=0.0)
        x_ant.append(xi)
    return np.nan_to_num(np.stack(x_ant, axis=0), nan=0.0, posinf=0.0, neginf=0.0)


@torch.no_grad()
def compute_zero_leak_val_mae(
    encoder_cal,
    sow_head,
    slen_head,
    encoder_ant,
    student_parallel,
    series_list,
    loader,
    device: str,
    config: Dict,
    scaler_ant: Dict,
) -> float:
    if len(loader.dataset) == 0:
        return float("inf")

    ranges = config["ranges"]
    encoder_cal.eval(); sow_head.eval(); slen_head.eval(); encoder_ant.eval(); student_parallel.eval()
    total_abs_err, total_cnt = 0.0, 0.0

    for x_cal, y_unit, mask, meta_true, idxs in loader:
        x_cal = x_cal.to(device)
        y_unit = y_unit.to(device)
        mask = mask.to(device)
        meta_true = meta_true.to(device)

        e_cal = encoder_cal(x_cal)
        d_pred = sow_head(e_cal)
        l_pred = slen_head(e_cal)
        d_pred_np = np.clip(d_pred.squeeze(-1).cpu().numpy(), ranges["sow_min"], ranges["sow_max"])
        l_pred_np = np.clip(l_pred.squeeze(-1).cpu().numpy(), ranges["slen_min"], ranges["slen_max"])

        x_ant_np = ant_aggregate_batch(series_list, idxs, d_pred_np, l_pred_np, config, scaler_ant)
        x_ant = torch.tensor(x_ant_np, dtype=torch.float32, device=device)
        e_ant = encoder_ant(x_ant)
        y_student = student_parallel(e_ant)

        d_true = meta_true[:, 0:1]
        l_true = meta_true[:, 1:2]
        y_pred = y_student * l_pred + d_pred
        y_true = y_unit * l_true + d_true

        abs_err = torch.abs(y_pred - y_true) * mask
        total_abs_err += abs_err.sum().item()
        total_cnt += mask.sum().item()

    return total_abs_err / max(total_cnt, 1.0)


@torch.no_grad()
def evaluate_zero_leak(
    encoder_cal,
    sow_head,
    slen_head,
    encoder_ant,
    student_parallel,
    series_list,
    loader,
    device: str,
    config: Dict,
    scaler_ant: Dict,
    desc: str = "Eval",
):
    if len(loader.dataset) == 0:
        print(f"[{desc}] Empty dataset. Skipped.")
        return

    stage_names = config["stages"]["names"]
    ranges = config["ranges"]
    n_stages = len(stage_names)

    encoder_cal.eval(); sow_head.eval(); slen_head.eval(); encoder_ant.eval(); student_parallel.eval()
    total_abs_err, total_sq_err, total_cnt = 0.0, 0.0, 0.0
    stage_abs_err = np.zeros(n_stages)
    stage_sq_err = np.zeros(n_stages)
    stage_cnt = np.zeros(n_stages)

    for x_cal, y_unit, mask, meta_true, idxs in loader:
        x_cal = x_cal.to(device)
        y_unit = y_unit.to(device)
        mask = mask.to(device)
        meta_true = meta_true.to(device)

        e_cal = encoder_cal(x_cal)
        d_pred = sow_head(e_cal)
        l_pred = slen_head(e_cal)
        d_pred_np = np.clip(d_pred.squeeze(-1).cpu().numpy(), ranges["sow_min"], ranges["sow_max"])
        l_pred_np = np.clip(l_pred.squeeze(-1).cpu().numpy(), ranges["slen_min"], ranges["slen_max"])

        x_ant_np = ant_aggregate_batch(series_list, idxs, d_pred_np, l_pred_np, config, scaler_ant)
        x_ant = torch.tensor(x_ant_np, dtype=torch.float32, device=device)
        e_ant = encoder_ant(x_ant)
        y_student = student_parallel(e_ant)

        d_true = meta_true[:, 0:1]
        l_true = meta_true[:, 1:2]
        y_pred = y_student * l_pred + d_pred
        y_true = y_unit * l_true + d_true

        abs_err = torch.abs(y_pred - y_true) * mask
        sq_err = ((y_pred - y_true) ** 2) * mask
        total_abs_err += abs_err.sum().item()
        total_sq_err += sq_err.sum().item()
        total_cnt += mask.sum().item()

        for s in range(n_stages):
            stage_abs_err[s] += abs_err[:, s].sum().item()
            stage_sq_err[s] += sq_err[:, s].sum().item()
            stage_cnt[s] += mask[:, s].sum().item()

    overall_mae = total_abs_err / max(total_cnt, 1.0)
    overall_rmse = math.sqrt(total_sq_err / max(total_cnt, 1.0))
    print(f"[{desc}] Overall -> MAE={overall_mae:.2f} d, RMSE={overall_rmse:.2f} d")
    for i, name in enumerate(stage_names):
        mae = stage_abs_err[i] / stage_cnt[i] if stage_cnt[i] > 0 else float("nan")
        rmse = math.sqrt(stage_sq_err[i] / stage_cnt[i]) if stage_cnt[i] > 0 else float("nan")
        print(f"  - {name}: MAE={mae:.2f} d, RMSE={rmse:.2f} d")


@torch.no_grad()
def export_predictions(
    encoder_cal,
    sow_head,
    slen_head,
    encoder_ant,
    student_parallel,
    series_list,
    dataset,
    meta,
    out_csv: str,
    device: str,
    config: Dict,
    scaler_ant: Dict,
):
    if len(dataset) == 0:
        print("[Export] Empty dataset. Skipped.")
        return

    stage_cols = [f"{s}_DOY" for s in config["stages"]["names"]]
    ranges = config["ranges"]
    loader = DataLoader(dataset, batch_size=256, shuffle=False, collate_fn=collate_fn)
    rows = []

    encoder_cal.eval(); sow_head.eval(); slen_head.eval(); encoder_ant.eval(); student_parallel.eval()

    for x_cal, y_unit, mask, meta_true, idxs in loader:
        x_cal = x_cal.to(device)
        y_unit = y_unit.to(device)
        mask = mask.to(device)
        meta_true = meta_true.to(device)

        e_cal = encoder_cal(x_cal)
        d_pred = sow_head(e_cal)
        l_pred = slen_head(e_cal)
        d_pred_np = np.clip(d_pred.squeeze(-1).cpu().numpy(), ranges["sow_min"], ranges["sow_max"])
        l_pred_np = np.clip(l_pred.squeeze(-1).cpu().numpy(), ranges["slen_min"], ranges["slen_max"])

        x_ant_np = ant_aggregate_batch(series_list, idxs, d_pred_np, l_pred_np, config, scaler_ant)
        x_ant = torch.tensor(x_ant_np, dtype=torch.float32, device=device)
        e_ant = encoder_ant(x_ant)
        y_student = student_parallel(e_ant)

        d_true = meta_true[:, 0:1]
        l_true = meta_true[:, 1:2]
        y_pred = y_student * l_pred + d_pred
        y_true = y_unit * l_true + d_true

        mask_np = mask.cpu().numpy().astype(bool)
        y_pred_np = y_pred.cpu().numpy()
        y_true_np = y_true.cpu().numpy()
        d_true_np = d_true.squeeze(-1).cpu().numpy()
        l_true_np = l_true.squeeze(-1).cpu().numpy()

        for i in range(y_pred_np.shape[0]):
            gidx = idxs[i].item()
            lat, lon, year_dt, _, _ = meta[gidx]
            row = {
                "Lat": float(lat),
                "Lon": float(lon),
                "Year": pd.to_datetime(year_dt).date(),
                "d_true": float(d_true_np[i]) if d_true_np[i] == d_true_np[i] else np.nan,
                "l_true": float(l_true_np[i]) if l_true_np[i] == l_true_np[i] else np.nan,
                "d_pred": float(d_pred_np[i]),
                "l_pred": float(l_pred_np[i]),
            }
            for k, col in enumerate(stage_cols):
                row[f"pred_{col}"] = float(y_pred_np[i, k])
            for k, col in enumerate(stage_cols):
                row[f"true_{col}"] = float(y_true_np[i, k]) if mask_np[i, k] else np.nan
            rows.append(row)

    pd.DataFrame(rows).to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"[Export] Saved predictions to: {out_csv}")
