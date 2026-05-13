from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


def doy_to_unit_by_true(y_doy: torch.Tensor):
    y = y_doy.clone()
    mask = ~torch.isnan(y)
    if torch.isnan(y[0]):
        sow = torch.tensor(200.0, device=y.device)
        length = torch.tensor(120.0, device=y.device)
    else:
        sow = y[0]
        y_filled = torch.where(mask, y, torch.full_like(y, -1e9))
        max_doy = torch.max(y_filled)
        if max_doy < sow:
            max_doy = sow + 1.0
        length = torch.clamp(max_doy - sow, min=1.0)
    y_unit = (y - sow) / length
    y_unit = torch.nan_to_num(y_unit, nan=0.0)
    return y_unit, mask.float(), torch.stack([sow, length])


class PhenologyDataset(Dataset):
    def __init__(self, x_cal, y, meta, scaler=None):
        self.x_cal = x_cal
        self.y = y
        self.meta = meta
        self.scaler = scaler

    def __len__(self):
        return len(self.x_cal)

    def __getitem__(self, idx):
        x = self.x_cal[idx]
        if self.scaler is not None and self.scaler["mean"] is not None:
            x = (x - self.scaler["mean"]) / self.scaler["std"]
        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(self.y[idx], dtype=torch.float32)
        y_unit, mask, meta_true = doy_to_unit_by_true(y)
        return x, y_unit, mask, meta_true, idx


def collate_fn(batch):
    x, y_unit, mask, meta_true, idxs = zip(*batch)
    return (
        torch.stack(x, 0),
        torch.stack(y_unit, 0),
        torch.stack(mask, 0),
        torch.stack(meta_true, 0),
        torch.tensor(idxs, dtype=torch.long),
    )


def fit_scaler(x_train: np.ndarray):
    x_flat = np.vstack([x for x in x_train])
    feat_mean = np.nanmean(x_flat, axis=0)
    feat_std = np.nanstd(x_flat, axis=0) + 1e-6
    feat_mean = np.where(np.isfinite(feat_mean), feat_mean, 0.0)
    feat_std = np.where(np.isfinite(feat_std) & (feat_std > 0), feat_std, 1.0)
    return {"mean": feat_mean, "std": feat_std}
