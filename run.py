from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from src.data.dataset import PhenologyDataset, collate_fn, fit_scaler
from src.data.preprocessing import (
    build_data,
    build_train_sample_weights,
    resolve_stage2_train_years,
    resolve_train_years,
)
from src.engine.evaluate import evaluate_zero_leak, export_predictions, fit_ant_scaler
from src.engine.stage1 import evaluate_calendar_detail, train_calendar_head
from src.engine.stage2 import train_student_ant, train_teacher_ant
from src.models.modules import (
    NonLinearActivation,
    SeasonLengthHead,
    SowingHead,
    StudentParallelDecoder,
    TeacherARDecoder,
    TemporalEncoder,
)
from src.utils.io import load_config
from src.utils.reproducibility import get_device, set_seed


def summarize_year_counts(meta_array, tag: str):
    if len(meta_array) == 0:
        print(f"[{tag}] Year distribution: {{}}")
        return
    years = [pd.to_datetime(m[2]).year for m in meta_array]
    uniq, cnt = np.unique(years, return_counts=True)
    print(f"[{tag}] Year distribution:", dict(sorted(zip(uniq.tolist(), cnt.tolist()))))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    device = get_device()
    print("DEVICE =", device)

    os.makedirs(cfg["data"]["output_dir"], exist_ok=True)

    x_cal, y, meta, series, _stage_cols, input_feats_cal, input_feats_ant = build_data(cfg)
    if len(x_cal) == 0:
        raise RuntimeError("No valid sample was built. Check labels, ERA tables, VI tables, and coordinate/date matching.")

    years_all = np.array([pd.to_datetime(y0).year for (_, _, y0, _, _) in meta])
    uniq = sorted(np.unique(years_all).tolist())
    if len(uniq) >= 2:
        test_years = [uniq[-1]]
        val_years = [uniq[-2]]
    else:
        test_years = [uniq[-1]]
        val_years = []

    train_years_stage1 = resolve_train_years(uniq, val_years, test_years, cfg)
    train_years_stage2 = resolve_stage2_train_years(train_years_stage1, cfg)
    print(f"[split] TRAIN_STAGE1={train_years_stage1} | TRAIN_STAGE2={train_years_stage2} | VAL={val_years} | TEST={test_years}")

    def subset(year_list):
        idx = [i for i, yy in enumerate(years_all) if yy in year_list]
        return x_cal[idx], y[idx], meta[idx], [series[i] for i in idx]

    x_tr_cal, y_tr_cal, meta_tr_cal, ser_tr_cal = subset(train_years_stage1)
    x_tr_ant, y_tr_ant, meta_tr_ant, ser_tr_ant = subset(train_years_stage2)
    x_val, y_val, meta_val, ser_val = subset(val_years)
    x_test, y_test, meta_test, ser_test = subset(test_years)

    summarize_year_counts(meta_tr_cal, "train_stage1")
    summarize_year_counts(meta_tr_ant, "train_stage2")
    summarize_year_counts(meta_val, "val")
    summarize_year_counts(meta_test, "test")

    if len(x_tr_cal) == 0:
        raise RuntimeError("Stage-1 training set is empty.")
    if len(x_tr_ant) == 0:
        raise RuntimeError("Stage-2 training set is empty.")

    # Adaptive ranges estimated from stage-1 training samples.
    d_train = np.array([m[3] for m in meta_tr_cal], dtype=float)
    l_train = np.array([m[4] for m in meta_tr_cal], dtype=float)
    d_train = d_train[~np.isnan(d_train)]
    l_train = l_train[~np.isnan(l_train)]
    if len(d_train) > 0 and len(l_train) > 0:
        d_min, d_max = np.percentile(d_train, [2, 98])
        l_min, l_max = np.percentile(l_train, [2, 98])
        cfg["ranges"]["sow_min"] = float(max(1, d_min - 3))
        cfg["ranges"]["sow_max"] = float(d_max + 3)
        cfg["ranges"]["slen_min"] = float(max(15, l_min - 5))
        cfg["ranges"]["slen_max"] = float(l_max + 5)
    print(
        f"[ranges] sow={cfg['ranges']['sow_min']:.1f}..{cfg['ranges']['sow_max']:.1f} | "
        f"season_len={cfg['ranges']['slen_min']:.1f}..{cfg['ranges']['slen_max']:.1f}"
    )

    scaler_cal = fit_scaler(x_tr_cal)
    scaler_ant = fit_ant_scaler(ser_tr_ant, meta_tr_ant, cfg)

    ds_tr_cal = PhenologyDataset(x_tr_cal, y_tr_cal, meta_tr_cal, scaler=scaler_cal)
    ds_tr_ant = PhenologyDataset(x_tr_ant, y_tr_ant, meta_tr_ant, scaler=scaler_cal)
    ds_val = PhenologyDataset(x_val, y_val, meta_val, scaler=scaler_cal)
    ds_test = PhenologyDataset(x_test, y_test, meta_test, scaler=scaler_cal)

    pin_mem = torch.cuda.is_available()
    batch_size = cfg["training"]["batch_size"]
    split_mode = cfg["split"]["train_year_mode"]

    if split_mode == "downweight":
        weights = build_train_sample_weights(meta_tr_cal, cfg)
        sampler = WeightedRandomSampler(
            weights=torch.as_tensor(weights, dtype=torch.double),
            num_samples=max(1, int(round(len(weights) * cfg["split"]["max_sampler_multiplier"]))),
            replacement=True,
        )
        ld_tr_cal = DataLoader(ds_tr_cal, batch_size=batch_size, sampler=sampler, shuffle=False, collate_fn=collate_fn, pin_memory=pin_mem)
    else:
        ld_tr_cal = DataLoader(ds_tr_cal, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, pin_memory=pin_mem)

    ld_tr_ant = DataLoader(ds_tr_ant, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, pin_memory=pin_mem)
    ld_val = DataLoader(ds_val, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, pin_memory=pin_mem)
    ld_test = DataLoader(ds_test, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, pin_memory=pin_mem)

    enc_cal = TemporalEncoder(in_dim=len(input_feats_cal), d_model=128, nhead=4, num_layers=2, dropout=0.15)
    sow_head = SowingHead(enc_dim=enc_cal.out_dim, hidden=256, sow_min=cfg["ranges"]["sow_min"], sow_max=cfg["ranges"]["sow_max"])
    slen_head = SeasonLengthHead(enc_dim=enc_cal.out_dim, hidden=256, sl_min=cfg["ranges"]["slen_min"], sl_max=cfg["ranges"]["slen_max"])

    enc_ant = TemporalEncoder(in_dim=len(input_feats_ant), d_model=128, nhead=4, num_layers=2, dropout=0.15)
    teacher_ar = TeacherARDecoder(enc_dim=enc_ant.out_dim, hidden=256, n_stages=len(cfg["stages"]["names"]))
    bridge_nl = NonLinearActivation(n_stages=len(cfg["stages"]["names"]), hidden=64)
    student_parallel = StudentParallelDecoder(enc_dim=enc_ant.out_dim, hidden=256, n_stages=len(cfg["stages"]["names"]))

    print("\n== Stage 1: Calendar -> sowing date / season length ==")
    train_calendar_head(enc_cal, sow_head, slen_head, ld_tr_cal, ld_val, device, cfg)
    val_cal = evaluate_calendar_detail(enc_cal, sow_head, slen_head, ld_val, device, cfg)
    test_cal = evaluate_calendar_detail(enc_cal, sow_head, slen_head, ld_test, device, cfg)
    print(f"[Stage1-VAL] loss={val_cal['loss']:.4f}, MAE(d)={val_cal['mae_d']:.3f}, MAE(L)={val_cal['mae_l']:.3f}")
    print(f"[Stage1-TEST] loss={test_cal['loss']:.4f}, MAE(d)={test_cal['mae_d']:.3f}, MAE(L)={test_cal['mae_l']:.3f}")

    print("\n== Stage 2A: Train teacher decoder ==")
    train_teacher_ant(enc_ant, teacher_ar, ser_tr_ant, ld_tr_ant, device, cfg, scaler_ant)

    print("\n== Stage 2B: Train student decoder ==")
    train_student_ant(
        enc_cal,
        sow_head,
        slen_head,
        enc_ant,
        teacher_ar,
        bridge_nl,
        student_parallel,
        ser_tr_ant,
        ser_val,
        ld_tr_ant,
        ld_val,
        device,
        cfg,
        scaler_ant,
    )

    print("\n== Validation (zero leak) ==")
    evaluate_zero_leak(enc_cal, sow_head, slen_head, enc_ant, student_parallel, ser_val, ld_val, device, cfg, scaler_ant, desc="VAL")

    print("\n== Test (zero leak) ==")
    evaluate_zero_leak(enc_cal, sow_head, slen_head, enc_ant, student_parallel, ser_test, ld_test, device, cfg, scaler_ant, desc="TEST")

    out_csv = os.path.join(
        cfg["data"]["output_dir"],
        f"predictions_CAL{cfg['windows']['cal_k']}_ANT{cfg['windows']['ant_k']}_public_framework.csv",
    )
    export_predictions(
        enc_cal,
        sow_head,
        slen_head,
        enc_ant,
        student_parallel,
        ser_test,
        ds_test,
        meta_test,
        out_csv,
        device,
        cfg,
        scaler_ant,
    )


if __name__ == "__main__":
    main()
