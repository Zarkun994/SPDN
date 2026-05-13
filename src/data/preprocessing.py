from __future__ import annotations

import os
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from glob import glob
from typing import Dict, List, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd

from src.utils.io import detect_date_col, parse_dates_inplace, read_any


@dataclass
class SampleSeries:
    df: pd.DataFrame
    year: int
    sow_true: float
    slen_true: float
    lat: float
    lon: float


def calc_photoperiod_hours(lat_deg, doy):
    doy = np.asarray(doy, dtype=float)
    lat_rad = np.deg2rad(lat_deg)
    decl = 0.409 * np.sin(2.0 * np.pi * doy / 365.0 - 1.39)
    x = -np.tan(lat_rad) * np.tan(decl)
    x = np.clip(x, -1.0, 1.0)
    ws = np.arccos(x)
    return 24.0 / np.pi * ws


def add_photoperiod(df_point: pd.DataFrame, lat_deg: float) -> pd.DataFrame:
    df = df_point.copy()
    doy = df["date"].dt.dayofyear.values
    df["Photoperiod"] = calc_photoperiod_hours(lat_deg, doy)
    return df


def add_cumgdd_since_sow(df_point: pd.DataFrame, year: int, sow_doy: float) -> pd.DataFrame:
    df = df_point.copy().sort_values("date").reset_index(drop=True)
    if "GDD_sum" not in df.columns:
        df["cumGDD_since_sow"] = 0.0
        return df

    base = datetime(year, 1, 1)
    sow_dt = base + timedelta(days=int(round(float(sow_doy))) - 1)
    gdd_daily = pd.to_numeric(df["GDD_sum"], errors="coerce").fillna(0.0).values
    mask_after = (df["date"] >= sow_dt).values

    cum = np.zeros(len(df), dtype=float)
    running = 0.0
    for i in range(len(df)):
        if mask_after[i]:
            running += max(float(gdd_daily[i]), 0.0)
            cum[i] = running
    df["cumGDD_since_sow"] = cum
    return df


def make_calendar_bins(year: int, base_month: int, base_day: int, k: int) -> List[Tuple[datetime, datetime]]:
    start = datetime(year - 1, base_month, base_day)
    end = datetime(year, 7, 31)
    total_days = (end - start).days + 1
    edges = [start + timedelta(days=int(round(i * total_days / k))) for i in range(k + 1)]
    return [(edges[i], edges[i + 1] - timedelta(seconds=1)) for i in range(k)]


def make_ant_bins(sow_doy: float, slen: float, year: int, k: int) -> List[Tuple[datetime, datetime]]:
    base = datetime(year, 1, 1)
    sow_dt = base + timedelta(days=int(round(float(sow_doy))) - 1)
    slen = max(float(slen), 1.0)
    edges = [sow_dt + timedelta(days=int(round(u * slen))) for u in np.linspace(0, 1, k + 1)]
    return [(edges[i], edges[i + 1] - timedelta(seconds=1)) for i in range(k)]


def _safe_first_last(x: pd.Series) -> Tuple[float, float]:
    d = x.dropna()
    if d.empty:
        return np.nan, np.nan
    return d.iloc[0], d.iloc[-1]


def aggregate_slice(
    df_point: pd.DataFrame,
    start_dt: datetime,
    end_dt: datetime,
    vi_cols: List[str],
) -> Dict[str, float]:
    sl = df_point[(df_point["date"] >= start_dt) & (df_point["date"] <= end_dt)]
    out: Dict[str, float] = {}
    n_days = max(int((end_dt - start_dt).days) + 1, 1)

    out["ET_mm"] = sl["ET_mm"].sum(skipna=True) if "ET_mm" in sl.columns else np.nan
    out["GDD_sum"] = sl["GDD_sum"].sum(skipna=True) if "GDD_sum" in sl.columns else np.nan
    out["Rad_MJ"] = sl["Rad_MJ"].sum(skipna=True) if "Rad_MJ" in sl.columns else np.nan
    out["soil_water"] = sl["soil_water"].mean(skipna=True) if "soil_water" in sl.columns else np.nan

    for c in vi_cols:
        if c in sl.columns:
            s = pd.to_numeric(sl[c], errors="coerce")
            v_mean = s.mean(skipna=True)
            v_max = s.max(skipna=True)
            v_min = s.min(skipna=True)
            v_range = (v_max - v_min) if (pd.notna(v_max) and pd.notna(v_min)) else np.nan
            v_sum = s.sum(skipna=True)
            v_std = s.std(skipna=True)
            v_first, v_last = _safe_first_last(s)
            v_slope = (v_last - v_first) / n_days if (pd.notna(v_first) and pd.notna(v_last)) else np.nan
            out[f"{c}_mean"] = v_mean
            out[f"{c}_max"] = v_max
            out[f"{c}_range"] = v_range
            out[f"{c}_sum"] = v_sum
            out[f"{c}_slope"] = v_slope
            out[f"{c}_std"] = v_std
        else:
            for suffix in ["mean", "max", "range", "sum", "slope", "std"]:
                out[f"{c}_{suffix}"] = np.nan

    if "Photoperiod" in sl.columns:
        p = pd.to_numeric(sl["Photoperiod"], errors="coerce")
        out["Photo_mean"] = p.mean(skipna=True)
        p_first, p_last = _safe_first_last(p)
        out["Photo_slope"] = (p_last - p_first) / n_days if (pd.notna(p_first) and pd.notna(p_last)) else np.nan
    else:
        out["Photo_mean"] = np.nan
        out["Photo_slope"] = np.nan

    if "cumGDD_since_sow" in sl.columns:
        out["cumGDD_since_sow_end"] = sl["cumGDD_since_sow"].max(skipna=True)
    else:
        out["cumGDD_since_sow_end"] = np.nan

    return out


def ffill_bins(mat: np.ndarray) -> np.ndarray:
    a = mat.copy()
    for j in range(a.shape[1]):
        last = np.nan
        for i in range(a.shape[0]):
            if np.isnan(a[i, j]):
                a[i, j] = last
            else:
                last = a[i, j]
    return np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)


def build_data(config: Dict):
    data_cfg = config["data"]
    stage_names = config["stages"]["names"]
    meteo_feats = config["features"]["meteo"]
    vi_cols = config["features"]["vi"]
    vi_stats = config["features"]["vi_stats"]
    photo_stats = config["features"]["photo_stats"]
    ant_extra_stats = config["features"]["ant_extra_stats"]
    win_cfg = config["windows"]

    stage_cols = [f"{s}_DOY" for s in stage_names]
    input_feats_cal = meteo_feats + [f"{c}_{s}" for c in vi_cols for s in vi_stats] + photo_stats
    input_feats_ant = input_feats_cal + ant_extra_stats

    df_label_list = []
    point_id_to_loc = {}

    for item in data_cfg["label_configs"]:
        df_lbl = pd.read_excel(item["label_xlsx"])
        df_lbl.columns = df_lbl.columns.str.strip()
        df_label_list.append(df_lbl)

        shp_path = item.get("point_shp")
        if shp_path and os.path.exists(shp_path):
            gdf = gpd.read_file(shp_path)
            id_col = "point_id" if "point_id" in gdf.columns else gdf.columns[0]
            for _, row in gdf.iterrows():
                if row.geometry is not None:
                    point_id_to_loc[str(row[id_col])] = (round(float(row.geometry.y), 6), round(float(row.geometry.x), 6))

    df_label = pd.concat(df_label_list, ignore_index=True)
    df_label = df_label.loc[(~df_label[stage_cols[0]].isna())].reset_index(drop=True)
    df_label["Lat"] = df_label["Lat"].astype(float).round(6)
    df_label["Lon"] = df_label["Lon"].astype(float).round(6)
    loc_keys = set(zip(df_label["Lat"], df_label["Lon"]))

    era_files = sorted(glob(os.path.join(data_cfg["era_dir"], data_cfg["era_file_glob"])))
    vi_files = sorted(glob(os.path.join(data_cfg["vi_dir"], data_cfg["vi_file_glob"])))

    store_meteo = defaultdict(list)
    date_col_detected = None
    for f in era_files:
        df = pd.read_csv(f, low_memory=False)
        if date_col_detected is None:
            date_col_detected = detect_date_col(df.columns)
        cols_keep = [c for c in ([date_col_detected, "Lat", "Lon"] + meteo_feats) if c in df.columns]
        df = parse_dates_inplace(df[cols_keep].copy(), date_col_detected)
        for col in ("Lat", "Lon"):
            if col not in df.columns and col.lower() in df.columns:
                df.rename(columns={col.lower(): col}, inplace=True)
        df["Lat"] = df["Lat"].astype(float).round(6)
        df["Lon"] = df["Lon"].astype(float).round(6)
        df = df[df[["Lat", "Lon"]].apply(tuple, axis=1).isin(loc_keys)]
        if df.empty:
            continue
        for c in meteo_feats:
            if c not in df.columns:
                df[c] = np.nan
        for (lat, lon), g in df.groupby(["Lat", "Lon"], sort=False):
            store_meteo[(lat, lon)].append(g[[date_col_detected, *meteo_feats]].copy())

    for key in list(store_meteo.keys()):
        d = pd.concat(store_meteo[key], ignore_index=True)
        d = d.dropna(subset=[date_col_detected]).sort_values(by=date_col_detected)
        d = d.drop_duplicates(subset=[date_col_detected], keep="last").reset_index(drop=True)
        d.rename(columns={date_col_detected: "date"}, inplace=True)
        store_meteo[key] = d

    store_vi = defaultdict(list)
    date_col_vi_detected = None
    for f in vi_files:
        df = read_any(f)
        if date_col_vi_detected is None:
            date_col_vi_detected = detect_date_col(df.columns)
        df.rename(columns={"NDVI_interp": "NDVI", "NDWI_interp": "NDWI"}, inplace=True)
        if "Lat" not in df.columns and "Lon" not in df.columns and "point_id" in df.columns:
            df["Lat"] = df["point_id"].astype(str).map(lambda x: point_id_to_loc.get(x, (np.nan, np.nan))[0])
            df["Lon"] = df["point_id"].astype(str).map(lambda x: point_id_to_loc.get(x, (np.nan, np.nan))[1])
            df.dropna(subset=["Lat", "Lon"], inplace=True)
        else:
            for col in ("Lat", "Lon"):
                if col not in df.columns and col.lower() in df.columns:
                    df.rename(columns={col.lower(): col}, inplace=True)

        cols_keep = [c for c in ([date_col_vi_detected, "Lat", "Lon"] + vi_cols) if c in df.columns]
        df = parse_dates_inplace(df[cols_keep].copy(), date_col_vi_detected)
        df["Lat"] = df["Lat"].astype(float).round(6)
        df["Lon"] = df["Lon"].astype(float).round(6)
        df = df[df[["Lat", "Lon"]].apply(tuple, axis=1).isin(loc_keys)]
        if df.empty:
            continue
        df.rename(columns={date_col_vi_detected: "date"}, inplace=True)
        for (lat, lon), g in df.groupby(["Lat", "Lon"], sort=False):
            store_vi[(lat, lon)].append(g[["date", *vi_cols]].copy())

    store_merged = {}
    union_keys = set(store_meteo.keys()) | set(store_vi.keys())
    for key in union_keys:
        df_m = store_meteo.get(key)
        df_v = store_vi.get(key)
        if df_m is not None and df_v is not None:
            df = pd.merge(df_m, pd.concat(df_v, ignore_index=True), on="date", how="outer")
        else:
            df = df_m if df_m is not None else (pd.concat(df_v, ignore_index=True) if df_v is not None else None)
        if df is None:
            continue
        df = df.dropna(subset=["date"]).sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
        df = add_photoperiod(df, key[0])
        store_merged[key] = df

    x_cal_list, y_list, meta_list, series_list = [], [], [], []
    for _, row in df_label.iterrows():
        lat, lon = float(row["Lat"]), float(row["Lon"])
        key = (round(lat, 6), round(lon, 6))
        if key not in store_merged:
            continue
        df_point = store_merged[key].copy()
        year = pd.to_datetime(row["Year"]).year

        d_true = float(row[stage_cols[0]]) if not pd.isna(row[stage_cols[0]]) else np.nan
        d_valid = [row[c] for c in stage_cols if not pd.isna(row[c])]
        l_true = (max(d_valid) - d_true) if (len(d_valid) > 0 and d_true == d_true) else np.nan

        bins_cal = make_calendar_bins(
            year=year,
            base_month=win_cfg["cal_base_month"],
            base_day=win_cfg["cal_base_day"],
            k=win_cfg["cal_k"],
        )
        feat_rows = []
        for st, ed in bins_cal:
            agg = aggregate_slice(df_point, st, ed, vi_cols=vi_cols)
            vec = [agg["ET_mm"], agg["GDD_sum"], agg["Rad_MJ"], agg["soil_water"]]
            for n in [f"{c}_{s}" for c in vi_cols for s in vi_stats]:
                vec.append(agg.get(n, np.nan))
            for n in photo_stats:
                vec.append(agg.get(n, np.nan))
            feat_rows.append(vec)
        x_cal_list.append(ffill_bins(np.array(feat_rows, dtype=float)))

        y_list.append([np.nan if pd.isna(row[c]) else float(row[c]) for c in stage_cols])
        keep_cols = ["date", *meteo_feats, *vi_cols, "Photoperiod"]
        series_df = df_point[[c for c in keep_cols if c in df_point.columns]].copy()
        series_list.append(SampleSeries(series_df, year, d_true, l_true, lat, lon))
        meta_list.append((lat, lon, pd.to_datetime(row["Year"]), d_true, l_true))

    return (
        np.array(x_cal_list, dtype=float),
        np.array(y_list, dtype=float),
        np.array(meta_list, dtype=object),
        series_list,
        np.array(stage_cols),
        input_feats_cal,
        input_feats_ant,
    )


def build_ant_matrix_for_sample(ss: SampleSeries, d_used: float, l_used: float, config: Dict) -> np.ndarray:
    vi_cols = config["features"]["vi"]
    vi_stats = config["features"]["vi_stats"]
    photo_stats = config["features"]["photo_stats"]
    ant_extra_stats = config["features"]["ant_extra_stats"]
    ant_k = config["windows"]["ant_k"]

    dfp = add_cumgdd_since_sow(ss.df, year=ss.year, sow_doy=float(d_used))
    bins = make_ant_bins(float(d_used), float(l_used), ss.year, k=ant_k)
    rows = []
    for st, ed in bins:
        agg = aggregate_slice(dfp, st, ed, vi_cols=vi_cols)
        vec = [agg["ET_mm"], agg["GDD_sum"], agg["Rad_MJ"], agg["soil_water"]]
        for n in [f"{c}_{s}" for c in vi_cols for s in vi_stats]:
            vec.append(agg.get(n, np.nan))
        for n in photo_stats:
            vec.append(agg.get(n, np.nan))
        for n in ant_extra_stats:
            vec.append(agg.get(n, np.nan))
        rows.append(vec)
    return ffill_bins(np.array(rows, dtype=float))


def resolve_train_years(sorted_years: List[int], val_years: List[int], test_years: List[int], config: Dict) -> List[int]:
    split_cfg = config["split"]
    mode = split_cfg["train_year_mode"]
    recent_start = split_cfg["recent_train_start_year"]

    holdout = set(val_years) | set(test_years)
    candidate_years = [y for y in sorted_years if y not in holdout]
    if mode == "recent_only":
        recent_years = [y for y in candidate_years if y >= recent_start]
        if len(recent_years) == 0:
            raise RuntimeError(f"No training year >= {recent_start} is available.")
        return recent_years
    if mode in {"downweight", "all"}:
        return candidate_years
    raise ValueError(f"Unknown train_year_mode: {mode}")


def resolve_stage2_train_years(stage1_train_years: List[int], config: Dict) -> List[int]:
    recent_start = config["split"]["recent_train_start_year"]
    years = [y for y in stage1_train_years if y >= recent_start]
    if len(years) == 0:
        raise RuntimeError(f"No stage-2 training year >= {recent_start} is available.")
    return years


def build_train_sample_weights(meta_array: np.ndarray, config: Dict) -> np.ndarray:
    split_cfg = config["split"]
    w_old_1 = split_cfg["old_year_weight_2009_2011"]
    w_old_2 = split_cfg["old_year_weight_2012_2014"]
    weights = []
    for m in meta_array:
        year = pd.to_datetime(m[2]).year
        if year <= 2011:
            w = w_old_1
        elif year <= 2014:
            w = w_old_2
        else:
            w = 1.0
        weights.append(w)
    weights = np.asarray(weights, dtype=float)
    return np.clip(weights, 1e-6, None)
