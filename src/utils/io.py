from __future__ import annotations

import os
from typing import Any, Dict

import pandas as pd
import yaml


DATE_COL_CANDIDATES = [
    "date", "Date", "DATE", "time", "Time", "system:time_start", "system_time_start"
]


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def detect_date_col(columns) -> str:
    for c in DATE_COL_CANDIDATES:
        if c in columns:
            return c
    raise ValueError(f"No valid date column found. Candidates: {DATE_COL_CANDIDATES}")


def parse_dates_inplace(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    try:
        df[date_col] = pd.to_datetime(df[date_col])
    except Exception:
        df[date_col] = pd.to_datetime(df[date_col], unit="ms", errors="coerce")
    return df


def read_any(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext in [".csv", ".txt"]:
        return pd.read_csv(path, low_memory=False)
    if ext in [".xls", ".xlsx"]:
        return pd.read_excel(path)
    raise ValueError(f"Unsupported file type: {path}")
