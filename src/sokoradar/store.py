"""The forecast store: precomputed tables every channel (API, dashboard, USSD) reads from."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import pandas as pd

from .config import STORE_DIR


def path(name: str) -> Path:
    return STORE_DIR / name


def write_df(name: str, df: pd.DataFrame) -> Path:
    STORE_DIR.mkdir(parents=True, exist_ok=True)
    p = path(f"{name}.parquet")
    df.to_parquet(p, index=False)
    return p


def read_df(name: str) -> pd.DataFrame:
    return pd.read_parquet(path(f"{name}.parquet"))


def exists(name: str) -> bool:
    return path(f"{name}.parquet").exists() or path(name).exists()


def write_json(name: str, obj) -> Path:
    STORE_DIR.mkdir(parents=True, exist_ok=True)
    p = path(f"{name}.json")
    p.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")
    return p


def read_json(name: str):
    return json.loads(path(f"{name}.json").read_text(encoding="utf-8"))


@lru_cache(maxsize=32)
def cached_df(name: str, mtime: float) -> pd.DataFrame:
    return read_df(name)


def load(name: str) -> pd.DataFrame:
    """Read a table, re-reading only when the file changes (for long-running servers)."""
    p = path(f"{name}.parquet")
    return cached_df(name, p.stat().st_mtime)
