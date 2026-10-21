"""Monthly county rainfall and vegetation indicators from the HDX dekadal (10-day) files.

The HDX files report old district units (adm_level 2) whose PCODE starts with today's
county code (e.g. KE009041 sits in Mandera, KE009). Districts are aggregated to counties,
weighted by satellite pixel count, and anomalies are recomputed from the aggregated totals
against the long-term average (1989-2018 for CHIRPS).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import RAW_DIR, settings


def _read(key: str) -> pd.DataFrame:
    df = pd.read_csv(RAW_DIR / settings()["sources"][key]["file"], dtype={"PCODE": str, "adm_level": str})
    df = df[~df["date"].astype(str).str.startswith("#")].copy()
    df["date"] = pd.to_datetime(df["date"])
    for c in df.columns.difference(["date", "adm_level", "PCODE", "version"]):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df[df["adm_level"] == "2"].copy()
    df["county_pcode"] = df["PCODE"].str[:5]
    if "version" in df:  # keep final over preliminary values for the same dekad
        df["_rank"] = df["version"].map({"final": 0}).fillna(1)
        df = df.sort_values("_rank").drop_duplicates(["PCODE", "date"]).drop(columns="_rank")
    return df


def _weighted(df: pd.DataFrame, cols: list[str], keys: list[str]) -> pd.DataFrame:
    w = df["n_pixels"].fillna(1.0)
    tmp = df[keys].copy()
    for c in cols:
        tmp[c + "_w"] = df[c] * w
        tmp[c + "_n"] = w.where(df[c].notna(), 0.0)
    g = tmp.groupby(keys).sum(numeric_only=True)
    out = pd.DataFrame({c: g[c + "_w"] / g[c + "_n"].replace(0, np.nan) for c in cols})
    return out.reset_index()


def rainfall_monthly() -> pd.DataFrame:
    df = _read("rainfall")
    df["month"] = df["date"].dt.to_period("M").dt.to_timestamp()
    # Rolling 1- and 3-month totals are reported at every dekad; the month's last available
    # dekad closes the month (the 21st, or an earlier dekad while the month is in progress).
    last = df.groupby(["PCODE", "month"])["date"].transform("max")
    df = df[df["date"] == last]
    df["dekads_in_month"] = df["date"].dt.day.map({1: 1, 11: 2, 21: 3})
    cty = _weighted(df, ["r1h", "r1h_avg", "r3h", "r3h_avg"], ["county_pcode", "month"])
    complete = df.groupby(["county_pcode", "month"])["dekads_in_month"].min().rename("rain_dekads").reset_index()
    cty = cty.merge(complete, on=["county_pcode", "month"])
    cty["rain_1m_mm"] = cty["r1h"]
    cty["rain_1m_pct"] = 100 * cty["r1h"] / cty["r1h_avg"]
    cty["rain_3m_pct"] = 100 * cty["r3h"] / cty["r3h_avg"]
    return cty[["county_pcode", "month", "rain_1m_mm", "rain_1m_pct", "rain_3m_pct", "rain_dekads"]]


def ndvi_monthly() -> pd.DataFrame:
    df = _read("ndvi")
    df["month"] = df["date"].dt.to_period("M").dt.to_timestamp()
    cty = _weighted(df, ["vim", "vim_avg"], ["county_pcode", "month"])
    cty["ndvi"] = cty["vim"]
    cty["ndvi_pct"] = 100 * cty["vim"] / cty["vim_avg"]
    return cty[["county_pcode", "month", "ndvi", "ndvi_pct"]]


def climate_monthly() -> pd.DataFrame:
    rain = rainfall_monthly()
    ndvi = ndvi_monthly()
    out = rain.merge(ndvi, on=["county_pcode", "month"], how="outer").sort_values(["county_pcode", "month"])
    # Extreme ratios happen when the long-term average is close to zero in the dry season.
    for c in ("rain_1m_pct", "rain_3m_pct"):
        out[c] = out[c].clip(0, 400)
    return out.reset_index(drop=True)
