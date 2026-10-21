"""Macro drivers: KES exchange rates and global commodity prices (World Bank Pink Sheet)."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from .config import RAW_DIR, settings

PINK_SERIES = {
    "maize": "Maize",
    "wheat": "Wheat, US HRW",
    "crude_oil": "Crude oil, average",
    "urea": "Urea",
    "dap": "DAP",
}


def cbk_fx() -> pd.DataFrame:
    """Monthly mean CBK indicative rates (KES per USD, UGX and TZS per KES)."""
    path = RAW_DIR / settings()["sources"]["cbk_fx"]["file"]
    if not path.exists():
        return pd.DataFrame(columns=["month", "kes_usd", "ugx_kes", "tzs_kes"])
    df = pd.read_csv(path, header=None, names=["date", "currency", "mean", "buy", "sell"], dtype=str,
                     on_bad_lines="skip")
    df["currency"] = df["currency"].str.replace('"', "").str.replace(r"\s+", " ", regex=True).str.strip().str.upper()
    df["currency"] = df["currency"].str.replace(" / ", "/")
    iso = df["date"].str.contains("-", na=False)
    dates = pd.Series(pd.NaT, index=df.index)
    dates[iso] = pd.to_datetime(df.loc[iso, "date"], format="%Y-%m-%d", errors="coerce")
    dates[~iso] = pd.to_datetime(df.loc[~iso, "date"], format="%d/%m/%Y", errors="coerce")
    df["date"] = dates
    df["mean"] = pd.to_numeric(df["mean"], errors="coerce")
    df = df.dropna(subset=["date", "mean"]).drop_duplicates(["date", "currency"])
    df["month"] = df["date"].dt.to_period("M").dt.to_timestamp()
    wanted = {"US DOLLAR": "kes_usd", "KES/USHS": "ugx_kes", "KES/TSHS": "tzs_kes"}
    df = df[df["currency"].isin(wanted)]
    out = df.pivot_table(index="month", columns="currency", values="mean", aggfunc="mean").rename(columns=wanted)
    return out.reset_index()


def wfp_implied_fx(raw: pd.DataFrame) -> pd.DataFrame:
    """KES per USD implied by WFP's own price and USD price columns (large prices only, to avoid rounding)."""
    df = raw[(raw["usdprice"] >= 5) & (raw["price"] > 0) & (raw["currency"] == "KES")].copy()
    df["month"] = df["date"].dt.to_period("M").dt.to_timestamp()
    df["rate"] = df["price"] / df["usdprice"]
    return df.groupby("month")["rate"].median().rename("kes_usd_wfp").reset_index()


def fx_monthly(raw_wfp: pd.DataFrame) -> pd.DataFrame:
    cbk = cbk_fx()
    wfp = wfp_implied_fx(raw_wfp)
    months = pd.date_range(min(cbk["month"].min() if len(cbk) else wfp["month"].min(), wfp["month"].min()),
                           max(cbk["month"].max() if len(cbk) else wfp["month"].max(), wfp["month"].max()),
                           freq="MS")
    out = pd.DataFrame({"month": months}).merge(cbk, on="month", how="left").merge(wfp, on="month", how="left")
    if "kes_usd" not in out:
        out["kes_usd"] = np.nan
    both = out.dropna(subset=["kes_usd", "kes_usd_wfp"])
    scale = float((both["kes_usd"] / both["kes_usd_wfp"]).median()) if len(both) else 1.0
    out["fx_source"] = np.where(out["kes_usd"].notna(), "CBK", np.where(out["kes_usd_wfp"].notna(), "WFP", None))
    out["kes_usd"] = out["kes_usd"].fillna(out["kes_usd_wfp"] * scale)
    out["kes_usd"] = out["kes_usd"].interpolate(limit=3, limit_area="inside")
    return out[["month", "kes_usd", "fx_source"] + [c for c in ("ugx_kes", "tzs_kes") if c in out]]


def pink_sheet() -> pd.DataFrame:
    """Monthly global prices in USD for maize, wheat, crude oil and fertilizers."""
    path = RAW_DIR / settings()["sources"]["pink_sheet"]["file"]
    raw = pd.read_excel(path, sheet_name="Monthly Prices", header=None)
    header_row = raw.index[raw.iloc[:, 1].astype(str).str.startswith("Crude oil")][0]
    names = raw.iloc[header_row].fillna("").astype(str).str.strip()
    body = raw.iloc[header_row + 2:].copy()
    body.columns = names
    body = body.rename(columns={names.iloc[0]: "period"})
    body = body[body["period"].astype(str).str.match(r"^\d{4}M\d{2}$")]
    out = pd.DataFrame({"month": pd.to_datetime(body["period"].str.replace("M", "-") + "-01")})
    for key, label in PINK_SERIES.items():
        col = next((c for c in names if re.sub(r"\s+\*+$", "", c).strip() == label), None)
        out[f"global_{key}_usd"] = pd.to_numeric(body[col], errors="coerce").values if col else np.nan
    return out.reset_index(drop=True)


def macro_monthly(raw_wfp: pd.DataFrame) -> pd.DataFrame:
    fx = fx_monthly(raw_wfp)
    try:
        pink = pink_sheet()
    except FileNotFoundError:
        pink = pd.DataFrame(columns=["month"] + [f"global_{k}_usd" for k in PINK_SERIES])
    out = fx.merge(pink, on="month", how="outer").sort_values("month")
    out = out[out["month"] >= "2000-01-01"]
    return out.reset_index(drop=True)
