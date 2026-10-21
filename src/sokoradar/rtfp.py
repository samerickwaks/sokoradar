"""World Bank Real-Time Food Prices (RTFP): matched to WFP markets.

RTFP is gap-filled: only 3-5% of its market-months are observed and the rest are model
estimates that are revised with every release. SokoRadar therefore never trains on it;
it is used for (1) live-goat prices in the goat-to-maize terms of trade and (2) a
benchmark scored on the same observed WFP prices.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import geo
from .config import RAW_DIR, settings


def load_rtfp() -> pd.DataFrame:
    cfg = settings()
    path = RAW_DIR / cfg["sources"]["rtfp_food"]["file"]
    cols = cfg["rtfp_columns"]
    want = {"mkt_name", "lat", "lon", "DATES", *cols.values(), *(c[2:] for c in cols.values())}
    df = pd.read_csv(path, usecols=lambda c: c in want, low_memory=False)
    out = pd.DataFrame({
        "rtfp_market": df["mkt_name"], "rtfp_lat": df["lat"], "rtfp_lon": df["lon"],
        "month": pd.to_datetime(df["DATES"]).dt.to_period("M").dt.to_timestamp(),
    })
    for key, col in cols.items():
        out[f"rtfp_{key}"] = pd.to_numeric(df[col], errors="coerce")
        out[f"rtfp_{key}_observed"] = pd.to_numeric(df[col[2:]], errors="coerce").notna()
    return out


def match_markets(markets: pd.DataFrame, rtfp: pd.DataFrame, max_km: float | None = None) -> pd.DataFrame:
    max_km = max_km or settings()["prep"]["rtfp_match_km"]
    locs = rtfp.drop_duplicates("rtfp_market")[["rtfp_market", "rtfp_lat", "rtfp_lon"]]
    rows = []
    for m in markets.itertuples():
        d = geo.haversine_km(m.latitude, m.longitude, locs.rtfp_lat.values, locs.rtfp_lon.values)
        i = int(np.nanargmin(d))
        if d[i] <= max_km:
            rows.append({"market_key": m.market_key, "rtfp_market": locs.rtfp_market.iloc[i],
                         "rtfp_km": round(float(d[i]), 1)})
    return pd.DataFrame(rows, columns=["market_key", "rtfp_market", "rtfp_km"])


def rtfp_for_markets(markets: pd.DataFrame) -> pd.DataFrame:
    """RTFP estimates by WFP market and month (empty if the file is unavailable)."""
    path = RAW_DIR / settings()["sources"]["rtfp_food"]["file"]
    if not path.exists():
        return pd.DataFrame(columns=["market_key", "month", "rtfp_maize", "rtfp_beans", "rtfp_goat"])
    rtfp = load_rtfp()
    link = match_markets(markets, rtfp)
    out = link.merge(rtfp, on="rtfp_market", how="left")
    return out.drop(columns=["rtfp_lat", "rtfp_lon"])


def terms_of_trade(prices: pd.DataFrame, rtfp_m: pd.DataFrame, markets: pd.DataFrame) -> pd.DataFrame:
    """Kilograms of maize one goat buys, by pastoral market and month.

    WFP does not report live goats in Kenya, so goat prices come from RTFP. The headline
    series divides by the RTFP maize estimate too, so both sides come from one consistent
    source; `tot_wfp_kg` uses the observed WFP retail maize price where there is one.
    """
    if rtfp_m.empty:
        return pd.DataFrame()
    maize = (prices[(prices.commodity == "maize") & (prices.pricetype == "Retail")]
             .groupby(["market_key", "month"])["price"].median().rename("maize_wfp").reset_index())
    tot = rtfp_m[["market_key", "month", "rtfp_goat", "rtfp_maize", "rtfp_goat_observed"]].merge(
        maize, on=["market_key", "month"], how="left")
    tot["tot_kg_per_goat"] = tot["rtfp_goat"] / tot["rtfp_maize"]
    tot["tot_wfp_kg"] = tot["rtfp_goat"] / tot["maize_wfp"]
    pastoral = markets.loc[markets["pastoral"], "market_key"]
    tot = tot[tot["market_key"].isin(pastoral) & tot["tot_kg_per_goat"].notna()]
    return tot.sort_values(["market_key", "month"]).reset_index(drop=True)
