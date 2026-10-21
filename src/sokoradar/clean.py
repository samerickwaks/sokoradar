"""Standardise WFP market prices into clean monthly KES-per-kg series.

Output grain: one row per (market, commodity, price type, month) with an observed price.
Targets are never gap-filled; missing months simply have no row.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import geo
from .config import RAW_DIR, settings

NAIROBI = (-1.2864, 36.8172)

# Pre-2013 district names WFP still uses in admin2, mapped to today's counties.
OLD_DISTRICTS = {"Ijara": "Garissa", "Meru North": "Meru", "Meru South": "Tharaka-Nithi", "Moyale": "Marsabit"}

_UNIT_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)?\s*(KG|G|L|ML)\s*$", re.IGNORECASE)
_UNIT_KG = {"kg": 1.0, "g": 0.001, "l": 1.0, "ml": 0.001}


@dataclass
class CleanResult:
    prices: pd.DataFrame
    markets: pd.DataFrame
    quality: dict = field(default_factory=dict)
    dropped_outliers: pd.DataFrame | None = None


def unit_to_kg(unit: str) -> float:
    """'90 KG' -> 90, '200 G' -> 0.2, 'KG' -> 1; NaN for counts such as 'Unit' or 'Bunch'."""
    if not isinstance(unit, str):
        return np.nan
    m = _UNIT_RE.match(unit)
    if not m:
        return np.nan
    qty = float(m.group(1)) if m.group(1) else 1.0
    return qty * _UNIT_KG[m.group(2).lower()]


def commodity_map() -> dict[str, str]:
    out = {}
    for key, spec in settings()["commodities"].items():
        for name in spec["wfp_names"]:
            out[name] = key
    return out


def market_base(name: str) -> str:
    """'Garissa town (Garissa)' and 'Garissa' share the base 'garissa'.

    Parentheses are place hints and are dropped, unless they carry a number:
    'Kalobeyei (Village 2)' and 'Kakuma 3' stay distinct markets.
    """
    base = re.sub(r"\([^)\d]*\)", "", str(name)).lower()
    base = re.sub(r"[()]", " ", base)
    base = re.sub(r"\b(town|market|centre|center)\b", "", base)
    return re.sub(r"\s+", " ", base).strip()


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", str(text).lower()).strip("-")


def read_wfp_raw(path=None) -> pd.DataFrame:
    path = path or RAW_DIR / settings()["sources"]["wfp_prices"]["file"]
    df = pd.read_csv(path, dtype=str)
    df = df[~df["date"].str.startswith("#")].copy()
    for col in ("price", "usdprice", "latitude", "longitude"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["date"] = pd.to_datetime(df["date"])
    return df


def _merge_markets(raw: pd.DataFrame, merge_km: float) -> pd.DataFrame:
    """Give every raw market name a canonical market key (same base name and within merge_km)."""
    mk = (raw.groupby("market")
          .agg(latitude=("latitude", "median"), longitude=("longitude", "median"),
               admin2=("admin2", "first"), n=("price", "size"), market_id=("market_id", "first"))
          .reset_index())
    mk["base"] = mk["market"].map(market_base)
    mk = mk.sort_values("n", ascending=False)
    canon: dict[str, str] = {}
    anchors: list[tuple[str, str, float, float]] = []  # base, canonical name, lat, lon
    for r in mk.itertuples():
        match = None
        for base, name, lat, lon in anchors:
            if base == r.base and geo.haversine_km(lat, lon, r.latitude, r.longitude) <= merge_km:
                match = name
                break
        if match is None:
            anchors.append((r.base, r.market, r.latitude, r.longitude))
            match = r.market
        canon[r.market] = match
    mk["market_canon"] = mk["market"].map(canon)
    return mk


def _drop_outliers(df: pd.DataFrame, k_mad: float, min_ratio: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Drop prices that sit far from their own series' rolling median (unit slips, typos)."""
    df = df.sort_values(["series_id", "month"]).copy()
    df["logp"] = np.log(df["price"])
    med = df.groupby("series_id")["logp"].transform(
        lambda s: s.rolling(7, center=True, min_periods=3).median())
    dev = df["logp"] - med
    mad = dev.groupby(df["series_id"]).transform(lambda s: (s - s.median()).abs().median()) * 1.4826
    mad = mad.clip(lower=0.05)
    bad = (dev.abs() > k_mad * mad) & (dev.abs() > np.log(min_ratio))
    return df.loc[~bad].drop(columns="logp"), df.loc[bad].drop(columns="logp")


def clean_prices(raw: pd.DataFrame | None = None) -> CleanResult:
    cfg = settings()["prep"]
    raw = read_wfp_raw() if raw is None else raw
    q: dict = {"raw_rows": len(raw), "raw_markets": raw["market"].nunique(),
               "raw_commodities": raw["commodity"].nunique(),
               "raw_period": f"{raw['date'].min():%Y-%m} to {raw['date'].max():%Y-%m}"}

    cmap = commodity_map()
    df = raw[raw["commodity"].isin(cmap) & raw["pricetype"].isin(settings()["price_types"])].copy()
    df = df[df["currency"] == settings()["country"]["currency"]]
    q["rows_in_scope"] = len(df)
    df["commodity_raw"] = df["commodity"]
    df["commodity"] = df["commodity"].map(cmap)
    df["kg"] = df["unit"].map(unit_to_kg)
    q["rows_unit_unparsed"] = int(df["kg"].isna().sum())
    df = df[df["kg"].notna() & (df["price"] > 0)].copy()
    df["price_kg"] = df["price"] / df["kg"]
    df["month"] = df["date"].dt.to_period("M").dt.to_timestamp()

    markets = _merge_markets(raw, cfg["market_merge_km"])
    q["markets_merged"] = int((markets["market"] != markets["market_canon"]).sum())
    q["merged_pairs"] = (markets.loc[markets["market"] != markets["market_canon"], ["market", "market_canon"]]
                         .to_dict("records"))
    df["market"] = df["market"].map(dict(zip(markets["market"], markets["market_canon"])))

    # One canonical market table.
    canon = markets[markets["market"] == markets["market_canon"]].copy()
    canon["market_key"] = canon["market"].map(slug)
    canon["county_pcode"] = [geo.locate_county(a, b) for a, b in zip(canon.latitude, canon.longitude)]
    cty = geo.counties().set_index("pcode")
    by_name = geo.counties().set_index("county")["pcode"]
    # Fallbacks: a county named in the market's parentheses, then WFP's admin2 (old districts mapped).
    hint = canon["market"].str.extract(r"\(([^)]*)\)", expand=False).str.strip()
    missing = canon["county_pcode"].isna()
    canon.loc[missing, "county_pcode"] = hint[missing].map(by_name)
    missing = canon["county_pcode"].isna()
    canon.loc[missing, "county_pcode"] = (canon.loc[missing, "admin2"].replace(OLD_DISTRICTS).map(by_name))
    canon["county"] = canon["county_pcode"].map(cty["county"])
    # Missing coordinates: WFP's market list, then the county centre.
    no_xy = canon["latitude"].isna()
    if no_xy.any():
        mfile = RAW_DIR / settings()["sources"]["wfp_markets"]["file"]
        if mfile.exists():
            ml = pd.read_csv(mfile, dtype=str)
            ml = ml[~ml["market_id"].str.startswith("#")]
            xy = ml.set_index("market")[["latitude", "longitude"]].apply(pd.to_numeric, errors="coerce")
            xy = xy[~xy.index.duplicated()]
            canon.loc[no_xy, "latitude"] = canon.loc[no_xy, "market"].map(xy["latitude"])
            canon.loc[no_xy, "longitude"] = canon.loc[no_xy, "market"].map(xy["longitude"])
        no_xy = canon["latitude"].isna()
        canon.loc[no_xy, "latitude"] = canon.loc[no_xy, "county_pcode"].map(cty["center_lat"])
        canon.loc[no_xy, "longitude"] = canon.loc[no_xy, "county_pcode"].map(cty["center_lon"])
    canon["asal"] = canon["county"].map(geo.asal_class)
    canon["pastoral"] = canon["asal"].eq("arid")
    canon["dist_nairobi_km"] = geo.haversine_km(canon.latitude, canon.longitude, *NAIROBI).round(1)
    canon["dist_border_km"] = [round(geo.distance_to_border_km(a, b), 1)
                               for a, b in zip(canon.latitude, canon.longitude)]
    canon = canon.rename(columns={"market_id": "wfp_market_id"})
    canon = canon[["market_key", "market", "wfp_market_id", "latitude", "longitude", "county_pcode", "county",
                   "asal", "pastoral", "dist_nairobi_km", "dist_border_km"]]

    # Monthly median per series; several raw names/units can describe the same series.
    agg = (df.groupby(["market", "commodity", "pricetype", "month"])
           .agg(price=("price_kg", "median"), n_raw=("price_kg", "size"),
                sources=("commodity_raw", lambda s: ";".join(sorted(set(s)))))
           .reset_index())
    agg = agg.merge(canon[["market", "market_key"]], on="market", how="left")
    agg["series_id"] = agg["market_key"] + "|" + agg["commodity"] + "|" + agg["pricetype"].str.lower()
    q["series_before_filter"] = agg["series_id"].nunique()

    agg, dropped = _drop_outliers(agg, cfg["outlier_log_mad"], cfg["outlier_min_ratio"])
    q["outliers_dropped"] = len(dropped)

    counts = agg.groupby("series_id")["month"].nunique()
    keep = counts[counts >= cfg["min_months"]].index
    q["series_kept"] = len(keep)
    q["series_short"] = int((counts < cfg["min_months"]).sum())
    prices = agg[agg["series_id"].isin(keep)].sort_values(["series_id", "month"]).reset_index(drop=True)
    prices = prices[["series_id", "market_key", "market", "commodity", "pricetype", "month", "price", "n_raw",
                     "sources"]]
    q["observations_kept"] = len(prices)
    q["markets_kept"] = prices["market_key"].nunique()
    q["kept_period"] = (f"{prices['month'].min():%Y-%m} to {prices['month'].max():%Y-%m}" if len(prices) else "none")
    used = canon[canon["market_key"].isin(prices["market_key"])].reset_index(drop=True)
    q["markets_without_county"] = used.loc[used["county"].isna(), "market"].tolist()
    return CleanResult(prices=prices, markets=used, quality=q, dropped_outliers=dropped)


def series_table(prices: pd.DataFrame, markets: pd.DataFrame) -> pd.DataFrame:
    """One row per series with coverage statistics."""
    s = (prices.groupby(["series_id", "market_key", "commodity", "pricetype"])
         .agg(first_month=("month", "min"), last_month=("month", "max"), n_obs=("month", "nunique"),
              last_price=("price", "last"))
         .reset_index())
    span = ((s["last_month"].dt.year - s["first_month"].dt.year) * 12
            + s["last_month"].dt.month - s["first_month"].dt.month + 1)
    s["coverage"] = (s["n_obs"] / span).round(3)
    return s.merge(markets, on="market_key", how="left")
