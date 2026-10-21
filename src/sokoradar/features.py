"""Feature table: one row per (series, origin month t) with an observed price at t.

Every feature uses information available at the end of month t only: prices up to t,
ALPS refitted on prices up to t, climate and macro values up to t. Targets are the log
price change from t to t+h and the current-month ALPS at t+h.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from .config import CONFIG_DIR, settings

HORIZONS = (1, 2, 3)
LAGS = (1, 2, 3, 4, 5, 6, 9, 12)

PRICE_FEATURES = ([f"lag{k}" for k in LAGS]
                  + ["rel_ma3", "rel_ma6", "rel_ma12", "rel_ma36", "vol6", "vol12", "prev_gap", "n_hist"])
LEVEL_FEATURES = ["alps", "alps_l1", "alps_chg", "rel_sd"]
SEASON_FEATURES = ["seas_chg", "seas_avg", "month", "tgt_month", "tgt_sin", "tgt_cos", "tgt_lean",
                   "tgt_harvest", "tgt_dry"]
NETWORK_FEATURES = ["nat_chg1", "nat_chg3", "nat_chg12", "gap_nbo", "nbo_chg3", "near_gap"]
RAIN_FEATURES = [f"rain1_l{k}" for k in range(6)] + ["rain3_l0", "rain3_l3", "rain6"]
NDVI_FEATURES = ["ndvi_l0", "ndvi_l1", "ndvi_l2", "ndvi_l3", "ndvi_l5"]
FX_FEATURES = ["fx_chg1", "fx_chg3", "fx_chg12"]
GLOBAL_FEATURES = ["gmaize_chg3", "gmaize_chg12", "gwheat_chg12", "crude_chg3", "crude_chg12", "urea_chg12"]
MACRO_FEATURES = FX_FEATURES + GLOBAL_FEATURES
POLICY_FEATURES = ["policy_now", "policy_ended"]
MARKET_FEATURES = ["dist_nairobi_km", "dist_border_km", "asal_code", "commodity_code", "pricetype_code"]

_PRICE = PRICE_FEATURES + LEVEL_FEATURES + SEASON_FEATURES + MARKET_FEATURES
# Ablation sets for research question 2. "sokoradar" is the production set: global commodity
# prices (Pink Sheet) added noise in the 2015-2025 backtests, so they only appear in "full".
FEATURE_SETS = {
    "price_only": _PRICE,
    "price_network": _PRICE + NETWORK_FEATURES,
    "price_climate": _PRICE + RAIN_FEATURES + NDVI_FEATURES,
    "sokoradar": _PRICE + NETWORK_FEATURES + RAIN_FEATURES + NDVI_FEATURES + FX_FEATURES + POLICY_FEATURES,
    "full": _PRICE + NETWORK_FEATURES + RAIN_FEATURES + NDVI_FEATURES + MACRO_FEATURES + POLICY_FEATURES,
}
FEATURES = FEATURE_SETS["sokoradar"]
CATEGORICAL = ["commodity_code", "pricetype_code", "asal_code"]

FAMILY = {}
for _fam, _cols in [("price_momentum", PRICE_FEATURES), ("price_level", LEVEL_FEATURES),
                    ("seasonality", SEASON_FEATURES), ("market_network", NETWORK_FEATURES),
                    ("rainfall", RAIN_FEATURES), ("vegetation", NDVI_FEATURES), ("macro", MACRO_FEATURES),
                    ("policy", POLICY_FEATURES), ("market_profile", MARKET_FEATURES)]:
    FAMILY.update({c: _fam for c in _cols})

COMMODITY_CODES = {"maize": 0, "beans": 1, "maize_flour": 2, "goat_meat": 3}
ASAL_CODES = {"non-ASAL": 0, "semi-arid": 1, "arid": 2}


def _series_grid(prices: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for sid, g in prices.groupby("series_id", sort=True):
        idx = pd.date_range(g["month"].min(), g["month"].max(), freq="MS")
        s = g.set_index("month")["price"].reindex(idx)
        frames.append(pd.DataFrame({"series_id": sid, "month": idx, "price": s.to_numpy()}))
    return pd.concat(frames, ignore_index=True)


def _shift(df: pd.DataFrame, col: str, k: int) -> pd.Series:
    return df.groupby("series_id", sort=False)[col].shift(k)


def _rolling(df: pd.DataFrame, col: str, window: int, fn: str, min_periods: int) -> pd.Series:
    r = df.groupby("series_id", sort=False)[col].rolling(window, min_periods=min_periods)
    return getattr(r, fn)().reset_index(level=0, drop=True)


def _index_changes(g: pd.DataFrame, keys: list[str], prefix: str, windows=(1, 3, 12)) -> pd.DataFrame:
    """Chain-linked price index from the median monthly log change across series."""
    ret = g.dropna(subset=["dlog"]).groupby(keys + ["month"])["dlog"].median().rename("ret").reset_index()
    out = []
    for key, h in ret.groupby(keys):
        idx = pd.date_range(h["month"].min(), h["month"].max(), freq="MS")
        s = h.set_index("month")["ret"].reindex(idx).fillna(0.0)
        frame = pd.DataFrame({"month": idx})
        for i, k in enumerate(keys):
            frame[k] = key[i] if isinstance(key, tuple) else key
        cum = s.cumsum()
        for w in windows:
            frame[f"{prefix}_chg{w}"] = (cum - cum.shift(w)).to_numpy()
        out.append(frame)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=keys + ["month"])


def _policy_flags(months: pd.Series) -> pd.DataFrame:
    ev = pd.read_csv(CONFIG_DIR / "policy_events.csv")
    now = np.zeros(len(months), dtype=float)
    ended = np.zeros(len(months), dtype=float)
    m = pd.to_datetime(months).to_numpy()
    for e in ev.itertuples():
        start, end = np.datetime64(pd.Timestamp(e.start)), np.datetime64(pd.Timestamp(e.end))
        now = np.maximum(now, ((m >= start) & (m <= end)).astype(float))
        end3 = np.datetime64(pd.Timestamp(e.end) + pd.DateOffset(months=3))
        ended = np.maximum(ended, ((m > end) & (m <= end3)).astype(float))
    return pd.DataFrame({"policy_now": now, "policy_ended": ended})


def _climate_lags(climate: pd.DataFrame) -> pd.DataFrame:
    c = climate.sort_values(["county_pcode", "month"]).copy()
    # Re-grid so shifts are true calendar lags.
    frames = []
    for pc, g in c.groupby("county_pcode"):
        idx = pd.date_range(g["month"].min(), g["month"].max(), freq="MS")
        frames.append(g.set_index("month").reindex(idx).assign(county_pcode=pc).rename_axis("month").reset_index())
    c = pd.concat(frames, ignore_index=True)
    gb = c.groupby("county_pcode", sort=False)
    out = c[["county_pcode", "month"]].copy()
    for k in range(6):
        out[f"rain1_l{k}"] = gb["rain_1m_pct"].shift(k)
    out["rain3_l0"] = c["rain_3m_pct"]
    out["rain3_l3"] = gb["rain_3m_pct"].shift(3)
    avg = c["rain_1m_mm"] * 100 / c["rain_1m_pct"].replace(0, np.nan)
    mm6 = c.assign(_mm=c["rain_1m_mm"]).groupby("county_pcode")["_mm"].rolling(6, min_periods=6).sum()
    avg6 = c.assign(_avg=avg).groupby("county_pcode")["_avg"].rolling(6, min_periods=6).sum()
    out["rain6"] = (100 * mm6.reset_index(level=0, drop=True) / avg6.reset_index(level=0, drop=True)).clip(0, 400)
    for k in (0, 1, 2, 3, 5):
        out[f"ndvi_l{k}"] = gb["ndvi_pct"].shift(k)
    return out


def _macro_changes(macro: pd.DataFrame) -> pd.DataFrame:
    m = macro.set_index("month").sort_index()
    m = m.reindex(pd.date_range(m.index.min(), m.index.max(), freq="MS")).ffill(limit=2)
    out = pd.DataFrame(index=m.index)
    lfx = np.log(m["kes_usd"])
    out["lfx_l3"] = lfx.shift(3)
    for k in (1, 3, 12):
        out[f"fx_chg{k}"] = lfx - lfx.shift(k)
    for name, col in [("gmaize", "global_maize_usd"), ("gwheat", "global_wheat_usd"),
                      ("crude", "global_crude_oil_usd"), ("urea", "global_urea_usd")]:
        lk = np.log(m[col] * m["kes_usd"])  # global price converted to shillings
        for k in (3, 12):
            out[f"{name}_chg{k}"] = lk - lk.shift(k)
    return out.rename_axis("month").reset_index()


def _nearest_gaps(grid: pd.DataFrame, meta: pd.DataFrame, max_km: float = 500) -> pd.Series:
    from .geo import haversine_km
    wide = grid.pivot(index="month", columns="series_id", values="logp")
    gaps = pd.Series(np.nan, index=grid.index)
    m = meta.set_index("series_id")
    for sid, g in grid.groupby("series_id", sort=False):
        me = m.loc[sid]
        cand = m[(m["commodity"] == me["commodity"]) & (m["market_key"] != me["market_key"])
                 & (m["pricetype"] == me["pricetype"])].copy()
        if cand.empty:
            continue
        cand["d"] = haversine_km(me["latitude"], me["longitude"], cand["latitude"], cand["longitude"])
        cand = cand[cand["d"] <= max_km].sort_values("d")
        ref = pd.Series(np.nan, index=wide.index)
        for csid in cand.index:
            ref = ref.combine_first(wide[csid])
        gaps.loc[g.index] = g["logp"].to_numpy() - ref.reindex(g["month"]).to_numpy()
    return gaps


def build_panel(prices: pd.DataFrame, markets: pd.DataFrame, climate: pd.DataFrame, macro: pd.DataFrame,
                alps: pd.DataFrame) -> pd.DataFrame:
    """Full monthly panel per series (observed and missing months) with features and targets."""
    seas = settings()["seasons"]
    meta = (prices.groupby("series_id")[["market_key", "commodity", "pricetype"]].first().reset_index()
            .merge(markets, on="market_key", how="left"))
    g = _series_grid(prices).merge(meta, on="series_id", how="left")
    g = g.sort_values(["series_id", "month"]).reset_index(drop=True)
    g["logp"] = np.log(g["price"])
    g["observed"] = g["logp"].notna()

    # Price memory, relative to the current log price so one model fits every price level.
    for k in LAGS:
        g[f"lag{k}"] = _shift(g, "logp", k) - g["logp"]
    for w in (3, 6, 12, 36):
        g[f"rel_ma{w}"] = _rolling(g, "logp", w, "mean", max(2, w // 3)) - g["logp"]
    # A smoothed current level (last price averaged with the 3-month mean) damps one-off spikes.
    g["smooth_logp"] = 0.5 * g["logp"] + 0.5 * (g["logp"] + g["rel_ma3"].fillna(0.0))
    g["dlog"] = g.groupby("series_id", sort=False)["logp"].diff()
    g["vol6"] = _rolling(g, "dlog", 6, "std", 3)
    g["vol12"] = _rolling(g, "dlog", 12, "std", 6)
    pos = g.groupby("series_id", sort=False).cumcount()
    last_obs = pos.where(g["observed"]).groupby(g["series_id"]).shift(1)
    last_obs = last_obs.groupby(g["series_id"]).ffill()
    g["prev_gap"] = pos - last_obs
    g["n_hist"] = np.log1p(g.groupby("series_id", sort=False)["observed"].cumsum())

    # Seasonality seen in the series' own history: change from month t to t+h in past years.
    for h in HORIZONS:
        g[f"seas_chg_h{h}"] = _shift(g, "logp", 12 - h) - _shift(g, "logp", 12)
        past = np.column_stack([(_shift(g, "logp", 12 * y - h) - _shift(g, "logp", 12 * y)).to_numpy()
                                for y in range(1, 6)])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            g[f"seas_avg_h{h}"] = np.nanmean(past, axis=1)

    # ALPS at the origin (and the expected prices that define phases at t+h).
    g = g.merge(alps.drop(columns=["phase"], errors="ignore"), on=["series_id", "month"], how="left")
    g["alps_l1"] = _shift(g, "alps", 1)
    g["alps_chg"] = g["alps"] - g["alps_l1"]
    g["rel_sd"] = g["alps_sd"] / g["expected"]

    # Market network: national index for the same commodity and price type, Nairobi, nearest market.
    nat = _index_changes(g, ["commodity", "pricetype"], "nat")
    g = g.merge(nat, on=["commodity", "pricetype", "month"], how="left")
    nbo_rows = g[(g["county"] == settings()["country"]["capital_market_county"]) & (g["pricetype"] == "Wholesale")]
    nbo_level = nbo_rows.groupby(["commodity", "month"])["logp"].median().rename("nbo_logp").reset_index()
    nbo_level = (nbo_level.set_index("month").groupby("commodity")["nbo_logp"]
                 .apply(lambda s: s.reindex(pd.date_range(s.index.min(), s.index.max(), freq="MS")).ffill(limit=2))
                 .rename("nbo_logp").reset_index().rename(columns={"level_1": "month"}))
    nbo_idx = _index_changes(nbo_rows, ["commodity"], "nbo", windows=(3,))
    g = g.merge(nbo_level, on=["commodity", "month"], how="left").merge(nbo_idx, on=["commodity", "month"], how="left")
    g["gap_nbo"] = g["logp"] - g["nbo_logp"]
    g["near_gap"] = _nearest_gaps(g, meta)

    # Climate for the market's county, and macro drivers.
    g = g.merge(_climate_lags(climate), on=["county_pcode", "month"], how="left")
    g = g.merge(_macro_changes(macro), on="month", how="left")
    g = pd.concat([g, _policy_flags(g["month"])], axis=1)

    # Calendar and static market descriptors.
    g["month_num"] = g["month"].dt.month
    g["high_potential"] = g["county"].isin(seas["high_potential_counties"])
    g["commodity_code"] = g["commodity"].map(COMMODITY_CODES)
    g["pricetype_code"] = (g["pricetype"] == "Wholesale").astype(int)
    g["asal_code"] = g["asal"].map(ASAL_CODES)

    # Targets, and the seasonal-naive baseline (same month last year).
    for h in HORIZONS:
        g[f"snaive_h{h}"] = _shift(g, "logp", 12 - h)
        g[f"tgt_h{h}"] = _shift(g, "logp", -h) - g["logp"]
        g[f"tgt_alps_h{h}"] = _shift(g, "alps", -h)
        g[f"tgt_month_h{h}"] = g["month"] + pd.DateOffset(months=h)
    return g


def design(panel: pd.DataFrame, h: int, features: list[str] | None = None) -> pd.DataFrame:
    """Model matrix for horizon h (horizon-specific seasonal and calendar features)."""
    seas = settings()["seasons"]
    features = features or FEATURES
    X = pd.DataFrame(index=panel.index)
    tgt_m = (panel["month"].dt.month - 1 + h) % 12 + 1
    derived = {
        "seas_chg": panel[f"seas_chg_h{h}"], "seas_avg": panel[f"seas_avg_h{h}"],
        "month": panel["month"].dt.month, "tgt_month": tgt_m,
        "tgt_sin": np.sin(2 * np.pi * tgt_m / 12), "tgt_cos": np.cos(2 * np.pi * tgt_m / 12),
        "tgt_lean": tgt_m.isin(seas["lean"]).astype(int),
        "tgt_harvest": np.where(panel["high_potential"], tgt_m.isin(seas["harvest_high_potential"]),
                                tgt_m.isin(seas["harvest_marginal"])).astype(int),
        "tgt_dry": (tgt_m.isin(seas["pastoral_dry"]) & panel["pastoral"].fillna(False).astype(bool)).astype(int),
    }
    for f in features:
        X[f] = derived[f] if f in derived else panel[f]
    return X.astype(float)


def origins(panel: pd.DataFrame) -> pd.DataFrame:
    """Rows usable as forecast origins: the price at t is observed."""
    return panel[panel["observed"]].copy()
