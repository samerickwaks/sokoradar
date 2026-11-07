"""SokoRadar forecast: the frozen final model, ready to run.

What it does, for every county price series (47 counties x 26 commodity/price-type combinations):
  1. forecast the price 1, 2 and 3 months ahead           (ARIMA, Daniella's orders)
  2. give the chance of a spike (a rise of more than 15%)  (M8a, a LightGBM spike classifier)
  3. raise an alert when that chance is above a threshold   (20% false-alarm budget, set on 2011-2014)
  4. list the three inputs that move the forecast most      (M5, explained with SHAP values)
  5. after an alert, suggest a cheaper nearby county         (Daniella's "where to buy" rule, notebook 04_recommender)

Steps 1-4 were chosen and tested in notebooks/modelling/02_model_build.ipynb (backtest 2015-2024);
step 5 in Daniella's 04_recommender notebook (same backtest, real prices only).
The models are trained on targets up to TRAIN_END (Dec 2024) and nothing else; the settings below are
frozen and must not be changed without re-running that notebook.

Run (from Anaconda Prompt):
    python sokoradar_forecast.py                         # forecast from the latest month in the data
    python sokoradar_forecast.py --origins 2024-06:2024-12
    python sokoradar_forecast.py --check                  # rebuild the 2024 backtest year and compare with the notebook
    python sokoradar_forecast.py --origins 2024-01:2024-12 --chart "Nakuru | Dry Maize | Wholesale"   # also draw a chart

Output: a CSV (one row per series, origin month and horizon) and a small JSON describing the run.
See MODEL_CARD.md in this folder for what the columns mean and the model's limits.
"""
import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- frozen settings (from the notebook)
TRAIN_END = "2024-12"            # the models learn only from answers (target months) up to here
HORIZONS = [1, 2, 3]             # months ahead
SPIKE = 0.15                     # a spike = the price rises more than 15% from the origin month
DATA_FILE = "SokoRadar_County_2006_2024_long.csv"

# Alert thresholds: M8a's spike probability above which an alert is raised, per horizon. Each is the value that
# alerted in 20% of calm (no-spike) months on 2011-2014 real prices; 2015-2024 only judged them.
ALERT_THRESHOLD = {1: 0.08230874618305906, 2: 0.2025086377495205, 3: 0.35055914314124076}

# ARIMA orders per commodity (Daniella's, chosen by BIC on 2006-2012), fitted on log prices
ARIMA_ORDERS = {"Dry Maize": (1, 1, 0), "Sorghum": (2, 1, 0), "Beans Rosecoco": (2, 1, 2), "Red Irish Potatoes": (2, 1, 1)}
ARIMA_DEFAULT = (1, 1, 0)

# LightGBM settings. M8a: standard settings, not tuned. M5: tuned per horizon by time-series CV on 2011-2014.
# The learning rate is written out on purpose, so it is always visible.
BASE = dict(objective="l1", n_estimators=200, learning_rate=0.05, num_leaves=31,
            min_child_samples=200, random_state=42, verbose=-1, n_jobs=-1)
M8A_PARAMS = {**BASE, "objective": "binary"}
M5_PARAMS = {1: {**BASE, "num_leaves": 15, "min_child_samples": 1000, "learning_rate": 0.05, "n_estimators": 200},
             2: {**BASE, "num_leaves": 63, "min_child_samples": 200, "learning_rate": 0.025, "n_estimators": 400},
             3: {**BASE, "num_leaves": 15, "min_child_samples": 200, "learning_rate": 0.05, "n_estimators": 200}}
# Real market prices count more in training: each observed answer weighs (estimated rows / observed rows)
# ("full balancing weight", chosen in the notebook for M1, M5 and M8a).

# "Where to buy" (Daniella's recommender, 04_recommender): after an alert, the cheapest nearby county on recent REAL
# (market-recorded) prices, within RADIUS_KM, same commodity and price type, if at least MIN_GAIN cheaper.
# Her backtest showed "cheapest today" saves as much as using the forecast, so no forecast is used. Only the four crops
# she evaluated. Real prices must be from the origin month itself (RECENT_MONTHS = 1), her tested rule.
# A 3-month window was tried (8 Nov 2026): replayed on the 2015-2024 backtest, same-month suggestions were really cheaper
# a month later 85% of the time (average saving 20%), older-price ones only 71% (8%), so it was dropped.
RECOMMEND_CROPS = ["Dry Maize", "Sorghum", "Beans Rosecoco", "Red Irish Potatoes"]
RADIUS_KM = 200
MIN_GAIN = 0.05
RECENT_MONTHS = 1                # 1 = real prices from the origin month only

# Inputs
PRICE_HISTORY = ["change_1m", "change_3m", "change_6m", "change_12m", "distance_from_normal", "jumpiness"]
M1_FEATURES = PRICE_HISTORY + ["month_of_year", "commodity", "price_type", "county"]          # M8a uses these
WEATHER = ["rain_1m_pct", "rain_3m_pct", "ndvi_pct", "el_nino_index"]
RELATED = ["related_change_1m", "related_change_3m"]
ECONOMY = ["murban_change_3m", "diesel_change_3m", "shilling_change_3m", "shilling_change_12m", "inflation_12m", "cpi_change_3m"]
M4_FEATURES = M1_FEATURES + WEATHER + RELATED + ECONOMY                                     # M5 uses these

CROP_GROUPS = {"Dry Maize": "cereals", "Sorghum": "cereals", "Wheat": "cereals", "Wheat flour": "cereals",
               "Beans Canadian": "beans", "Beans Dolichos (Njahi)": "beans", "Beans Mwitemania": "beans",
               "Beans Rosecoco": "beans", "Mwezi Moja": "beans",
               "Red Irish Potatoes": "potatoes", "White Irish Potatoes": "potatoes",
               "Beef": "meat", "Goat meat": "meat"}

# Plain-language names for the dashboard: (family, label)
DRIVER_LABELS = {
    "change_1m": ("recent price moves", "price change over the last month"),
    "change_3m": ("recent price moves", "price change over the last 3 months"),
    "change_6m": ("recent price moves", "price change over the last 6 months"),
    "change_12m": ("recent price moves", "price change over the last 12 months"),
    "distance_from_normal": ("recent price moves", "price compared with its 12-month average"),
    "jumpiness": ("recent price moves", "how jumpy the price has been (last 12 months)"),
    "month_of_year": ("season", "time of year"),
    "commodity": ("which market", "which commodity"),
    "price_type": ("which market", "retail or wholesale"),
    "county": ("which market", "which county"),
    "rain_1m_pct": ("weather", "rainfall last month compared with normal"),
    "rain_3m_pct": ("weather", "rainfall over the last 3 months compared with normal"),
    "ndvi_pct": ("weather", "vegetation (greenness) compared with normal"),
    "el_nino_index": ("weather", "El Niño index (2 months earlier)"),
    "related_change_1m": ("related crops", "price change of related crops, last month"),
    "related_change_3m": ("related crops", "price change of related crops, last 3 months"),
    "murban_change_3m": ("economy", "crude oil price (Murban), 3-month change"),
    "diesel_change_3m": ("economy", "diesel pump price, 3-month change"),
    "shilling_change_3m": ("economy", "shilling against the dollar, 3-month change"),
    "shilling_change_12m": ("economy", "shilling against the dollar, 12-month change"),
    "inflation_12m": ("economy", "overall inflation, 12-month"),
    "cpi_change_3m": ("economy", "consumer price index, 3-month change"),
}
assert sorted(DRIVER_LABELS) == sorted(M4_FEATURES), "every M5 input needs a plain-language label"


# --------------------------------------------------------------------------- 1. load and check the data
def find_data(data_arg, raw_arg):
    """The pipeline CSV and the raw-data folder: from the arguments, SOKORADAR_OUTPUT, or the folders above this one."""
    if data_arg:
        data = Path(data_arg)
    else:
        places = [Path(os.environ["SOKORADAR_OUTPUT"])] if os.environ.get("SOKORADAR_OUTPUT") else []
        here = Path(__file__).resolve().parent
        places += [up / "output" for up in [here, *here.parents][:6]]
        data = next((p / DATA_FILE for p in places if (p / DATA_FILE).exists()), None)
        if data is None:
            sys.exit(f"Could not find {DATA_FILE}. Pass --data, or set SOKORADAR_OUTPUT to the folder that holds it.")
    raw = Path(raw_arg) if raw_arg else data.parent.parent / "sokoradar_raw_data"
    for need in ["counties.csv", "county_climate_chirps_ndvi.csv", "national_drivers_2006_2024.csv"]:
        if not (raw / need).exists():
            sys.exit(f"Missing raw file {raw / need}. Pass --raw with the folder that holds it.")
    return data, raw


def load_prices(data_path):
    prices = pd.read_csv(data_path, parse_dates=["month"])
    for col in ["month", "county_code", "county", "commodity", "price_type", "price_kes", "flag"]:
        if col not in prices.columns:
            sys.exit(f"The price file has no '{col}' column")
    if prices["price_kes"].isna().any() or (prices["price_kes"] <= 0).any():
        sys.exit("Some prices are blank, zero or negative")
    if prices["month"].min() > pd.Timestamp("2006-01-01") or prices["month"].max() < pd.Timestamp(TRAIN_END):
        sys.exit(f"The price file must cover Jan 2006 to at least {TRAIN_END}")
    prices["observed"] = prices["flag"].str.startswith("Observed")
    key = ["county", "commodity", "price_type"]
    prices = prices.sort_values(key + ["month"]).reset_index(drop=True)
    prices["series"] = prices["county"] + " | " + prices["commodity"] + " | " + prices["price_type"]
    n_months = prices["month"].nunique()
    if not prices.groupby("series")["month"].nunique().eq(n_months).all():
        sys.exit("Some series are missing months; every series needs every month")
    return prices


# --------------------------------------------------------------------------- 2. inputs, known at the origin month only
def build_inputs(prices, raw):
    """One row per series and month, with every input the models use (all from information up to that month)."""
    past = prices.groupby("series")["price_kes"]
    X = prices[["series", "month"]].rename(columns={"month": "origin_month"})
    for k in [1, 3, 6, 12]:
        X[f"change_{k}m"] = prices["price_kes"] / past.shift(k) - 1
    X["distance_from_normal"] = prices["price_kes"] / past.transform(lambda s: s.rolling(12).mean()) - 1
    X["jumpiness"] = X.groupby("series")["change_1m"].transform(lambda s: s.rolling(12).std())
    X["month_of_year"] = prices["month"].dt.month

    # weather: county rainfall and vegetation at the origin month, El Niño from 2 months earlier
    counties = pd.read_csv(raw / "counties.csv")[["code", "pcode"]]
    climate = pd.read_csv(raw / "county_climate_chirps_ndvi.csv", parse_dates=["month"])
    drivers = pd.read_csv(raw / "national_drivers_2006_2024.csv")
    drivers["month"] = pd.to_datetime(drivers["Month"])
    drivers = drivers.sort_values("month").reset_index(drop=True)
    el_nino = drivers[["month", "ONI (El Nino index)"]].rename(columns={"ONI (El Nino index)": "el_nino_index"})
    el_nino["month"] = el_nino["month"] + pd.DateOffset(months=2)
    weather = (prices[["series", "county_code", "month"]]
               .merge(counties, left_on="county_code", right_on="code", how="left")
               .merge(climate[["county_pcode", "month", "rain_1m_pct", "rain_3m_pct", "ndvi_pct"]],
                      left_on=["pcode", "month"], right_on=["county_pcode", "month"], how="left")
               .merge(el_nino, on="month", how="left"))
    weather = weather[["series", "month"] + WEATHER].rename(columns={"month": "origin_month"})

    # related crops: average recent change of the other crops in the same group and county
    rel = prices[["series", "county", "commodity", "month"]].copy()
    rel["group"] = rel["commodity"].map(CROP_GROUPS)
    for k in [1, 3]:
        rel["chg"] = X[f"change_{k}m"].values
        group_sum = rel.groupby(["county", "group", "month"])["chg"].transform("sum")
        group_n = rel.groupby(["county", "group", "month"])["chg"].transform("count")
        own_sum = rel.groupby(["county", "commodity", "month"])["chg"].transform("sum")
        own_n = rel.groupby(["county", "commodity", "month"])["chg"].transform("count")
        rel[f"related_change_{k}m"] = ((group_sum - own_sum) / (group_n - own_n)).replace([np.inf, -np.inf], np.nan)
    related = rel[["series", "month"] + RELATED].rename(columns={"month": "origin_month"})

    # economy, each lagged to when it is published
    pct = lambda col, k: drivers[col].pct_change(k, fill_method=None)
    macro = pd.DataFrame({"origin_month": drivers["month"]})
    macro["murban_change_3m"] = pct("Murban crude (USD/bbl)", 3)
    macro["diesel_change_3m"] = pct("Diesel (KES/L)", 3)
    macro["shilling_change_3m"] = pct("KES per USD", 3)
    macro["shilling_change_12m"] = pct("KES per USD", 12)
    macro["inflation_12m"] = drivers["Overall inflation, 12-month (%)"].shift(1)
    macro["cpi_change_3m"] = pct("CPI index (Feb 2019=100, chained)", 3).shift(1)

    for part in [weather, related]:
        X = X.merge(part, on=["series", "origin_month"], how="left")
    return X.merge(macro, on="origin_month", how="left")


def as_categories(table, prices):
    """Commodity, price type and county as labels, with the same label list in training and forecasting."""
    for col in ["commodity", "price_type", "county"]:
        table[col] = pd.Categorical(table[col], categories=sorted(prices[col].unique()))
    return table


def learning_rows(prices, X, train_end):
    """The answers the models learn from: every (series, origin, horizon) whose target month is <= train_end."""
    g = prices.groupby("series")
    parts = []
    for h in HORIZONS:
        part = prices[["series", "county", "commodity", "price_type", "month", "price_kes"]].rename(
            columns={"month": "origin_month", "price_kes": "origin_price"})
        part["horizon"] = h
        part["target_month"] = part["origin_month"] + pd.DateOffset(months=h)
        part["target_price"] = g["price_kes"].shift(-h).values
        part["target_observed"] = g["observed"].shift(-h).values
        part["season_price"] = g["price_kes"].shift(12 - h).values      # kept so the rows match the notebook's
        parts.append(part)
    T = pd.concat(parts, ignore_index=True).dropna(subset=["target_price", "season_price"])
    T["target_observed"] = T["target_observed"].astype(bool)
    T["target_change"] = T["target_price"] / T["origin_price"] - 1
    T = as_categories(T.merge(X, on=["series", "origin_month"], how="left"), prices)
    return T[T["target_month"] <= pd.Timestamp(train_end)].reset_index(drop=True)


def forecast_rows(prices, X, origins):
    """One row per series, origin month and horizon to forecast."""
    F = prices.loc[prices["month"].isin(origins), ["series", "county", "commodity", "price_type", "month", "price_kes", "observed"]]
    F = F.rename(columns={"month": "origin_month", "price_kes": "origin_price", "observed": "origin_observed"})
    F = pd.concat([F.assign(horizon=h, target_month=F["origin_month"] + pd.DateOffset(months=h)) for h in HORIZONS],
                  ignore_index=True)
    return as_categories(F.merge(X, on=["series", "origin_month"], how="left"), prices)


def balancing_weights(L):
    obs = L["target_observed"].values
    balancing = max((~obs).sum() / max(obs.sum(), 1), 1)          # estimated rows / observed rows
    return np.where(obs, balancing, 1.0)


# --------------------------------------------------------------------------- 3. the three models
def arima_forecast(prices, F, train_end):
    """ARIMA per series: parameters fitted on log prices up to train_end, then run forward to each origin month."""
    from statsmodels.tsa.arima.model import ARIMA
    log_price = np.log(prices.pivot(index="month", columns="series", values="price_kes"))
    commodity_of = prices.groupby("series")["commodity"].first()
    out = pd.Series(F["origin_price"].values, index=F.index, dtype=float)       # naive if a fit fails
    failed = 0
    last_origin = F["origin_month"].max()
    for series, rows in F.groupby("series"):
        y = log_price[series].dropna()
        train, upto = y[:pd.Timestamp(train_end)], y[:last_origin]
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fitted = ARIMA(train.values, order=ARIMA_ORDERS.get(commodity_of[series], ARIMA_DEFAULT)).fit()
                filtered = fitted.apply(upto.values)
            for origin, orows in rows.groupby("origin_month"):
                t = upto.index.get_loc(origin)
                path = filtered.get_prediction(start=t + 1, end=t + max(HORIZONS), dynamic=True).predicted_mean
                out[orows.index] = np.exp(path[orows["horizon"].values - 1])
        except Exception:
            failed += 1
    return out.values, failed


def m8a_probability(T, F):
    """Chance of a spike (rise > 15%), one LightGBM classifier per horizon on the price-history inputs."""
    prob = pd.Series(np.nan, index=F.index)
    for h in HORIZONS:
        L, rows = T[T["horizon"] == h], F[F["horizon"] == h]
        fitted = lgb.LGBMClassifier(**M8A_PARAMS).fit(L[M1_FEATURES], (L["target_change"] > SPIKE).astype(int),
                                                       sample_weight=balancing_weights(L))
        prob[rows.index] = fitted.predict_proba(rows[M1_FEATURES])[:, 1]
    return prob.values


def m5_drivers(T, F):
    """M5's predicted change and each input's push on it (SHAP values, in percentage points), per horizon."""
    change = pd.Series(np.nan, index=F.index)
    pushes = pd.DataFrame(np.nan, index=F.index, columns=M4_FEATURES + ["baseline"])
    for h in HORIZONS:
        L, rows = T[T["horizon"] == h], F[F["horizon"] == h]
        fitted = lgb.LGBMRegressor(**M5_PARAMS[h]).fit(L[M4_FEATURES], L["target_change"], sample_weight=balancing_weights(L))
        contrib = fitted.predict(rows[M4_FEATURES], pred_contrib=True)
        pred = fitted.predict(rows[M4_FEATURES])
        assert np.allclose(contrib.sum(axis=1), pred), "SHAP values must add up to the prediction"
        change[rows.index] = pred
        pushes.loc[rows.index] = contrib * 100
    return change.values, pushes


# --------------------------------------------------------------------------- 4. the output table
def build_output(F, arima_price, prob, m5_change, pushes):
    out = F[["series", "county", "commodity", "price_type", "origin_month", "horizon", "target_month",
             "origin_price", "origin_observed"]].copy()
    for col in ["county", "commodity", "price_type"]:
        out[col] = out[col].astype(str)
    out = out.rename(columns={"origin_price": "origin_price_kes", "origin_observed": "origin_price_is_real"})
    out["forecast_price_kes"] = np.round(arima_price, 2)
    out["forecast_change_pct"] = np.round((arima_price / F["origin_price"].values - 1) * 100, 2)
    out["spike_probability_pct"] = np.round(prob * 100, 1)
    out["alert_threshold_pct"] = np.round(F["horizon"].map(ALERT_THRESHOLD).values * 100, 1)
    out["alert"] = prob > F["horizon"].map(ALERT_THRESHOLD).values
    out["alert_message"] = np.where(out["alert"], "Higher risk of a price rise above 15%", "")
    out["drivers_model_change_pct"] = np.round(m5_change * 100, 2)
    out["drivers_typical_change_pct"] = np.round(pushes["baseline"].values, 2)
    P = pushes[M4_FEATURES]
    order = np.argsort(-P.abs().values, axis=1)[:, :3]
    for k in range(3):
        cols = np.array(M4_FEATURES)[order[:, k]]
        out[f"driver_{k + 1}"] = [DRIVER_LABELS[c][1] for c in cols]
        out[f"driver_{k + 1}_family"] = [DRIVER_LABELS[c][0] for c in cols]
        out[f"driver_{k + 1}_push_points"] = np.round(P.values[np.arange(len(P)), order[:, k]], 2)
    out["origin_month"] = out["origin_month"].dt.strftime("%Y-%m")
    out["target_month"] = out["target_month"].dt.strftime("%Y-%m")
    return out.sort_values(["series", "origin_month", "horizon"]).reset_index(drop=True)


def where_to_buy(out, prices, raw):
    """Daniella's rule: for each alert, the cheapest nearby county on recent real prices, if at least MIN_GAIN cheaper.
    Each county's price is its latest real price within RECENT_MONTHS up to the origin month; its month is shown."""
    cty = pd.read_csv(raw / "counties.csv")[["county", "center_lat", "center_lon"]]
    lat, lon = np.radians(cty["center_lat"].values), np.radians(cty["center_lon"].values)
    a = (np.sin((lat[:, None] - lat[None, :]) / 2) ** 2
         + np.cos(lat)[:, None] * np.cos(lat)[None, :] * np.sin((lon[:, None] - lon[None, :]) / 2) ** 2)
    km = 2 * 6371 * np.arcsin(np.sqrt(a))
    i, j = np.where((km <= RADIUS_KM) & (km > 0))
    pairs = pd.DataFrame({"county": cty["county"].values[i], "buy_county": cty["county"].values[j], "km": km[i, j]})

    # latest real price per county, commodity and price type, within the recent window of each origin month
    real = prices.loc[prices["observed"] & prices["commodity"].isin(RECOMMEND_CROPS),
                      ["county", "commodity", "price_type", "month", "price_kes"]]
    parts = []
    for origin in pd.to_datetime(out["origin_month"].unique()):
        r = real[(real["month"] <= origin) & (real["month"] > origin - pd.DateOffset(months=RECENT_MONTHS))]
        r = r.sort_values("month").groupby(["county", "commodity", "price_type"]).tail(1)
        parts.append(r.assign(origin_month=f"{origin:%Y-%m}", month=r["month"].dt.strftime("%Y-%m")))
    latest = pd.concat(parts, ignore_index=True)

    key = ["series", "origin_month", "horizon"]
    A = out.loc[out["alert"] & out["commodity"].isin(RECOMMEND_CROPS), key + ["county", "commodity", "price_type"]]
    A = A.merge(latest.rename(columns={"price_kes": "base_price", "month": "base_month"}),
                on=["county", "commodity", "price_type", "origin_month"])           # the alert county's own recent real price
    C = A.merge(pairs, on="county").merge(
        latest.rename(columns={"county": "buy_county", "price_kes": "buy_price", "month": "buy_month"}),
        on=["buy_county", "commodity", "price_type", "origin_month"])
    C = C[C["buy_price"] <= C["base_price"] * (1 - MIN_GAIN)]
    best = C.loc[C.groupby(key)["buy_price"].idxmin()]
    best = pd.DataFrame({**{k: best[k] for k in key},
                         "where_to_buy_county": best["buy_county"],
                         "where_to_buy_price_kes": best["buy_price"].round(2),
                         "where_to_buy_price_month": best["buy_month"],
                         "where_to_buy_compared_with_kes": best["base_price"].round(2),
                         "where_to_buy_compared_month": best["base_month"],
                         "where_to_buy_saving_pct": ((1 - best["buy_price"] / best["base_price"]) * 100).round(1),
                         "where_to_buy_km": best["km"].round(0)})
    return out.merge(best, on=key, how="left")


def chart(out, prices, series_list, path):
    """One panel per series: price history, the 1-month alerts in this run, and the forecast path from the last origin.
    Where the real price one month later is already known, each alert month is marked caught / false alarm / missed."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    unknown = [s for s in series_list if s not in set(out["series"])]
    if unknown:
        sys.exit(f"Unknown series for --chart: {unknown}. Use the exact name, e.g. 'Nakuru | Dry Maize | Wholesale'.")
    INK, MUTED, BLUE, RED, GRID = "#1f1f1e", "#8a8985", "#2a6fdb", "#e34948", "#e6e5e1"
    real = prices.set_index(["series", "month"])
    fig, axes = plt.subplots(len(series_list), 1, figsize=(13, 3.6 * len(series_list)), squeeze=False)
    for ax, s in zip(axes[:, 0], series_list):
        o = out[(out["series"] == s)].copy()
        o["origin"] = pd.to_datetime(o["origin_month"])
        first, last = o["origin"].min(), o["origin"].max()
        p = prices[(prices["series"] == s) & (prices["month"] >= first - pd.DateOffset(months=24))]
        ax.plot(p["month"], p["price_kes"], color="#c9c8c3", lw=1.2, label="price (estimated months included)")
        r = p[p["observed"]]
        ax.plot(r["month"], r["price_kes"], "o", ms=3, color=MUTED, label="real market price")
        h1 = o[o["horizon"] == 1]
        nxt = [real["observed"].get((s, t + pd.DateOffset(months=1)), False) for t in h1["origin"]]
        later = [real["price_kes"].get((s, t + pd.DateOffset(months=1)), np.nan) for t in h1["origin"]]
        h1 = h1.assign(known=nxt, spike=np.array(later) / h1["origin_price_kes"].values - 1 > SPIKE)
        k, u = h1[h1["known"]], h1[~h1["known"] & h1["alert"]]
        groups = [(k[k["alert"] & k["spike"]], dict(marker="o", ms=9, color=BLUE, mec="white", mew=1.5), "alert, and a spike came (caught)"),
                  (k[k["alert"] & ~k["spike"]], dict(marker="o", ms=8, mfc="none", mec=MUTED, mew=1.3), "alert, no spike (false alarm)"),
                  (k[~k["alert"] & k["spike"]], dict(marker="x", ms=9, color=RED, mew=2), "spike came, no alert (missed)"),
                  (u, dict(marker="^", ms=10, color=BLUE), "alert, no real price next month to check it")]
        for g, style, label in groups:
            if len(g):
                ax.plot(g["origin"], g["origin_price_kes"], ls="none", label=label, **style)
        path_rows = o[o["origin"] == last].sort_values("horizon")
        ax.plot([last] + list(pd.to_datetime(path_rows["target_month"])),
                [path_rows["origin_price_kes"].iloc[0]] + list(path_rows["forecast_price_kes"]),
                ls="--", color=INK, lw=1.2, label=f"ARIMA forecast from {last:%b %Y}")
        n_spikes = int(k["spike"].sum())
        caught = int((k["alert"] & k["spike"]).sum())
        note = f"caught {caught} of {n_spikes} known spikes, {int((k['alert'] & ~k['spike']).sum())} false alarms" if len(k) else "no outcomes known yet"
        ax.set_title(f"{s}:  {note}", loc="left", fontsize=11, color=INK)
        ax.set_ylabel("KES per kg", color=INK)
        ax.grid(axis="y", color=GRID)
        ax.set_axisbelow(True)
        for side in ["top", "right"]:
            ax.spines[side].set_visible(False)
        ax.legend(loc="upper left", fontsize=8, frameon=False, ncol=2)
    title = "SokoRadar: spike alerts 1 month ahead (M8a) and the price forecast (ARIMA)"
    if pd.to_datetime(out["origin_month"]).min() < pd.Timestamp(TRAIN_END):
        title += f"\nIn-sample: the models learned outcomes up to {TRAIN_END}, so past alerts here look better than a fair test"
    fig.suptitle(title, x=0.01, ha="left", fontsize=13, color=INK)
    plt.tight_layout(rect=(0, 0, 1, 1 - 0.05 * title.count("\n") / len(series_list) - 0.03))
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=110)
    plt.close(fig)
    print(f"Chart saved to {path.resolve()}")


def write_safely(df, path, info):
    """Write to a temporary file first and rename it once complete, so a crash never leaves a half-written file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    df.to_csv(tmp, index=False, encoding="utf-8-sig")     # utf-8-sig so Excel shows "El Niño" correctly
    os.replace(tmp, path)
    meta, tmp = path.with_suffix(".json"), path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(info, indent=1, default=str))
    os.replace(tmp, meta)


# --------------------------------------------------------------------------- 5. run
def parse_origins(text, prices):
    if not text:
        return [prices["month"].max()]
    start, _, end = text.partition(":")
    return list(pd.date_range(pd.Timestamp(start), pd.Timestamp(end or start), freq="MS"))


def run(prices, raw, origins, train_end):
    available = set(prices["month"])
    missing = [o for o in origins if o not in available]
    if missing:
        sys.exit(f"No prices for origin month(s) {[f'{m:%Y-%m}' for m in missing]}")
    if min(origins) < pd.Timestamp(train_end):
        print(f"Note: origin months before {train_end} have answers the models learned from (in-sample, not a fair test).")
    t0 = time.time()
    X = build_inputs(prices, raw)
    T = learning_rows(prices, X, train_end)
    F = forecast_rows(prices, X, origins)
    print(f"Learning from {len(T):,} answers (target months up to {train_end}); forecasting {len(F):,} rows "
          f"({F['series'].nunique():,} series x {len(origins)} origin month(s) x {len(HORIZONS)} horizons)")
    gaps = F[WEATHER + ECONOMY].isna().mean()
    if gaps.max() > 0.5:
        print(f"Warning: weather/economy inputs missing for most forecast rows ({', '.join(gaps[gaps > 0.5].index)}); "
              "update the raw files. Forecasts and alerts are unaffected; the driver list leans on what is there.")
    print("1/3 ARIMA forecasts ...", flush=True)
    arima_price, failed = arima_forecast(prices, F, train_end)
    print(f"    done ({failed} series fell back to 'no change' because the fit failed)")
    print("2/3 Spike probabilities (M8a) ...", flush=True)
    prob = m8a_probability(T, F)
    print("3/3 Drivers (M5 + SHAP) ...", flush=True)
    m5_change, pushes = m5_drivers(T, F)
    out = where_to_buy(build_output(F, arima_price, prob, m5_change, pushes), prices, raw)
    n_alert = int((out["alert"] & out["commodity"].isin(RECOMMEND_CROPS)).sum())
    print(f"Where to buy: {int(out['where_to_buy_county'].notna().sum()):,} suggestions for {n_alert:,} alerts on the four "
          f"recommender crops (a suggestion needs real prices from the last {RECENT_MONTHS} months in both counties)")
    print(f"Finished in {time.time() - t0:.0f} s")
    return out


def check(prices, raw, forecasts_dir):
    """Rebuild the 2024 backtest year (learning from answers up to Dec 2023) and compare with the notebook's saved forecasts."""
    print("Check: rebuilding the 2024 backtest year and comparing with the notebook's saved forecasts")
    origins = list(pd.date_range("2024-01-01", "2024-12-01", freq="MS"))
    out = run(prices, raw, origins, "2023-12")
    key = ["series", "origin_month", "horizon"]
    def saved(name):
        f = pd.read_pickle(Path(forecasts_dir) / f"{name}.pkl")
        f = f[f["fold year"] == 2024].copy()
        f["origin_month"] = f["origin_month"].dt.strftime("%Y-%m")
        return f[key + ["forecast_price", "origin_price"]]
    both = (out.merge(saved("ARIMA_Daniella_s_orders").rename(columns={"forecast_price": "nb_arima"}), on=key)
               .merge(saved("M8a_full_balancing_weight")[key + ["forecast_price"]].rename(columns={"forecast_price": "nb_prob"}), on=key)
               .merge(saved("M5_tuned_by_time_series_CV")[key + ["forecast_price"]].rename(columns={"forecast_price": "nb_m5"}), on=key))
    expected = (out["target_month"] <= "2024-12").sum()           # the notebook only keeps rows whose answer is known
    assert len(both) == expected, f"only {len(both):,} of {expected:,} rows matched the notebook's rows"
    print(f"    {len(both):,} rows compared")
    # The CSV rounds prices to 0.01 KES, probabilities to 0.1 point and changes to 0.01 point,
    # so each difference is allowed up to half its rounding step (plus a hair for floating point).
    nb_alert = both["nb_prob"] > both["horizon"].map(ALERT_THRESHOLD)
    nb_m5_change = (both["nb_m5"] / both["origin_price"] - 1) * 100
    results = {   # name: (largest difference, allowed)
        "ARIMA forecast, largest difference (KES)": ((both["forecast_price_kes"] - both["nb_arima"]).abs().max(), 0.0051),
        "spike probability, largest difference (points)": ((both["spike_probability_pct"] - both["nb_prob"] * 100).abs().max(), 0.051),
        "M5 predicted change, largest difference (points)": ((both["drivers_model_change_pct"].clip(lower=-90) - nb_m5_change).abs().max(), 0.0051),
        "alerts that disagree (rows)": ((both["alert"] != nb_alert).sum(), 0),
    }
    for k, (v, allowed) in results.items():
        print(f"    {k}: {v:.4f} (allowed {allowed})")
    ok = all(v <= allowed for v, allowed in results.values())
    print("CHECK PASSED: the script reproduces the notebook" if ok else "CHECK FAILED: the script does not match the notebook")
    return ok


def main():
    ap = argparse.ArgumentParser(description="SokoRadar: price forecasts, spike alerts and drivers")
    ap.add_argument("--data", help=f"path to {DATA_FILE} (default: found automatically)")
    ap.add_argument("--raw", help="folder with counties.csv, county_climate_chirps_ndvi.csv, national_drivers_2006_2024.csv")
    ap.add_argument("--origins", help="origin month or range, e.g. 2024-12 or 2024-06:2024-12 (default: latest month in the data)")
    ap.add_argument("--out", default="sokoradar_forecasts.csv", help="output CSV (default: sokoradar_forecasts.csv)")
    ap.add_argument("--check", action="store_true", help="rebuild the 2024 backtest year and compare with the notebook")
    ap.add_argument("--forecasts-dir", help="folder with the notebook's saved forecasts (for --check; default: <output>/forecasts)")
    ap.add_argument("--chart", nargs="+", metavar="SERIES",
                    help="also draw a chart for these series, e.g. --chart \"Nakuru | Dry Maize | Wholesale\" (saved next to the CSV)")
    args = ap.parse_args()

    data_path, raw = find_data(args.data, args.raw)
    print(f"Prices: {data_path}\nRaw data: {raw}")
    prices = load_prices(data_path)
    print(f"{len(prices):,} rows, {prices['series'].nunique():,} series, {prices['month'].min():%b %Y} to {prices['month'].max():%b %Y}")

    if args.check:
        sys.exit(0 if check(prices, raw, args.forecasts_dir or data_path.parent / "forecasts") else 1)

    origins = parse_origins(args.origins, prices)
    out = run(prices, raw, origins, TRAIN_END)
    info = {"created": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"), "train_end": TRAIN_END,
            "origin_months": [f"{o:%Y-%m}" for o in origins], "rows": len(out),
            "data_file": str(data_path), "data_rows": len(prices), "data_modified": round(os.path.getmtime(data_path)),
            "alert_threshold": ALERT_THRESHOLD, "arima_orders": ARIMA_ORDERS, "m5_params": M5_PARAMS, "m8a_params": M8A_PARAMS,
            "versions": {"lightgbm": lgb.__version__, "pandas": pd.__version__, "numpy": np.__version__}}
    path = Path(args.out)
    write_safely(out, path, info)
    print(f"Saved {len(out):,} rows to {path.resolve()} (alerts: {int(out['alert'].sum()):,})")
    if args.chart:
        chart(out, prices, args.chart, path.with_name(path.stem + "_chart.png"))


if __name__ == "__main__":
    main()
