"""Per-market statistical models: SARIMAX with climate and exchange-rate regressors, and Prophet.

Both work on log prices. Regressors are lagged three months, so their values for t+1..t+3
are already known at the forecast origin t and no regressor has to be forecast.
SARIMAX handles missing months natively through the Kalman filter.
"""

from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd

EXOG = ["ex_rain3", "ex_ndvi", "ex_lfx"]
MIN_TRAIN_OBS = 24


def stat_exog(climate: pd.DataFrame, macro: pd.DataFrame) -> pd.DataFrame:
    """Regressors by county and month, each lagged three months."""
    c = climate[["county_pcode", "month", "rain_3m_pct", "ndvi_pct"]].copy()
    frames = []
    for pc, g in c.groupby("county_pcode"):
        idx = pd.date_range(g["month"].min(), g["month"].max() + pd.DateOffset(months=3), freq="MS")
        g = g.set_index("month").reindex(idx)
        frames.append(pd.DataFrame({
            "county_pcode": pc, "month": idx,
            "ex_rain3": ((g["rain_3m_pct"] - 100) / 100).shift(3).to_numpy(),
            "ex_ndvi": ((g["ndvi_pct"] - 100) / 100).shift(3).to_numpy(),
        }))
    ex = pd.concat(frames, ignore_index=True)
    m = macro.set_index("month")["kes_usd"].sort_index()
    idx = pd.date_range(m.index.min(), m.index.max() + pd.DateOffset(months=3), freq="MS")
    lfx = np.log(m.reindex(idx).ffill()).shift(3)
    ex = ex.merge(pd.DataFrame({"month": idx, "ex_lfx": lfx.to_numpy()}), on="month", how="left")
    return ex


def _exog_for(ex: pd.DataFrame, county: str, months: pd.DatetimeIndex) -> np.ndarray:
    e = ex[ex["county_pcode"] == county].set_index("month")[EXOG].reindex(months)
    e[["ex_rain3", "ex_ndvi"]] = e[["ex_rain3", "ex_ndvi"]].fillna(0.0)
    e["ex_lfx"] = e["ex_lfx"].ffill().bfill()
    if e["ex_lfx"].isna().all():
        e["ex_lfx"] = 0.0
    return e.to_numpy(dtype=float)


def sarimax_series(grid: pd.DataFrame, ex: pd.DataFrame, train_end: pd.Timestamp,
                   origins: list[pd.Timestamp], horizons=(1, 2, 3)) -> pd.DataFrame:
    """Fit on months <= train_end; forecast from each origin, conditioning on prices up to it."""
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    grid = grid.sort_values("month")
    sid, county = grid["series_id"].iloc[0], grid["county_pcode"].iloc[0]
    months = pd.date_range(grid["month"].min(), max(grid["month"].max(), max(origins)), freq="MS")
    y = grid.set_index("month")["logp"].reindex(months)
    train = y[y.index <= train_end]
    if train.notna().sum() < MIN_TRAIN_OBS:
        return pd.DataFrame()
    full_months = pd.date_range(months[0], months[-1] + pd.DateOffset(months=max(horizons)), freq="MS")
    X = _exog_for(ex, county, full_months)
    seasonal = (1, 0, 0, 12) if train.notna().sum() >= 60 else (0, 0, 0, 0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            res = SARIMAX(train.to_numpy(), exog=X[: len(train)], order=(1, 1, 0), seasonal_order=seasonal,
                          enforce_stationarity=False, enforce_invertibility=False).fit(disp=False, maxiter=100)
        except Exception:
            return pd.DataFrame()
        rows = []
        for t in origins:
            i = months.get_loc(t)
            if np.isnan(y.iloc[i]):
                continue
            try:
                r = res.apply(y.iloc[: i + 1].to_numpy(), exog=X[: i + 1])
                fc = r.forecast(steps=max(horizons), exog=X[i + 1: i + 1 + max(horizons)])
            except Exception:
                continue
            for h in horizons:
                rows.append({"series_id": sid, "month": t, "h": h, "pred_sarimax": float(fc[h - 1])})
    return pd.DataFrame(rows)


def prophet_series(grid: pd.DataFrame, train_end: pd.Timestamp, origins: list[pd.Timestamp],
                   horizons=(1, 2, 3)) -> pd.DataFrame:
    """Direct Prophet: one model per horizon with the price h months earlier as a regressor."""
    from prophet import Prophet

    for name in ("cmdstanpy", "prophet"):
        logging.getLogger(name).setLevel(logging.ERROR)
        logging.getLogger(name).disabled = True
    grid = grid.sort_values("month")
    sid = grid["series_id"].iloc[0]
    months = pd.date_range(grid["month"].min(), max(grid["month"].max(), max(origins)), freq="MS")
    y = grid.set_index("month")["logp"].reindex(months)
    rows = []
    for h in horizons:
        df = pd.DataFrame({"ds": months, "y": y.to_numpy(), "lag": y.shift(h).to_numpy()})
        df = df[(df["ds"] <= train_end)].dropna()
        if len(df) < MIN_TRAIN_OBS:
            continue
        fut = pd.DataFrame({"ds": [t + pd.DateOffset(months=h) for t in origins],
                            "lag": [y.get(t, np.nan) for t in origins], "origin": origins}).dropna()
        if fut.empty:
            continue
        m = Prophet(yearly_seasonality=3, weekly_seasonality=False, daily_seasonality=False,
                    changepoint_prior_scale=0.05, uncertainty_samples=0)
        m.add_regressor("lag")
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                m.fit(df)
                fc = m.predict(fut[["ds", "lag"]])
        except Exception:
            continue
        rows += [{"series_id": sid, "month": o, "h": h, "pred_prophet": float(v)}
                 for o, v in zip(fut["origin"], fc["yhat"])]
    return pd.DataFrame(rows)


def run_statistical(panel: pd.DataFrame, ex: pd.DataFrame, train_end: pd.Timestamp, test_origins: pd.DataFrame,
                    use_prophet: bool = True, n_jobs: int = -1) -> pd.DataFrame:
    """SARIMAX (and Prophet) forecasts for every series with origins in `test_origins`."""
    from joblib import Parallel, delayed

    jobs = []
    for sid, o in test_origins.groupby("series_id"):
        grid = panel[panel["series_id"] == sid][["series_id", "month", "logp", "county_pcode"]]
        jobs.append((grid, list(o["month"])))

    def work(grid, origins):
        a = sarimax_series(grid, ex, train_end, origins)
        if use_prophet:
            b = prophet_series(grid, train_end, origins)
            if len(a) and len(b):
                return a.merge(b, on=["series_id", "month", "h"], how="outer")
            return a if len(a) else b
        return a

    out = Parallel(n_jobs=n_jobs)(delayed(work)(g, o) for g, o in jobs)
    out = [d for d in out if len(d)]
    cols = ["series_id", "month", "h", "pred_sarimax"] + (["pred_prophet"] if use_prophet else [])
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=cols)
