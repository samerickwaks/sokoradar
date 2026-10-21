"""WFP's Alert for Price Spikes (ALPS), computed without look-ahead.

ALPS = (observed price - expected price) / residual standard deviation, where the expected
price comes from regressing price on a linear time trend and 12 month dummies. WFP's
guidance fits that regression on the whole history, which leaks the future into historical
ALPS values; SokoRadar refits it on an expanding window so the ALPS for month t only uses
prices up to t. Phases: Normal < 0.25 <= Stress < 1 <= Alert < 2 <= Crisis.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import PHASES, settings


def phase_code(alps) -> np.ndarray:
    t = settings()["alps"]["thresholds"]
    a = np.asarray(alps, dtype=float)
    code = np.select([a >= t["crisis"], a >= t["alert"], a >= t["stress"]], [3, 2, 1], default=0)
    return np.where(np.isnan(a), -1, code)


def phase_name(alps) -> np.ndarray:
    codes = phase_code(alps)
    names = np.array(PHASES + [None], dtype=object)
    return names[np.where(codes < 0, 4, codes)]


def _design(trend: np.ndarray, month: np.ndarray) -> np.ndarray:
    dummies = (month[:, None] == np.arange(2, 13)[None, :]).astype(float)
    return np.column_stack([np.ones_like(trend, dtype=float), trend.astype(float), dummies])


def alps_series(months: pd.Series, prices: pd.Series, min_obs: int | None = None,
                horizons: tuple[int, ...] = (1, 2, 3)) -> pd.DataFrame:
    """Expanding-window ALPS for one series of observed monthly prices.

    Returns, for every observed month t with at least `min_obs` prices up to t: the ALPS
    value, the expected price at t, the residual SD, and the expected prices at t+h that a
    forecast must exceed to reach a given phase.
    """
    min_obs = min_obs or settings()["alps"]["min_obs"]
    months = pd.to_datetime(pd.Series(months)).reset_index(drop=True)
    y = np.asarray(prices, dtype=float)
    order = np.argsort(months.values)
    months, y = months.iloc[order].reset_index(drop=True), y[order]
    t0 = months.iloc[0]
    trend = ((months.dt.year - t0.year) * 12 + months.dt.month - t0.month).to_numpy()
    X = _design(trend, months.dt.month.to_numpy())
    rows = []
    for i in range(min_obs - 1, len(y)):
        Xi, yi = X[: i + 1], y[: i + 1]
        beta, *_ = np.linalg.lstsq(Xi, yi, rcond=None)
        resid = yi - Xi @ beta
        dof = max(len(yi) - Xi.shape[1], 1)
        sd = float(np.sqrt(np.sum(resid ** 2) / dof))
        expected = float(X[i] @ beta)
        row = {"month": months.iloc[i], "alps": (y[i] - expected) / sd if sd > 0 else np.nan,
               "expected": expected, "alps_sd": sd}
        for h in horizons:
            fut_month = (months.iloc[i].month - 1 + h) % 12 + 1
            xf = _design(np.array([trend[i] + h]), np.array([fut_month]))[0]
            row[f"exp_h{h}"] = float(xf @ beta)
        rows.append(row)
    return pd.DataFrame(rows)


def alps_panel(prices: pd.DataFrame) -> pd.DataFrame:
    """ALPS for every series in a long price table (series_id, month, price)."""
    out = []
    for sid, g in prices.groupby("series_id", sort=False):
        a = alps_series(g["month"], g["price"])
        if len(a):
            a.insert(0, "series_id", sid)
            out.append(a)
    res = pd.concat(out, ignore_index=True)
    res["phase"] = phase_name(res["alps"])
    return res


def forecast_alps(pred_price, expected_future, sd) -> np.ndarray:
    """ALPS a forecast price would have, against the trend fitted at the forecast origin."""
    return (np.asarray(pred_price, float) - np.asarray(expected_future, float)) / np.asarray(sd, float)
