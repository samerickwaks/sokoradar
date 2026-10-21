"""Accuracy metrics for price forecasts, intervals and alerts."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import settings
from .models.alerts import alert_metrics


def mase_scales(panel: pd.DataFrame, folds: list[int]) -> pd.DataFrame:
    """Per series and fold: mean |p_m - p_(m-12)| in KES over history before the fold (MASE denominator)."""
    g = panel[["series_id", "month", "price"]].sort_values(["series_id", "month"]).copy()
    g["sdiff"] = (g["price"] - g.groupby("series_id")["price"].shift(12)).abs()
    whole = g.groupby("series_id")["sdiff"].mean()
    rows = []
    for fold in folds:
        cutoff = pd.Timestamp(year=fold - 1, month=12, day=1)
        hist = g[g["month"] <= cutoff].groupby("series_id")["sdiff"].agg(["mean", "count"])
        for sid in whole.index:
            if sid in hist.index and hist.loc[sid, "count"] >= 6:
                scale = hist.loc[sid, "mean"]
            else:
                scale = whole[sid]
            rows.append({"series_id": sid, "fold": fold, "scale": scale})
    return pd.DataFrame(rows)


def point_metrics(preds: pd.DataFrame, models: list[str], scales: pd.DataFrame | None = None,
                  by: list[str] | None = None, common: bool = True) -> pd.DataFrame:
    """MAE in KES and in % (log), MASE, and MAE relative to the naive and seasonal-naive forecasts."""
    by = by or ["h"]
    cols = [f"pred_{m}" for m in models if f"pred_{m}" in preds and preds[f"pred_{m}"].notna().any()]
    d = preds[preds["y_true"].notna()].copy()
    if common:
        d = d.dropna(subset=cols)
    if scales is not None:
        d = d.merge(scales, on=["series_id", "fold"], how="left")
    y = np.exp(d["y_true"])
    rows = []
    for keys, g in d.groupby(by):
        keys = keys if isinstance(keys, tuple) else (keys,)
        yk = np.exp(g["y_true"])
        base = {}
        for c in ("pred_naive", "pred_snaive"):
            base[c] = np.mean(np.abs(np.exp(g[c]) - yk))
        for c in cols:
            err = np.abs(np.exp(g[c]) - yk)
            row = dict(zip(by, keys))
            row.update({
                "model": c.replace("pred_", ""), "n": len(g),
                "mae_kes": err.mean(), "mae_pct": 100 * np.mean(np.abs(g[c] - g["y_true"])),
                "rel_mae_naive": err.mean() / base["pred_naive"],
                "rel_mae_snaive": err.mean() / base["pred_snaive"],
            })
            if scales is not None:
                row["mase"] = float(np.mean(err / g["scale"]))
            rows.append(row)
    del y
    return pd.DataFrame(rows)


def interval_metrics(p: pd.DataFrame, by: list[str] | None = None) -> pd.DataFrame:
    by = by or ["h"]
    d = p[p["y_true"].notna() & p["lo"].notna()]
    rows = []
    for keys, g in d.groupby(by):
        keys = keys if isinstance(keys, tuple) else (keys,)
        inside = (g["y_true"] >= g["lo"]) & (g["y_true"] <= g["hi"])
        raw_inside = (g["y_true"] >= g["raw_lo"]) & (g["y_true"] <= g["raw_hi"])
        rows.append({**dict(zip(by, keys)), "n": len(g), "coverage": inside.mean(),
                     "raw_quantile_coverage": raw_inside.mean(),
                     "median_width_pct": 100 * float(np.median(np.exp(g["hi"] - g["lo"]) - 1))})
    return pd.DataFrame(rows)


def alert_table(p: pd.DataFrame, routes=("int", "clf", "stack"), by: list[str] | None = None) -> pd.DataFrame:
    by = by or ["h"]
    d = p[p["event"].notna() & p["alps_pred"].notna()]
    rows = []
    for keys, g in d.groupby(by):
        keys = keys if isinstance(keys, tuple) else (keys,)
        alert = settings()["alps"]["thresholds"]["alert"]
        # Onsets: the market is below Alert today but reaches it at t+h. Persistence can never catch these.
        onset = g["event"].astype(bool) & (g["alps_now"] < alert)
        for r in routes:
            m = alert_metrics(g[f"flag_{r}"], g["event"].astype(bool))
            m["onset_recall"] = float((g[f"flag_{r}"].astype(bool) & onset).sum() / onset.sum()) if onset.sum() else np.nan
            rows.append({**dict(zip(by, keys)), "route": r, **m})
        # Persistence: today's ALPS phase carried forward (what current-month ALPS implies).
        now = g["alps_now"] >= alert
        m = alert_metrics(now, g["event"].astype(bool))
        m["onset_recall"] = 0.0
        rows.append({**dict(zip(by, keys)), "route": "persistence", **m})
    return pd.DataFrame(rows)


def in_years(p: pd.DataFrame, years: tuple[int, int]) -> pd.DataFrame:
    return p[p["fold"].between(years[0], years[1])]
