"""Crisis replays: rerun the system month by month through past drought years.

For each market series, an episode starts in the first month E when the current-month ALPS
(what WFP's monitoring would show) reaches Alert after a month below it. SokoRadar's warning
month W is the earliest origin in the three months before E at which any of its 1-3 month
forecasts flagged Alert or Crisis. Lead time = E - W months; 0 means no earlier warning
than ALPS itself. All forecasts come from backtest folds trained only on earlier years.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import settings


def replay_window(year: int) -> tuple[pd.Timestamp, pd.Timestamp]:
    return pd.Timestamp(year=year - 1, month=7, day=1), pd.Timestamp(year=year, month=12, day=1)


def episodes(alps: pd.DataFrame, preds: pd.DataFrame, year: int) -> pd.DataFrame:
    alert = settings()["alps"]["thresholds"]["alert"]
    start, end = replay_window(year)
    rows = []
    flags = preds[preds["alert_flag"]].groupby(["series_id", "month"]).size()
    for sid, a in alps[alps["month"].between(start - pd.DateOffset(months=1), end)].groupby("series_id"):
        a = a.sort_values("month").reset_index(drop=True)
        above = a["alps"] >= alert
        for i in range(1, len(a)):
            if above[i] and not above[i - 1] and a["month"][i] >= start:
                onset = a["month"][i]
                window = [onset - pd.DateOffset(months=k) for k in (3, 2, 1)]
                warned = [t for t in window if (sid, t) in flags.index]
                first = warned[0] if warned else None
                lead = 0 if first is None else (onset.year - first.year) * 12 + onset.month - first.month
                peak = a.loc[i:, "alps"].head(6).max()
                rows.append({"year": year, "series_id": sid, "onset": onset, "first_warning": first,
                             "lead_months": lead, "peak_alps_6m": round(float(peak), 2)})
                break  # first episode per series and replay year
    return pd.DataFrame(rows)


def monthly_view(preds: pd.DataFrame, alps: pd.DataFrame, year: int) -> pd.DataFrame:
    """Month-by-month state of every market for the dashboard's replay mode."""
    start, end = replay_window(year)
    p = preds[preds["month"].between(start, end)]
    order = {"Normal": 0, "Stress": 1, "Alert": 2, "Crisis": 3}
    g = p.assign(phase_rank=p["phase_pred"].map(order)).groupby(["series_id", "month"])
    view = g.agg(market=("market", "first"), commodity=("commodity", "first"), pricetype=("pricetype", "first"),
                 county=("county", "first"), max_phase_rank=("phase_rank", "max"), p_alert_max=("p_alert", "max"),
                 flagged=("alert_flag", "any")).reset_index()
    inv = {v: k for k, v in order.items()}
    view["sokoradar_phase"] = view["max_phase_rank"].map(inv)
    cur = alps[["series_id", "month", "alps", "phase"]].rename(columns={"phase": "alps_phase"})
    view = view.merge(cur, on=["series_id", "month"], how="left")
    view["year"] = year
    return view.drop(columns="max_phase_rank")


def run_replays(preds: pd.DataFrame, alps: pd.DataFrame, years=None) -> dict:
    years = years or settings()["backtest"]["replay_years"]
    eps = pd.concat([episodes(alps, preds, y) for y in years], ignore_index=True)
    views = pd.concat([monthly_view(preds, alps, y) for y in years], ignore_index=True)
    summary = []
    for y in years:
        e = eps[eps["year"] == y]
        start, end = replay_window(y)
        p = preds[preds["month"].between(start, end) & preds["alert_flag"]]
        # A flag is a false alarm if the market never reached Alert in the following three months.
        hits = 0
        for r in p.itertuples():
            fut = alps[(alps["series_id"] == r.series_id) & alps["month"].between(
                r.month + pd.DateOffset(months=1), r.month + pd.DateOffset(months=3))]
            hits += bool((fut["alps"] >= settings()["alps"]["thresholds"]["alert"]).any())
        summary.append({
            "year": y, "episodes": len(e), "warned_earlier": int((e["lead_months"] >= 1).sum()) if len(e) else 0,
            "share_warned_earlier": float((e["lead_months"] >= 1).mean()) if len(e) else np.nan,
            "mean_lead_months": float(e["lead_months"].mean()) if len(e) else np.nan,
            "flags": len(p), "flags_followed_by_alert": hits,
        })
    return {"episodes": eps, "monthly": views, "summary": pd.DataFrame(summary)}
