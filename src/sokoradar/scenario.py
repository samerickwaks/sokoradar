"""What-if rainfall scenarios for the dashboard slider and the API.

The scenario replaces the rainfall of the most recent `months` months in every market's county
with `rain_pct` % of normal, then re-scores the gradient-boosting components and the alert
classifier. SARIMAX and Prophet components keep their stored forecasts: their rainfall
regressor is lagged three months, so recent rain does not reach them within the horizon.
Vegetation is left as observed. This is a sensitivity test of the models, not a weather forecast.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import store
from .forecast import assemble, component_forecasts, load_bundle


def apply_rain_scenario(latest: pd.DataFrame, rain_pct: float, months: int = 3) -> pd.DataFrame:
    s = latest.copy()
    older = s[[f"rain1_l{k}" for k in range(months, 6)]].mean(axis=1).fillna(rain_pct)
    for k in range(months):
        s[f"rain1_l{k}"] = float(rain_pct)
    s["rain3_l0"] = float(rain_pct)
    s["rain6"] = (rain_pct * months + older * (6 - months)) / 6
    return s


def what_if(rain_pct: float, months: int = 3, bundle: dict | None = None, latest: pd.DataFrame | None = None,
            base: pd.DataFrame | None = None) -> pd.DataFrame:
    bundle = bundle or load_bundle()
    latest = store.read_df("latest_features") if latest is None else latest
    base = store.read_df("forecasts") if base is None else base
    scen = apply_rain_scenario(latest, rain_pct, months)
    stat_cols = [c for c in ("pred_sarimax", "pred_prophet") if c in base]
    stat = base[["series_id", "month", "h"] + stat_cols] if stat_cols else None
    f = assemble(scen, component_forecasts(scen, bundle, stat), bundle)
    keep = ["series_id", "h", "phase", "price_fcst", "price_lo", "price_hi", "p_alert", "change_pct"]
    b = base[["series_id", "h", "phase", "price_fcst", "p_alert", "live", "market", "county", "commodity",
              "pricetype", "latitude", "longitude", "target_month", "price_now"]].rename(
        columns={"phase": "phase_base", "price_fcst": "price_base", "p_alert": "p_alert_base"})
    out = b.merge(f[keep], on=["series_id", "h"], how="left")
    out["price_shift_pct"] = (100 * (out["price_fcst"] / out["price_base"] - 1)).round(2)
    out["rain_pct"] = rain_pct
    return out


def sensitivity(levels=(40, 60, 80, 100, 120, 150), months: int = 3) -> pd.DataFrame:
    """Share of live market forecasts in Alert or Crisis under each rainfall level."""
    bundle, latest, base = load_bundle(), store.read_df("latest_features"), store.read_df("forecasts")
    rows = []
    for lv in levels:
        w = what_if(lv, months, bundle, latest, base)
        w = w[w["live"]]
        rows.append({"rain_pct": lv, "alert_or_crisis": int(w["phase"].isin(["Alert", "Crisis"]).sum()),
                     "mean_price_shift_pct": float(np.nanmean(w["price_shift_pct"]))})
    return pd.DataFrame(rows)
