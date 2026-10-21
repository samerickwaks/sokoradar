"""Live forecasts: train on all data, forecast from each series' latest observed month.

Calibration (ensemble weights, conformal correction, alert stacker and thresholds) comes
from the most recent backtest years, so live forecasts use exactly the recipe that was
scored out of sample.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd

from .alps import forecast_alps
from .backtest import GBM_SPECS
from .conformal import apply_correction, cqr_correction
from .config import MODELS_DIR, settings
from .explain import explain_rows
from .features import FEATURES, HORIZONS, origins
from .models.alerts import (AlertStacker, PhaseClassifier, alert_decision, choose_threshold, display_phase,
                            interval_alert_prob)
from .models.baselines import naive, seasonal_naive, smooth
from .models.ensemble import combine, production_weights, weights_for
from .models.gbm import GlobalForecaster
from .models.statistical import run_statistical

log = logging.getLogger(__name__)
BUNDLE = "production.joblib"


def months_between(a: pd.Timestamp, b: pd.Timestamp) -> int:
    return (b.year - a.year) * 12 + b.month - a.month


def build_calibration(bt: pd.DataFrame, route: str, model: str = "sokoradar") -> dict:
    """Production calibration from the latest backtest years (bt is calibrate()-d backtest output)."""
    cfg = settings()["backtest"]
    last = int(bt["fold"].max())
    recent = bt[bt["fold"].between(last - cfg["calibration_years"] + 1, last)]
    calib = {"route": route, "model": model, "last_fold": last, "cqr_q": {}, "tau": {}, "stacker": {}}
    for h in HORIZONS:
        r = recent[recent["h"] == h]
        calib["cqr_q"][h] = cqr_correction(r["score"], settings()["forecast"]["interval_level"])
        lab = r[r["event"].notna() & r["alps_pred"].notna()]
        calib["tau"][h] = choose_threshold(lab[f"p_alert_{route}"], lab["event"].astype(bool))
        all_lab = bt[(bt["h"] == h) & bt["event"].notna() & bt["alps_pred"].notna()]
        calib["stacker"][h] = AlertStacker().fit(all_lab, all_lab["event"].astype(bool))
    calib["weights"] = production_weights(bt, last)
    return calib


def train_production(panel: pd.DataFrame, calib: dict, features: list[str] | None = None) -> dict:
    features = features or FEATURES
    org = origins(panel)
    forecasters = {}
    for kind, anchor in GBM_SPECS:
        m = GlobalForecaster(kind, features, anchor=anchor, quantiles=(0.1, 0.9) if anchor == "naive" else None)
        forecasters[m.name] = m.fit(org)
    bundle = {
        "forecasters": forecasters,
        "classifier": PhaseClassifier(features).fit(org),
        "calibration": calib,
        "features": features,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "data_end": str(org["month"].max().date()),
    }
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, MODELS_DIR / BUNDLE, compress=3)
    return bundle


def load_bundle() -> dict:
    return joblib.load(MODELS_DIR / BUNDLE)


def latest_origins(panel: pd.DataFrame) -> pd.DataFrame:
    return origins(panel).sort_values("month").groupby("series_id").tail(1).reset_index(drop=True)


def component_forecasts(latest: pd.DataFrame, bundle: dict, stat: pd.DataFrame | None = None) -> pd.DataFrame:
    """Every component's log-price forecast for each latest origin and horizon."""
    rows = []
    for h in HORIZONS:
        rows.append(pd.DataFrame({
            "series_id": latest["series_id"].to_numpy(), "month": latest["month"].to_numpy(), "h": h,
            "pred_naive": naive(latest, h), "pred_snaive": seasonal_naive(latest, h), "pred_smooth": smooth(latest, h),
        }))
    comp = pd.concat(rows, ignore_index=True)
    keys = ["series_id", "month", "h"]
    for m in bundle["forecasters"].values():
        comp = comp.merge(m.predict(latest), on=keys, how="left")
    comp = comp.merge(bundle["classifier"].predict_proba(latest)[keys + ["p_alert_clf"]], on=keys, how="left")
    if stat is not None and len(stat):
        comp = comp.merge(stat, on=keys, how="left")
    return comp


def assemble(latest: pd.DataFrame, comp: pd.DataFrame, bundle: dict) -> pd.DataFrame:
    """Ensemble, intervals, ALPS phases and alerts from component forecasts."""
    calib = bundle["calibration"]
    info = latest[["series_id", "market_key", "market", "commodity", "pricetype", "county", "county_pcode", "asal",
                   "pastoral", "latitude", "longitude", "price", "logp", "alps", "alps_sd"]
                  + [f"exp_h{h}" for h in HORIZONS]].rename(columns={"alps": "alps_now", "price": "price_now"})
    f = comp.merge(info, on="series_id", how="left")
    f["pred"] = np.nan
    for (h, com, pt), g in f.groupby(["h", "commodity", "pricetype"]):
        f.loc[g.index, "pred"] = combine(g, weights_for(calib["weights"], h, com, pt))
    f["pred"] = f["pred"].fillna(f["pred_lgbm"])
    f["exp_price"] = [row[f"exp_h{row['h']}"] for _, row in f.iterrows()]
    f = f.drop(columns=[f"exp_h{h}" for h in HORIZONS])
    f["lo"], f["hi"] = np.nan, np.nan
    for h in HORIZONS:
        idx = f["h"] == h
        raw_lo = f.loc[idx, "pred"] + (f.loc[idx, "q_lo"] - f.loc[idx, "pred_lgbm"])
        raw_hi = f.loc[idx, "pred"] + (f.loc[idx, "q_hi"] - f.loc[idx, "pred_lgbm"])
        f.loc[idx, "lo"], f.loc[idx, "hi"] = apply_correction(raw_lo, raw_hi, calib["cqr_q"][h])
    f["alps_pred"] = forecast_alps(np.exp(f["pred"]), f["exp_price"], f["alps_sd"])
    f["p_alert_int"] = interval_alert_prob(f["pred"], f["hi"], f["exp_price"], f["alps_sd"])
    f["p_alert_stack"] = np.nan
    for h in HORIZONS:
        idx = f["h"] == h
        f.loc[idx, "p_alert_stack"] = calib["stacker"][h].predict_proba(f[idx])
    route = calib["route"]
    f["p_alert"] = f[f"p_alert_{route}"]
    f["tau"] = f["h"].map(calib["tau"])
    f["alert_flag"] = alert_decision(f["p_alert"], f["alps_pred"], f["tau"]) & f["alps_pred"].notna()
    f["phase"] = display_phase(f["alert_flag"], f["alps_pred"])
    f["target_month"] = [m + pd.DateOffset(months=int(h)) for m, h in zip(f["month"], f["h"])]
    for c, src in (("price_fcst", "pred"), ("price_lo", "lo"), ("price_hi", "hi")):
        f[c] = np.exp(f[src]).round(2)
    f["change_pct"] = (100 * (np.exp(f["pred"] - f["logp"]) - 1)).round(1)
    return f


def add_explanations(f: pd.DataFrame, latest: pd.DataFrame, bundle: dict) -> pd.DataFrame:
    labels = {k: v["label"] for k, v in settings()["commodities"].items()}
    model = bundle["forecasters"]["lgbm"]
    parts = [explain_rows(model.models[(h, "point")], latest, h, model.features, labels) for h in HORIZONS]
    ex = pd.concat(parts, ignore_index=True)
    return f.merge(ex, on=["series_id", "month", "h"], how="left")


def add_freshness(f: pd.DataFrame, data_end: pd.Timestamp) -> pd.DataFrame:
    cfg = settings()["forecast"]
    f["months_old"] = [months_between(m, data_end) for m in f["month"]]
    f["freshness"] = np.select([f["months_old"] <= cfg["fresh_months"], f["months_old"] <= cfg["live_months"]],
                               ["fresh", "aging"], default="stale")
    f["live"] = f["months_old"] <= cfg["live_months"]
    return f


def tot_outlook(f: pd.DataFrame, tot: pd.DataFrame) -> pd.DataFrame:
    """Goat-to-maize terms of trade now and at t+h, applying SokoRadar's maize forecast."""
    if tot is None or tot.empty:
        return pd.DataFrame()
    last = tot.sort_values("month").groupby("market_key").tail(1)
    maize = f[(f["commodity"] == "maize") & (f["pricetype"] == "Retail")][
        ["market_key", "h", "target_month", "change_pct", "live"]]
    out = last.merge(maize, on="market_key", how="inner")
    out["tot_fcst_kg"] = (out["tot_kg_per_goat"] / (1 + out["change_pct"] / 100)).round(1)
    out["tot_now_kg"] = out["tot_kg_per_goat"].round(1)
    return out[["market_key", "month", "h", "target_month", "tot_now_kg", "tot_fcst_kg", "rtfp_goat", "rtfp_maize",
                "live"]].rename(columns={"month": "tot_month"})


def run_forecast(panel: pd.DataFrame, ex: pd.DataFrame, bundle: dict, tot: pd.DataFrame | None = None,
                 statistical: bool = True) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    latest = latest_origins(panel)
    data_end = panel.loc[panel["observed"], "month"].max()
    stat = None
    weights = bundle["calibration"]["weights"]
    needs_stat = weights["model"].isin(["sarimax", "prophet"]).any() if len(weights) else False
    if statistical and needs_stat:
        use_prophet = bool(weights["model"].eq("prophet").any())
        frames = []
        for t, g in latest.groupby("month"):
            frames.append(run_statistical(panel, ex, t, g[["series_id", "month"]], use_prophet=use_prophet))
        stat = pd.concat(frames, ignore_index=True) if frames else None
    comp = component_forecasts(latest, bundle, stat)
    f = assemble(latest, comp, bundle)
    f = add_explanations(f, latest, bundle)
    f = add_freshness(f, data_end)
    tout = tot_outlook(f, tot)
    return f, latest, tout


def archive(forecasts: pd.DataFrame, existing: pd.DataFrame | None, run_id: str) -> pd.DataFrame:
    cols = ["series_id", "month", "h", "target_month", "pred", "lo", "hi", "p_alert", "alert_flag", "phase",
            "price_fcst", "price_lo", "price_hi"]
    new = forecasts[cols].copy()
    new["run_id"] = run_id
    new["source"] = "live"
    if existing is not None and len(existing):
        existing = existing[existing["run_id"] != run_id]
        new = pd.concat([existing, new], ignore_index=True)
    return new.drop_duplicates(["series_id", "month", "h", "run_id"], keep="last")


def scorecard(arch: pd.DataFrame, prices: pd.DataFrame, alps: pd.DataFrame) -> pd.DataFrame:
    """Score archived forecasts against prices that have since been observed."""
    act = prices[["series_id", "month", "price"]].rename(columns={"month": "target_month", "price": "actual"})
    a = alps[["series_id", "month", "alps"]].rename(columns={"month": "target_month", "alps": "alps_actual"})
    s = arch.merge(act, on=["series_id", "target_month"], how="inner").merge(a, on=["series_id", "target_month"],
                                                                              how="left")
    s["abs_pct_error"] = (100 * np.abs(np.log(s["actual"]) - s["pred"])).round(2)
    s["inside_interval"] = (s["actual"] >= s["price_lo"]) & (s["actual"] <= s["price_hi"])
    s["alert_actual"] = s["alps_actual"] >= settings()["alps"]["thresholds"]["alert"]
    return s
