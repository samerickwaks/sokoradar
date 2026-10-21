"""Rolling-origin backtests.

Fold Y trains on every (origin, target) pair whose target month is on or before December of
Y-1, then forecasts from every origin in year Y. Ensemble weights, conformal intervals, the
alert stacker and alert thresholds for fold Y are fitted only on out-of-sample forecasts
from earlier folds, so every number in the backtest is one the system could have produced
at the time.
"""

from __future__ import annotations

import logging
import time

import numpy as np
import pandas as pd

from .alps import phase_code
from .conformal import apply_correction, cqr_correction, cqr_scores
from .config import settings
from .features import FEATURES, HORIZONS, origins
from .models.alerts import (AlertStacker, PhaseClassifier, alert_decision, choose_threshold, display_phase,
                            interval_alert_prob)
from .models.baselines import naive, seasonal_naive, smooth
from .models.ensemble import adaptive_ensemble
from .models.gbm import GlobalForecaster
from .models.statistical import run_statistical

log = logging.getLogger(__name__)

POINT_MODELS = ["naive", "snaive", "smooth", "sarimax", "prophet", "xgb", "lgbm", "lgbm_smooth", "sokoradar"]
ALERT_ROUTES = ["int", "clf", "stack"]
GBM_SPECS = [("lgbm", "naive"), ("lgbm", "smooth"), ("xgb", "naive")]


def training_rows(org: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    """Origins up to the cutoff, with targets that fall after it hidden."""
    tr = org[org["month"] <= cutoff].copy()
    for h in HORIZONS:
        late = tr[f"tgt_month_h{h}"] > cutoff
        tr.loc[late, [f"tgt_h{h}", f"tgt_alps_h{h}"]] = np.nan
    return tr


def base_frame(test: pd.DataFrame) -> pd.DataFrame:
    """Actuals, baselines and ALPS reference values for each test origin and horizon."""
    out = []
    for h in HORIZONS:
        f = pd.DataFrame({
            "series_id": test["series_id"].to_numpy(), "month": test["month"].to_numpy(), "h": h,
            "target_month": test[f"tgt_month_h{h}"].to_numpy(),
            "market_key": test["market_key"].to_numpy(), "market": test["market"].to_numpy(),
            "commodity": test["commodity"].to_numpy(), "pricetype": test["pricetype"].to_numpy(),
            "county": test["county"].to_numpy(), "asal": test["asal"].to_numpy(),
            "logp": test["logp"].to_numpy(), "y_true": (test["logp"] + test[f"tgt_h{h}"]).to_numpy(),
            "alps_now": test["alps"].to_numpy(), "alps_true": test[f"tgt_alps_h{h}"].to_numpy(),
            "exp_price": test[f"exp_h{h}"].to_numpy(), "alps_sd": test["alps_sd"].to_numpy(),
            "pred_naive": naive(test, h), "pred_snaive": seasonal_naive(test, h), "pred_smooth": smooth(test, h),
            "snaive_fallback": test[f"snaive_h{h}"].isna().to_numpy(),
        })
        out.append(f)
    return pd.concat(out, ignore_index=True)


def run_fold(panel: pd.DataFrame, org: pd.DataFrame, ex: pd.DataFrame, year: int, features: list[str],
             statistical: bool, prophet: bool, gbm_specs=GBM_SPECS, classifier: bool = True,
             n_jobs: int = -1) -> tuple[pd.DataFrame, pd.DataFrame]:
    cutoff = pd.Timestamp(year=year - 1, month=12, day=1)
    test = org[org["month"].dt.year == year]
    train = training_rows(org, cutoff)
    if test.empty or train["tgt_h1"].notna().sum() < 50:
        return pd.DataFrame(), pd.DataFrame()
    preds = base_frame(test)
    keys = ["series_id", "month", "h"]
    imps = []
    for kind, anchor in gbm_specs:
        m = GlobalForecaster(kind, features, anchor=anchor, quantiles=(0.1, 0.9) if anchor == "naive" else None)
        m.fit(train)
        preds = preds.merge(m.predict(test), on=keys, how="left")
        imp = m.importance()
        imp["model"], imp["fold"] = m.name, year
        imps.append(imp)
    if classifier:
        clf = PhaseClassifier(features).fit(train)
        preds = preds.merge(clf.predict_proba(test)[keys + ["p_alert_clf"]], on=keys, how="left")
    if statistical:
        st = run_statistical(panel, ex, cutoff, test[["series_id", "month"]], use_prophet=prophet, n_jobs=n_jobs)
        if len(st):
            preds = preds.merge(st, on=keys, how="left")
    preds["fold"] = year
    return preds, (pd.concat(imps, ignore_index=True) if imps else pd.DataFrame())


def run_backtest(panel: pd.DataFrame, ex: pd.DataFrame, years: list[int] | None = None,
                 features: list[str] | None = None, statistical: bool | None = None, prophet: bool | None = None,
                 gbm_specs=GBM_SPECS, classifier: bool = True, n_jobs: int = -1) -> dict:
    cfg = settings()["backtest"]
    years = years or list(range(cfg["first_test_year"], cfg["holdout_year"] + 1))
    statistical = cfg["statistical_models"] if statistical is None else statistical
    prophet = cfg["prophet"] if prophet is None else prophet
    features = features or FEATURES
    org = origins(panel)
    all_preds, all_imp = [], []
    for y in years:
        t0 = time.time()
        p, imp = run_fold(panel, org, ex, y, features, statistical, prophet, gbm_specs, classifier, n_jobs)
        if len(p):
            all_preds.append(p)
            all_imp.append(imp)
        log.info("fold %s: %d forecasts in %.0fs", y, len(p), time.time() - t0)
    preds = pd.concat(all_preds, ignore_index=True)
    for m in POINT_MODELS:
        if f"pred_{m}" not in preds and m != "sokoradar":
            preds[f"pred_{m}"] = np.nan
    if "p_alert_clf" not in preds:
        preds["p_alert_clf"] = np.nan
    preds["pred_sokoradar"], weights = adaptive_ensemble(preds)
    return {"preds": preds, "importance": pd.concat(all_imp, ignore_index=True), "weights": weights}


def calibrate(preds: pd.DataFrame, model: str = "sokoradar", calib_years: int | None = None,
              level: float | None = None) -> pd.DataFrame:
    """Add conformal intervals, alert probabilities, rolling thresholds and flags."""
    calib_years = calib_years or settings()["backtest"]["calibration_years"]
    level = level or settings()["forecast"]["interval_level"]
    alert = settings()["alps"]["thresholds"]["alert"]
    p = preds.copy()
    p["pred"] = p[f"pred_{model}"].fillna(p["pred_lgbm"])
    p["raw_lo"] = p["pred"] + (p["q_lo"] - p["pred_lgbm"])
    p["raw_hi"] = p["pred"] + (p["q_hi"] - p["pred_lgbm"])
    p["score"] = cqr_scores(p["y_true"], p["raw_lo"], p["raw_hi"])
    p["alps_pred"] = (np.exp(p["pred"]) - p["exp_price"]) / p["alps_sd"]
    p["event"] = np.where(p["alps_true"].notna(), p["alps_true"] >= alert, np.nan)
    for c in ("lo", "hi", "cqr_q", "p_alert_int", "p_alert_stack"):
        p[c] = np.nan
    for r in ALERT_ROUTES:
        p[f"tau_{r}"], p[f"flag_{r}"] = np.nan, False
    folds = sorted(p["fold"].unique())
    for h in HORIZONS:
        # Intervals and the interval-route probability, fold by fold.
        for fold in folds:
            cur = (p["h"] == h) & (p["fold"] == fold)
            past = (p["h"] == h) & p["fold"].between(fold - calib_years, fold - 1)
            q = cqr_correction(p.loc[past, "score"], level)
            lo, hi = apply_correction(p.loc[cur, "raw_lo"], p.loc[cur, "raw_hi"], q)
            p.loc[cur, "lo"], p.loc[cur, "hi"], p.loc[cur, "cqr_q"] = lo, hi, q
            p.loc[cur, "p_alert_int"] = interval_alert_prob(p.loc[cur, "pred"], hi, p.loc[cur, "exp_price"],
                                                            p.loc[cur, "alps_sd"])
        # Stacked probability from all earlier folds, then thresholds for every route.
        for fold in folds:
            cur = (p["h"] == h) & (p["fold"] == fold)
            lab_all = (p["h"] == h) & (p["fold"] < fold) & p["event"].notna() & p["alps_pred"].notna()
            st = AlertStacker().fit(p[lab_all], p.loc[lab_all, "event"].astype(bool))
            p.loc[cur, "p_alert_stack"] = st.predict_proba(p[cur])
            lab = lab_all & p["fold"].between(fold - calib_years, fold - 1)
            for r in ALERT_ROUTES:
                tau = choose_threshold(p.loc[lab, f"p_alert_{r}"], p.loc[lab, "event"].astype(bool))
                p.loc[cur, f"tau_{r}"] = tau
                p.loc[cur, f"flag_{r}"] = alert_decision(p.loc[cur, f"p_alert_{r}"], p.loc[cur, "alps_pred"], tau)
    return p


def finalize_alerts(p: pd.DataFrame, route: str) -> pd.DataFrame:
    p = p.copy()
    p["alert_route"] = route
    p["p_alert"] = p[f"p_alert_{route}"]
    p["alert_flag"] = p[f"flag_{route}"].astype(bool) & p["alps_pred"].notna()
    p["phase_pred"] = display_phase(p["alert_flag"], p["alps_pred"])
    p["phase_true_code"] = phase_code(p["alps_true"])
    return p
