"""Alert models: will a market's ALPS phase reach Alert (ALPS >= 1) at t+h?

Three routes are compared:
  * interval route - the probability that the forecast price clears the Alert threshold,
    read off the calibrated predictive distribution of the price forecast;
  * classifier route - a class-weighted LightGBM classifier of the ALPS phase at t+h;
  * stacked route - a logistic regression on today's ALPS, the forecast ALPS and the two
    probabilities above, fitted on earlier out-of-sample predictions (calibrated output).
The decision threshold on the alert probability is tuned on past out-of-sample predictions
for recall on Alert and Crisis first, then precision.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

from ..alps import phase_code
from ..config import PHASES, settings
from ..features import FEATURES, HORIZONS, design

Z80 = norm.ppf(0.9)


class PhaseClassifier:
    def __init__(self, features: list[str] | None = None, horizons=HORIZONS):
        self.features = list(features or FEATURES)
        self.horizons = tuple(horizons)
        self.models: dict = {}

    def fit(self, rows: pd.DataFrame) -> "PhaseClassifier":
        import lightgbm as lgb

        params = dict(settings()["lightgbm"])
        params.update(n_estimators=300)
        for h in self.horizons:
            d = rows[rows[f"tgt_alps_h{h}"].notna()]
            y = phase_code(d[f"tgt_alps_h{h}"])
            if len(np.unique(y)) < 2:
                continue
            clf = lgb.LGBMClassifier(class_weight="balanced", random_state=42, n_jobs=4, **params)
            self.models[h] = clf.fit(design(d, h, self.features), y)
        return self

    def predict_proba(self, rows: pd.DataFrame) -> pd.DataFrame:
        out = []
        for h in self.horizons:
            frame = pd.DataFrame({"series_id": rows["series_id"].to_numpy(), "month": rows["month"].to_numpy(), "h": h})
            probs = np.zeros((len(rows), 4))
            if h in self.models:
                m = self.models[h]
                p = m.predict_proba(design(rows, h, self.features))
                for j, c in enumerate(m.classes_):
                    probs[:, int(c)] = p[:, j]
            for k, name in enumerate(PHASES):
                frame[f"p_{name.lower()}"] = probs[:, k]
            frame["p_alert_clf"] = probs[:, 2] + probs[:, 3]
            out.append(frame)
        return pd.concat(out, ignore_index=True)


def interval_alert_prob(pred_log, hi_log, expected_price, alps_sd, alert: float | None = None) -> np.ndarray:
    """P(price at t+h >= expected + alert * sd) under a normal predictive law in log space.

    The spread comes from the upper half of the calibrated 80% interval, so the upper tail
    (the one that matters for spikes) carries its own width.
    """
    alert = settings()["alps"]["thresholds"]["alert"] if alert is None else alert
    pred_log, hi_log = np.asarray(pred_log, float), np.asarray(hi_log, float)
    thr = np.asarray(expected_price, float) + alert * np.asarray(alps_sd, float)
    sigma = np.maximum((hi_log - pred_log) / Z80, 1e-3)
    with np.errstate(invalid="ignore", divide="ignore"):
        p = 1 - norm.cdf((np.log(np.where(thr > 0, thr, np.nan)) - pred_log) / sigma)
    p = np.where(thr <= 0, 1.0, p)
    return np.where(np.isnan(pred_log) | np.isnan(thr), np.nan, p)


def _logit(p) -> np.ndarray:
    p = np.clip(np.asarray(p, float), 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def stack_features(d: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "alps_now": d["alps_now"].clip(-4, 6).to_numpy(),
        "alps_pred": d["alps_pred"].clip(-4, 6).to_numpy(),
        "logit_int": _logit(d["p_alert_int"]),
        "logit_clf": _logit(d["p_alert_clf"]),
    }).fillna(0.0)


class AlertStacker:
    """Logistic regression combining the alert signals; returns calibrated probabilities."""

    def __init__(self):
        self.model = None

    def fit(self, d: pd.DataFrame, event) -> "AlertStacker":
        from sklearn.linear_model import LogisticRegression

        y = np.asarray(event, bool)
        if y.sum() >= 10 and (~y).sum() >= 10:
            self.model = LogisticRegression(C=1.0, max_iter=1000).fit(stack_features(d), y.astype(int))
        return self

    def predict_proba(self, d: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            return d["p_alert_int"].to_numpy(dtype=float)
        return self.model.predict_proba(stack_features(d))[:, 1]

    def coefficients(self) -> dict:
        if self.model is None:
            return {}
        cols = ["alps_now", "alps_pred", "logit_int", "logit_clf"]
        return {"intercept": float(self.model.intercept_[0]), **dict(zip(cols, map(float, self.model.coef_[0])))}


def alert_decision(p_alert, alps_pred, tau) -> np.ndarray:
    """Alert when the route's probability clears its threshold, or when the point forecast itself
    already lands in Alert (so a forecast in Alert territory is never shown as calm).

    The threshold is tuned on the probability alone, so the early-warning sensitivity of the
    alert model is kept and forecasts in Alert territory only add to it.
    """
    alert = settings()["alps"]["thresholds"]["alert"]
    a = np.asarray(alps_pred, float)
    pr = np.nan_to_num(np.asarray(p_alert, float), nan=-1.0)
    return (pr >= np.asarray(tau, float)) | (np.nan_to_num(a, nan=-np.inf) >= alert)


def choose_threshold(p, y, target_recall: float | None = None, min_precision: float | None = None) -> float:
    """Highest threshold that still reaches the recall target (so precision is as high as it can be)."""
    cfg = settings()["alerts"]
    target_recall = cfg["target_recall"] if target_recall is None else target_recall
    p, y = np.asarray(p, float), np.asarray(y, bool)
    ok = ~np.isnan(p)
    p, y = p[ok], y[ok]
    if y.sum() < 5:
        return 0.5
    best = 0.05
    for tau in np.round(np.arange(0.05, 0.96, 0.01), 2):
        flag = p >= tau
        recall = (flag & y).sum() / y.sum()
        if recall >= target_recall:
            best = tau
    return float(best)


def alert_metrics(flag, y) -> dict:
    flag, y = np.asarray(flag, bool), np.asarray(y, bool)
    tp, fp, fn = int((flag & y).sum()), int((flag & ~y).sum()), int((~flag & y).sum())
    recall = tp / (tp + fn) if tp + fn else np.nan
    precision = tp / (tp + fp) if tp + fp else np.nan
    f1 = 2 * precision * recall / (precision + recall) if precision and recall else np.nan
    return {"recall": recall, "precision": precision, "f1": f1, "positives": tp + fn, "flags": tp + fp, "n": len(y)}


def display_phase(alert_flag, alps_point) -> np.ndarray:
    """Phase shown to users: the alert decision sets Alert/Crisis; the point forecast sets Normal/Stress."""
    t = settings()["alps"]["thresholds"]
    a = np.asarray(alps_point, float)
    flag = np.asarray(alert_flag, bool)
    crisis = np.where(a >= t["crisis"], "Crisis", "Alert")
    calm = np.where(a >= t["stress"], "Stress", "Normal")
    out = np.where(flag, crisis, calm).astype(object)
    out[np.isnan(a) & ~flag] = None
    return out
