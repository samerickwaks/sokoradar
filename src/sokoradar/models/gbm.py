"""Global gradient-boosting forecasters: one model per horizon, trained across all markets.

The target is the log price at t+h minus an anchor known at t: the last price ("naive"
anchor) or the smoothed level ("smooth" anchor). One model therefore serves every market and
commodity regardless of price level, and when the features carry no signal the forecast falls
back to its anchor. Point models minimise absolute error. The naive-anchored LightGBM also
fits 10% and 90% quantile models, which conformal calibration turns into 80% intervals.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import settings
from ..features import FEATURES, HORIZONS, design

ANCHORS = {"naive": "logp", "smooth": "smooth_logp"}


def _lgbm(objective: str = "l1", **extra):
    import lightgbm as lgb

    params = dict(settings()["lightgbm"])
    params.update(extra)
    return lgb.LGBMRegressor(objective=objective, random_state=42, n_jobs=4, **params)


def _xgb():
    import xgboost as xgb

    return xgb.XGBRegressor(objective="reg:absoluteerror", random_state=42, n_jobs=4, tree_method="hist",
                            **settings()["xgboost"])


class GlobalForecaster:
    """Direct multi-horizon forecaster. kind = 'lgbm' or 'xgb'; anchor = 'naive' or 'smooth'."""

    def __init__(self, kind: str = "lgbm", features: list[str] | None = None, horizons=HORIZONS,
                 anchor: str = "naive", quantiles: tuple[float, float] | None = (0.1, 0.9), name: str | None = None):
        self.kind = kind
        self.anchor = anchor
        self.features = list(features or FEATURES)
        self.horizons = tuple(horizons)
        self.quantiles = quantiles if kind == "lgbm" else None
        self.name = name or (kind if anchor == "naive" else f"{kind}_{anchor}")
        self.models: dict = {}

    def _target(self, d: pd.DataFrame, h: int) -> np.ndarray:
        return (d["logp"] + d[f"tgt_h{h}"] - d[ANCHORS[self.anchor]]).to_numpy()

    def fit(self, rows: pd.DataFrame) -> "GlobalForecaster":
        for h in self.horizons:
            d = rows[rows[f"tgt_h{h}"].notna()]
            X = design(d, h, self.features)
            point = _lgbm() if self.kind == "lgbm" else _xgb()
            self.models[(h, "point")] = point.fit(X, self._target(d, h))
            if self.quantiles:
                y = d[f"tgt_h{h}"].to_numpy()  # quantiles are anchored on the last price
                for q in self.quantiles:
                    self.models[(h, q)] = _lgbm("quantile", alpha=q).fit(X, y)
        return self

    def predict(self, rows: pd.DataFrame) -> pd.DataFrame:
        """Predicted log prices (and raw quantiles) for t+h, one row per origin and horizon."""
        out = []
        anchor = rows[ANCHORS[self.anchor]].to_numpy()
        for h in self.horizons:
            X = design(rows, h, self.features)
            frame = pd.DataFrame({"series_id": rows["series_id"].to_numpy(), "month": rows["month"].to_numpy(), "h": h,
                                  f"pred_{self.name}": anchor + self.models[(h, "point")].predict(X)})
            if self.quantiles:
                base = rows["logp"].to_numpy()
                lo, hi = (base + self.models[(h, q)].predict(X) for q in self.quantiles)
                frame["q_lo"], frame["q_hi"] = np.minimum(lo, hi), np.maximum(lo, hi)
            out.append(frame)
        return pd.concat(out, ignore_index=True)

    def importance(self) -> pd.DataFrame:
        rows = []
        for (h, kind), m in self.models.items():
            if kind != "point":
                continue
            imp = m.booster_.feature_importance("gain") if self.kind == "lgbm" else m.feature_importances_
            rows += [{"h": h, "feature": f, "gain": float(g)} for f, g in zip(self.features, imp)]
        return pd.DataFrame(rows)
