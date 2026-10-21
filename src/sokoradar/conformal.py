"""Conformalized quantile regression (CQR) for 80% price intervals.

Romano, Patterson & Candes (2019), the method behind MAPIE's ConformalizedQuantileRegressor.
Scores are taken from earlier out-of-sample forecasts (rolling calibration), so intervals
for a year are calibrated only on errors the system could have seen before that year.
"""

from __future__ import annotations

import numpy as np


def cqr_scores(y, lo, hi) -> np.ndarray:
    """How far each actual fell outside its raw band (negative when inside)."""
    y, lo, hi = (np.asarray(a, float) for a in (y, lo, hi))
    return np.maximum(lo - y, y - hi)


def cqr_correction(scores, level: float = 0.8, min_scores: int = 30) -> float:
    """Finite-sample conformal quantile of the scores: ceil((n+1)*level)/n."""
    s = np.asarray(scores, float)
    s = s[~np.isnan(s)]
    n = len(s)
    if n < min_scores:
        return 0.0
    q = min(1.0, np.ceil((n + 1) * level) / n)
    return float(np.quantile(s, q, method="higher"))


def apply_correction(lo, hi, q: float):
    lo, hi = np.asarray(lo, float) - q, np.asarray(hi, float) + q
    # A negative correction can cross the band; keep it ordered around its midpoint.
    mid = (lo + hi) / 2
    return np.minimum(lo, mid), np.maximum(hi, mid)


def coverage(y, lo, hi) -> float:
    y, lo, hi = (np.asarray(a, float) for a in (y, lo, hi))
    ok = ~np.isnan(y) & ~np.isnan(lo) & ~np.isnan(hi)
    return float(np.mean((y[ok] >= lo[ok]) & (y[ok] <= hi[ok]))) if ok.any() else np.nan
