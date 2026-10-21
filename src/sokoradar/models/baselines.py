"""The bar every model must clear: naive (last month) and seasonal naive (same month last year).

`smooth` (last price averaged with the 3-month mean) is not a proposal baseline; it is a
cheap ensemble component that damps one-off spikes in noisy retail series.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def naive(rows: pd.DataFrame, h: int) -> np.ndarray:
    """Log price forecast for t+h = the price observed at t."""
    return rows["logp"].to_numpy()


def seasonal_naive(rows: pd.DataFrame, h: int) -> np.ndarray:
    """Log price forecast for t+h = the price in the same month a year earlier.

    When that month was not observed the naive forecast stands in (flagged in backtests).
    """
    s = rows[f"snaive_h{h}"].to_numpy()
    return np.where(np.isnan(s), rows["logp"].to_numpy(), s)


def smooth(rows: pd.DataFrame, h: int) -> np.ndarray:
    return rows["smooth_logp"].to_numpy()
