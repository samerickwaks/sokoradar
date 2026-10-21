"""SokoRadar's adaptive ensemble.

For each horizon and segment (commodity x price type), every candidate forecast gets a weight
proportional to 1 / MAE^2 of its log errors over the previous `window` backtest years; rows
where a candidate has no forecast (e.g. SARIMAX on a short series) renormalise the remaining
weights. Segments without enough history use equal weights over the default components.
Weights for year Y only use errors from years before Y.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import settings

SEGMENT = ["commodity", "pricetype"]


def _weights_from(past: pd.DataFrame, candidates: list[str], min_rows: int) -> dict[str, float] | None:
    maes = {}
    for c in candidates:
        col = f"pred_{c}"
        if col not in past:
            continue
        ok = past[col].notna() & past["y_true"].notna()
        if ok.sum() >= min_rows:
            maes[c] = float(np.mean(np.abs(past.loc[ok, col] - past.loc[ok, "y_true"])))
    if not maes:
        return None
    w = {c: 1.0 / max(m, 1e-6) ** 2 for c, m in maes.items()}
    total = sum(w.values())
    return {c: v / total for c, v in w.items()}


def combine(frame: pd.DataFrame, weights: dict[str, float]) -> np.ndarray:
    num = np.zeros(len(frame))
    den = np.zeros(len(frame))
    for c, w in weights.items():
        col = f"pred_{c}"
        if col not in frame:
            continue
        v = frame[col].to_numpy(dtype=float)
        ok = ~np.isnan(v)
        num[ok] += w * v[ok]
        den[ok] += w
    with np.errstate(invalid="ignore"):
        return np.where(den > 0, num / den, np.nan)


def default_weights() -> dict[str, float]:
    comps = settings()["ensemble"]["default"]
    return {c: 1.0 / len(comps) for c in comps}


def adaptive_ensemble(preds: pd.DataFrame, window: int | None = None) -> tuple[pd.Series, pd.DataFrame]:
    """Out-of-sample ensemble forecast for every backtest row, plus the weights used."""
    cfg = settings()["ensemble"]
    window = window or cfg["window"]
    out = pd.Series(np.nan, index=preds.index)
    log = []
    for (h, fold), g in preds.groupby(["h", "fold"]):
        past_all = preds[(preds["h"] == h) & preds["fold"].between(fold - window, fold - 1)]
        for seg, gs in g.groupby(SEGMENT):
            past = past_all[(past_all["commodity"] == seg[0]) & (past_all["pricetype"] == seg[1])]
            w = _weights_from(past, cfg["candidates"], cfg["min_rows"]) or default_weights()
            out.loc[gs.index] = combine(gs, w)
            log += [{"fold": fold, "h": h, "commodity": seg[0], "pricetype": seg[1], "model": c, "weight": v}
                    for c, v in w.items()]
    return out, pd.DataFrame(log)


def production_weights(preds: pd.DataFrame, last_fold: int, window: int | None = None) -> pd.DataFrame:
    """Weights for live forecasts, from the most recent `window` backtest years."""
    cfg = settings()["ensemble"]
    window = window or cfg["window"]
    past_all = preds[preds["fold"].between(last_fold - window + 1, last_fold)]
    rows = []
    for (h, com, pt), past in past_all.groupby(["h"] + SEGMENT):
        w = _weights_from(past, cfg["candidates"], cfg["min_rows"]) or default_weights()
        rows += [{"h": h, "commodity": com, "pricetype": pt, "model": c, "weight": v} for c, v in w.items()]
    return pd.DataFrame(rows)


def weights_for(table: pd.DataFrame, h: int, commodity: str, pricetype: str) -> dict[str, float]:
    t = table[(table["h"] == h) & (table["commodity"] == commodity) & (table["pricetype"] == pricetype)]
    return dict(zip(t["model"], t["weight"])) if len(t) else default_weights()
