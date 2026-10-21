"""Plain-language explanations of each forecast from SHAP values.

SHAP values of the LightGBM forecaster are summed by driver family (rainfall, vegetation,
price momentum, seasonality, ...). The three families that move the forecast most become one
sentence in English and one in Kiswahili, each with its direction and the number behind it,
e.g. "below-normal rains over the last 3 months (62% of normal) - pushing prices up".
"""

from __future__ import annotations

import calendar

import numpy as np
import pandas as pd

from .features import FAMILY, design

SW_MONTHS = ["Januari", "Februari", "Machi", "Aprili", "Mei", "Juni", "Julai", "Agosti", "Septemba",
             "Oktoba", "Novemba", "Desemba"]


def shap_values(model, X: pd.DataFrame) -> np.ndarray:
    """SHAP values for a fitted LightGBM regressor (TreeSHAP, via LightGBM's own contrib output)."""
    contrib = model.predict(X, pred_contrib=True)
    return np.asarray(contrib)[:, :-1]


def family_contributions(shap: np.ndarray, features: list[str]) -> pd.DataFrame:
    fam = pd.Series([FAMILY.get(f, "other") for f in features])
    df = pd.DataFrame(shap, columns=features)
    return df.T.groupby(fam.values).sum().T


def _pct(x) -> str:
    return f"{abs(100 * x):.0f}%"


def describe(family: str, row: pd.Series, h: int, commodity_label: str) -> tuple[str, str, str, str]:
    """(English phrase, Kiswahili phrase, short English label, short Kiswahili label) for one driver family."""
    tgt = (row["month"].month - 1 + h) % 12 + 1
    m0, m1 = calendar.month_abbr[row["month"].month], calendar.month_abbr[tgt]
    if family == "rainfall":
        v = row.get("rain3_l0", np.nan)
        if np.isnan(v):
            v = row.get("rain6", np.nan)
        if np.isnan(v):
            return "recent rainfall", "mvua ya hivi karibuni", "rainfall", "mvua"
        kind = "below" if v < 90 else "above" if v > 110 else "near"
        sw = {"below": "chini ya", "above": "juu ya", "near": "karibu na"}[kind]
        return (f"{kind}-normal rains over the last 3 months ({v:.0f}% of normal)",
                f"mvua {sw} kawaida miezi 3 iliyopita ({v:.0f}%)", f"{kind}-normal rains ({v:.0f}%)",
                f"mvua {sw} kawaida ({v:.0f}%)")
    if family == "vegetation":
        v = row.get("ndvi_l0", np.nan)
        if np.isnan(v):
            return "crop and pasture condition", "hali ya mimea na malisho", "pasture condition", "hali ya malisho"
        kind = "poor" if v < 95 else "good" if v > 105 else "average"
        sw = {"poor": "duni", "good": "nzuri", "average": "ya wastani"}[kind]
        return (f"{kind} crop and pasture condition (vegetation {v:.0f}% of normal)",
                f"hali {sw} ya mimea na malisho ({v:.0f}%)", f"{kind} pasture ({v:.0f}%)",
                f"malisho {sw} ({v:.0f}%)")
    if family == "price_momentum":
        v = -row.get("lag3", np.nan)
        if np.isnan(v):
            v = -row.get("lag1", np.nan)
        if np.isnan(v):
            return "recent price movements", "mwenendo wa bei", "recent prices", "mwenendo wa bei"
        up = v >= 0
        return (f"prices have {'risen' if up else 'fallen'} {_pct(v)} over the last 3 months",
                f"bei {'imepanda' if up else 'imeshuka'} {_pct(v)} miezi 3 iliyopita",
                f"prices {'up' if up else 'down'} {_pct(v)} in 3 months",
                f"bei {'imepanda' if up else 'imeshuka'} {_pct(v)} miezi 3")
    if family == "price_level":
        a = row.get("alps", np.nan)
        if np.isnan(a):
            return "price level against its usual trend", "kiwango cha bei", "price level", "kiwango cha bei"
        above = a >= 0
        return (f"prices are already {'above' if above else 'below'} their seasonal trend (ALPS {a:+.1f})",
                f"bei tayari {'iko juu' if above else 'iko chini'} ya mwenendo wa msimu (ALPS {a:+.1f})",
                f"prices {'above' if above else 'below'} trend",
                f"bei {'juu' if above else 'chini'} ya kawaida")
    if family == "seasonality":
        v = row.get(f"seas_avg_h{h}", np.nan)
        if np.isnan(v):
            return f"the usual seasonal pattern between {m0} and {m1}", "mwenendo wa msimu", "season", "msimu"
        up = v >= 0
        return (f"prices usually {'rise' if up else 'fall'} about {_pct(v)} between {m0} and {m1}",
                f"kwa kawaida bei {'hupanda' if up else 'hushuka'} kati ya {SW_MONTHS[row['month'].month - 1]} "
                f"na {SW_MONTHS[tgt - 1]}", f"seasonal {'rise' if up else 'fall'}",
                f"msimu wa bei {'kupanda' if up else 'kushuka'}")
    if family == "market_network":
        v = row.get("gap_nbo", np.nan)
        if not np.isnan(v):
            above = v >= 0
            return (f"{commodity_label.lower()} here costs {_pct(v)} {'more' if above else 'less'} than the "
                    f"Nairobi wholesale price",
                    f"bei hapa iko {'juu' if above else 'chini'} kwa {_pct(v)} kuliko bei ya jumla Nairobi",
                    f"{'above' if above else 'below'} Nairobi price",
                    f"bei {'juu' if above else 'chini'} kuliko Nairobi")
        v = row.get("nat_chg3", np.nan)
        if np.isnan(v):
            return "prices in other markets", "bei katika masoko mengine", "other markets", "masoko mengine"
        up = v >= 0
        return (f"{commodity_label.lower()} prices across Kenya {'rose' if up else 'fell'} {_pct(v)} in 3 months",
                f"bei nchini {'imepanda' if up else 'imeshuka'} {_pct(v)} miezi 3",
                f"national prices {'up' if up else 'down'}", f"bei nchini {'inapanda' if up else 'inashuka'}")
    if family == "macro":
        fx, gm, oil = (row.get(c, np.nan) for c in ("fx_chg12", "gmaize_chg12", "crude_chg12"))
        cands = [(abs(x), n, x) for n, x in (("fx", fx), ("maize", gm), ("oil", oil)) if not np.isnan(x)]
        if not cands:
            return ("global prices and the exchange rate", "bei za dunia na thamani ya shilingi", "global prices",
                    "bei za dunia")
        _, name, x = max(cands)
        if name == "fx":
            weak = x > 0
            return (f"a {'weaker' if weak else 'stronger'} shilling ({_pct(x)} against the dollar in a year)",
                    f"shilingi {'imedhoofika' if weak else 'imeimarika'} ({_pct(x)} dhidi ya dola)",
                    f"{'weaker' if weak else 'stronger'} shilling", f"shilingi {'dhaifu' if weak else 'imara'}")
        what = {"maize": ("global maize prices", "bei ya mahindi duniani"),
                "oil": ("fuel (crude oil) prices", "bei ya mafuta")}[name]
        up = x > 0
        return (f"{what[0]} {'up' if up else 'down'} {_pct(x)} in a year",
                f"{what[1]} {'imepanda' if up else 'imeshuka'} {_pct(x)}", f"{what[0]} {'up' if up else 'down'}",
                f"{what[1]} {'juu' if up else 'chini'}")
    if family == "policy":
        return ("a recent government subsidy or import-duty waiver", "ruzuku au msamaha wa ushuru wa hivi karibuni",
                "government policy", "sera ya serikali")
    return "the market's location and type", "eneo na aina ya soko", "market profile", "aina ya soko"


def explain_rows(model, rows: pd.DataFrame, h: int, features: list[str], labels: dict[str, str],
                 top: int = 3) -> pd.DataFrame:
    """Top driver families and sentences for each origin row at horizon h."""
    X = design(rows, h, features)
    fam = family_contributions(shap_values(model, X), features)
    out = []
    for i, (_, row) in enumerate(rows.iterrows()):
        contrib = fam.iloc[i]
        order = contrib.abs().sort_values(ascending=False).index[:top]
        label = labels.get(row["commodity"], row["commodity"])
        en, sw, short, short_sw, drivers = [], [], [], [], []
        for f in order:
            e, s, u, us = describe(f, row, h, label)
            arrow = contrib[f] >= 0
            en.append(f"{e} ({'pushing prices up' if arrow else 'pushing prices down'})")
            sw.append(f"{s} ({'inaongeza bei' if arrow else 'inapunguza bei'})")
            short.append(u)
            short_sw.append(us)
            drivers.append({"family": f, "effect_pct": round(100 * float(contrib[f]), 2)})
        out.append({
            "series_id": row["series_id"], "month": row["month"], "h": h,
            "why_en": "Main reasons: " + "; ".join(en) + ".",
            "why_sw": "Sababu kuu: " + "; ".join(sw) + ".",
            "top_driver": short[0] if short else "",
            "top_driver_sw": short_sw[0] if short_sw else "",
            "drivers": drivers,
        })
    return pd.DataFrame(out)


def shap_family_summary(model, rows: pd.DataFrame, h: int, features: list[str], by: str = "asal") -> pd.DataFrame:
    """Mean |SHAP| per driver family, split by a grouping column (research question 3)."""
    X = design(rows, h, features)
    fam = family_contributions(shap_values(model, X), features).abs()
    fam[by] = rows[by].to_numpy()
    out = fam.groupby(by).mean().reset_index().melt(id_vars=by, var_name="family", value_name="mean_abs_shap")
    out["share"] = out["mean_abs_shap"] / out.groupby(by)["mean_abs_shap"].transform("sum")
    out["h"] = h
    return out
