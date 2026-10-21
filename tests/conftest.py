"""Synthetic fixtures: tests never need the network or the real data files."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def synthetic_prices(n_months: int = 60, seed: int = 0) -> pd.DataFrame:
    """Three series with trend, seasonality and noise (two markets, two commodities)."""
    rng = np.random.default_rng(seed)
    months = pd.date_range("2015-01-01", periods=n_months, freq="MS")
    rows = []
    for market, commodity, pricetype, base in [("alpha", "maize", "Retail", 40.0), ("alpha", "beans", "Retail", 90.0),
                                                ("beta", "maize", "Wholesale", 30.0)]:
        season = 1 + 0.08 * np.sin(2 * np.pi * (months.month - 3) / 12)
        trend = np.exp(0.004 * np.arange(n_months))
        price = base * season * trend * np.exp(rng.normal(0, 0.04, n_months))
        for m, p in zip(months, price):
            rows.append({"series_id": f"{market}|{commodity}|{pricetype.lower()}", "market_key": market,
                         "market": market.title(), "commodity": commodity, "pricetype": pricetype, "month": m,
                         "price": float(p), "n_raw": 1, "sources": commodity})
    return pd.DataFrame(rows)


def synthetic_markets() -> pd.DataFrame:
    return pd.DataFrame([
        {"market_key": "alpha", "market": "Alpha", "wfp_market_id": "1", "latitude": 3.93, "longitude": 41.86,
         "county_pcode": "KE009", "county": "Mandera", "asal": "arid", "pastoral": True, "dist_nairobi_km": 800.0,
         "dist_border_km": 1.0},
        {"market_key": "beta", "market": "Beta", "wfp_market_id": "2", "latitude": -1.28, "longitude": 36.82,
         "county_pcode": "KE047", "county": "Nairobi", "asal": "non-ASAL", "pastoral": False, "dist_nairobi_km": 1.0,
         "dist_border_km": 140.0},
    ])


def synthetic_climate(months: pd.DatetimeIndex) -> pd.DataFrame:
    rng = np.random.default_rng(1)
    rows = []
    for pc in ("KE009", "KE047"):
        for m in months:
            rows.append({"county_pcode": pc, "month": m, "rain_1m_mm": 50 + rng.normal(0, 10),
                         "rain_1m_pct": 100 + rng.normal(0, 20), "rain_3m_pct": 100 + rng.normal(0, 15),
                         "rain_dekads": 3, "ndvi": 0.5, "ndvi_pct": 100 + rng.normal(0, 5)})
    return pd.DataFrame(rows)


def synthetic_macro(months: pd.DatetimeIndex) -> pd.DataFrame:
    n = len(months)
    return pd.DataFrame({"month": months, "kes_usd": 100 * np.exp(0.003 * np.arange(n)), "fx_source": "CBK",
                         "global_maize_usd": 200.0, "global_wheat_usd": 250.0, "global_crude_oil_usd": 70.0,
                         "global_urea_usd": 300.0, "global_dap_usd": 500.0})


@pytest.fixture(scope="session")
def prices() -> pd.DataFrame:
    return synthetic_prices()


@pytest.fixture(scope="session")
def markets() -> pd.DataFrame:
    return synthetic_markets()


@pytest.fixture(scope="session")
def panel(prices, markets):
    from sokoradar.alps import alps_panel
    from sokoradar.features import build_panel

    months = pd.date_range("2010-01-01", "2020-12-01", freq="MS")
    return build_panel(prices, markets, synthetic_climate(months), synthetic_macro(months), alps_panel(prices))


def synthetic_forecasts(n_counties: int = 11) -> pd.DataFrame:
    """Live forecast rows shaped like data/store/forecasts.parquet."""
    rows = []
    for c in range(n_counties):
        for mk in range(2):
            for com in ("maize", "beans"):
                for h in (1, 2, 3):
                    phase = "Alert" if (c == 0 and com == "maize" and h == 2) else "Normal"
                    rows.append({
                        "series_id": f"m{c}{mk}|{com}|retail", "market_key": f"m{c}{mk}",
                        "market": f"Market {c}-{mk} (County {c})", "county": f"County {c:02d}", "commodity": com,
                        "pricetype": "Retail", "month": pd.Timestamp("2026-08-01"), "h": h,
                        "target_month": pd.Timestamp("2026-08-01") + pd.DateOffset(months=h),
                        "price_now": 50.0, "price_fcst": 52.0, "price_lo": 45.0, "price_hi": 60.0,
                        "change_pct": 4.0, "phase": phase, "p_alert": 0.8 if phase == "Alert" else 0.05,
                        "top_driver": "below-normal rains (60%)", "top_driver_sw": "mvua chini ya kawaida (60%)",
                        "live": True, "latitude": 0.0, "longitude": 37.0,
                    })
    return pd.DataFrame(rows)


@pytest.fixture()
def forecasts_df() -> pd.DataFrame:
    return synthetic_forecasts()
