"""SokoRadar open API.

Every endpoint answers from the precomputed forecast store (data/store), so responses are fast
even on free hosting. Run locally with:

    uvicorn api.main:app --reload
"""

from __future__ import annotations

import os
import time
from typing import Literal

import numpy as np
import pandas as pd
from fastapi import FastAPI, Form, Header, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse, RedirectResponse

from sokoradar import __version__, store
from sokoradar.config import settings
from sokoradar.ussd import UssdData, UssdSession

DESCRIPTION = """
Forecasts of staple food prices in Kenyan markets 1-3 months ahead, on WFP's ALPS scale
(Normal / Stress / Alert / Crisis), with 80% intervals and the drivers behind each forecast.

**Data:** WFP market prices (CC BY-IGO) via HDX, CHIRPS rainfall and MODIS NDVI via HDX,
Central Bank of Kenya, World Bank Pink Sheet and RTFP. **Forecasts are early warning, not
certainty and not trading advice.**
"""

app = FastAPI(title="SokoRadar API", version=__version__, description=DESCRIPTION,
              contact={"name": "SokoRadar team"}, license_info={"name": "MIT (code); data per source licences"})
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET", "POST"], allow_headers=["*"])

PHASE_ORDER = {"Normal": 0, "Stress": 1, "Alert": 2, "Crisis": 3}


def records(df: pd.DataFrame) -> list[dict]:
    out = df.copy()
    for c in out.columns:
        if pd.api.types.is_datetime64_any_dtype(out[c]):
            out[c] = out[c].dt.strftime("%Y-%m")
    out = out.astype(object).where(pd.notna(out), None)
    return out.to_dict("records")


FORECAST_FIELDS = ["series_id", "market", "market_key", "county", "commodity", "pricetype", "month", "h",
                   "target_month", "price_now", "price_fcst", "price_lo", "price_hi", "change_pct", "alps_now",
                   "alps_pred", "p_alert", "phase", "why_en", "why_sw", "top_driver", "freshness", "months_old",
                   "live"]


def forecasts() -> pd.DataFrame:
    return store.load("forecasts")


@app.middleware("http")
async def timing(request: Request, call_next):
    t0 = time.perf_counter()
    response = await call_next(request)
    response.headers["X-Response-Time-ms"] = f"{1000 * (time.perf_counter() - t0):.1f}"
    return response


@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse("/docs")


@app.get("/health", tags=["meta"])
def health():
    return {"status": "ok", "version": __version__, "data_end": store.read_json("meta")["data_end"]}


@app.get("/meta", tags=["meta"], summary="Run metadata, data freshness and sources")
def meta():
    return store.read_json("meta")


@app.get("/markets", tags=["markets"], summary="Markets with county, ASAL class and data freshness")
def markets():
    m = store.load("markets")
    return records(m)


@app.get("/counties", tags=["markets"], summary="Counties with live forecasts")
def counties():
    f = forecasts()
    f = f[f["live"]]
    g = f.groupby("county").agg(markets=("market_key", "nunique"), series=("series_id", "nunique")).reset_index()
    return records(g)


@app.get("/forecast", tags=["forecasts"], summary="Latest forecasts, filterable")
def forecast(market: str | None = Query(None, description="market key or name, e.g. kakuma-3"),
             county: str | None = None,
             commodity: Literal["maize", "beans", "maize_flour", "goat_meat"] | None = None,
             pricetype: Literal["Retail", "Wholesale"] | None = None,
             h: int | None = Query(None, ge=1, le=3, description="months ahead"),
             live_only: bool = True):
    f = forecasts()
    if live_only:
        f = f[f["live"]]
    if market:
        f = f[(f["market_key"] == market) | (f["market"].str.lower() == market.lower())]
    if county:
        f = f[f["county"].str.lower() == county.lower()]
    if commodity:
        f = f[f["commodity"] == commodity]
    if pricetype:
        f = f[f["pricetype"] == pricetype]
    if h:
        f = f[f["h"] == h]
    if f.empty:
        raise HTTPException(404, "no forecasts match these filters")
    return records(f[FORECAST_FIELDS].sort_values(["county", "market", "commodity", "h"]))


@app.get("/alerts", tags=["forecasts"], summary="Markets forecast to reach a phase within the horizon")
def alerts(county: str | None = None, min_phase: Literal["Stress", "Alert", "Crisis"] = "Alert",
           h: int | None = Query(None, ge=1, le=3)):
    f = forecasts()
    f = f[f["live"] & (f["phase"].map(PHASE_ORDER).fillna(-1) >= PHASE_ORDER[min_phase])]
    if county:
        f = f[f["county"].str.lower() == county.lower()]
    if h:
        f = f[f["h"] == h]
    return records(f[FORECAST_FIELDS].sort_values(["target_month", "p_alert"], ascending=[True, False]))


@app.get("/counties/{county}/alerts", tags=["forecasts"], summary="Alerts for one county")
def county_alerts(county: str):
    return alerts(county=county, min_phase="Alert", h=None)


@app.get("/history", tags=["history"], summary="Observed prices and current-month ALPS for one series")
def history(market: str, commodity: str, pricetype: str = "Retail"):
    sid = f"{market}|{commodity}|{pricetype.lower()}"
    p = store.load("prices")
    p = p[p["series_id"] == sid][["month", "price"]]
    if p.empty:
        raise HTTPException(404, f"unknown series {sid}")
    a = store.load("alps")
    a = a[a["series_id"] == sid][["month", "alps", "expected", "phase"]]
    return {"series_id": sid, "points": records(p.merge(a, on="month", how="left"))}


@app.get("/terms-of-trade", tags=["pastoral"], summary="Goat-to-maize terms of trade, now and forecast")
def terms_of_trade(market: str | None = None):
    if not store.exists("tot_outlook.parquet"):
        return []
    t = store.load("tot_outlook")
    if market:
        t = t[t["market_key"] == market]
    return records(t)


@app.get("/accuracy", tags=["accuracy"], summary="Backtest metrics and the live scorecard")
def accuracy():
    m = store.read_json("metrics")
    sc = store.load("scorecard") if store.exists("scorecard.parquet") else pd.DataFrame()
    live = {}
    if len(sc):
        live = (sc.groupby(["source", "h"]).agg(n=("abs_pct_error", "size"), mean_abs_pct_error=("abs_pct_error", "mean"),
                                                 interval_coverage=("inside_interval", "mean")).reset_index())
        live = records(live)
    return {"selected_model": m["selected_model"], "alert_route": m["alert_route"], "targets": m["targets"],
            "point": m["point"], "intervals": m["intervals"], "alerts": m["alerts"], "replays": m["replays"],
            "scorecard": live}


@app.get("/replay/{year}", tags=["accuracy"], summary="Month-by-month crisis replay (2011, 2017, 2022)")
def replay(year: int):
    r = store.load("replay_monthly")
    r = r[r["year"] == year]
    if r.empty:
        raise HTTPException(404, "replay years: " + ", ".join(map(str, settings()["backtest"]["replay_years"])))
    e = store.load("replay_episodes")
    return {"monthly": records(r), "episodes": records(e[e["year"] == year])}


@app.get("/what-if", tags=["forecasts"], summary="Re-score live forecasts with recent rainfall at X% of normal")
def what_if(rain_pct: float = Query(100, ge=0, le=300), months: int = Query(3, ge=1, le=6)):
    from sokoradar.scenario import what_if as run

    w = run(rain_pct, months)
    w = w[w["live"]]
    return records(w[["series_id", "market", "county", "commodity", "h", "target_month", "phase_base", "phase",
                      "price_base", "price_fcst", "price_shift_pct", "p_alert_base", "p_alert"]])


@app.get("/bulletin", tags=["meta"], response_class=PlainTextResponse, summary="Monthly bulletin (Markdown)")
def bulletin():
    return store.path("bulletin.md").read_text(encoding="utf-8")


# --------------------------------------------------------------------------- USSD and SMS

_ussd_cache: dict = {}


def ussd_session() -> UssdSession:
    mtime = store.path("forecasts.parquet").stat().st_mtime
    if _ussd_cache.get("mtime") != mtime:
        from sokoradar.sms import subscribe

        _ussd_cache.update(mtime=mtime, session=UssdSession(UssdData.from_store(), subscribe=subscribe))
    return _ussd_cache["session"]


@app.post("/ussd", tags=["ussd"], response_class=PlainTextResponse,
          summary="Africa's Talking USSD callback (form-encoded)")
def ussd(sessionId: str = Form(""), serviceCode: str = Form(""), phoneNumber: str = Form(""),
         text: str = Form("")):
    return ussd_session().handle(text, phoneNumber)


@app.post("/sms/broadcast", tags=["ussd"], summary="Send SMS alerts to subscribers (admin token required)")
def sms_broadcast(x_admin_token: str = Header(""), dry_run: bool = True):
    token = os.environ.get("SOKORADAR_ADMIN_TOKEN")
    if not token or x_admin_token != token:
        raise HTTPException(403, "admin token required")
    from sokoradar.sms import broadcast

    return broadcast(forecasts(), dry_run=dry_run)


@app.get("/stats/latency", tags=["meta"], include_in_schema=False)
def latency_probe():
    t0 = time.perf_counter()
    ussd_session().handle("2*1*1", "")
    return {"ussd_ms": round(1000 * (time.perf_counter() - t0), 2), "rows": int(np.int64(len(forecasts())))}
