# SokoRadar API

Live instance: `https://sokoradar-api.onrender.com`, with interactive documentation (OpenAPI) at
[/docs](https://sokoradar-api.onrender.com/docs); a local run serves the same at `http://127.0.0.1:8000/docs`. The free
server sleeps when idle, so the first request can take up to a minute. All responses are JSON
unless noted; months are `YYYY-MM`.
Every response carries an `X-Response-Time-ms` header.

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | Liveness and the latest data month |
| GET | `/meta` | Run metadata: data end, series counts, model, sources |
| GET | `/markets` | Markets with county, ASAL class, coordinates and last price month |
| GET | `/counties` | Counties with live forecasts |
| GET | `/forecast` | Latest forecasts. Filters: `market` (key or name), `county`, `commodity` (`maize`, `beans`, `maize_flour`, `goat_meat`), `pricetype` (`Retail`, `Wholesale`), `h` (1–3), `live_only` (default true) |
| GET | `/alerts` | Live forecasts at or above a phase: `min_phase` (`Stress`, `Alert`, `Crisis`; default `Alert`), `county`, `h` |
| GET | `/counties/{county}/alerts` | Alerts for one county |
| GET | `/history` | Observed prices and current-month ALPS: `market`, `commodity`, `pricetype` |
| GET | `/terms-of-trade` | Goat-to-maize terms of trade now and forecast (pastoral markets): optional `market` |
| GET | `/what-if` | Re-score live forecasts with recent rainfall at `rain_pct` % of normal over `months` months |
| GET | `/accuracy` | Backtest metrics, proposal targets and the live scorecard |
| GET | `/replay/{year}` | Crisis replay (2011, 2017, 2022): month-by-month states and episodes |
| GET | `/bulletin` | The monthly bulletin (Markdown) |
| POST | `/ussd` | Africa's Talking USSD callback (`sessionId`, `serviceCode`, `phoneNumber`, `text`, form-encoded); replies `CON …` or `END …` as plain text |
| POST | `/sms/broadcast` | Send SMS alerts to subscribers; header `X-Admin-Token` required; `dry_run` (default true) |

## Examples

```bash
curl "https://sokoradar-api.onrender.com/forecast?county=Turkana&commodity=maize&h=1"
curl "https://sokoradar-api.onrender.com/alerts?min_phase=Alert"
curl "https://sokoradar-api.onrender.com/what-if?rain_pct=50"
curl -X POST https://sokoradar-api.onrender.com/ussd -d "sessionId=1&serviceCode=*384*79278#&phoneNumber=%2B254700000000&text=2*1*1"
```

A forecast record:

```json
{
  "series_id": "kakuma-3|maize|retail", "market": "Kakuma 3", "county": "Turkana",
  "commodity": "maize", "pricetype": "Retail", "month": "2026-08", "h": 1, "target_month": "2026-09",
  "price_now": 80.0, "price_fcst": 79.1, "price_lo": 70.2, "price_hi": 88.4, "change_pct": -1.1,
  "alps_now": 0.42, "alps_pred": 0.31, "p_alert": 0.06, "phase": "Stress",
  "why_en": "Main reasons: ...", "why_sw": "Sababu kuu: ...", "top_driver": "prices down 4% in 3 months",
  "freshness": "fresh", "months_old": 1, "live": true
}
```

(Values illustrative.) `price_lo`–`price_hi` is an 80% interval: in backtests about 79% of
actual prices fell inside it. `p_alert` is the probability that the market reaches ALPS Alert
or Crisis in `target_month`. `phase` is what SokoRadar shows: Alert/Crisis when the alert model
fires, otherwise Normal/Stress from the forecast price.

## Licence and attribution

Code: MIT. Data: WFP (CC BY-IGO) via HDX; CHIRPS (CHC) and MODIS NDVI (NASA) via WFP/HDX; Central
Bank of Kenya; World Bank (CC BY 4.0). Credit these sources when you reuse the forecasts.
Forecasts are early warning, not certainty and not trading advice.
