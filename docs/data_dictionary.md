# SokoRadar data dictionary

Every table below is written by `python -m sokoradar.pipeline run`. Tables in `data/store/` are
what the API, dashboard and USSD read; `data/processed/` holds intermediate tables that are
rebuilt each run and are not committed. Prices are **KES per kg**; months are the first day
of the month.

## Identifiers

| Field | Meaning |
|---|---|
| `market_key` | Slug of the canonical WFP market name, e.g. `kakuma-3`, `eldoret-town-uasin-gishu` |
| `series_id` | `market_key|commodity|pricetype` (price type lower-case), e.g. `mandera|maize|retail` |
| `commodity` | `maize`, `beans`, `maize_flour`, `goat_meat` (WFP names are mapped in `config/settings.yaml`) |
| `pricetype` | `Retail` or `Wholesale` (kept separate, never mixed) |
| `county_pcode` | OCHA county code `KE001`–`KE047` |
| `asal` | NDMA class of the county: `arid`, `semi-arid`, `non-ASAL`; `pastoral` = arid |

## Source tables (`data/raw/`, downloaded)

| File | Provider | Notes |
|---|---|---|
| `wfp_food_prices_ken.csv` | WFP via HDX | One row per market, commodity, unit, price type and month. Second line holds HXL tags. |
| `wfp_markets_ken.csv` | WFP via HDX | Market coordinates and admin names |
| `ken-rainfall-subnat-full.csv` | WFP/CHC CHIRPS via HDX | Dekadal rainfall `rfh`, rolling 1- and 3-month totals `r1h`, `r3h`, their 1989–2018 averages `*_avg` and anomalies `rfq`, `r1q`, `r3q` (% of normal). `adm_level` 2 units are pre-2013 districts. |
| `ken-ndvi-subnat-full.csv` | WFP/NASA MODIS via HDX | Dekadal NDVI `vim`, average `vim_avg`, anomaly `viq` (%) |
| `cbk_fx_historical.csv` | Central Bank of Kenya | Daily indicative rates; no header; mixed date formats |
| `CMO-Historical-Data-Monthly.xlsx` | World Bank Pink Sheet | Monthly global prices in USD |
| `real-time-food-prices-for-kenya.csv` | World Bank RTFP via HDX | Wide table; `c_<item>` = estimate, plain `<item>` = observed |
| `ipc_ken_level1_long_latest.csv` | IPC via HDX | Latest IPC analysis (context only) |
| `ken_admin1.geojson` | OCHA COD via HDX | County boundaries (simplified copy in `config/ken_counties.geojson`) |

## `prices` — cleaned observed prices

| Column | Type | Meaning |
|---|---|---|
| `series_id`, `market_key`, `market`, `commodity`, `pricetype` | str | Identifiers (see above) |
| `month` | date | Month of the price |
| `price` | float | Median KES/kg across the WFP rows mapped to this series and month |
| `n_raw` | int | Number of raw WFP rows behind the value |
| `sources` | str | WFP commodity names merged into the series |

Only series with at least 36 monthly prices are kept (WFP's minimum for a seasonal trend).
Prices further than 5 MADs **and** 2.5× from the series' rolling median are dropped as unit slips.

## `markets`

`market_key`, `market`, `wfp_market_id`, `latitude`, `longitude`, `county_pcode`, `county`, `asal`,
`pastoral`, `dist_nairobi_km` (great-circle), `dist_border_km` (to the nearest crossing in
`config/border_crossings.csv`), `last_price_month`, `live_series` (series with a live forecast).

## `alps` — Alert for Price Spikes, no look-ahead

| Column | Meaning |
|---|---|
| `alps` | (observed − expected) ÷ residual SD, from a regression of price on a linear trend and 12 month dummies fitted on prices **up to that month only** |
| `expected` | Expected price at that month |
| `alps_sd` | Residual standard deviation |
| `exp_h1`–`exp_h3` | Expected price 1–3 months ahead from the same fit (used to place forecasts on the ALPS scale) |
| `phase` | Normal < 0.25 ≤ Stress < 1 ≤ Alert < 2 ≤ Crisis |

## `climate` — county-month climate

`rain_1m_mm` (1-month total, mm), `rain_1m_pct` and `rain_3m_pct` (% of the 1989–2018 normal,
capped at 400), `rain_dekads` (dekads available in the month; 3 = complete), `ndvi`, `ndvi_pct`
(% of the long-term average). Districts are aggregated to counties weighted by pixel count.

## `macro` (processed)

`kes_usd` (CBK monthly mean, filled with the WFP-implied rate), `fx_source`, `ugx_kes`, `tzs_kes`,
`global_maize_usd`, `global_wheat_usd`, `global_crude_oil_usd`, `global_urea_usd`, `global_dap_usd`.

## `panel` (processed) — model features, one row per series and month

| Group | Columns |
|---|---|
| Price memory | `lag{1,2,3,4,5,6,9,12}` = log p(t−k) − log p(t); `rel_ma{3,6,12,36}` = rolling mean of log price − log p(t); `vol6`, `vol12` = SD of monthly log changes; `prev_gap` = months since the previous price; `n_hist` = log(1 + prices so far); `smooth_logp` |
| Level | `alps`, `alps_l1`, `alps_chg`, `rel_sd` (= `alps_sd` / `expected`) |
| Seasonality | `seas_chg_h{h}` = change from month t to t+h one year earlier; `seas_avg_h{h}` = mean of that change over the past 5 years; calendar of the target month (`tgt_month`, `tgt_sin`, `tgt_cos`, `tgt_lean`, `tgt_harvest`, `tgt_dry`) |
| Market network | `nat_chg{1,3,12}` = national price index for the commodity and price type; `gap_nbo` = log gap to the Nairobi wholesale price; `nbo_chg3`; `near_gap` = log gap to the nearest market with the same commodity |
| Climate | `rain1_l{0..5}` = 1-month rainfall % of normal, k months ago; `rain3_l0`, `rain3_l3`; `rain6` = 6-month cumulative % of normal; `ndvi_l{0,1,2,3,5}` |
| Macro | `fx_chg{1,3,12}` (log change in KES/USD); global prices in KES: `gmaize_chg{3,12}`, `gwheat_chg12`, `crude_chg{3,12}`, `urea_chg12` (ablation only); `lfx_l3` |
| Policy | `policy_now`, `policy_ended` from `config/policy_events.csv` |
| Market | `dist_nairobi_km`, `dist_border_km`, `asal_code`, `commodity_code`, `pricetype_code` |
| Targets | `tgt_h{h}` = log p(t+h) − log p(t); `tgt_alps_h{h}` = current-month ALPS at t+h; `snaive_h{h}` = log p(t+h−12) |

## `forecasts` — latest live forecasts (one row per series and horizon)

| Column | Meaning |
|---|---|
| `month` | Forecast origin: the series' latest observed month |
| `h`, `target_month` | Months ahead (1–3) and the month forecast |
| `price_now` | Latest observed price |
| `price_fcst`, `price_lo`, `price_hi` | Ensemble forecast and 80% conformal interval |
| `change_pct` | Forecast change from `price_now`, % |
| `pred_naive` … `pred_sarimax`, `pred_prophet` | Component forecasts (log price) |
| `q_lo`, `q_hi` | Raw LightGBM 10% / 90% quantiles (log) |
| `alps_now`, `alps_pred`, `exp_price` | ALPS today, ALPS of the forecast, and the expected price at t+h |
| `p_alert_int`, `p_alert_clf`, `p_alert_stack`, `p_alert` | Alert probability by route; `p_alert` is the selected route |
| `tau`, `alert_flag`, `phase` | Decision threshold, alert decision and the phase shown to users |
| `why_en`, `why_sw`, `top_driver`, `top_driver_sw`, `drivers` | Plain-language explanation from SHAP values grouped by driver family |
| `months_old`, `freshness`, `live` | Age of the latest price; `fresh` ≤ 2 months, `aging` ≤ 3, else `stale`; only live series are shown on USSD, SMS and the alert map |

## Other store tables

| Table | Contents |
|---|---|
| `backtest` | Every out-of-sample forecast from the rolling-origin backtest (2009–2026) with components, intervals, alert probabilities and phases |
| `metrics.json` | Backtest metrics, the selected model and alert route, and the proposal's success targets |
| `metrics_by_year`, `metrics_by_segment` | Accuracy by test year, and by price type and commodity |
| `ablation` | Research question 2: LightGBM accuracy with each feature set |
| `shap_summary` | Research question 3: mean absolute SHAP by driver family and ASAL class |
| `importance`, `feature_importance` | LightGBM gain importance per fold, and for the production model |
| `ensemble_weights`, `ensemble_weights_history` | Production ensemble weights; weights used in each backtest fold |
| `replay_episodes`, `replay_monthly`, `replay_summary` | Crisis replays for 2011, 2017 and 2022 |
| `tot`, `tot_outlook` | Goat-to-maize terms of trade (kg of maize per goat) and its 1–3 month outlook |
| `forecast_archive`, `scorecard` | Every issued forecast, and forecasts scored against prices observed since |
| `latest_features` | Feature rows at each series' latest origin (for what-if scenarios) |
| `meta.json`, `data_quality.json`, `calibration.json` | Run metadata, cleaning statistics, production calibration |
| `bulletin.md` | The monthly early-warning bulletin |
