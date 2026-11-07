# SokoRadar forecast: model card

For the deployment / dashboard team. Built and tested in `notebooks/modelling/02_model_build.ipynb` (Franklin Karimi, with Daniella's ARIMA orders and review).

## What it gives you

For each of the 1,222 county price series (47 counties x 26 commodity/price-type combinations), from an origin month, for **1, 2 and 3 months ahead**:

| What | Model | Notes |
|---|---|---|
| Price forecast | ARIMA per series on log prices (Daniella's orders for maize, sorghum, Rosecoco beans and red potatoes; (1,1,0) for the rest) | Best forecaster in the backtest, level with "no change" |
| Spike probability | M8a: LightGBM classifier ("will the price rise more than 15%?") on price history and calendar | |
| Alert | M8a probability above a threshold per horizon | Thresholds set so alerts fire in 20% of calm months (2011-2014) |
| Top 3 drivers | M5: LightGBM on price history, weather, related crops and the economy, explained with SHAP values | Explains *what the model leans on*, not cause and effect |

**The three columns come from different models on purpose** (each is the best at its job). So a flat price forecast can sit next to an alert: the forecast is the most likely path, the alert says the chance of a big rise is higher than usual. Show them as separate items, not as one story.

## How to run

Needs Python 3.11 with numpy, pandas, lightgbm, statsmodels (tested: numpy 2.4.6, pandas 3.0.5, lightgbm 4.7.0, statsmodels 0.15.0).

```
python sokoradar_forecast.py                              # from the latest month in the data
python sokoradar_forecast.py --origins 2024-06:2024-12    # several origin months
python sokoradar_forecast.py --out path/to/forecasts.csv
python sokoradar_forecast.py --check                       # proves the script reproduces the notebook (needs the saved backtest files)
```

Inputs (found automatically, or pass `--data` and `--raw`):
- `SokoRadar_County_2006_2024_long.csv` from the data pipeline (every series, every month)
- `sokoradar_raw_data/`: `counties.csv`, `county_climate_chirps_ndvi.csv`, `national_drivers_2006_2024.csv`

It takes about 3 minutes. It writes a CSV and a `.json` beside it (settings, data file, versions, time of the run). Don't commit the outputs to Git.

## Output columns

| Column | Meaning |
|---|---|
| `series`, `county`, `commodity`, `price_type` | Which price |
| `origin_month`, `horizon`, `target_month` | Forecast made at the origin month for `horizon` months later |
| `origin_price_kes`, `origin_price_is_real` | Price at the origin; `True` = recorded in a market, `False` = estimated by the pipeline (about 95% of prices) |
| `forecast_price_kes`, `forecast_change_pct` | ARIMA forecast and its change from the origin price |
| `spike_probability_pct` | M8a's chance of a rise above 15% |
| `alert_threshold_pct`, `alert`, `alert_message` | Alert when the probability is above the threshold (8.2% / 20.3% / 35.1% at 1 / 2 / 3 months) |
| `drivers_model_change_pct` | M5's predicted change (the number the drivers explain) |
| `drivers_typical_change_pct` | M5's starting point before any input (the average change) |
| `driver_1..3`, `driver_k_family`, `driver_k_push_points` | The three inputs that move M5's prediction most, in plain words; push in percentage points (+ = pushes the price up). Typical change + all pushes = `drivers_model_change_pct` |

## How well it works (backtest 2015-2024, real market prices only)

- **Forecast:** average error 9.0% / 12.3% / 14.5% at 1 / 2 / 3 months, level with "next month = this month" (ratio 1.016 / 1.005 / 1.003). No model we built beats that overall; prices are hard to predict from this data.
- **Alerts:** catch 48% / 45% / 34% of spikes; 22% / 32% / 39% of alerts are followed by a spike; 80% / 81% / 86% of calm months stay quiet. After an alert, a spike is about **twice as likely** as in an average month (e.g. 22% vs 11% at 1 month).
- **Drivers:** price history explains most of M5's forecasts; then the economy and weather. El Niño is the largest outside driver.

## Limits (please show these to users)

- **An alert means "higher risk", not "a spike is coming".** Most alerts are not followed by a spike.
- **No alert does not mean calm:** about half of spikes come without one. Most spikes can't be seen in this data (they need event information such as floods, conflict or policy changes).
- **Mostly estimated prices:** only about 5% of the county prices were recorded in a market. Scores are on those real prices; forecasts for series with no real prices are less certain.
- **Not yet tested on 2025-2026.** The sealed test runs once, on Day 5. Performance by group (ASAL counties, wholesale vs retail, commodity) and in past crises (2011, 2017, 2022) has not been checked.
- **Drivers explain the model, not the market.** "Diesel pushed the forecast up" means the model leans on diesel, not proof that diesel caused it.
- **Inputs must be current.** Weather and economy files end in Dec 2024. For later origin months, update the raw files; the forecast and alerts don't need them, but the drivers do (the script warns when they are missing).

## Frozen settings

All settings sit at the top of `sokoradar_forecast.py` (training end Dec 2024, ARIMA orders, LightGBM settings with their learning rates, alert thresholds, spike = 15%). They were chosen in the notebook on pre-2015 years; **don't change them** without re-running the notebook. Retraining on newer data (moving `TRAIN_END`) is a modelling decision for the group, after the Day 5 test.
