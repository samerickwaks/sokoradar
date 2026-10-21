"""SokoRadar's monthly pipeline, end to end from one command.

    python -m sokoradar.pipeline run                 # every stage
    python -m sokoradar.pipeline run --stages prepare,forecast,publish --offline
    python -m sokoradar.pipeline run --quick         # skip SARIMAX/Prophet and the ablation

Stages: ingest -> prepare -> backtest (+ evaluate) -> train -> forecast -> publish.
`--stages evaluate` re-scores a saved backtest without rerunning the models.
"""

from __future__ import annotations

import argparse
import logging
import platform
import sys
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from . import __version__, store
from .alps import alps_panel
from .backtest import ALERT_ROUTES, POINT_MODELS, calibrate, finalize_alerts, run_backtest
from .clean import clean_prices, read_wfp_raw, series_table
from .climate import climate_monthly
from .config import DOCS_DIR, PROCESSED_DIR, RAW_DIR, ensure_dirs, settings
from .evaluate import alert_table, in_years, interval_metrics, mase_scales, point_metrics
from .features import FEATURE_SETS, HORIZONS, build_panel, origins
from .macro import macro_monthly
from .models.statistical import stat_exog
from .replay import run_replays
from .reporting import bulletin, data_quality_report, write_doc
from .rtfp import rtfp_for_markets, terms_of_trade

log = logging.getLogger("sokoradar")
STAGES = ["ingest", "prepare", "backtest", "train", "forecast", "publish"]  # backtest ends with evaluate


def _p(name: str):
    return PROCESSED_DIR / f"{name}.parquet"


def _read(name: str) -> pd.DataFrame:
    return pd.read_parquet(_p(name))


# --------------------------------------------------------------------------- stages

def stage_ingest(args) -> None:
    from .ingest import fetch_all

    manifest = fetch_all()
    log.info("ingest: %s", {k: v["status"] for k, v in manifest["sources"].items()})


def stage_prepare(args) -> None:
    raw = read_wfp_raw()
    res = clean_prices(raw)
    alps = alps_panel(res.prices)
    clim = climate_monthly()
    mac = macro_monthly(raw)
    rt = rtfp_for_markets(res.markets)
    tot = terms_of_trade(res.prices, rt, res.markets)
    panel = build_panel(res.prices, res.markets, clim, mac, alps)
    series = series_table(res.prices, res.markets)
    for name, df in [("prices", res.prices), ("markets", res.markets), ("alps", alps), ("climate", clim),
                     ("macro", mac), ("rtfp", rt), ("tot", tot), ("panel", panel), ("series", series)]:
        df.to_parquet(_p(name), index=False)
    store.write_df("prices", res.prices)
    store.write_df("alps", alps)
    store.write_df("series", series)
    store.write_df("climate", clim[clim["month"] >= "2005-01-01"])
    store.write_df("tot", tot)
    store.write_json("data_quality", res.quality)
    manifest = None
    if (RAW_DIR / "manifest.json").exists():
        import json
        manifest = json.loads((RAW_DIR / "manifest.json").read_text(encoding="utf-8"))
    write_doc("data_quality_report.md", data_quality_report(res.quality, series, clim, mac, manifest))
    log.info("prepare: %d series, %d observations, %d panel rows", res.quality["series_kept"],
             res.quality["observations_kept"], len(panel))


def _select_model(pm: pd.DataFrame) -> str:
    cands = [m for m in ("lgbm", "lgbm_smooth", "xgb", "sarimax", "prophet", "sokoradar") if m in set(pm["model"])]
    score = pm[pm["model"].isin(cands)].groupby("model")["mae_pct"].mean()
    return str(score.idxmin())


def _f2(precision: float, recall: float) -> float:
    if not precision or not recall or np.isnan(precision) or np.isnan(recall):
        return 0.0
    return 5 * precision * recall / (4 * precision + recall)


def _select_route(at: pd.DataFrame) -> str:
    """Route with the best mean F2 score at 1-2 months.

    F2 weighs recall four times as much as precision, the proposal's stated priority
    ("a missed crisis costs more than a false alarm").
    """
    t = at[at["h"].isin([1, 2]) & at["route"].isin(ALERT_ROUTES)].copy()
    t["f2"] = [_f2(pr, rc) for pr, rc in zip(t["precision"], t["recall"])]
    return str(t.groupby("route")["f2"].mean().idxmax())


def _ablation(panel: pd.DataFrame, ex: pd.DataFrame, years: list[int]) -> pd.DataFrame:
    """Research question 2: what each block of drivers adds, LightGBM only."""
    rows = []
    for name, feats in FEATURE_SETS.items():
        bt = run_backtest(panel, ex, years=years, features=feats, statistical=False, prophet=False,
                          gbm_specs=(("lgbm", "naive"),), classifier=False)
        pm = point_metrics(bt["preds"], ["naive", "lgbm"], common=True)
        pm = pm[pm["model"] == "lgbm"].assign(feature_set=name, n_features=len(feats))
        rows.append(pm)
    return pd.concat(rows, ignore_index=True)


def stage_backtest(args) -> None:
    cfg = settings()["backtest"]
    panel, clim, mac = _read("panel"), _read("climate"), _read("macro")
    ex = stat_exog(clim, mac)
    t0 = time.time()
    bt = run_backtest(panel, ex, statistical=not args.quick, prophet=(not args.quick) and cfg["prophet"])
    log.info("backtest: %d forecasts in %.0fs", len(bt["preds"]), time.time() - t0)
    bt["preds"].to_parquet(_p("backtest_raw"), index=False)
    store.write_df("importance", bt["importance"])
    store.write_df("ensemble_weights_history", bt["weights"])
    if not args.quick and not args.no_ablation:
        my = tuple(cfg["metric_years"])
        _ablation(panel, ex, list(range(my[0], my[1] + 1))).to_parquet(_p("ablation"), index=False)
    elif _p("ablation").exists():
        _p("ablation").unlink()
    stage_evaluate(args)


def stage_evaluate(args) -> None:
    """Metrics, model and alert-route selection, intervals and replays from the saved backtest."""
    cfg = settings()["backtest"]
    panel, alps = _read("panel"), _read("alps")
    preds = _read("backtest_raw")
    folds = sorted(preds["fold"].unique())
    scales = mase_scales(panel, folds)
    my, hold = tuple(cfg["metric_years"]), (cfg["holdout_year"], cfg["holdout_year"])
    models = [m for m in POINT_MODELS if preds[f"pred_{m}"].notna().any()]
    pm_common = point_metrics(in_years(preds, my), models, scales, common=True)
    gbm_models = ["naive", "snaive", "smooth", "xgb", "lgbm", "lgbm_smooth", "sokoradar"]
    pm_all = point_metrics(in_years(preds, my), gbm_models, scales, common=True)
    pm_hold = point_metrics(in_years(preds, hold), gbm_models, scales, common=True)
    pm_year = point_metrics(preds, ["naive", "snaive", "lgbm", "sokoradar"], scales, by=["fold", "h"], common=True)
    pm_seg = point_metrics(in_years(preds, my), ["naive", "lgbm", "sokoradar"], scales,
                           by=["pricetype", "commodity", "h"], common=True)
    selected = _select_model(pm_all)
    cal = calibrate(preds, model=selected)
    at = alert_table(in_years(cal, my))
    route = _select_route(at)
    final = finalize_alerts(cal, route)
    at_hold = alert_table(in_years(cal, hold))
    im = interval_metrics(in_years(final, my))
    im_hold = interval_metrics(in_years(final, hold))
    rep = run_replays(final, alps)
    abl = _read("ablation") if _p("ablation").exists() else None

    final.to_parquet(_p("backtest_calibrated"), index=False)
    keep = ["series_id", "market", "market_key", "commodity", "pricetype", "county", "asal", "fold", "month", "h",
            "target_month", "logp", "y_true", "alps_now", "alps_true", "alps_pred", "exp_price", "alps_sd", "pred",
            "lo", "hi", "p_alert", "alert_flag", "phase_pred", "event"] + [f"pred_{m}" for m in models]
    store.write_df("backtest", final[keep])
    store.write_df("replay_episodes", rep["episodes"])
    store.write_df("replay_monthly", rep["monthly"])
    store.write_df("replay_summary", rep["summary"])
    store.write_df("metrics_by_year", pm_year)
    store.write_df("metrics_by_segment", pm_seg)
    if abl is not None:
        store.write_df("ablation", abl)

    targets = success_targets(pm_all, im, at, rep["summary"], route)
    metrics = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "metric_years": list(my), "holdout_year": cfg["holdout_year"],
        "selected_model": selected, "alert_route": route,
        "point_common": pm_common.to_dict("records"), "point": pm_all.to_dict("records"),
        "point_holdout": pm_hold.to_dict("records"),
        "intervals": im.to_dict("records"), "intervals_holdout": im_hold.to_dict("records"),
        "alerts": at.to_dict("records"), "alerts_holdout": at_hold.to_dict("records"),
        "replays": rep["summary"].to_dict("records"),
        "ablation": abl.to_dict("records") if abl is not None else None,
        "targets": targets,
    }
    store.write_json("metrics", metrics)
    from .tracking import log_run
    flat = {f"{r['model']}_h{r['h']}_mae_pct": r["mae_pct"] for r in metrics["point"]}
    flat.update({f"coverage_h{r['h']}": r["coverage"] for r in metrics["intervals"]})
    log_run("backtest", {"selected_model": selected, "alert_route": route, "quick": args.quick,
                         "lightgbm": settings()["lightgbm"]}, flat, [store.path("metrics.json")])
    log.info("evaluate: selected model %s, alert route %s", selected, route)


def success_targets(pm: pd.DataFrame, im: pd.DataFrame, at: pd.DataFrame, rep: pd.DataFrame, route: str) -> list:
    """The proposal's success metrics, checked automatically where data allows."""
    sel = pm[pm["model"] == "sokoradar"].set_index("h")
    out = []
    mase = {h: float(sel.loc[h, "mase"]) for h in HORIZONS}
    snaive_mase = pm[pm["model"] == "snaive"].set_index("h")["mase"].to_dict()
    out.append({"area": "Forecasting", "metric": "MASE (seasonal-naive scaled) at 1-3 months",
                "target": "< 1.0 and below seasonal naive",
                "value": ", ".join(f"h{h}: {mase[h]:.2f} (snaive {snaive_mase[h]:.2f})" for h in HORIZONS),
                "met": all(mase[h] < 1.0 and mase[h] < snaive_mase[h] for h in HORIZONS)})
    rel = {h: float(sel.loc[h, "rel_mae_naive"]) for h in HORIZONS}
    out.append({"area": "Forecasting", "metric": "MAE vs naive (last month) at 2-3 months",
                "target": "at least 10% lower",
                "value": ", ".join(f"h{h}: {100 * (1 - rel[h]):+.1f}%" for h in HORIZONS),
                "met": all(rel[h] <= 0.90 for h in (2, 3))})
    cov = im.set_index("h")["coverage"].to_dict()
    out.append({"area": "Uncertainty", "metric": "Share of actual prices inside the 80% interval",
                "target": "75-85%", "value": ", ".join(f"h{h}: {100 * cov[h]:.1f}%" for h in HORIZONS),
                "met": all(0.75 <= cov[h] <= 0.85 for h in HORIZONS)})
    a = at[(at["route"] == route)].set_index("h")
    out.append({"area": "Alerts", "metric": f"Recall / precision on Alert+Crisis ({route} route)",
                "target": "recall >= 0.70 and precision >= 0.50 at 1-2 months",
                "value": ", ".join(f"h{h}: {a.loc[h, 'recall']:.2f} / {a.loc[h, 'precision']:.2f}" for h in HORIZONS),
                "met": all(a.loc[h, "recall"] >= 0.70 and a.loc[h, "precision"] >= 0.50 for h in (1, 2))})
    if len(rep):
        lead = rep["mean_lead_months"].mean()
        share = (rep["warned_earlier"].sum() / max(rep["episodes"].sum(), 1))
        out.append({"area": "Alerts", "metric": "Warning lead time in crisis replays (2011, 2017, 2022)",
                    "target": "at least one month earlier than current-month ALPS",
                    "value": f"{share:.0%} of {int(rep['episodes'].sum())} episodes warned >= 1 month earlier; "
                             f"mean lead {lead:.1f} months",
                    "met": bool(share >= 0.5)})
    return out


def stage_train(args) -> None:
    from .explain import shap_family_summary
    from .forecast import build_calibration, train_production

    panel = _read("panel")
    bt = _read("backtest_calibrated")
    metrics = store.read_json("metrics")
    calib = build_calibration(bt, metrics["alert_route"], metrics["selected_model"])
    bundle = train_production(panel, calib)
    org = origins(panel)
    m = bundle["forecasters"]["lgbm"]
    shap = pd.concat([shap_family_summary(m.models[(h, "point")], org[org[f"tgt_h{h}"].notna()], h, m.features)
                      for h in HORIZONS], ignore_index=True)
    store.write_df("shap_summary", shap)
    store.write_df("ensemble_weights", calib["weights"])
    store.write_df("feature_importance", m.importance())
    store.write_json("calibration", {"route": calib["route"], "model": calib["model"], "cqr_q": calib["cqr_q"],
                                     "tau": calib["tau"],
                                     "stacker": {h: s.coefficients() for h, s in calib["stacker"].items()}})
    log.info("train: production models trained on data to %s", bundle["data_end"])


def stage_forecast(args) -> None:
    from .forecast import archive, load_bundle, run_forecast, scorecard

    panel, clim, mac, tot = _read("panel"), _read("climate"), _read("macro"), _read("tot")
    prices, alps = _read("prices"), _read("alps")
    bundle = load_bundle()
    f, latest, tout = run_forecast(panel, stat_exog(clim, mac), bundle, tot, statistical=not args.quick)
    markets = _read("markets")
    last_obs = latest.set_index("market_key")["month"].groupby(level=0).max()
    markets["last_price_month"] = markets["market_key"].map(last_obs)
    live = f[f["live"]].groupby("market_key").size()
    markets["live_series"] = markets["market_key"].map(live).fillna(0).astype(int)
    store.write_df("markets", markets)
    drop = [c for c in f.columns if c == "drivers"]
    f_out = f.drop(columns=drop).assign(drivers=f["drivers"].map(lambda d: str(d)))
    store.write_df("forecasts", f_out)
    store.write_df("latest_features", latest)
    if len(tout):
        store.write_df("tot_outlook", tout)
    run_id = f"live-{pd.Timestamp(bundle['data_end']):%Y-%m}"
    existing = store.read_df("forecast_archive") if store.exists("forecast_archive.parquet") else None
    if existing is None:
        existing = _seed_archive()
    arch = archive(f, existing, run_id)
    store.write_df("forecast_archive", arch)
    store.write_df("scorecard", scorecard(arch, prices, alps))
    log.info("forecast: %d forecasts (%d live series)", len(f), f.loc[f["live"], "series_id"].nunique())


def _seed_archive() -> pd.DataFrame | None:
    """Start the track record with the untouched holdout year's backtest forecasts."""
    if not _p("backtest_calibrated").exists():
        return None
    bt = _read("backtest_calibrated")
    hold = bt[bt["fold"] == settings()["backtest"]["holdout_year"]].copy()
    hold["price_fcst"], hold["price_lo"], hold["price_hi"] = np.exp(hold["pred"]), np.exp(hold["lo"]), np.exp(hold["hi"])
    hold["phase"] = hold["phase_pred"]
    hold["run_id"] = "backtest-" + hold["month"].dt.strftime("%Y-%m")
    hold["source"] = "backtest"
    cols = ["series_id", "month", "h", "target_month", "pred", "lo", "hi", "p_alert", "alert_flag", "phase",
            "price_fcst", "price_lo", "price_hi", "run_id", "source"]
    return hold[cols]


def stage_publish(args) -> None:
    f = store.read_df("forecasts")
    tot = store.read_df("tot_outlook") if store.exists("tot_outlook.parquet") else None
    metrics = store.read_json("metrics")
    q = store.read_json("data_quality")
    data_end = f["month"].max()
    meta = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "data_end": str(data_end.date()),
        "version": __version__, "python": platform.python_version(),
        "series": int(f["series_id"].nunique()), "live_series": int(f.loc[f["live"], "series_id"].nunique()),
        "markets": int(f["market_key"].nunique()),
        "observations": q["observations_kept"],
        "selected_model": metrics["selected_model"], "alert_route": metrics["alert_route"],
        "sources": "WFP (CC BY-IGO) via HDX; CHIRPS rainfall and MODIS NDVI via HDX (WFP/CHC/NASA); "
                   "Central Bank of Kenya; World Bank Pink Sheet and RTFP",
    }
    store.write_json("meta", meta)
    text = bulletin(f, tot, meta)
    store.path("bulletin.md").write_text(text, encoding="utf-8")
    (DOCS_DIR / "bulletins").mkdir(parents=True, exist_ok=True)
    (DOCS_DIR / "bulletins" / f"bulletin_{data_end:%Y-%m}.md").write_text(text, encoding="utf-8")
    log.info("publish: bulletin for data to %s", meta["data_end"])


RUNNERS = {"ingest": stage_ingest, "prepare": stage_prepare, "backtest": stage_backtest,
           "evaluate": stage_evaluate, "train": stage_train, "forecast": stage_forecast, "publish": stage_publish}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="sokoradar", description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("run", help="run pipeline stages")
    run.add_argument("--stages", default="all",
                     help="comma-separated: " + ",".join(STAGES) + " (or evaluate: re-score the saved backtest)")
    run.add_argument("--offline", action="store_true", help="skip ingest and use the files in data/raw")
    run.add_argument("--quick", action="store_true", help="skip SARIMAX/Prophet and the ablation")
    run.add_argument("--no-ablation", action="store_true", help="skip the feature ablation backtests")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stdout)
    for name in ("cmdstanpy", "prophet"):
        logging.getLogger(name).disabled = True
    ensure_dirs()
    stages = STAGES if args.stages == "all" else [s.strip() for s in args.stages.split(",")]
    if args.offline and "ingest" in stages:
        stages.remove("ingest")
    for s in stages:
        t0 = time.time()
        RUNNERS[s](args)
        log.info("stage %s done in %.0fs", s, time.time() - t0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
