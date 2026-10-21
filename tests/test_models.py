import numpy as np
import pandas as pd
import pytest

from sokoradar.conformal import apply_correction, coverage, cqr_correction, cqr_scores
from sokoradar.features import origins
from sokoradar.models.alerts import alert_decision, choose_threshold, display_phase, interval_alert_prob
from sokoradar.models.ensemble import _weights_from, combine
from sokoradar.models.gbm import GlobalForecaster


def test_cqr_reaches_nominal_coverage():
    rng = np.random.default_rng(0)
    y = rng.normal(size=4000)
    lo, hi = np.full(4000, -0.5), np.full(4000, 0.5)  # deliberately too narrow
    q = cqr_correction(cqr_scores(y[:2000], lo[:2000], hi[:2000]), level=0.8)
    lo2, hi2 = apply_correction(lo[2000:], hi[2000:], q)
    assert 0.77 < coverage(y[2000:], lo2, hi2) < 0.83


def test_cqr_matches_mapie_symmetric_cqr():
    mapie = pytest.importorskip("mapie.regression")
    from sklearn.ensemble import GradientBoostingRegressor

    rng = np.random.default_rng(0)
    X = rng.normal(size=(600, 2))
    y = X[:, 0] + rng.normal(size=600)
    ests = [GradientBoostingRegressor(loss="quantile", alpha=a, n_estimators=50, max_depth=2).fit(X[:300], y[:300])
            for a in (0.1, 0.9, 0.5)]
    m = mapie.ConformalizedQuantileRegressor(ests, confidence_level=0.8, prefit=True).conformalize(X[300:450],
                                                                                                    y[300:450])
    _, pis = m.predict_interval(X[450:], symmetric_correction=True)
    lc, hc = ests[0].predict(X[300:450]), ests[1].predict(X[300:450])
    q = cqr_correction(cqr_scores(y[300:450], np.minimum(lc, hc), np.maximum(lc, hc)), 0.8, min_scores=1)
    lt, ht = ests[0].predict(X[450:]), ests[1].predict(X[450:])
    lo, hi = apply_correction(np.minimum(lt, ht), np.maximum(lt, ht), q)
    assert np.abs(lo - pis[:, 0, 0]).max() < 0.02
    assert np.abs(hi - pis[:, 1, 0]).max() < 0.02


def test_interval_alert_probability_rises_with_forecast():
    p = interval_alert_prob(np.log([90, 100, 110, 120]), np.log([100, 110, 121, 132]), [100] * 4, [5] * 4)
    assert np.all(np.diff(p) > 0)
    assert p[0] < 0.1 and p[-1] > 0.9


def test_threshold_meets_recall_target():
    rng = np.random.default_rng(1)
    y = rng.random(1000) < 0.1
    p = np.clip(y * 0.5 + rng.random(1000) * 0.6, 0, 1)
    tau = choose_threshold(p, y, target_recall=0.7)
    assert ((p >= tau) & y).sum() / y.sum() >= 0.7


def test_display_phase_lets_alert_decision_set_alert_and_crisis():
    out = display_phase([True, True, False, False], [2.5, 0.5, 1.5, 0.1])
    assert list(out) == ["Crisis", "Alert", "Stress", "Normal"]


def test_ensemble_weights_favour_accurate_models_and_renormalise():
    past = pd.DataFrame({"y_true": np.zeros(100), "pred_a": np.full(100, 0.1), "pred_b": np.full(100, 0.2)})
    w = _weights_from(past, ["a", "b", "c"], min_rows=30)
    assert w["a"] == pytest.approx(0.8) and w["b"] == pytest.approx(0.2) and "c" not in w
    frame = pd.DataFrame({"pred_a": [1.0, np.nan], "pred_b": [2.0, 2.0]})
    out = combine(frame, w)
    assert out[0] == pytest.approx(1.2) and out[1] == pytest.approx(2.0)


def test_global_forecaster_fits_and_predicts(panel):
    org = origins(panel)
    m = GlobalForecaster("lgbm").fit(org)
    p = m.predict(org.tail(5))
    assert len(p) == 15
    assert (p["q_lo"] <= p["q_hi"]).all()
    assert p["pred_lgbm"].between(np.log(10), np.log(500)).all()


def test_forecast_in_alert_territory_is_always_an_alert():
    flags = alert_decision([0.05, 0.05, 0.6, np.nan], [1.4, 0.3, 0.3, np.nan], 0.5)
    assert list(flags) == [True, False, True, False]
