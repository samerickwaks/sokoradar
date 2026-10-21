import numpy as np
import pandas as pd

from sokoradar.alps import alps_panel
from sokoradar.backtest import training_rows
from sokoradar.features import FEATURE_SETS, FEATURES, build_panel, design, origins

from .conftest import synthetic_climate, synthetic_macro


def test_targets_are_future_log_changes(panel):
    s = panel[panel["series_id"] == "alpha|maize|retail"].reset_index(drop=True)
    i = 30
    assert np.isclose(s.loc[i, "tgt_h2"], np.log(s.loc[i + 2, "price"]) - np.log(s.loc[i, "price"]))
    assert np.isclose(s.loc[i, "lag1"], np.log(s.loc[i - 1, "price"]) - np.log(s.loc[i, "price"]))


def test_features_do_not_see_the_future(prices, markets):
    months = pd.date_range("2010-01-01", "2020-12-01", freq="MS")
    clim, mac = synthetic_climate(months), synthetic_macro(months)
    base = build_panel(prices, markets, clim, mac, alps_panel(prices))
    changed = prices.copy()
    late = changed["month"] >= "2018-06-01"
    changed.loc[late, "price"] *= 3  # a shock after the origin we check
    shocked = build_panel(changed, markets, clim, mac, alps_panel(changed))
    origin = pd.Timestamp("2018-03-01")
    a = design(base[base["month"] == origin], 1).reset_index(drop=True)
    b = design(shocked[shocked["month"] == origin], 1).reset_index(drop=True)
    pd.testing.assert_frame_equal(a, b)


def test_design_has_every_feature(panel):
    X = design(origins(panel), 2)
    assert list(X.columns) == FEATURES
    assert set(FEATURES) < set(FEATURE_SETS["full"])
    assert X["tgt_month"].between(1, 12).all()


def test_training_rows_hide_targets_after_cutoff(panel):
    org = origins(panel)
    cutoff = pd.Timestamp("2018-12-01")
    tr = training_rows(org, cutoff)
    assert tr["month"].max() <= cutoff
    for h in (1, 2, 3):
        visible = tr[tr[f"tgt_h{h}"].notna()]
        assert (visible[f"tgt_month_h{h}"] <= cutoff).all()
