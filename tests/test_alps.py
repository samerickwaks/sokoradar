import numpy as np
import pandas as pd

from sokoradar.alps import alps_series, forecast_alps, phase_code, phase_name


def test_phase_thresholds_follow_wfp_guidance():
    assert list(phase_name([0.1, 0.25, 0.99, 1.0, 1.99, 2.0])) == ["Normal", "Stress", "Stress", "Alert", "Alert",
                                                                  "Crisis"]
    assert phase_code([np.nan])[0] == -1


def _series(n=60, spike_at=None):
    months = pd.date_range("2015-01-01", periods=n, freq="MS")
    rng = np.random.default_rng(3)
    y = 40 + 0.2 * np.arange(n) + 3 * np.sin(2 * np.pi * months.month.to_numpy() / 12) + rng.normal(0, 1, n)
    if spike_at is not None:
        y[spike_at:] += 15
    return months, y


def test_alps_needs_minimum_history():
    months, y = _series(40)
    a = alps_series(months, y, min_obs=36)
    assert len(a) == 40 - 35
    assert a["month"].iloc[0] == months[35]


def test_alps_has_no_look_ahead():
    months, y = _series(60)
    _, y_spiked = _series(60, spike_at=50)
    a = alps_series(months, y, min_obs=36).set_index("month")
    b = alps_series(months, y_spiked, min_obs=36).set_index("month")
    before = a.index < months[50]
    # A later price spike must not change any ALPS value computed before it.
    np.testing.assert_allclose(a.loc[before, "alps"], b.loc[before, "alps"])
    assert b.loc[months[50], "alps"] > 2  # and the spike itself is flagged


def test_forecast_alps_uses_expected_price_and_sd():
    assert forecast_alps([110.0], [100.0], [5.0])[0] == 2.0
