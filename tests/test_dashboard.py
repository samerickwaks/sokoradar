import time

import pytest

from sokoradar import store


@pytest.mark.skipif(not store.path("forecasts.parquet").exists(), reason="run the pipeline first")
def test_dashboard_renders_quickly():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file("../dashboard/app.py", default_timeout=60)
    at.run()  # cold start: imports and first reads
    assert not at.exception
    t0 = time.perf_counter()
    at.run()  # what a user waits for on each interaction
    warm = time.perf_counter() - t0
    assert not at.exception
    assert warm < 3.0, f"dashboard rerun took {warm:.1f}s"
    assert at.title[0].value.endswith("SokoRadar")
