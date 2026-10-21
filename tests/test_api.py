import json

import pandas as pd
import pytest

from .conftest import synthetic_forecasts, synthetic_prices


@pytest.fixture()
def client(tmp_path, monkeypatch):
    from sokoradar import store

    monkeypatch.setattr(store, "STORE_DIR", tmp_path)
    f = synthetic_forecasts(3)
    for c in ("market_key",):
        f[c] = f[c]
    f = f.assign(alps_now=0.1, alps_pred=0.3, why_en="Main reasons: x.", why_sw="Sababu kuu: x.", freshness="fresh",
                 months_old=1)
    store.write_df("forecasts", f)
    p = synthetic_prices()
    store.write_df("prices", p)
    store.write_df("alps", pd.DataFrame({"series_id": p["series_id"], "month": p["month"], "alps": 0.0,
                                         "expected": p["price"], "phase": "Normal"}))
    store.write_df("markets", f.groupby("market_key").first().reset_index()[["market_key", "market", "county"]])
    store.write_json("meta", {"data_end": "2026-08-01", "generated_at": "2026-10-21T00:00:00"})
    store.write_json("metrics", {"selected_model": "sokoradar", "alert_route": "clf", "targets": [], "point": [],
                                 "intervals": [], "alerts": [], "replays": []})
    (tmp_path / "bulletin.md").write_text("# Bulletin", encoding="utf-8")
    from fastapi.testclient import TestClient

    from api.main import _ussd_cache, app

    _ussd_cache.clear()
    return TestClient(app)


def test_health_and_meta(client):
    assert client.get("/health").json()["status"] == "ok"
    assert client.get("/meta").json()["data_end"] == "2026-08-01"


def test_forecast_filters(client):
    r = client.get("/forecast", params={"county": "County 00", "commodity": "maize", "h": 2})
    assert r.status_code == 200
    rows = r.json()
    assert rows and all(x["h"] == 2 and x["commodity"] == "maize" for x in rows)
    assert client.get("/forecast", params={"county": "Nowhere"}).status_code == 404


def test_alerts_endpoint(client):
    rows = client.get("/alerts").json()
    assert len(rows) == 2 and {r["phase"] for r in rows} == {"Alert"}


def test_history_endpoint(client):
    r = client.get("/history", params={"market": "alpha", "commodity": "maize", "pricetype": "Retail"})
    assert r.status_code == 200 and len(r.json()["points"]) == 60


def test_ussd_callback(client):
    r = client.post("/ussd", data={"sessionId": "s1", "serviceCode": "*384#", "phoneNumber": "", "text": "2*1*1"})
    assert r.status_code == 200 and r.text.startswith("END")
    assert float(r.headers["X-Response-Time-ms"]) < 2000


def test_broadcast_requires_token(client, monkeypatch):
    monkeypatch.setenv("SOKORADAR_ADMIN_TOKEN", "secret")
    assert client.post("/sms/broadcast").status_code == 403
