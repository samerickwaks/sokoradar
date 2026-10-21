import json

import pandas as pd
import pytest

from sokoradar import sms
from sokoradar.ussd import MAX_CHARS, UssdData, UssdSession

LABELS = {"maize": ("Mahindi", "Maize"), "beans": ("Maharagwe", "Beans")}


@pytest.fixture()
def session(forecasts_df):
    calls = []
    s = UssdSession(UssdData(forecasts_df, LABELS), subscribe=lambda *a: calls.append(a))
    s.calls = calls
    return s


def test_root_menu(session):
    r = session.handle("")
    assert r.startswith("CON SokoRadar") and "Bei ijayo" in r


def test_outlook_answer_within_three_screens_and_one_screen_of_text(session):
    # Screen 1 root, screen 2 county, screen 3 market -> answer.
    assert session.handle("2").startswith("CON")
    assert session.handle("2*1").startswith("CON")
    r = session.handle("2*1*1")
    assert r.startswith("END") and "Mahindi" in r and "Sababu" in r
    assert len(r) <= MAX_CHARS


def test_county_list_pages(session):
    first = session.handle("1")
    assert "9. Zaidi" in first
    second = session.handle("1*9")
    assert "County 08" in second
    assert session.handle("1*9*1*1").startswith("END")


def test_alerts_for_county(session):
    r = session.handle("3*1")
    assert r.startswith("END") and "Tahadhari" in r
    assert "Hakuna" in session.handle("3*2")


def test_subscription_records_consent(session):
    assert session.handle("4*1", "+254700000001").startswith("CON")
    r = session.handle("4*1*1", "+254700000001")
    assert r.startswith("END Asante")
    assert session.calls[-1] == ("+254700000001", "County 00", "sw", True)


def test_english_toggle(session):
    assert "Price now" in session.handle("0")
    assert "Why:" in session.handle("0*2*1*1")


@pytest.mark.parametrize("text", ["7", "2*99", "2*x", "4*1*5"])
def test_invalid_input_ends_politely(session, text):
    r = session.handle(text)
    assert r.startswith("END") and len(r) <= MAX_CHARS


def test_every_reply_fits_one_screen(session, forecasts_df):
    for path in ["", "1", "2", "3", "4", "1*1", "2*1", "1*1*1", "2*1*2", "3*1", "4*1", "0", "0*2*1*1"]:
        assert len(session.handle(path)) <= MAX_CHARS, path


def test_sms_hash_is_salted_and_stable(monkeypatch):
    monkeypatch.setenv("SOKORADAR_HASH_SALT", "a")
    h1 = sms.phone_hash("+254700000001")
    assert h1 == sms.phone_hash("+254700000001") and "254700000001" not in h1
    monkeypatch.setenv("SOKORADAR_HASH_SALT", "b")
    assert sms.phone_hash("+254700000001") != h1


def test_sms_subscription_lifecycle(tmp_path, monkeypatch):
    from cryptography.fernet import Fernet

    monkeypatch.setattr(sms, "DB", tmp_path / "subs.db")
    monkeypatch.setenv("SOKORADAR_SECRET_KEY", Fernet.generate_key().decode())
    sms.subscribe("+254700000002", "Turkana", "sw")
    subs = sms.subscribers("Turkana")
    assert subs == [{"phone_hash": sms.phone_hash("+254700000002"), "phone": "+254700000002",
                     "county": "Turkana", "lang": "sw"}]
    raw = (tmp_path / "subs.db").read_bytes()
    assert b"254700000002" not in raw  # never stored in clear text
    sms.subscribe("+254700000002", "Turkana", "sw", active=False)
    assert sms.subscribers("Turkana") == []


def test_sms_broadcast_dry_run(tmp_path, monkeypatch, forecasts_df):
    monkeypatch.setattr(sms, "DB", tmp_path / "subs.db")
    monkeypatch.setattr(sms, "OUTBOX", tmp_path / "outbox.jsonl")
    res = sms.broadcast(forecasts_df, dry_run=True)
    assert res and all(len(r["message"]) <= 160 for r in res)
    lines = [json.loads(x) for x in (tmp_path / "outbox.jsonl").read_text().splitlines()]
    assert "County 00" in lines[0]["message"]


def test_market_names_stay_distinct_when_shortened(forecasts_df):
    data = UssdData(forecasts_df, LABELS)
    assert data.short_market("Kalobeyei (Village 1)") == "Kalobeyei V1"
    assert data.short_market("Kalobeyei (Village 2)") != data.short_market("Kalobeyei (Village 1)")
    assert data.short_market("Eldoret town (Uasin Gishu)") == "Eldoret town"
    assert data.short_market("Kakuma 3") == "Kakuma 3"
