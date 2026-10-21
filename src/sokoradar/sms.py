"""SMS alert subscriptions and broadcasts through Africa's Talking.

Privacy (Kenya Data Protection Act 2019): a subscriber is identified by a salted SHA-256
hash of their number; the number itself is kept only encrypted (Fernet, key held as a
secret) because it is needed to deliver the SMS, and it is deleted on unsubscribe. Only the
county is stored next to it. Without an Africa's Talking API key, messages go to a local
outbox file instead (dry run).

    python -m sokoradar.sms broadcast            # send alerts for the latest forecasts
    python -m sokoradar.sms broadcast --dry-run
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import requests

from .config import DATA_DIR, PHASES_SW, settings
from .ussd import UssdData

log = logging.getLogger(__name__)
DB = Path(os.environ.get("SOKORADAR_SUBSCRIBERS_DB", DATA_DIR / "subscribers.db"))
OUTBOX = DATA_DIR / "sms_outbox.jsonl"
SANDBOX_URL = "https://api.sandbox.africastalking.com/version1/messaging"
LIVE_URL = "https://api.africastalking.com/version1/messaging"


def _salt() -> bytes:
    return os.environ.get("SOKORADAR_HASH_SALT", "sokoradar-dev-salt").encode()


def phone_hash(phone: str) -> str:
    return hashlib.sha256(_salt() + phone.strip().encode()).hexdigest()


def _fernet():
    key = os.environ.get("SOKORADAR_SECRET_KEY")
    if not key:
        return None
    from cryptography.fernet import Fernet

    return Fernet(key.encode())


def _connect() -> sqlite3.Connection:
    DB.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(DB)
    con.execute("""CREATE TABLE IF NOT EXISTS subscribers (
        phone_hash TEXT PRIMARY KEY, phone_enc TEXT, county TEXT NOT NULL, lang TEXT NOT NULL,
        consent_at TEXT NOT NULL, active INTEGER NOT NULL)""")
    return con


def subscribe(phone: str, county: str, lang: str = "sw", active: bool = True) -> None:
    """Record consent (active=True) or withdraw it (active=False, number deleted)."""
    f = _fernet()
    enc = f.encrypt(phone.encode()).decode() if (f and active) else None
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with _connect() as con:
        if active:
            con.execute("INSERT OR REPLACE INTO subscribers VALUES (?,?,?,?,?,1)",
                        (phone_hash(phone), enc, county, lang, now))
        else:
            con.execute("UPDATE subscribers SET active=0, phone_enc=NULL WHERE phone_hash=?", (phone_hash(phone),))


def subscribers(county: str | None = None) -> list[dict]:
    f = _fernet()
    with _connect() as con:
        q = "SELECT phone_hash, phone_enc, county, lang FROM subscribers WHERE active=1"
        rows = con.execute(q + (" AND county=?" if county else ""), (county,) if county else ()).fetchall()
    out = []
    for h, enc, c, lang in rows:
        phone = f.decrypt(enc.encode()).decode() if (f and enc) else None
        out.append({"phone_hash": h, "phone": phone, "county": c, "lang": lang})
    return out


def compose(county: str, rows, lang: str) -> str:
    """One SMS (<= 160 characters) listing the county's Alert/Crisis markets."""
    labels = {k: (v["label_sw"] if lang == "sw" else v["label"]) for k, v in settings()["commodities"].items()}
    head = f"SokoRadar {county}: " + ("bei inatarajiwa kupanda - " if lang == "sw" else "price alert - ")
    parts = []
    for r in rows:
        phase = PHASES_SW.get(r["phase"], r["phase"]) if lang == "sw" else r["phase"]
        parts.append(f"{UssdData.short_market(r['market'])} {labels.get(r['commodity'], r['commodity'])} {phase} "
                     f"{r['target_month']:%b}")
    msg = head + "; ".join(parts)
    code = settings()["ussd"]["service_code"]
    tail = f" Piga {code}" if lang == "sw" else f" Dial {code}"
    if len(msg) + len(tail) > 160:
        msg = msg[: 160 - len(tail) - 3] + "..."
    return msg + tail


def send(numbers: list[str], message: str, dry_run: bool = False) -> dict:
    username = os.environ.get("AT_USERNAME", "sandbox")
    key = os.environ.get("AT_API_KEY")
    if dry_run or not key or not numbers:
        OUTBOX.parent.mkdir(parents=True, exist_ok=True)
        with OUTBOX.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({"at": datetime.now(timezone.utc).isoformat(), "to": len(numbers),
                                 "message": message}) + "\n")
        return {"status": "dry-run", "recipients": len(numbers)}
    url = SANDBOX_URL if username == "sandbox" or settings()["sms"]["sandbox"] else LIVE_URL
    data = {"username": username, "to": ",".join(numbers), "message": message}
    if settings()["sms"]["sender_id"]:
        data["from"] = settings()["sms"]["sender_id"]
    r = requests.post(url, data=data, headers={"apiKey": key, "Accept": "application/json"}, timeout=30)
    r.raise_for_status()
    return r.json()


def broadcast(forecasts, horizons=(1, 2), dry_run: bool = False) -> list[dict]:
    """Send each county's subscribers the markets expected in Alert or Crisis."""
    f = forecasts[forecasts["live"] & forecasts["phase"].isin(["Alert", "Crisis"]) & forecasts["h"].isin(horizons)]
    results = []
    for county, g in f.groupby("county"):
        first = g.sort_values("target_month").groupby(["market_key", "commodity"]).head(1)
        rows = first.to_dict("records")
        subs = subscribers(county)
        for lang in ("sw", "en"):
            numbers = [s["phone"] for s in subs if s["lang"] == lang and s["phone"]]
            if not numbers and not dry_run:
                continue
            msg = compose(county, rows, lang)
            results.append({"county": county, "lang": lang, "message": msg, **send(numbers, msg, dry_run)})
    return results


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="sokoradar.sms")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("broadcast")
    b.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    from . import store

    res = broadcast(store.read_df("forecasts"), dry_run=args.dry_run)
    for r in res:
        log.info("%s", r)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
