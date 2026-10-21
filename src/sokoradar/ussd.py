"""USSD menu for basic phones (Africa's Talking callback format).

Africa's Talking posts the whole session path in `text` ("", "2", "2*1", "2*1*3"); the reply
starts with "CON " (more input) or "END " (final screen). The menu is stateless: every reply
is rebuilt from the path, so any API replica can answer any step.

    root ── 1 Bei ya sasa (price now)        ── county ── market ── END
         ── 2 Bei ijayo (next 3 months)       ── county ── market ── END
         ── 3 Tahadhari (alerts in my county) ── county ── END
         ── 4 Jiandikishe (SMS alerts)         ── county ── consent ── END
         ── 0 English (switches language, then the same menu)

Every answer is reachable within three screens; lists page with "9" (Zaidi / More).
"""

from __future__ import annotations

import calendar
from dataclasses import dataclass, field

import pandas as pd

from .config import PHASES_SW, settings

MAX_CHARS = 182
PAGE = 8
SW_MONTHS = ["Jan", "Feb", "Mac", "Apr", "Mei", "Jun", "Jul", "Ago", "Sep", "Okt", "Nov", "Des"]

TEXT = {
    "sw": {
        "root": "SokoRadar\n1. Bei ya sasa (Price now)\n2. Bei ijayo (Next 3 months)\n3. Tahadhari (Alerts)\n"
                "4. Jiandikishe (SMS alerts)\n0. English",
        "county": "Chagua kaunti:", "market": "Chagua soko:", "more": "9. Zaidi",
        "invalid": "Chaguo si sahihi. Tafadhali jaribu tena.",
        "no_data": "Hakuna data ya hivi karibuni kwa soko hili.",
        "now_head": "{market} ({month}), KES/kg:", "fcst_head": "{market}, KES/kg sasa -> {month}:",
        "reason": "Sababu: {r}", "source": "Chanzo: WFP",
        "alerts_head": "Tahadhari {county}, miezi 3 ijayo:", "no_alerts": "Hakuna soko linalotarajiwa kuwa "
        "Tahadhari {county} miezi 3 ijayo.",
        "consent": "SMS za tahadhari kwa {county}. Nambari yako itahifadhiwa kwa usiri na unaweza kujiondoa "
                   "wakati wowote.\n1. Kubali\n2. Jiondoe",
        "subscribed": "Asante! Utapokea SMS za tahadhari za bei kwa {county}. Piga tena na uchague 4 > 2 "
                      "kujiondoa.",
        "unsubscribed": "Umejiondoa. Hutapokea SMS za SokoRadar tena.",
        "retail": "rej", "wholesale": "jum",
    },
    "en": {
        "root": "SokoRadar\n1. Price now\n2. Next 3 months\n3. Alerts in my county\n4. SMS alerts\n0. Kiswahili",
        "county": "Choose county:", "market": "Choose market:", "more": "9. More",
        "invalid": "Invalid choice. Please try again.",
        "no_data": "No recent data for this market.",
        "now_head": "{market} ({month}), KES/kg:", "fcst_head": "{market}, KES/kg now -> {month}:",
        "reason": "Why: {r}", "source": "Source: WFP",
        "alerts_head": "Alerts {county}, next 3 months:", "no_alerts": "No market in {county} is expected to "
        "reach Alert in the next 3 months.",
        "consent": "Price alert SMS for {county}. Your number is kept private and you can stop any time."
                   "\n1. Agree\n2. Unsubscribe",
        "subscribed": "Thank you! You will get price alert SMS for {county}. Dial again and choose 4 > 2 "
                      "to stop.",
        "unsubscribed": "You are unsubscribed and will get no more SokoRadar SMS.",
        "retail": "ret", "wholesale": "whl",
    },
}


def _month(ts: pd.Timestamp, lang: str) -> str:
    return SW_MONTHS[ts.month - 1] if lang == "sw" else calendar.month_abbr[ts.month]


def _phase(p: str | None, lang: str) -> str:
    if not p:
        return "-"
    return PHASES_SW.get(p, p) if lang == "sw" else p


def _fit(lines: list[str], prefix: str) -> str:
    """Join lines, dropping trailing optional lines until the reply fits one USSD screen."""
    body = "\n".join(lines)
    while len(prefix) + len(body) > MAX_CHARS and len(lines) > 1:
        lines = lines[:-1]
        body = "\n".join(lines)
    return (prefix + body)[:MAX_CHARS]


@dataclass
class UssdData:
    forecasts: pd.DataFrame  # live forecasts only
    labels: dict = field(default_factory=dict)

    @classmethod
    def from_store(cls) -> "UssdData":
        from . import store

        f = store.load("forecasts")
        f = f[f["live"]].copy()
        labels = {k: (v["label_sw"], v["label"]) for k, v in settings()["commodities"].items()}
        return cls(f, labels)

    def counties(self) -> list[str]:
        return sorted(self.forecasts["county"].dropna().unique())

    def markets(self, county: str) -> list[tuple[str, str]]:
        f = self.forecasts[self.forecasts["county"] == county]
        m = f.groupby("market_key")["market"].first().sort_values()
        return list(m.items())

    @staticmethod
    def short_market(name: str) -> str:
        """Fit a market name on one USSD line without making two markets look alike.

        'Eldoret town (Uasin Gishu)' -> 'Eldoret town'; 'Kalobeyei (Village 2)' -> 'Kalobeyei V2'.
        """
        if len(name) <= 18 or " (" not in name:
            return name
        base, inner = name.split(" (", 1)
        inner = inner.rstrip(")")
        if any(ch.isdigit() for ch in inner):
            return f"{base} {inner.replace('Village ', 'V')}"
        return base


class UssdSession:
    """Parses one Africa's Talking `text` path and returns the reply."""

    def __init__(self, data: UssdData, subscribe=None):
        self.data = data
        self.subscribe = subscribe  # callable(phone, county, lang, active) -> None

    def handle(self, text: str, phone: str = "") -> str:
        tokens = [t for t in (text or "").split("*")] if text else []
        lang = "sw"
        # Language toggles at the root; each "0" flips it.
        while tokens and tokens[0] == "0":
            lang = "en" if lang == "sw" else "sw"
            tokens = tokens[1:]
        T = TEXT[lang]
        if not tokens:
            return "CON " + T["root"]
        choice, rest = tokens[0], tokens[1:]
        if choice not in {"1", "2", "3", "4"}:
            return "END " + T["invalid"]
        counties = self.data.counties()
        county, rest, reply = self._pick(rest, counties, T["county"], T)
        if reply:
            return reply
        if county is None:
            return "END " + T["invalid"]
        if choice == "3":
            return self._alerts(county, lang)
        if choice == "4":
            return self._subscription(county, rest, phone, lang)
        markets = self.data.markets(county)
        names = [self.data.short_market(n) for _, n in markets]
        idx, rest, reply = self._pick(rest, list(range(len(markets))), T["market"], T, names)
        if reply:
            return reply
        if idx is None:
            return "END " + T["invalid"]
        key, name = markets[idx]
        return self._price_now(key, name, lang) if choice == "1" else self._outlook(key, name, lang)

    # -- list screens ------------------------------------------------------------
    def _pick(self, tokens, items, title, T, names=None):
        """Consume one list selection (with paging). Returns (item, remaining tokens, reply-or-None)."""
        names = names or [str(i) for i in items]
        page = 0
        while tokens and tokens[0] == "9" and (page + 1) * PAGE < len(items):
            page += 1
            tokens = tokens[1:]
        if not tokens:
            start = page * PAGE
            lines = [title] + [f"{i + 1}. {names[start + i]}" for i in range(min(PAGE, len(items) - start))]
            if start + PAGE < len(items):
                lines.append(T["more"])
            return None, [], "CON " + "\n".join(lines)
        try:
            n = int(tokens[0])
        except ValueError:
            return None, tokens[1:], None
        pos = page * PAGE + n - 1
        if n < 1 or n > PAGE or pos >= len(items):
            return None, tokens[1:], None
        return items[pos], tokens[1:], None

    # -- answers -------------------------------------------------------------------
    def _rows(self, key: str) -> pd.DataFrame:
        return self.data.forecasts[self.data.forecasts["market_key"] == key]

    def _label(self, commodity: str, lang: str) -> str:
        sw, en = self.data.labels.get(commodity, (commodity, commodity))
        return sw if lang == "sw" else en

    def _price_now(self, key: str, name: str, lang: str) -> str:
        T = TEXT[lang]
        f = self._rows(key)
        f = f[f["h"] == 1]
        if f.empty:
            return "END " + T["no_data"]
        head = T["now_head"].format(market=self.data.short_market(name), month=_month(f["month"].max(), lang))
        lines = [head]
        for r in f.sort_values(["commodity", "pricetype"]).itertuples():
            lines.append(f"{self._label(r.commodity, lang)} ({T[r.pricetype.lower()]}): {r.price_now:,.0f}")
        lines.append(T["source"])
        return _fit(lines, "END ")

    def _outlook(self, key: str, name: str, lang: str) -> str:
        T = TEXT[lang]
        f = self._rows(key)
        if f.empty:
            return "END " + T["no_data"]
        last = f[f["h"] == f["h"].max()]
        head = T["fcst_head"].format(market=self.data.short_market(name),
                                     month=_month(last["target_month"].max(), lang))
        order = {"Normal": 0, "Stress": 1, "Alert": 2, "Crisis": 3}
        lines, worst = [head], None
        for (com, pt), g in f.groupby(["commodity", "pricetype"]):
            g = g.sort_values("h")
            end = g.iloc[-1]
            peak = max(g["phase"].dropna(), key=lambda p: order.get(p, -1), default=None)
            lines.append(f"{self._label(com, lang)}: {end.price_now:,.0f} -> {end.price_lo:,.0f}-{end.price_hi:,.0f} "
                         f"{_phase(peak, lang)}")
            rank = order.get(peak, -1)
            if worst is None or rank > worst[0]:
                worst = (rank, g.iloc[0])
        if worst is not None and isinstance(worst[1].get("top_driver"), str):
            lines.append(T["reason"].format(r=_reason(worst[1], lang)))
        return _fit(lines, "END ")

    def _alerts(self, county: str, lang: str) -> str:
        T = TEXT[lang]
        f = self.data.forecasts[(self.data.forecasts["county"] == county)
                                & self.data.forecasts["phase"].isin(["Alert", "Crisis"])]
        if f.empty:
            return "END " + T["no_alerts"].format(county=county)
        lines = [T["alerts_head"].format(county=county)]
        first = f.sort_values("target_month").groupby(["market_key", "commodity"]).head(1)
        for r in first.sort_values(["target_month", "market"]).itertuples():
            lines.append(f"{self.data.short_market(r.market)}: {self._label(r.commodity, lang)} "
                         f"{_phase(r.phase, lang)} {_month(r.target_month, lang)}")
        return _fit(lines, "END ")

    def _subscription(self, county: str, tokens, phone: str, lang: str) -> str:
        T = TEXT[lang]
        if not tokens:
            return "CON " + T["consent"].format(county=county)
        if tokens[0] not in {"1", "2"}:
            return "END " + T["invalid"]
        active = tokens[0] == "1"
        if self.subscribe is not None and phone:
            self.subscribe(phone, county, lang, active)
        return "END " + (T["subscribed"].format(county=county) if active else T["unsubscribed"])


def _reason(row, lang: str) -> str:
    val = row.get("top_driver_sw") if lang == "sw" else None
    return str(val if isinstance(val, str) and val else row["top_driver"])
