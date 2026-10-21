"""SokoRadar dashboard (Streamlit).

    streamlit run dashboard/app.py

Reads only the precomputed forecast store, so it loads in well under three seconds.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import plotly.express as px  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

from sokoradar import store  # noqa: E402
from sokoradar.config import PHASES, settings  # noqa: E402
from sokoradar.geo import county_geojson  # noqa: E402

st.set_page_config(page_title="SokoRadar", page_icon="📡", layout="wide")

PHASE_COLORS = {"Normal": "#2e7d32", "Stress": "#f2b705", "Alert": "#e8590c", "Crisis": "#c92a2a", "No data": "#adb5bd"}
LABELS = {k: v["label"] for k, v in settings()["commodities"].items()}


# --------------------------------------------------------------------------- data

@st.cache_data(show_spinner=False)
def load(name: str, mtime: float) -> pd.DataFrame:
    return store.read_df(name)


def table(name: str) -> pd.DataFrame:
    p = store.path(f"{name}.parquet")
    return load(name, p.stat().st_mtime) if p.exists() else pd.DataFrame()


@st.cache_data(show_spinner=False)
def meta_json(name: str, mtime: float):
    return store.read_json(name)


def js(name: str):
    p = store.path(f"{name}.json")
    return meta_json(name, p.stat().st_mtime) if p.exists() else {}


@st.cache_resource(show_spinner=False)
def bundle():
    from sokoradar.forecast import load_bundle

    return load_bundle()


@st.cache_data(show_spinner="Re-scoring forecasts under this rainfall scenario…")
def scenario(rain_pct: int, months: int, mtime: float) -> pd.DataFrame:
    from sokoradar.scenario import what_if

    return what_if(rain_pct, months, bundle(), table("latest_features"), table("forecasts"))


if not store.path("forecasts.parquet").exists():
    st.error("No forecasts yet. Run `python -m sokoradar.pipeline run` first.")
    st.stop()

meta = js("meta")
metrics = js("metrics")
F = table("forecasts")
F["commodity_label"] = F["commodity"].map(LABELS)
LIVE = F[F["live"]]


def badge(freshness: str) -> str:
    return {"fresh": "🟢 fresh", "aging": "🟡 ageing", "stale": "⚪ stale"}.get(freshness, freshness)


def _hover(row: pd.Series, fields: list[tuple[str, str, str]]) -> str:
    parts = [f"<b>{row['market']}</b>"]
    for label, col, fmt in fields:
        v = row.get(col)
        if v is None or (isinstance(v, float) and np.isnan(v)):
            continue
        parts.append(f"{label}: {format(v, fmt) if fmt else v}")
    return "<br>".join(parts)


def phase_map(df: pd.DataFrame, color_col: str = "phase", title: str = "",
              hover: list[tuple[str, str, str]] | None = None, height: int = 560):
    """County outlines plus market markers coloured by phase.

    Uses Plotly's SVG geo projection: no map tiles and no WebGL, so it works offline and on
    any laptop used for a demo.
    """
    gj = county_geojson()
    ids = [f["id"] for f in gj["features"]]
    fig = go.Figure(go.Choropleth(geojson=gj, locations=ids, z=[0] * len(ids), featureidkey="id",
                                  colorscale=[[0, "#f8f9fa"], [1, "#f8f9fa"]], showscale=False,
                                  marker_line_color="#adb5bd", marker_line_width=0.7, hoverinfo="skip"))
    df = df.copy()
    df[color_col] = df[color_col].fillna("No data")
    for phase in PHASES + ["No data"]:
        d = df[df[color_col] == phase]
        if d.empty:
            continue
        fig.add_trace(go.Scattergeo(
            lat=d["latitude"], lon=d["longitude"], mode="markers", name=phase,
            marker=dict(size=13, color=PHASE_COLORS[phase], line=dict(width=1, color="white"), opacity=0.92),
            text=[_hover(r, hover or []) for _, r in d.iterrows()], hoverinfo="text"))
    fig.update_geos(fitbounds="locations", visible=False, bgcolor="rgba(0,0,0,0)")
    fig.update_layout(height=height, margin=dict(l=0, r=0, t=34 if title else 0, b=0), title=title,
                      legend=dict(title="Phase", orientation="h", y=-0.02), paper_bgcolor="rgba(0,0,0,0)")
    return fig


# --------------------------------------------------------------------------- header

st.title("📡 SokoRadar")
st.caption("Early warning of staple food price shocks in Kenyan markets, 1–3 months ahead, on WFP's ALPS scale. "
           f"Data to **{pd.Timestamp(meta['data_end']):%B %Y}** · model: *{metrics.get('selected_model', '?')}* · "
           f"alerts: *{metrics.get('alert_route', '?')}* route · updated {meta['generated_at'][:10]}")

tabs = st.tabs(["🗺️ Alert map", "📈 Market forecast", "🌧️ What-if rainfall", "⏪ Crisis replay", "🎯 Accuracy",
                "📱 USSD demo", "ℹ️ Data & method"])

# --------------------------------------------------------------------------- 1. map

with tabs[0]:
    c1, c2, c3 = st.columns([1, 2, 2])
    h = c1.radio("Months ahead", [1, 2, 3], horizontal=True, key="map_h")
    coms = c2.multiselect("Commodities", sorted(LIVE["commodity"].unique()), default=sorted(LIVE["commodity"].unique()),
                          format_func=lambda c: LABELS.get(c, c))
    pts = c3.multiselect("Price type", ["Retail", "Wholesale"], default=["Retail", "Wholesale"])
    view = LIVE[(LIVE["h"] == h) & LIVE["commodity"].isin(coms) & LIVE["pricetype"].isin(pts)]
    order = {p: i for i, p in enumerate(PHASES)}
    worst = (view.assign(rank=view["phase"].map(order).fillna(-1))
             .sort_values("rank", ascending=False).groupby("market_key").head(1))
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Live market series", view["series_id"].nunique())
    k2.metric("Alert", int((view["phase"] == "Alert").sum()))
    k3.metric("Crisis", int((view["phase"] == "Crisis").sum()))
    tgt = view["target_month"].max()
    k4.metric("Forecast month", f"{tgt:%b %Y}" if pd.notna(tgt) else "–")
    left, right = st.columns([3, 2])
    with left:
        st.plotly_chart(phase_map(worst, hover=[("County", "county", ""), ("Most severe", "commodity_label", ""),
                                                ("Forecast KES/kg", "price_fcst", ".1f"),
                                                ("P(Alert)", "p_alert", ".0%")]), width="stretch")
        st.caption("Each dot is a market, coloured by its most severe forecast phase across the selected commodities.")
    with right:
        st.subheader("Markets expected in Alert or Crisis")
        flagged = view[view["phase"].isin(["Alert", "Crisis"])].sort_values("p_alert", ascending=False)
        if flagged.empty:
            st.success("No live market is expected to reach Alert at this horizon.")
        else:
            st.dataframe(flagged[["county", "market", "commodity_label", "pricetype", "target_month", "phase",
                                  "price_now", "price_fcst", "p_alert", "top_driver"]].rename(
                columns={"commodity_label": "commodity", "price_now": "now (KES/kg)", "price_fcst": "forecast",
                         "p_alert": "P(Alert)", "top_driver": "main driver"}),
                hide_index=True, width="stretch",
                column_config={"P(Alert)": st.column_config.ProgressColumn(format="%.0f%%", min_value=0, max_value=1),
                               "target_month": st.column_config.DateColumn("month", format="MMM YYYY")})
        st.markdown("**All live forecasts at this horizon**")
        st.dataframe(view.sort_values(["county", "market"])[
            ["county", "market", "commodity_label", "pricetype", "price_now", "price_fcst", "change_pct", "phase",
             "freshness"]].rename(columns={"commodity_label": "commodity", "change_pct": "change %"}),
            hide_index=True, width="stretch", height=300)

# --------------------------------------------------------------------------- 2. market

with tabs[1]:
    c1, c2, c3 = st.columns(3)
    county = c1.selectbox("County", sorted(F["county"].dropna().unique()),
                          index=sorted(F["county"].dropna().unique()).index("Turkana")
                          if "Turkana" in set(F["county"]) else 0)
    mk = F[F["county"] == county].groupby("market_key")["market"].first().sort_values()
    market_key = c2.selectbox("Market", mk.index, format_func=lambda k: mk[k])
    ser = F[F["market_key"] == market_key].groupby("series_id")[["commodity", "pricetype"]].first()
    sid = c3.selectbox("Series", ser.index,
                       format_func=lambda s: f"{LABELS.get(ser.loc[s, 'commodity'])} · {ser.loc[s, 'pricetype']}")
    fc = F[F["series_id"] == sid].sort_values("h")
    hist = table("prices")
    hist = hist[hist["series_id"] == sid]
    alps = table("alps")
    alps = alps[alps["series_id"] == sid]
    r0 = fc.iloc[0]
    st.markdown(f"#### {r0['market']} · {LABELS.get(r0['commodity'])} ({r0['pricetype'].lower()}) "
                f"&nbsp; {badge(r0['freshness'])} — last price {r0['month']:%b %Y}")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Price now (KES/kg)", f"{r0['price_now']:,.1f}", help=f"ALPS now {r0['alps_now']:+.2f}")
    for col, (_, r) in zip((m2, m3, m4), fc.iterrows()):
        col.metric(f"{r['target_month']:%b %Y} · {r['phase'] or '–'}", f"{r['price_fcst']:,.1f}",
                   f"{r['change_pct']:+.1f}%", delta_color="inverse",
                   help=f"80% range {r['price_lo']:,.1f}–{r['price_hi']:,.1f}; P(Alert) {r['p_alert']:.0%}")
    recent = hist[hist["month"] >= hist["month"].max() - pd.DateOffset(years=4)]
    fig = go.Figure()
    a = alps[alps["month"].isin(recent["month"])]
    if len(a):
        fig.add_trace(go.Scatter(x=a["month"], y=a["expected"], name="Seasonal trend (expected price)",
                                 line=dict(color="#868e96", dash="dot")))
    fig.add_trace(go.Scatter(x=recent["month"], y=recent["price"], mode="lines+markers", name="Observed (WFP)",
                             line=dict(color="#1c7ed6")))
    x = [r0["month"]] + list(fc["target_month"])
    fig.add_trace(go.Scatter(x=x + x[::-1], y=[r0["price_now"]] + list(fc["price_hi"]) + list(fc["price_lo"])[::-1]
                             + [r0["price_now"]], fill="toself", fillcolor="rgba(232,89,12,0.15)", mode="lines",
                             line=dict(width=0), name="80% interval", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=x, y=[r0["price_now"]] + list(fc["price_fcst"]), mode="lines+markers",
                             name="SokoRadar forecast", line=dict(color="#e8590c", width=3),
                             marker=dict(color=[PHASE_COLORS.get(p, "#adb5bd") for p in [None] + list(fc["phase"])],
                                         size=11)))
    fig.update_layout(height=420, margin=dict(l=10, r=10, t=10, b=10), yaxis_title="KES per kg",
                      legend=dict(orientation="h", y=-0.15))
    st.plotly_chart(fig, width="stretch")
    st.markdown("**Why?** " + str(fc.iloc[0]["why_en"]))
    st.caption("Kiswahili: " + str(fc.iloc[0]["why_sw"]))
    tot = table("tot")
    if len(tot) and market_key in set(tot["market_key"]):
        t = tot[(tot["market_key"] == market_key) & (tot["month"] >= tot["month"].max() - pd.DateOffset(years=5))]
        st.markdown("##### Pastoral terms of trade: kilograms of maize one goat buys")
        fig2 = px.line(t, x="month", y="tot_kg_per_goat", labels={"tot_kg_per_goat": "kg maize per goat", "month": ""})
        out = table("tot_outlook")
        if len(out) and market_key in set(out["market_key"]):
            o = out[out["market_key"] == market_key]
            fig2.add_scatter(x=o["target_month"], y=o["tot_fcst_kg"], mode="markers+lines", name="outlook",
                             line=dict(dash="dash", color="#e8590c"))
        fig2.update_layout(height=280, margin=dict(l=10, r=10, t=10, b=10), showlegend=False)
        st.plotly_chart(fig2, width="stretch")
        st.caption("Goat prices are World Bank RTFP estimates (mostly modelled). The outlook holds the goat price and "
                   "applies SokoRadar's maize forecast.")

# --------------------------------------------------------------------------- 3. what-if

with tabs[2]:
    st.markdown("How would the alerts change if the current season's rains came in at a different level? "
                "This re-scores the machine-learning forecasts with the last months' rainfall set to the chosen "
                "share of normal. It is a sensitivity test of the model, not a weather forecast.")
    c1, c2, c3 = st.columns([3, 1, 1])
    rain = c1.slider("Rainfall, % of the long-term normal", 20, 250, 100, step=10)
    months = c2.selectbox("Over the last", [1, 2, 3], index=2, format_func=lambda m: f"{m} month(s)")
    hh = c3.radio("Months ahead", [1, 2, 3], horizontal=True, key="wi_h")
    w = scenario(rain, months, store.path("forecasts.parquet").stat().st_mtime)
    w = w[w["live"] & (w["h"] == hh)]
    n0 = int(w["phase_base"].isin(["Alert", "Crisis"]).sum())
    n1 = int(w["phase"].isin(["Alert", "Crisis"]).sum())
    k1, k2, k3 = st.columns(3)
    k1.metric("Series in Alert/Crisis (baseline)", n0)
    k2.metric(f"…with rains at {rain}% of normal", n1, n1 - n0, delta_color="inverse")
    k3.metric("Mean forecast price shift", f"{np.nanmean(w['price_shift_pct']):+.1f}%")
    left, right = st.columns([3, 2])
    with left:
        worst = w.assign(rank=w["phase"].map({p: i for i, p in enumerate(PHASES)}).fillna(-1)).sort_values(
            "rank", ascending=False).groupby("market").head(1)
        st.plotly_chart(phase_map(worst, hover=[("County", "county", ""), ("Commodity", "commodity", ""),
                                                ("Baseline phase", "phase_base", ""),
                                                ("Price shift %", "price_shift_pct", "+.1f")]), width="stretch")
    with right:
        changed = w[w["phase"] != w["phase_base"]]
        st.markdown(f"**{len(changed)} forecasts change phase**")
        st.dataframe(w.sort_values("price_shift_pct", ascending=False)[
            ["county", "market", "commodity", "phase_base", "phase", "price_base", "price_fcst", "price_shift_pct"]],
            hide_index=True, width="stretch", height=480)

# --------------------------------------------------------------------------- 4. replay

with tabs[3]:
    rep, eps, summ = table("replay_monthly"), table("replay_episodes"), table("replay_summary")
    if rep.empty:
        st.info("Run the backtest stage to build crisis replays.")
    else:
        years = sorted(rep["year"].unique())
        c1, c2 = st.columns([1, 3])
        year = c1.selectbox("Drought year", years, index=0)
        r = rep[rep["year"] == year]
        months_ = sorted(r["month"].unique())
        month = c2.select_slider("Month", options=months_, value=months_[len(months_) // 2],
                                 format_func=lambda m: f"{pd.Timestamp(m):%b %Y}")
        s = summ[summ["year"] == year].iloc[0]
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Alert episodes (ALPS)", int(s["episodes"]))
        k2.metric("Warned ≥ 1 month earlier", f"{int(s['warned_earlier'])}",
                  f"{s['share_warned_earlier']:.0%}" if pd.notna(s["share_warned_earlier"]) else None)
        k3.metric("Mean lead time", f"{s['mean_lead_months']:.1f} months" if pd.notna(s["mean_lead_months"]) else "–")
        k4.metric("Flags followed by Alert", f"{int(s['flags_followed_by_alert'])}/{int(s['flags'])}")
        cur = r[r["month"] == month].copy()
        cur["market_key"] = cur["series_id"].str.split("|").str[0]
        cur = cur.merge(table("markets")[["market_key", "latitude", "longitude"]], on="market_key", how="left")
        rank = {p: i for i, p in enumerate(PHASES)}

        def worst_by_market(d: pd.DataFrame, col: str) -> pd.DataFrame:
            d = d.assign(_r=d[col].map(rank).fillna(-1)).sort_values("_r", ascending=False)
            items = d.groupby("market_key")["commodity"].apply(lambda c: ", ".join(sorted(set(c.map(LABELS)))))
            return d.groupby("market_key").head(1).assign(series=lambda x: x["market_key"].map(items))

        left, right = st.columns(2)
        with left:
            st.plotly_chart(phase_map(worst_by_market(cur, "alps_phase"), "alps_phase",
                                      title=f"What WFP's ALPS showed in {pd.Timestamp(month):%b %Y}",
                                      hover=[("Series", "series", ""), ("Highest ALPS", "alps", ".2f")], height=460),
                            width="stretch")
        with right:
            st.plotly_chart(phase_map(worst_by_market(cur, "sokoradar_phase"), "sokoradar_phase",
                                      title="What SokoRadar forecast for the next 3 months",
                                      hover=[("Series", "series", ""), ("P(Alert)", "p_alert_max", ".0%")],
                                      height=460), width="stretch")
        e = eps[eps["year"] == year].copy()
        if len(e):
            e["series"] = e["series_id"].str.replace("|", " · ", regex=False)
            st.markdown("**When each market was first flagged**")
            st.dataframe(e[["series", "onset", "first_warning", "lead_months", "peak_alps_6m"]].rename(
                columns={"onset": "ALPS reached Alert", "first_warning": "SokoRadar first warned",
                         "lead_months": "months earlier", "peak_alps_6m": "peak ALPS (6 m)"}),
                hide_index=True, width="stretch",
                column_config={"ALPS reached Alert": st.column_config.DateColumn(format="MMM YYYY"),
                               "SokoRadar first warned": st.column_config.DateColumn(format="MMM YYYY")})
        st.caption("Replays use backtest forecasts from models trained only on data before each year.")

# --------------------------------------------------------------------------- 5. accuracy

with tabs[4]:
    st.subheader("Proposal success metrics")
    tg = pd.DataFrame(metrics.get("targets", []))
    if len(tg):
        tg["status"] = np.where(tg["met"], "✅ met", "❌ not met")
        st.dataframe(tg[["area", "metric", "target", "value", "status"]], hide_index=True, width="stretch")
    st.caption(f"Backtest {metrics['metric_years'][0]}–{metrics['metric_years'][1]}: rolling origin, each year forecast "
               f"by models trained only on earlier years. {metrics['holdout_year']} is the untouched holdout.")
    pt = pd.DataFrame(metrics.get("point", []))
    if len(pt):
        pt["h"] = pt["h"].astype(str)
        c1, c2 = st.columns(2)
        fig = px.bar(pt[pt["model"] != "snaive"], x="model", y="rel_mae_naive", color="h", barmode="group",
                     labels={"rel_mae_naive": "MAE ÷ naive MAE (lower is better)", "h": "months ahead"},
                     title="Price forecasts against the last-month (naive) forecast")
        fig.add_hline(y=1.0, line_dash="dot")
        fig.add_hline(y=0.9, line_dash="dash", line_color="#2e7d32", annotation_text="proposal target")
        fig.update_yaxes(range=[0.8, 1.3])
        c1.plotly_chart(fig, width="stretch")
        hold = pd.DataFrame(metrics.get("point_holdout", []))
        if len(hold):
            hold["h"] = hold["h"].astype(str)
            fig = px.bar(hold[hold["model"] != "snaive"], x="model", y="rel_mae_naive", color="h", barmode="group",
                         labels={"rel_mae_naive": "MAE ÷ naive MAE", "h": "months ahead"},
                         title=f"Holdout year {metrics['holdout_year']}")
            fig.add_hline(y=1.0, line_dash="dot")
            fig.update_yaxes(range=[0.8, 1.3])
            c2.plotly_chart(fig, width="stretch")
        st.dataframe(pt.pivot_table(index="model", columns="h", values=["mae_pct", "mase", "rel_mae_naive"]).round(3),
                     width="stretch")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**80% interval coverage**")
        st.dataframe(pd.DataFrame(metrics.get("intervals", [])).round(3), hide_index=True, width="stretch")
    with c2:
        st.markdown("**Alerts (Alert or Crisis at t+h)**")
        al = pd.DataFrame(metrics.get("alerts", []))
        if len(al):
            st.dataframe(al[["h", "route", "recall", "precision", "f1", "onset_recall", "positives"]].round(3)
                         if "onset_recall" in al else al.round(3), hide_index=True, width="stretch")
    by_year = table("metrics_by_year")
    if len(by_year):
        fig = px.line(by_year[by_year["model"].isin(["lgbm", "sokoradar"])], x="fold", y="rel_mae_naive",
                      color="model", facet_col="h", markers=True,
                      labels={"fold": "year", "rel_mae_naive": "MAE ÷ naive"}, title="Accuracy by year")
        fig.add_hline(y=1.0, line_dash="dot")
        st.plotly_chart(fig, width="stretch")
    sc = table("scorecard")
    if len(sc):
        st.markdown("**Track record: archived forecasts scored against prices observed since**")
        g = sc.groupby(["source", "h"]).agg(forecasts=("abs_pct_error", "size"),
                                             mean_abs_error_pct=("abs_pct_error", "mean"),
                                             inside_80pct_interval=("inside_interval", "mean")).reset_index()
        st.dataframe(g.round(3), hide_index=True, width="stretch")

# --------------------------------------------------------------------------- 6. USSD

with tabs[5]:
    from sokoradar.ussd import UssdData, UssdSession

    st.markdown(f"Simulates the USSD menu (`{settings()['ussd']['service_code']}`) exactly as Africa's Talking "
                "would call it. Type a choice and press **Send**; **Hang up** starts again.")
    if "ussd_path" not in st.session_state:
        st.session_state.ussd_path = []
    session = UssdSession(UssdData(LIVE.copy(), {k: (v["label_sw"], v["label"])
                                               for k, v in settings()["commodities"].items()}))
    reply = session.handle("*".join(st.session_state.ussd_path))
    col, _ = st.columns([1, 2])
    with col:
        st.markdown(f"""<div style="background:#212529;border-radius:28px;padding:22px 18px;max-width:300px">
<div style="background:#d3f9d8;color:#111;font-family:monospace;font-size:14px;border-radius:8px;padding:12px;
min-height:220px;white-space:pre-wrap">{reply[4:]}</div></div>""", unsafe_allow_html=True)
        with st.form("ussd", clear_on_submit=True):
            choice = st.text_input("Reply", disabled=reply.startswith("END"), placeholder="e.g. 2")
            b1, b2 = st.columns(2)
            send = b1.form_submit_button("Send", disabled=reply.startswith("END"))
            hang = b2.form_submit_button("Hang up")
        if send and choice.strip():
            st.session_state.ussd_path.append(choice.strip())
            st.rerun()
        if hang:
            st.session_state.ussd_path = []
            st.rerun()
        st.caption(f"Screen {len(st.session_state.ussd_path) + 1} · {len(reply) - 4} characters")

# --------------------------------------------------------------------------- 7. about

with tabs[6]:
    st.markdown("""
**What SokoRadar does.** For every WFP-monitored Kenyan market series with at least 36 months of prices, it forecasts
the retail or wholesale price 1, 2 and 3 months ahead, wraps each forecast in an 80% conformal interval, converts it to
WFP's Alert for Price Spikes (ALPS) phase — Normal, Stress, Alert, Crisis — and names the main drivers.

**How.** An adaptive ensemble of global LightGBM/XGBoost models (trained across all markets), per-market SARIMAX and
Prophet models and simple baselines, weighted by recent out-of-sample accuracy for each commodity and price type.
Alerts come from the route that scored best in backtests. ALPS is computed on an expanding window, so no future prices
leak into past phases.

**Read with care.** Forecasts are early warning, not certainty and not trading advice. Data are patchy in remote
markets: every forecast shows how old its latest price is, and series without a price in the last three months are
not shown as live. Policy shocks (subsidies, import waivers) are only partly visible to the models; published alerts
should pass a human review.
""")
    st.markdown("**Sources and licences**")
    st.markdown("""
- WFP market prices for Kenya, via the Humanitarian Data Exchange (CC BY-IGO)
- CHIRPS rainfall and MODIS NDVI subnational indicators (WFP, Climate Hazards Center, NASA), via HDX
- Central Bank of Kenya indicative exchange rates; World Bank Commodity Price Data (Pink Sheet)
- World Bank Real-Time Food Prices (benchmark and goat prices only; mostly modelled estimates)
- County boundaries: OCHA common operational dataset via HDX
""")
    q = js("data_quality")
    if q:
        st.markdown(f"**Data in this release:** {q['series_kept']} series, {q['observations_kept']:,} monthly prices, "
                    f"{q['markets_kept']} markets, {q['kept_period']}.")
    st.markdown("API: [`/docs`](/docs) on the SokoRadar API · Code and data dictionary in the project repository.")
