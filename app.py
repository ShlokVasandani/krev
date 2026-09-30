"""Krev - AI decision layer for public-health supply resilience.

Run:  streamlit run app.py
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pydeck as pdk
import streamlit as st

import ui
from core.brief import action_brief
from core.counterfactual import POLICIES, compare
from core.features import ALERT_WINDOW
from core.federated import fedavg
import time

from core.geo import load_districts
from core.forecast import (FED_L2, FED_LR, FED_STEPS, federated_benchmark, forecast_at, forecast_table,
                           state_clients, train_model)
from core.ingest import parse_sms
from core.network import DISTRICTS, ITEM_BY_ID, ITEMS, STATES, district_distance_matrix
from core.optimizer import escalations, recommend_transfers
from core.security import KeyRegistry
from core.simdata import DATA_POOR_HISTORY, DATA_POOR_STATE, T_NOW, Scenario, generate_world, outbreak_field

st.set_page_config(page_title="Krev · Supply resilience", page_icon=":material/monitor_heart:", layout="wide")
ui.inject_css()


def rgb(h, a=255):
    h = h.lstrip("#")
    return [int(h[k:k + 2], 16) for k in (0, 2, 4)] + [a]


# ----------------------------------------------------------------- cached computation
@st.cache_resource(show_spinner="Training the federated outbreak model across three state nodes")
def get_model():
    return train_model(generate_world(Scenario()))


@st.cache_resource(show_spinner="Simulating the network")
def get_world(epi, days, inten):
    return generate_world(Scenario(epi, days, inten) if epi else Scenario())


@st.cache_resource(show_spinner="Replaying the next 21 days under three policies")
def get_impact(epi, days, inten):
    return compare(get_world(epi, days, inten), get_model(), horizon=21)


@st.cache_resource(show_spinner="Benchmarking local, federated and centralized training")
def get_benchmark():
    return federated_benchmark(generate_world(Scenario()), DATA_POOR_STATE)


@st.cache_resource(show_spinner="Running federated rounds")
def get_security_run(kind):
    w = generate_world(Scenario())
    clients = state_clients(w, district_distance_matrix(), 0, T_NOW)
    attack = None if kind == "none" else {"state": "Karnataka", "round": 3, "kind": kind}
    _, log = fedavg(clients, rounds=6, local_steps=FED_STEPS, lr=FED_LR, l2=FED_L2, registry=KeyRegistry(),
                    attack=attack)
    return log


# ----------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown('<div class="kv-brand" style="margin:4px 0 2px 0"><span class="kv-mark">K</span>Krev</div>'
                '<div class="kv-crumb" style="margin-bottom:6px">Scenario controls</div>', unsafe_allow_html=True)
    ui.sidebar_label("Scenario")
    mode = st.segmented_control("Scenario", ["Normal season", "Dengue outbreak"], default="Dengue outbreak",
                                label_visibility="collapsed") or "Dengue outbreak"
    if mode == "Dengue outbreak":
        epi = st.selectbox("Epicentre district", [d[0] for d in DISTRICTS], index=1)
        days = st.slider("Days since onset", 4, 14, 9, help="IDSP issues an outbreak alert about 3 days after local onset")
        inten = st.slider("Intensity (× peak-season cases)", 2.0, 6.0, 4.0, 0.5)
    else:
        epi, days, inten = None, 9, 4.0
    ui.sidebar_label("Map")
    item_sel = st.selectbox("Medicine shown on map", ["Worst across all medicines"] + [it.name for it in ITEMS])
    ui.sidebar_label("Integrations")
    api_key = st.text_input("Gemini API key", type="password", value=os.environ.get("GEMINI_API_KEY", ""),
                            help="Optional. Used only to write the plain-language action brief.")
    st.markdown('<div class="kv-side-foot">Synthetic network: 60 facilities across 12 districts in 3 states, '
                'two years of daily history with state-specific dengue seasons and past outbreaks. '
                'No real patient data.</div>', unsafe_allow_html=True)

model = get_model()
world = get_world(epi, days, inten)
fac = world.fac

key = (epi, days, inten)
updates = st.session_state.setdefault("sms_updates", {}).setdefault(key, [])
stock = world.stock.copy()
for u in updates:
    f = int(np.where(fac.code.values == u["code"])[0][0])
    stock[f, [it.id for it in ITEMS].index(u["item_id"])] = u["qty"]

fc = forecast_at(model, world, T_NOW, stock, world.incoming)
risks = forecast_table(world, fc, stock)
transfers = recommend_transfers(world, fc, stock)
esc = escalations(world, fc, stock, transfers)
impact = get_impact(epi, days, inten)
active_alerts = sorted({DISTRICTS[d][0] for day, d in world.alerts if 0 <= T_NOW - day < ALERT_WINDOW})
so = {p: impact[p]["so_days"].sum() for p in POLICIES}
un = {p: impact[p]["unmet"].sum() for p in POLICIES}

# ----------------------------------------------------------------- header
today = world.dates[T_NOW]
if epi:
    title = f"Dengue outbreak · {epi}"
    meta = [f"Day {days} since onset", f"Intensity {inten:g}×", f"As of {today:%d %b %Y}"]
else:
    title = "Normal season"
    meta = ["No outbreak injected", f"As of {today:%d %b %Y}"]
status = ((ui.RISK["HIGH"], f"{len(active_alerts)} districts under IDSP outbreak alert") if active_alerts
          else (ui.RISK["LOW"], "No active outbreak alerts"))
ui.brand_bar("<b>3</b> states · <b>12</b> districts · <b>60</b> facilities monitored")
ui.page_header(title, meta, status)

n_high, n_med = int((risks.risk == "HIGH").sum()), int((risks.risk == "MEDIUM").sum())
n_inter = int(transfers.interstate.sum()) if len(transfers) else 0
avoided = 1 - so["forecast+redistribution"] / max(so["static"], 1)
ui.stat_strip([
    dict(label="High stock-out risk", value=n_high, unit="items", sub=f"{n_med} more at medium risk", tone=ui.RISK["HIGH"]),
    dict(label="Districts under alert", value=len(active_alerts), sub=", ".join(active_alerts) or "None", tone="#C8612F"),
    dict(label="Transfers recommended", value=len(transfers), sub=f"{n_inter} need inter-state sign-off"),
    dict(label="Unresolved shortfalls", value=len(esc), sub="Escalated to state procurement", tone=ui.RISK["MEDIUM"]),
    dict(label="Stock-out days avoided", value=f"{avoided:.0%}",
         sub=f"{so['static']:.0f} → {so['forecast+redistribution']:.0f} over the next 21 days", tone=ui.RISK["LOW"]),
])

tabs = st.tabs(["Overview", "Redistribution", "Impact", "Federated model", "Facility", "Field reporting", "Method"])

# ----------------------------------------------------------------- overview
TOOLTIP = {"html": "<div style='font-weight:700;margin-bottom:4px'>{name}</div>{tip}",
           "style": {"backgroundColor": "#FFFFFF", "color": ui.INK, "fontFamily": "Inter, sans-serif",
                     "fontSize": "12px", "lineHeight": "1.5", "border": f"1px solid {ui.LINE}", "borderRadius": "6px",
                     "padding": "10px 12px", "boxShadow": "0 6px 20px rgba(0,0,0,0.10)"}}
DAYS_AHEAD = 14


def spread_layer(day_since_onset):
    """District choropleth of outbreak intensity on a given day since onset."""
    feats, lat, lon = load_districts()
    sc = world.scenario
    vals = outbreak_field(sc, lat, lon, day_since_onset) if sc.epicentre else np.zeros(len(feats))
    alert_d = {DISTRICTS[d][0] for day, d in world.alerts if 0 <= T_NOW - (days - day_since_onset) - day < ALERT_WINDOW}
    out = []
    for f, v in zip(feats, vals):
        col = ui.spread_color(v)
        mon = bool(f["monitored"])
        alert = any(m in alert_d for m in f["monitored"])
        fill = rgb(col, 205) if col else (rgb("#FFFFFF", 150) if mon else rgb("#FFFFFF", 40))
        line = rgb(ui.RISK["HIGH"], 255) if alert else (rgb(ui.NAV, 190) if mon else rgb("#9A948A", 90))
        width = 2500 if alert else (1200 if mon else 250)          # metres, clamped to 0.3-2.5 px
        if v >= 0.1:
            tip = f"{f['state']}<br/>Outbreak intensity {v:.1f}× seasonal peak"
        else:
            tip = f"{f['state']}<br/>No outbreak activity"
        tip += ("<br/><b>IDSP alert active</b>" if alert else "") + ("<br/>Monitored by Krev" if mon else "")
        out.append({"type": "Feature", "geometry": f["geometry"], "name": f["name"], "tip": tip,
                    "properties": {"fill": fill, "line": line, "w": width}})
    return pdk.Layer("GeoJsonLayer", {"type": "FeatureCollection", "features": out}, stroked=True, filled=True,
                     get_fill_color="properties.fill", get_line_color="properties.line",
                     get_line_width="properties.w", line_width_min_pixels=0.3, line_width_max_pixels=2.5,
                     pickable=True,
                     update_triggers={"get_fill_color": [day_since_onset]})


def facility_layers():
    if item_sel.startswith("Worst"):
        worst = risks.loc[risks.groupby("fid").stockout_prob.idxmax()].set_index("fid")
    else:
        worst = risks[risks.item == item_sel].set_index("fid")
    m = fac.loc[worst.index].copy()
    m["color"] = [rgb(ui.RISK[l], 255) for l in worst.risk]
    m["radius"] = m.tier.map({"DH": 9500, "CHC": 7000, "PHC": 5000})
    m["tip"] = [
        f"{ui.RISK_LABEL[r.risk]} risk · {r.item}<br/>{r.stockout_prob:.0%} chance of stock-out before resupply<br/>"
        f"{r.stock:,.0f} {r.unit} on hand · {r.days_cover:.1f} of {r.lead_days:.0f} days covered<br/>"
        f"Beds {min(world.occupancy[f, T_NOW], 1):.0%} occupied · staff {world.staff_present[f, T_NOW]}/{fac.staff_sanctioned.iloc[f]}"
        for f, r in zip(worst.index, worst.itertuples())]
    layers = []
    tr_map = transfers if item_sel.startswith("Worst") or transfers.empty else transfers[transfers.item == item_sel]
    if len(tr_map):
        a = tr_map.copy()
        a["slat"], a["slon"] = fac.lat.values[a.from_fid], fac.lon.values[a.from_fid]
        a["tlat"], a["tlon"] = fac.lat.values[a.to_fid], fac.lon.values[a.to_fid]
        a["name"] = a.frm + " → " + a.to
        a["tip"] = (a.qty.map("{:,}".format) + " " + a.unit + " · " + a.item + "<br/>"
                    + a.road_km.round().astype(int).astype(str) + " km, about " + a.days.astype(str) + " day(s)")
        a["c"] = [rgb(ui.VIOLET if x else ui.ACCENT, 200) for x in a.interstate]
        layers.append(pdk.Layer("ArcLayer", a, get_source_position="[slon, slat]", get_target_position="[tlon, tlat]",
                                get_source_color="c", get_target_color="c", get_width=1.8, get_height=0.35,
                                pickable=True))
    layers.append(pdk.Layer("ScatterplotLayer", m.reset_index(), get_position="[lon, lat]", get_radius="radius",
                            get_fill_color="color", get_line_color=[255, 255, 255, 255], stroked=True,
                            line_width_min_pixels=1.5, pickable=True, radius_min_pixels=4))
    return layers


def draw_map(target, day_since_onset):
    layers = [spread_layer(day_since_onset)]
    if not epi or day_since_onset == days:
        layers += facility_layers()
    if epi:
        ed = next(d for d in DISTRICTS if d[0] == epi)
        view = pdk.ViewState(latitude=ed[2] - 0.6, longitude=ed[3] + 0.6, zoom=5.35)
    else:
        view = pdk.ViewState(latitude=20.6, longitude=76.6, zoom=4.45)
    target.pydeck_chart(pdk.Deck(layers=layers, initial_view_state=view, tooltip=TOOLTIP,
                                 map_style=pdk.map_styles.CARTO_LIGHT), height=560)


with tabs[0]:
    left, right = st.columns([13, 9], gap="medium")
    with left:
        with st.container(border=True):
            ui.section("Outbreak spread and facility risk",
                       "Districts are shaded by dengue outbreak intensity; dots are facilities coloured by their chance "
                       "of running out before the next delivery.")
            if epi:
                c1, c2 = st.columns([6, 1], vertical_alignment="bottom")
                day_sel = c1.slider("Outbreak timeline", 0, days + DAYS_AHEAD, days, format="Day %d",
                                    help=f"Day {days} is today. Earlier days replay the outbreak; later days show the "
                                         f"simulated spread ahead.")
                play = c2.button("Play", width="stretch", help="Animate the spread from onset to two weeks ahead")
                banner_slot, map_slot = st.empty(), st.empty()

                def banner_for(dd):
                    if dd == days:
                        return (f"<b>Today, day {days} of the outbreak.</b> Facility risk and recommended transfers "
                                f"are shown for today.")
                    rel = dd - days
                    when = f"{-rel} days ago" if rel < 0 else f"{rel} days ahead"
                    return (f"<b>Day {dd} ({when}).</b> Showing district spread only; facility risk and transfers "
                            f"are shown for today.")

                if play:
                    for dd in range(0, days + DAYS_AHEAD + 1):
                        banner_slot.markdown(f'<div class="kv-banner">{banner_for(dd)}</div>', unsafe_allow_html=True)
                        draw_map(map_slot, dd)
                        time.sleep(0.45)
                banner_slot.markdown(f'<div class="kv-banner">{banner_for(day_sel)}</div>', unsafe_allow_html=True)
                draw_map(map_slot, day_sel)
            else:
                draw_map(st, days)
            ui.legend([
                ("Outbreak", [(c, l, "sq") for (_, c), l in zip(ui.SPREAD[1:], ["0.5×", "1×", "2×", "3×", "3×+"])]),
                ("Facility risk", [(ui.RISK["HIGH"], "High", "dot"), (ui.RISK["MEDIUM"], "Medium", "dot"),
                                   (ui.RISK["LOW"], "Low", "dot")]),
                ("", [(ui.RISK["HIGH"], "IDSP alert", "outline"), (ui.ACCENT, "Transfer", "line"),
                      (ui.VIOLET, "Inter-state", "line")]),
            ])
            ui.note("Outbreak intensity is shown as a multiple of each district's normal peak-season dengue cases, "
                    "from the scenario's spread model. District boundaries are simplified and for illustration.")
    with right:
        with st.container(border=True):
            ui.section("Early warnings", "Ranked by probability of stock-out before the next resupply arrives.")
            top = risks[risks.risk != "LOW"].sort_values("stockout_prob", ascending=False).head(40)
            short = {"pcm": "Paracetamol", "ors": "ORS", "ivns": "IV saline", "ns1": "NS1 test kits", "plt": "Platelets"}
            rows = [[str(k + 1),
                     f'<b style="font-weight:600">{ui.e(short[r.item_id])}</b>'
                     f'<span class="sub">{ui.e(r.facility)} · {ui.e(r.district)}</span>',
                     ui.risk_cell(r.risk, r.stockout_prob),
                     f'{r.days_cover:.1f} d<span class="sub">resupply {r.lead_days:.0f} d</span>']
                    for k, r in enumerate(top.itertuples())]
            if rows:
                ui.html_table([("#", "idx"), ("Medicine and facility", ""), ("Stock-out risk", ""), ("Stock left", "num")],
                              rows, max_height=720)
            else:
                st.markdown('<div class="kv-callout">No facility is at medium or high risk.</div>',
                            unsafe_allow_html=True)

    st.write("")
    ui.section("Risk by district and medicine", "Worst facility in each district. Read across a row to see which "
               "medicines a district will run out of first.")
    grid = risks.pivot_table(index="district", columns="item", values="stockout_prob", aggfunc="max")
    grid = grid.reindex([d[0] for d in DISTRICTS])[[it.name for it in ITEMS]]
    z = grid.values
    fig = go.Figure(go.Heatmap(
        z=z, x=[it.name for it in ITEMS], y=grid.index, zmin=0, zmax=1, xgap=2, ygap=2,
        colorscale=[[0, "#F3F2EE"], [0.2, "#F6E3B4"], [0.5, "#E9A25B"], [0.8, "#D2563F"], [1, "#A82A22"]],
        text=[[("" if np.isnan(v) else f"{v:.0%}") for v in row] for row in z], texttemplate="%{text}",
        textfont=dict(size=11), colorbar=dict(title="", tickformat=".0%", thickness=8, len=0.9, outlinewidth=0,
                                              tickfont=dict(color=ui.FAINT, size=11)),
        hovertemplate="%{y} · %{x}<br>%{z:.0%} stock-out probability<extra></extra>"))
    ui.style_fig(fig, height=420, legend_top=False)
    fig.update_xaxes(side="top", showline=False, ticks="", tickfont=dict(color=ui.MUTED))
    fig.update_yaxes(autorange="reversed", gridcolor="rgba(0,0,0,0)", tickfont=dict(color=ui.MUTED))
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
    ui.note("Platelets are held only at district hospital blood banks, so other tiers show no value.")

# ----------------------------------------------------------------- redistribution
with tabs[1]:
    ui.section("Recommended transfers", "Solved as a minimum-cost transport problem. Donors keep enough for their "
               "own resupply window plus 7 days; receivers are topped up to the 90th-percentile forecast. Respects "
               "distance and shelf life, blood-bank tier, minimum batch size, and prefers stock close to expiry.")
    if transfers.empty:
        st.markdown('<div class="kv-callout">No transfers needed. Every at-risk facility is covered by incoming supply.</div>',
                    unsafe_allow_html=True)
    else:
        ed = pd.DataFrame({
            "Approve": ~transfers.interstate, "Medicine": transfers.item, "Quantity": transfers.qty,
            "Unit": transfers.unit, "From": transfers.frm, "To": transfers.to,
            "Distance (km)": transfers.road_km.round().astype(int), "ETA (days)": transfers.days,
            "Scope": np.where(transfers.interstate, "Inter-state", "Intra-state"),
            "Risk before": (transfers.prob_before * 100).round().astype(int),
            "Risk after": (transfers.prob_after * 100).round().astype(int), "Rationale": transfers.why})
        out = st.data_editor(ed, hide_index=True, width="stretch", height=400,
                             disabled=[c for c in ed.columns if c != "Approve"],
                             column_config={
                                 "Approve": st.column_config.CheckboxColumn(width=70),
                                 "Quantity": st.column_config.NumberColumn(format="%d"),
                                 "Risk before": st.column_config.NumberColumn(format="%d%%"),
                                 "Risk after": st.column_config.NumberColumn(format="%d%%"),
                                 "Rationale": st.column_config.TextColumn(width="large")})
        ui.note(f"{int(out.Approve.sum())} of {len(out)} approved. Inter-state transfers start unapproved because "
                f"they need sign-off from both state drug stores.")

    st.write("")
    c1, c2 = st.columns(2, gap="large")
    with c1:
        ui.section("Escalations", "Shortfall that no nearby facility can cover. Raised as emergency procurement.")
        if esc.empty:
            st.markdown('<div class="kv-callout">Redistribution covers every projected shortfall.</div>',
                        unsafe_allow_html=True)
        else:
            ui.html_table([("District", ""), ("Medicine", ""), ("Shortfall", "num"), ("Facilities", "num")],
                          [[f'{ui.e(r.district)}<span class="sub">{ui.e(r.state)}</span>', ui.e(r.item),
                            f"{r.shortfall:,.0f} {ui.e(r.unit)}", str(r.facilities)] for r in esc.itertuples()],
                          max_height=360)
    with c2:
        ui.section("Action brief", "A plain-language summary for the District Health Officer, written by Gemini "
                   "from the figures above. It explains; it never computes.")
        if st.button("Generate brief", type="primary"):
            with st.spinner("Writing"):
                txt, src = action_brief(title, risks, transfers, esc, active_alerts, api_key or None)
            st.session_state["brief"] = (txt, src)
        if "brief" in st.session_state:
            txt, src = st.session_state["brief"]
            with st.container(border=True):
                st.markdown(txt)
            ui.note(f"Generated with {src}.")

# ----------------------------------------------------------------- impact
with tabs[2]:
    ui.section("Same outbreak, three policies", "The next 21 days of true demand are replayed under each policy. "
               "Central resupply takes 7 to 12 days in all three. Forecasts are re-run each day using only the "
               "data available that day.")
    items = []
    for p in POLICIES:
        sub = (f"{un[p]:,.0f} doses or units unavailable" if p == "static"
               else f"{1 - un[p] / max(un['static'], 1):.0%} less unmet demand than without")
        items.append(dict(label=ui.POLICY_SHORT[p], value=f"{so[p]:.0f}", unit="stock-out days", sub=sub,
                          good=p != "static"))
    ui.stat_strip(items)

    def policy_chart(col, ytitle):
        fig = go.Figure()
        for p in POLICIES:
            d = impact[p]["daily"]
            fig.add_trace(go.Scatter(x=d.date, y=d[col], name=ui.POLICY_SHORT[p], mode="lines",
                                     line=dict(width=2.2 if p != "forecast" else 1.6, color=ui.POLICY_COLOR[p]),
                                     hovertemplate="%{y:,.0f}"))
        ui.style_fig(fig, height=320)
        fig.update_layout(hovermode="x unified", yaxis_title=ytitle)
        fig.update_xaxes(tickformat="%d %b")
        fig.update_yaxes(rangemode="tozero")
        return fig

    for p in POLICIES:
        impact[p]["daily"]["cum_unmet"] = impact[p]["daily"].unmet_units.cumsum()
    c1, c2 = st.columns(2, gap="large")
    with c1:
        ui.section("Stocked-out facility-medicines per day")
        st.plotly_chart(policy_chart("stockouts", "count"), width="stretch", config={"displayModeBar": False})
    with c2:
        ui.section("Cumulative doses and units patients could not get")
        st.plotly_chart(policy_chart("cum_unmet", "units"), width="stretch", config={"displayModeBar": False})

    ui.section("Unmet demand by medicine")
    per = impact["static"]["per_item"]
    rows = []
    for k, it in enumerate(ITEMS):
        vals = [impact[p]["per_item"].unmet_units.iloc[k] for p in POLICIES]
        cut = 1 - vals[2] / vals[0] if vals[0] > 0 else 0
        rows.append([ui.e(it.name)] + [f"{v:,.0f}" for v in vals] + [f"{cut:.0%}" if vals[0] > 0 else "–"])
    ui.html_table([("Medicine", "")] + [(ui.POLICY_SHORT[p], "num") for p in POLICIES] + [("Reduction", "num")], rows)

# ----------------------------------------------------------------- federated
with tabs[3]:
    bench, curve = get_benchmark()
    ui.section("Outbreak model trained across states without pooling data",
               f"Cold-start test: a state joins with only {DATA_POOR_HISTORY} days of its own data. Each model is "
               f"scored on that state's two earlier dengue seasons, which were withheld. Error is WAPE on the "
               f"14-day case forecast; lower is better.")
    d = bench.set_index("state").loc[DATA_POOR_STATE]
    c1, c2 = st.columns([3, 2], gap="large")
    with c1:
        cols = [("persistence", "No model", "#D6D0C2"), ("local_only", "Local data only", "#C8612F"),
                ("federated", "Federated", ui.ACCENT), ("centralized", "Centralized", ui.ACCENT_SOFT)]
        fig = go.Figure(go.Bar(x=[l for _, l, _ in cols], y=[d[c] for c, _, _ in cols],
                               marker=dict(color=[k for _, _, k in cols], line=dict(width=0)),
                               text=[f"{d[c]:.0%}" for c, _, _ in cols], textposition="outside",
                               textfont=dict(color=ui.INK, size=12), cliponaxis=False,
                               hovertemplate="%{x}: %{y:.1%}<extra></extra>"))
        ui.style_fig(fig, height=330, legend_top=False)
        fig.update_yaxes(tickformat=".0%", rangemode="tozero")
        fig.update_xaxes(showline=False, ticks="", tickfont=dict(color=ui.MUTED, size=12))
        ui.section(f"{DATA_POOR_STATE}: forecast error by training approach")
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
    with c2:
        ui.section("All states, same test")
        rows = [[ui.e(r.state), f"{r.persistence:.0%}", f"{r.local_only:.0%}",
                 f'<b style="color:{ui.ACCENT};font-weight:600">{r.federated:.0%}</b>', f"{r.centralized:.0%}"]
                for r in bench.itertuples()]
        ui.html_table([("State", ""), ("No model", "num"), ("Local", "num"), ("Federated", "num"),
                       ("Central", "num")], rows)
        ui.note("Shared: signed weight updates and aggregate scaling statistics. Not shared: surveillance or "
                "patient records. Facility-level demand coefficients stay local. Aggregate stock levels are shared "
                "for redistribution; federation protects training data, not operational stock signals.")

    st.write("")
    ui.section("Securing the federated channel", "Every update is signed per node and round. The aggregator "
               "verifies signatures and rejects statistical outliers, then revokes and rotates the node's key.")
    kind = st.segmented_control("Attack", ["none", "tamper", "poison"], default="none",
                                format_func={"none": "No attack", "tamper": "Update altered in transit",
                                             "poison": "Compromised node, poisoned update"}.get,
                                label_visibility="collapsed") or "none"
    log = get_security_run(kind)
    rd = pd.DataFrame(log.rounds)
    c1, c2 = st.columns([3, 2], gap="large")
    with c1:
        piv = rd.pivot(index="round", columns="state", values="accepted")[STATES]
        rows = [[f"Round {r}"] + [(f'<span class="kv-status" style="color:{ui.RISK["LOW"]}">Accepted</span>' if ok else
                                   f'<span class="kv-status" style="color:{ui.RISK["HIGH"]}">Rejected</span>')
                                  for ok in piv.loc[r]] for r in piv.index]
        ui.html_table([("", "")] + [(s, "") for s in STATES], rows)
    with c2:
        if log.security:
            lines = "".join(f"<div><code>Round {x['round']}</code><span><b>{ui.e(x['state'])}</b> · {ui.e(x['event'])}</span></div>"
                            for x in log.security)
        else:
            lines = "<div><code>Rounds 1–6</code><span>All updates verified and aggregated.</span></div>"
        st.markdown(f'<div class="kv-callout kv-log">{lines}</div>', unsafe_allow_html=True)
        ui.note("Transport: mutual TLS 1.3 with short-lived certificates, rotated on schedule and immediately after a "
                "security event. Post-quantum path: hybrid ML-KEM key exchange.")

# ----------------------------------------------------------------- facility
with tabs[4]:
    c1, c2, c3 = st.columns(3)
    dnames = [d[0] for d in DISTRICTS]
    dist_sel = c1.selectbox("District", dnames, index=dnames.index(epi) if epi else 1)
    fnames = fac.name[fac.district == dist_sel].tolist()
    worst_f = risks[risks.district == dist_sel].sort_values("stockout_prob").facility.iloc[-1]
    fsel = c2.selectbox("Facility", fnames, index=fnames.index(worst_f))
    f = int(np.where(fac.name.values == fsel)[0][0])
    fr = risks[risks.fid == f].sort_values("stockout_prob")
    inames = [it.name for k, it in enumerate(ITEMS) if world.elig[f, k]]
    isel = c3.selectbox("Medicine", inames, index=inames.index(fr.item.iloc[-1]))
    i = [it.name for it in ITEMS].index(isel)
    it = ITEMS[i]
    row = risks[(risks.fid == f) & (risks.item_id == it.id)].iloc[0]

    ui.stat_strip([
        dict(label="Stock-out risk", value=f"{row.stockout_prob:.0%}", sub=f"{ui.RISK_LABEL[row.risk]} · before resupply"),
        dict(label="On hand", value=f"{row.stock:,.0f}", unit=it.unit, sub=f"{row.incoming:,.0f} arriving in window"),
        dict(label="Days of cover", value=f"{row.days_cover:.1f}", sub=f"Resupply takes {row.lead_days:.0f} days"),
        dict(label="Dengue-driven demand", value=f"{row.dengue_share:.0%}", sub="Share of the forecast"),
        dict(label="Bed occupancy", **({"value": "Full", "sub": f"{occ - 1:.0%} over {fac.beds.iloc[f]} beds, overflow"}
                                       if (occ := world.occupancy[f, T_NOW]) > 1 else
                                       {"value": f"{occ:.0%}", "sub": f"{fac.beds.iloc[f]} beds"})),
        dict(label="Staff present", value=f"{world.staff_present[f, T_NOW]}", unit=f"/ {fac.staff_sanctioned.iloc[f]}",
             sub="Today"),
    ])
    hist = slice(T_NOW - 89, T_NOW + 1)
    dates_h, fut = world.dates[hist], world.dates[T_NOW + 1:T_NOW + 15]
    mu, sg = fc.mu[f, i], fc.sig[f, i]
    p10, p50, p90 = np.expm1(mu - 1.2816 * sg), np.expm1(mu), np.expm1(mu + 1.2816 * sg)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(fut) + list(fut[::-1]), y=[p90] * 14 + [p10] * 14, fill="toself",
                             fillcolor="rgba(31,110,102,0.12)", line=dict(width=0), name="80% forecast range",
                             hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=dates_h, y=world.demand[f, i, hist], mode="lines", name="Daily demand",
                             line=dict(width=1.6, color=ui.INK), hovertemplate="%{y:.0f}"))
    fig.add_trace(go.Scatter(x=fut, y=[p50] * 14, mode="lines", name="Median forecast",
                             line=dict(width=2, dash="dot", color=ui.ACCENT), hovertemplate="%{y:.0f}"))
    ui.style_fig(fig, height=320)
    fig.update_layout(hovermode="x unified", yaxis_title=f"{it.unit} per day")
    fig.update_xaxes(tickformat="%d %b")
    fig.update_yaxes(rangemode="tozero")
    ui.section(f"{it.name} demand", "Last 90 days and the next 14-day forecast.")
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

    d = fac.district_idx.iloc[f]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=dates_h, y=world.reported[d, hist], name="Reported cases", marker_color="#D6D0C2",
                         hovertemplate="%{y:.0f}"))
    fig.add_trace(go.Scatter(x=fut, y=[fc.cases_fut[d]] * 14, mode="lines", name="Forecast (14-day average)",
                             line=dict(width=2, dash="dot", color=ui.RISK["HIGH"]), hovertemplate="%{y:.0f}"))
    for day, dd in world.alerts:
        if dd == d and T_NOW - 89 <= day <= T_NOW:
            fig.add_vline(x=world.dates[day], line_width=1, line_dash="dot", line_color=ui.RISK["HIGH"])
            fig.add_annotation(x=world.dates[day], y=1, yref="paper", text="IDSP alert", showarrow=False,
                               xanchor="left", xshift=4, font=dict(size=11, color=ui.RISK["HIGH"]))
    ui.style_fig(fig, height=250)
    fig.update_layout(hovermode="x unified", bargap=0.25)
    fig.update_xaxes(tickformat="%d %b")
    ui.section(f"Dengue surveillance · {dist_sel}", "IDSP counts arrive with a two-day reporting lag.")
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

# ----------------------------------------------------------------- field reporting
with tabs[5]:
    ui.section("Stock reports by SMS", "A PHC pharmacist sends stock from any phone, no app or data connection "
               "needed. Krev parses the message, updates the ledger and re-runs the forecast.")
    c1, c2 = st.columns([2, 3], gap="large")
    with c1:
        txt = st.text_area("Incoming message", "STK PHC-PUN-1 PCM 60 ORS 25 NS1 8\nSTK CHC-NAS IVNS 40", height=110)
        ui.note("Format: <code>STK &lt;facility-code&gt; &lt;medicine&gt; &lt;qty&gt; …</code> · "
                "medicines PCM, ORS, IVNS, NS1, PLT")
        parsed, errs = parse_sms(txt, fac.code.tolist())
        for er in errs:
            st.warning(er)
        b1, b2 = st.columns(2)
        if b1.button("Receive message", type="primary", disabled=not parsed, width="stretch"):
            updates.extend(parsed)
            st.rerun()
        if b2.button("Clear updates", disabled=not updates, width="stretch"):
            updates.clear()
            st.rerun()
    with c2:
        if parsed:
            ui.section("Parsed")
            ui.html_table([("Facility", ""), ("Medicine", ""), ("Quantity", "num")],
                          [[ui.e(dict(zip(fac.code, fac.name))[p["code"]]), ui.e(ITEM_BY_ID[p["item_id"]].name),
                            f'{p["qty"]:,}'] for p in parsed])
        if updates:
            st.write("")
            ui.section("Applied this session", "Risk and transfers across the app now use these values.")
            aff = {(int(np.where(fac.code.values == u["code"])[0][0]), u["item_id"]) for u in updates}
            sub = risks[[(r.fid, r.item_id) in aff for r in risks.itertuples()]]
            ui.html_table([("Facility", ""), ("Medicine", ""), ("On hand", "num"), ("Cover", "num"), ("Risk", "")],
                          [[ui.e(r.facility), ui.e(r.item), f"{r.stock:,.0f}", f"{r.days_cover:.1f} d",
                            ui.risk_cell(r.risk, r.stockout_prob)] for r in sub.itertuples()])
    st.write("")
    with st.expander("Facility codes"):
        st.dataframe(fac[["code", "name", "tier", "district", "state"]].rename(columns=str.title),
                     hide_index=True, width="stretch", height=260)

# ----------------------------------------------------------------- method
with tabs[6]:
    ui.section("How Krev works", "An intelligence layer on top of existing systems (e-Aushadhi, eVIN, IDSP, "
               "eBloodServices), not a replacement.")
    steps = [
        ("Data", "Facility stock, consumption, beds and staff attendance by API sync or SMS; IDSP dengue surveillance "
                 "and outbreak alerts. Tiered like the real network: PHC, CHC, district hospital. Platelets only at "
                 "district blood banks."),
        ("Outbreak forecast", "Federated across state nodes; only signed weight updates leave a state. Uses the "
                              "district's own curve, IDSP alerts and neighbouring districts' outbreak pressure, so a "
                              "district is warned before its own cases rise."),
        ("Demand forecast", "Each facility learns its baseline and how much each dengue case adds, per medicine. "
                            "Forecast equals baseline plus response times forecast cases, with uncertainty that grows "
                            "with how dengue-driven the medicine is."),
        ("Early warning", "Probability of stock-out before the next delivery arrives, rather than a fixed "
                          "low-stock threshold."),
        ("Redistribution", "Minimum-cost transport problem with donor safety buffers, distance and shelf-life limits, "
                           "blood-bank tier, near-expiry preference and inter-state approval. What it cannot cover "
                           "becomes an emergency procurement request."),
        ("Proof", "The same outbreak is replayed with and without Krev, so impact is measured, not asserted."),
    ]
    st.markdown('<div class="kv-steps">' + "".join(
        f'<div class="kv-step"><div class="n">0{k + 1}</div><h5>{ui.e(t)}</h5><p>{ui.e(b)}</p></div>'
        for k, (t, b) in enumerate(steps)) + "</div>", unsafe_allow_html=True)
    st.write("")
    c1, c2 = st.columns(2, gap="large")
    with c1:
        ui.section("Security")
        st.markdown('<div class="kv-callout">Mutual TLS 1.3, signed model updates, a poisoning guard, key rotation on '
                    'schedule and on security events, and optional differential-privacy noise on updates. Hybrid '
                    'post-quantum (ML-KEM) key exchange is the path for national backbone links.</div>',
                    unsafe_allow_html=True)
    with c2:
        ui.section("Limits")
        st.markdown('<div class="kv-callout" style="border-left-color:#D08A0E">Synthetic data. Early-phase outbreaks '
                    'are hard to forecast; accuracy improves sharply once an IDSP alert is a few days old. '
                    'Redistribution cannot help when every nearby facility is short, which is what escalation is '
                    'for.</div>', unsafe_allow_html=True)
