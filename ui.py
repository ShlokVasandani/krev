"""Presentation layer - "government-grade" direction.

Calm and official: warm cream page, white cards, a dark teal navigation bar, one teal accent, high-contrast
type (Inter), tabular numerals. Status colour is reserved for risk and always paired with a label or number.
No emoji, no pill badges, no gradients.
"""
from __future__ import annotations

import html

import plotly.graph_objects as go
import streamlit as st

# ---- tokens
PAGE = "#F4F1EA"
CARD = "#FFFFFF"
SIDEBAR = "#EDE9DF"
INK = "#15302D"
MUTED = "#566663"
FAINT = "#8A9492"
LINE = "#E0DBCD"
LINE_SOFT = "#ECE8DE"
NAV = "#1E4642"
ACCENT = "#1F6E66"          # primary teal
ACCENT_SOFT = "#8FBDB6"
VIOLET = "#5A4AA6"          # inter-state transfers
SUBTLE = "#F8F6F0"

RISK = {"HIGH": "#B3261E", "MEDIUM": "#C4860A", "LOW": "#2F7D5B"}
RISK_LABEL = {"HIGH": "High", "MEDIUM": "Medium", "LOW": "Low"}
# outbreak intensity (sequential, warm, distinct from the risk dots)
SPREAD = [(0.10, "#F6E2CB"), (0.5, "#EFC29B"), (1.0, "#E39A6E"), (2.0, "#CF6B4E"), (3.0, "#A9432F"), (99, "#7E2A1F")]

POLICY_COLOR = {"static": "#C8612F", "forecast": "#A7AFAD", "forecast+redistribution": ACCENT}
POLICY_SHORT = {"static": "Without Krev", "forecast": "Forecast only", "forecast+redistribution": "With Krev"}

CSS = f"""
<style>
:root {{ --ink:{INK}; --muted:{MUTED}; --faint:{FAINT}; --line:{LINE}; --line-soft:{LINE_SOFT};
        --card:{CARD}; --page:{PAGE}; --accent:{ACCENT}; --nav:{NAV}; }}
html, body, [class*="css"] {{ -webkit-font-smoothing: antialiased; }}
[data-testid="stAppViewContainer"], .main {{ background: var(--page); }}
[data-testid="stHeader"] {{ background: transparent; height: 0; }}
[data-testid="stToolbar"], [data-testid="stDecoration"], [data-testid="stStatusWidget"] {{ display: none !important; }}
.block-container {{ padding-top: 1.4rem; padding-bottom: 3rem; max-width: 1480px; }}
[data-testid="stSidebarContent"] {{ padding-top: 0.4rem; }}
[data-testid="stSidebarHeader"] {{ height: 1.2rem; }}

/* navigation bar (tabs) */
[data-testid="stTabs"] [role="tablist"] {{ background: var(--nav); border-radius: 8px; padding: 0 10px; gap: 2px;
        border: none; }}
[data-testid="stTab"] {{ padding: 12px 16px !important; margin: 0 !important; border-radius: 0;
        background: transparent !important; }}
[data-testid="stTab"] p {{ font-size: 13.5px; font-weight: 500; color: #C9DAD6; }}
[data-testid="stTab"]:hover p {{ color: #FFFFFF; }}
[data-testid="stTab"][aria-selected="true"] {{ background: rgba(255,255,255,0.10) !important;
        box-shadow: inset 0 -3px 0 #E8C872; }}
[data-testid="stTab"][aria-selected="true"] p {{ color: #FFFFFF; font-weight: 600; }}
[data-testid="stTabs"] [role="tabpanel"] {{ padding-top: 20px; }}

/* bordered containers become cards */
[data-testid="stVerticalBlockBorderWrapper"] {{ background: var(--card); border-color: var(--line) !important;
        border-radius: 10px !important; }}
[data-testid="stDeckGlJsonChart"] {{ border: 1px solid var(--line); border-radius: 8px; overflow: hidden; }}
[data-testid="stWidgetLabel"] p {{ font-size: 12.5px; color: var(--muted); font-weight: 500; }}
.stButton button {{ font-weight: 600; font-size: 13px; min-height: 36px; }}

/* header */
.kv-top {{ display:flex; align-items:center; justify-content:space-between; margin-bottom: 14px;
          padding-bottom: 12px; border-bottom: 1px solid var(--line); }}
.kv-brand {{ display:flex; align-items:center; gap:10px; font-weight:700; font-size:16px; color:var(--ink); }}
.kv-mark {{ width:22px; height:22px; border-radius:5px; background:var(--nav); display:inline-flex; align-items:center;
           justify-content:center; color:#E8C872; font-size:13px; font-weight:700; }}
.kv-brand small {{ font-weight:500; color:var(--muted); font-size:12.5px; margin-left:4px; }}
.kv-crumb {{ font-size:12.5px; color:var(--faint); }}
.kv-crumb b {{ color:var(--muted); font-weight:600; }}
.kv-title {{ font-size:28px; font-weight:700; color:var(--ink); letter-spacing:-0.02em; margin:0 0 6px 0; line-height:1.2; }}
.kv-meta {{ font-size:13.5px; color:var(--muted); display:flex; gap:20px; flex-wrap:wrap; margin-bottom:18px; }}
.kv-dot {{ display:inline-block; width:9px; height:9px; border-radius:2px; margin-right:7px; vertical-align:0px; }}

/* metric cards */
.kv-cards {{ display:grid; gap:14px; margin-bottom: 18px; }}
.kv-card {{ background:var(--card); border:1px solid var(--line); border-radius:10px; padding:14px 16px 15px 16px;
           border-top:3px solid var(--accent); min-width:0; }}
.kv-card .kv-label {{ font-size:12.5px; color:var(--muted); font-weight:600; margin-bottom:8px; white-space:nowrap;
                     overflow:hidden; text-overflow:ellipsis; }}
.kv-card .kv-value {{ font-size:30px; font-weight:700; color:var(--ink); letter-spacing:-0.02em; line-height:1.05;
                     font-variant-numeric: tabular-nums; }}
.kv-card .kv-value small {{ font-size:14px; font-weight:500; color:var(--muted); margin-left:5px; letter-spacing:0; }}
.kv-card .kv-sub {{ font-size:12px; color:var(--faint); margin-top:7px; white-space:nowrap; overflow:hidden;
                   text-overflow:ellipsis; }}

.kv-section {{ margin: 2px 0 12px 0; }}
.kv-section h4 {{ font-size:15px; font-weight:700; color:var(--ink); margin:0 0 3px 0; padding:0; }}
.kv-section p {{ font-size:12.5px; color:var(--muted); margin:0; line-height:1.5; }}

/* tables */
.kv-table-wrap {{ border:1px solid var(--line); border-radius:8px; overflow:auto; background:#fff; }}
table.kv-table {{ width:100%; border-collapse:collapse; font-size:13px; }}
table.kv-table th {{ position:sticky; top:0; z-index:1; background:#F8F6F0; text-align:left; font-weight:600; font-size:12px;
                    color:var(--muted); padding:9px 12px; border-bottom:1px solid var(--line); white-space:nowrap; }}
table.kv-table td {{ padding:9px 12px; border-bottom:1px solid var(--line-soft); color:var(--ink); vertical-align:middle; }}
table.kv-table tr:last-child td {{ border-bottom:none; }}
table.kv-table tr:hover td {{ background:#FBFAF6; }}
table.kv-table td.num, table.kv-table th.num {{ text-align:right; font-variant-numeric:tabular-nums; white-space:nowrap; }}
table.kv-table td.idx {{ color:var(--faint); font-variant-numeric:tabular-nums; width:28px; }}
table.kv-table .sub {{ display:block; font-size:11.5px; color:var(--faint); margin-top:1px; }}
.kv-bar {{ display:inline-block; width:52px; height:5px; background:var(--line-soft); border-radius:3px;
          vertical-align:middle; margin-right:8px; overflow:hidden; }}
.kv-bar i {{ display:block; height:100%; border-radius:3px; }}
.kv-status {{ font-size:13px; font-weight:700; white-space:nowrap; font-variant-numeric:tabular-nums; }}

/* legends and notes */
.kv-legend {{ display:flex; gap:16px; flex-wrap:wrap; font-size:12px; color:var(--muted); margin-top:10px; align-items:center; }}
.kv-legend span {{ display:inline-flex; align-items:center; gap:6px; }}
.kv-legend .ttl {{ font-weight:600; color:var(--ink); margin-right:2px; }}
.kv-sw {{ width:10px; height:10px; border-radius:50%; display:inline-block; border:1.5px solid #fff;
         box-shadow:0 0 0 1px rgba(0,0,0,0.12); }}
.kv-sw.sq {{ border-radius:2px; border:none; box-shadow:none; width:14px; height:10px; }}
.kv-sw.line {{ width:16px; height:2px; border-radius:1px; border:none; box-shadow:none; }}
.kv-sw.outline {{ background:transparent !important; border:2px solid; border-radius:2px; box-shadow:none; width:12px; height:10px; }}
.kv-note {{ font-size:12px; color:var(--faint); line-height:1.55; margin-top:8px; }}
.kv-callout {{ border:1px solid var(--line); border-left:4px solid var(--accent); border-radius:8px; padding:12px 14px;
              font-size:13px; color:var(--ink); background:#fff; line-height:1.55; }}
.kv-banner {{ font-size:12.5px; color:var(--ink); background:#FBF3E4; border:1px solid #EAD9B8; border-radius:6px;
             padding:7px 11px; margin-bottom:8px; }}
.kv-side-label {{ font-size:11px; font-weight:700; color:var(--faint); letter-spacing:0.07em; text-transform:uppercase;
                 margin: 18px 0 6px 0; }}
.kv-side-foot {{ font-size:11.5px; color:var(--faint); line-height:1.55; margin-top:22px; border-top:1px solid var(--line);
                padding-top:14px; }}
.kv-log {{ font-size:12.5px; }}
.kv-log div {{ padding:7px 0; border-bottom:1px solid var(--line-soft); display:flex; gap:14px; }}
.kv-log div:last-child {{ border-bottom:none; }}
.kv-log code {{ font-size:11.5px; color:var(--muted); background:transparent; padding:0; min-width:64px; }}
.kv-steps {{ display:grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap:14px; }}
.kv-step {{ padding:16px 18px; border:1px solid var(--line); border-radius:10px; background:#fff; border-top:3px solid var(--accent); }}
.kv-step .n {{ font-size:11.5px; color:var(--accent); font-weight:700; font-variant-numeric:tabular-nums; }}
.kv-step h5 {{ font-size:14px; font-weight:700; margin:6px 0 6px 0; padding:0; color:var(--ink); }}
.kv-step p {{ font-size:12.5px; color:var(--muted); margin:0; line-height:1.55; }}
.kv-daybar {{ display:flex; justify-content:space-between; font-size:11.5px; color:var(--faint); margin-top:-6px; }}
</style>
"""


def e(x) -> str:
    return html.escape(str(x))


def inject_css():
    st.markdown(CSS, unsafe_allow_html=True)


def brand_bar(right: str):
    st.markdown(f'<div class="kv-top"><div class="kv-brand"><span class="kv-mark">K</span>Krev'
                f'<small>Public health supply resilience</small></div><div class="kv-crumb">{right}</div></div>',
                unsafe_allow_html=True)


def page_header(title: str, meta: list[str], status: tuple[str, str] | None = None):
    status_html = ""
    if status:
        color, text = status
        status_html = f'<span><i class="kv-dot" style="background:{color}"></i><b style="color:{INK};font-weight:600">{e(text)}</b></span>'
    meta_html = "".join(f"<span>{e(m)}</span>" for m in meta)
    st.markdown(f'<div class="kv-title">{e(title)}</div><div class="kv-meta">{status_html}{meta_html}</div>',
                unsafe_allow_html=True)


def stat_strip(items: list[dict]):
    """items: {label, value, unit?, sub?, tone?}  tone colours the card's top rule."""
    cells = []
    for it in items:
        unit = f"<small>{e(it['unit'])}</small>" if it.get("unit") else ""
        sub = f'<div class="kv-sub">{e(it["sub"])}</div>' if it.get("sub") else ""
        top = f' style="border-top-color:{it["tone"]}"' if it.get("tone") else ""
        cells.append(f'<div class="kv-card"{top}><div class="kv-label">{e(it["label"])}</div>'
                     f'<div class="kv-value">{e(it["value"])}{unit}</div>{sub}</div>')
    st.markdown(f'<div class="kv-cards" style="grid-template-columns:repeat({len(items)},minmax(0,1fr))">'
                f'{"".join(cells)}</div>', unsafe_allow_html=True)


def section(title: str, desc: str | None = None):
    d = f"<p>{desc}</p>" if desc else ""
    st.markdown(f'<div class="kv-section"><h4>{e(title)}</h4>{d}</div>', unsafe_allow_html=True)


def note(text: str):
    st.markdown(f'<div class="kv-note">{text}</div>', unsafe_allow_html=True)


def banner(text: str):
    st.markdown(f'<div class="kv-banner">{text}</div>', unsafe_allow_html=True)


def legend(groups: list[tuple[str, list[tuple[str, str, str]]]]):
    """groups: (title, [(color, label, kind)]) kind in {'dot','sq','line','outline'}"""
    parts = []
    for title, items in groups:
        if title:
            parts.append(f'<span class="ttl">{e(title)}</span>')
        for color, label, kind in items:
            cls = "" if kind == "dot" else kind
            style = f"border-color:{color}" if kind == "outline" else f"background:{color}"
            parts.append(f'<span><i class="kv-sw {cls}" style="{style}"></i>{e(label)}</span>')
    st.markdown(f'<div class="kv-legend">{"".join(parts)}</div>', unsafe_allow_html=True)


def risk_cell(level: str, prob: float) -> str:
    c = RISK[level]
    return (f'<span class="kv-bar"><i style="width:{max(prob, 0.02) * 100:.0f}%;background:{c}"></i></span>'
            f'<span class="kv-status" style="color:{c}">{prob:.0%}</span>')


def html_table(columns: list[tuple[str, str]], rows: list[list[str]], max_height: int | None = None):
    head = "".join(f'<th class="{c}">{e(h)}</th>' for h, c in columns)
    body = "".join("<tr>" + "".join(f'<td class="{columns[k][1]}">{v}</td>' for k, v in enumerate(r)) + "</tr>"
                   for r in rows)
    style = f' style="max-height:{max_height}px"' if max_height else ""
    st.markdown(f'<div class="kv-table-wrap"{style}><table class="kv-table"><thead><tr>{head}</tr></thead>'
                f'<tbody>{body}</tbody></table></div>', unsafe_allow_html=True)


def spread_color(v: float) -> str | None:
    for thr, c in SPREAD:
        if v < thr:
            return None if thr == SPREAD[0][0] else c
    return SPREAD[-1][1]


def style_fig(fig: go.Figure, height: int = 300, legend_top: bool = True) -> go.Figure:
    fig.update_layout(
        height=height, margin=dict(l=0, r=6, t=30 if legend_top else 8, b=0),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, system-ui, sans-serif", size=12, color=MUTED),
        hoverlabel=dict(bgcolor="#FFFFFF", bordercolor=LINE, font=dict(family="Inter, sans-serif", size=12, color=INK)),
        legend=dict(orientation="h", x=0, y=1.02, yanchor="bottom", xanchor="left", font=dict(size=12, color=MUTED),
                    bgcolor="rgba(0,0,0,0)", itemclick=False, itemdoubleclick=False) if legend_top else dict(),
        showlegend=legend_top, title=None, bargap=0.45,
    )
    fig.update_xaxes(showgrid=False, showline=True, linecolor=LINE, ticks="outside", tickcolor=LINE, ticklen=4,
                     tickfont=dict(color=FAINT), title_font=dict(color=MUTED, size=12))
    fig.update_yaxes(gridcolor=LINE_SOFT, zeroline=False, showline=False, tickfont=dict(color=FAINT),
                     title_font=dict(color=MUTED, size=12))
    return fig


def sidebar_label(text: str):
    st.markdown(f'<div class="kv-side-label">{e(text)}</div>', unsafe_allow_html=True)
