"""Plain-language action brief for a District / State Health Officer, via Gemini.

The LLM only *explains*: every number comes from the forecaster and optimiser, and the prompt forbids
inventing figures. Without an API key we fall back to a deterministic template, so the demo never breaks.
"""
from __future__ import annotations

import os

import pandas as pd

DEFAULT_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.8-flash")


def _facts(scenario_label: str, risks: pd.DataFrame, transfers: pd.DataFrame, escalations: pd.DataFrame,
           alerts: list[str]) -> str:
    lines = [f"Situation: {scenario_label}.", f"Districts with active IDSP outbreak alerts: {', '.join(alerts) or 'none'}."]
    hi = risks[risks.risk == "HIGH"].sort_values("stockout_prob", ascending=False).head(8)
    lines.append(f"High stock-out risk items: {int((risks.risk == 'HIGH').sum())}. Top ones:")
    for r in hi.itertuples():
        lines.append(f"- {r.facility} ({r.district}): {r.item}, {r.stock:.0f} {r.unit} on hand, forecast "
                     f"{r.p50:.0f}/day, {r.days_cover:.1f} days cover vs {r.lead_days:.0f}-day resupply, "
                     f"stock-out probability {r.stockout_prob:.0%}, {r.dengue_share:.0%} dengue-driven.")
    if len(transfers):
        lines.append(f"Recommended transfers ({len(transfers)}):")
        for r in transfers.head(10).itertuples():
            lines.append(f"- {r.qty} {r.unit} {r.item}: {r.frm} -> {r.to}, {r.road_km:.0f} km, ~{r.days} day(s)"
                         f"{', INTER-STATE' if r.interstate else ''}; receiver risk {r.prob_before:.0%} -> {r.prob_after:.0%}.")
    if len(escalations):
        lines.append("Need that redistribution cannot cover (emergency procurement required):")
        for r in escalations.head(6).itertuples():
            lines.append(f"- {r.district}: {r.item} short by ~{r.shortfall:.0f} {r.unit}.")
    return "\n".join(lines)


PROMPT = """You are an operations assistant to a District Health Officer in India.
Write a concise action brief (max 180 words) from the FACTS below.
Structure: 1) one-line situation, 2) 'Act today' - numbered actions (transfers to approve, procurement to raise),
3) 'Watch' - districts likely to be affected next.
Rules: use ONLY numbers present in FACTS; never invent figures, names or places; plain language, no jargon.

FACTS:
{facts}"""


def template_brief(facts: str) -> str:
    return "**Action brief (template - add a Gemini API key for a written brief)**\n\n```\n" + facts + "\n```"


def action_brief(scenario_label, risks, transfers, escalations, alerts, api_key: str | None = None):
    facts = _facts(scenario_label, risks, transfers, escalations, alerts)
    key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        return template_brief(facts), "template"
    try:
        from google import genai
        client = genai.Client(api_key=key)
        resp = client.models.generate_content(model=DEFAULT_MODEL, contents=PROMPT.format(facts=facts))
        return resp.text, DEFAULT_MODEL
    except Exception as e:  # network, quota, bad key -> never break the demo
        return template_brief(facts) + f"\n\n_Gemini call failed: {type(e).__name__}: {str(e)[:160]}_", "template"
