# Krev — AI decision layer for PHC supply resilience

GDG Build with AI · Code for Communities 2nd Edition · Health · Challenge 03 (Smart Health & Supply Chain Resilience)

> Forecasts medicine demand with uncertainty, warns of stock-outs **before the next resupply arrives**, and recommends
> explainable, constraint-aware redistribution between facilities and states. State outbreak models learn together
> through federated training, so a data-poor state benefits without pooling data. Then it proves the impact by
> replaying the same outbreak **with and without** the system.

## Run it

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
export GEMINI_API_KEY=...                              # optional; Windows: set GEMINI_API_KEY=...
streamlit run app.py
```

First load takes ~15 s (trains the federated model and the benchmark, then caches). Without a Gemini key, the
action brief falls back to a template, so the demo never breaks. Model name defaults to `gemini-3.8-flash`;
override with `GEMINI_MODEL`.

## What's inside

| File | What it does | Challenge requirement |
|---|---|---|
| `core/network.py` | 12 districts / 3 states, tiered PHC → CHC → District Hospital, 5 items (platelets only at DH blood banks) | national network, realism |
| `core/simdata.py` | 2 years of daily data: state-specific dengue seasons, past outbreaks, IDSP alerts (2-day reporting lag), footfall, beds, staff | real-time visibility (simulated) |
| `core/features.py` | outbreak features: own curve, IDSP alerts, **neighbouring-district pressure** | early warning |
| `core/federated.py` | FedAvg across state nodes, federated feature scaling, signed updates, poisoning guard, DP-style noise | shared predictive modelling |
| `core/forecast.py` | stage 1 federated case trajectory + stage 2 local demand response → P50/P90, lead-time-aware stock-out probability; cold-start benchmark | forecasting + early warnings |
| `core/optimizer.py` | min-cost transportation LP (HiGHS): donor safety buffer, P90 receiver need, distance/shelf-life, tier, near-expiry, inter-state penalty; escalations | cross-district redistribution |
| `core/counterfactual.py` | 21-day replay: static reorder points vs AI forecast vs forecast + redistribution | proof of impact |
| `core/security.py` | HMAC-signed updates, key registry with scheduled + event-based rotation | secure federation |
| `core/ingest.py` | SMS stock reports from feature phones (`STK PHC-PUN-1 PCM 60 ORS 25`) | data ingestion story |
| `core/brief.py` | Gemini writes a DHO action brief from computed facts only | Build with AI |
| `app.py` | Streamlit dashboard, 7 tabs | demo |
| `ui.py`, `.streamlit/config.toml` | design system (government-grade: teal, cream, Inter), cards, tables, chart styling | demo polish |
| `core/geo.py`, `data/india_districts.geojson` | district boundaries (simplified from GADM via github.com/geohacker/india) for the outbreak-spread map and timeline | real-time visibility |

## Current numbers (default scenario: Pune outbreak, day 9, intensity 4)

- 21-day replay (Pune outbreak): stock-out facility-item-days **310 → 46** (85% fewer); unmet demand 82% lower; forecast alone (no transfers) gets to 129
- Across all 12 epicentres Krev beats static reorder points every time: 54–97% fewer stock-out days in Maharashtra/Karnataka; Delhi only ≈36% (no donors within reach, so escalations take over)
- Donors never ship below what they need on the shelf to reach their own next delivery
- Forecast error (14-day demand, all facilities): model beats last-7-days persistence in most outbreak scenarios; weakest at
  outbreak day ≤5 and for isolated districts (Nagpur) — say so if asked
- Federated cold-start (each state joins with 150 days): federated ≈ centralized, far better than local-only for every state

Numbers are from synthetic data. Say that first, before a judge asks.

## Demo script (7 minutes)

1. **Problem (30 s)** — stock-outs at PHCs happen because resupply takes 7–12 days and nobody sees the spike coming.
2. **Normal season (45 s)** — sidebar → *Normal season*. Mostly green map, few warnings. "This is a calm network."
3. **Trigger the outbreak (1.5 min)** — *Dengue outbreak*, Pune, day 9. Red rings = IDSP alerts; warnings jump; point out
   Nashik/Mumbai flagged **before** their own cases peak (neighbour pressure). Hover a facility: risk before resupply, cover vs lead time.
4. **Redistribution (1.5 min)** — tab 2. Read one *Why*. Show inter-state rows pending approval. Show escalations. Click
   *Generate brief* (Gemini).
5. **The moment (1 min)** — tab 3: without vs with. 310 vs 46 stock-out days, same outbreak, same demand.
6. **Federated (1 min)** — tab 4: Delhi local-only vs federated. Then inject *poisoned update* → rejected, key rotated.
7. **Close (30 s)** — SMS ingestion tab (works on a feature phone), and the one-line architecture.

Pre-load every tab once before presenting (caching). Keep a screen recording as backup.

## Likely judge questions

- *Is the data real?* No — synthetic, structured on real seasonality and India's tiered system. Next step: IDSP + e-Aushadhi/eVIN.
- *Why federated if you share stock levels anyway?* Federation protects training data (surveillance, patient footfall). Aggregate stock is operational and must be shared to redistribute.
- *Why not just procure more?* Procurement takes weeks; a transfer takes 1–2 days. Escalations cover what transfers can't.
- *What about QKD / quantum?* We use mTLS 1.3 and signed updates; hybrid ML-KEM is the post-quantum roadmap. QKD needs dedicated fibre hardware and isn't meaningful for PHC links.
- *Why linear models?* Tested a small neural net: no accuracy gain here, less stable, harder to explain.
