# 12-hour plan · 3 people

The working prototype already exists. The 12 hours go to **owning it, hardening it, and the pitch**, not rebuilding.

- **Shlok (Dev)**: all code: `core/` and `app.py`, Gemini key, deployment, live demo. Owns every number and the technical Q&A.
- **Member 2 (PPT: story + problem)**: problem slides, research on India's PHC supply chain, impact framing, script for the talk.
- **Member 3 (PPT: design + product)**: deck design, architecture diagram, screenshots, backup demo video.

The two PPT members start from `README.md` (what it does, numbers, likely judge questions) and should run the app once so they know every tab.

| Hour | Shlok (Dev) | Member 2 (PPT: story) | Member 3 (PPT: design) |
|---|---|---|---|
| 0–1 | All: run the app, click every tab, read README. Agree the demo scenario (Pune, day 9). | ← same | ← same |
| 1–3 | Read `forecast.py` + `optimizer.py` until every formula is explainable. Try 3–4 scenarios; note anything odd. Get a Gemini key and test the brief. | Problem research: PHC stock-outs, resupply delays, eVIN / e-Aushadhi / IDSP. Find 2–3 cited facts for slide 1–2. | Pick deck template + colours. Draft the slide skeleton (below). Draw the architecture diagram. |
| 3–5 | Tune realism (item rates, lead times) and optimiser penalties. Re-check README numbers after every change. Send updated numbers to both. | Write slide text: problem, solution, how it works, impact, federated, security, next steps. Keep each slide ≤ 25 words. | Build slides as text arrives. Placeholder screenshots for now. |
| 5–6 | **Checkpoint, all three.** Freeze model/data logic. Numbers in the deck are locked after this. | ← same | ← same |
| 6–8 | Deploy (Cloud Run for Google points, or Streamlit Community Cloud). Test on a phone. UI wording fixes from the PPT team. | Write the 7-min speaking script + Q&A cheat-sheet (use README "Likely judge questions"). | Final screenshots from the deployed app. Record 3-min backup demo video (Shlok drives, Member 3 records). |
| 8–10 | Stretch, only if everything above is done: expiry wastage in the replay *or* a scale test. Otherwise bug-fix. | Rehearse script alone, time it, cut to fit. | Finish deck polish, export PDF backup. |
| 10–11 | All: full dry run ×2 with a timer. Fix only showstoppers. | ← same | ← same |
| 11–12 | Buffer. Submission form, repo, video link. | ← same | ← same |

## Suggested deck (8 slides)

1. Problem: PHC stock-outs, 7–12 day resupply, nobody sees the spike coming
2. Krev in one line + the with/without number (~310 → ~50 stock-out days)
3. How it works: forecast → early warning → redistribution (architecture diagram)
4. Demo screenshot: map + early warnings
5. Impact: with vs without chart
6. Federated learning: cold-start chart + poisoned-update rejection
7. Security & honesty: mTLS, signed updates, post-quantum roadmap; synthetic data; limits
8. Next steps: IDSP + e-Aushadhi/eVIN integration, pilot district

## Rules

- No new features after hour 6. Bugs and wording only.
- Every number on a slide or said aloud must come from the app. Deck numbers come from Shlok, not from memory.
- Say "synthetic data" before anyone asks.
- If something breaks mid-demo, switch to the recorded video; don't debug live.
