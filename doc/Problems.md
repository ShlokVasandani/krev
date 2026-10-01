The problem Krev solves
In one line

Primary Health Centres (PHCs) run out of essential medicines and test kits right when an outbreak hits, because resupply takes weeks and nobody sees the spike coming. Nearby facilities often have spare stock, but no one tells them to share it.

How big the problem is

Medicines are often not on the shelf.

A systematic review of Indian public facilities found essential-medicine availability well below the WHO benchmark of 80%: 41.3% in Delhi, 45.2% in Punjab, 51.1% in Haryana and 58% in Chhattisgarh. In Maharashtra it ranged from 22.6% to 91.9% depending on the medicine. (Wadhwa et al., 2024)
It names supply-chain inefficiency, inaccurate demand forecasting and weak budgeting as the main causes. (Wadhwa et al., 2024)

Resupply is slow, and stock-outs last for months.

After a PHC places an indent, medicines took 25 days to arrive in Punjab and 7 days in Haryana. (Prinja et al., 2015)
In Haryana, nearly 60% of the medicines that were out of stock stayed out for 3–6 months, and 8% for more than 6 months. In Punjab, 19% were out for more than 6 months. (Prinja et al., 2015)

Outbreaks keep coming.

India recorded 2,33,519 dengue cases in 2024 and 2,89,235 in 2023. (NCVBDC)
The three states in our demo were all hit in 2024: Karnataka had 32,886 cases, Maharashtra 19,385 and Delhi 10,585. (NCVBDC)
Outbreaks reported through IDSP rose from 554 in 2020 to 3,020 in 2024. (NCDC)

When dengue arrives, demand for paracetamol, ORS, IV fluids, NS1 test kits and platelets jumps within days. A PHC that reorders only after it runs low will be empty for the whole wait.

What already exists, and the gap
System	What it does	What it does not do
e-Aushadhi / DVDMS	Tracks drug stock, indents and issues across state warehouses and facilities	Doesn't forecast outbreak-driven demand or warn before a stock-out
IDSP / IHIP	Weekly disease surveillance and outbreak alerts (S, P and L forms)	Isn't linked to medicine stock, so an alert doesn't trigger resupply
eVIN	Real-time vaccine cold-chain stock	Covers vaccines only, not outbreak medicines

The gap: surveillance knows an outbreak is starting, and inventory systems know what is on the shelf, but nothing joins the two. No system tells a district officer "this PHC will run out of NS1 kits in 2 days, the resupply takes 12, and this CHC 18 km away has spare."

What Krev adds
Forecasts demand using each facility's consumption, local case curves and IDSP outbreak alerts.
Warns before a stock-out by comparing days of stock left with resupply time, as a probability.
Recommends transfers of surplus stock from nearby facilities at the lowest cost, without leaving the donor short. If no donor can cover the gap, it escalates.
Keeps data private using federated learning: each state trains locally and shares only signed model updates.
Who uses it
District Health Officer: sees which facilities are at risk and approves transfers
PHC Medical Officer: gets early warning and reports stock by SMS if offline
State drug store / procurement: handles escalations no nearby facility can cover
Honest limits
The demo runs on synthetic data shaped to match real dengue seasons and these supply delays.
The next step is connecting live IDSP and e-Aushadhi data, then piloting in one district.
Sources
Wadhwa et al. (2024). Factors Affecting the Availability and Utilization of Essential Medicines in India: A Systematic Review. Journal of Pharmacy & Bioallied Sciences.
Prinja S, Bahuguna P, Tripathy JP, Kumar R (2015). Availability of medicines in public sector health facilities of two North Indian States. BMC Pharmacology and Toxicology.
National Center for Vector Borne Diseases Control. Dengue situation in India.
National Centre for Disease Control. IDSP key activities and achievements.
Comptroller and Auditor General of India. Performance audit of e-Aushadhi.
Team Krev
Shlok Vasandani
Yakshi Thakkar
Prisha Mathur
