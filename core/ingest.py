"""Offline-first data ingestion: a PHC pharmacist reports stock by plain SMS / WhatsApp text.

Format (case-insensitive, any order after the code):
    STK PHC-PUN-1 PCM 120 ORS 40 IVNS 15 NS1 30
    USE PHC-PUN-1 PCM 35 ORS 12            (today's consumption, optional)

Item codes: PCM paracetamol, ORS, IVNS IV normal saline, NS1 dengue test kits, PLT platelets.
This is the answer to "how does data get in from 30,000 low-connectivity PHCs?" -- works on a feature phone,
and maps directly onto existing registers; a real deployment would also sync from e-Aushadhi / eVIN APIs.
"""
from __future__ import annotations

import re

from .network import ITEM_IDS

ALIASES = {"PCM": "pcm", "PARA": "pcm", "PARACETAMOL": "pcm", "ORS": "ors", "IVNS": "ivns", "NS": "ivns",
           "IV": "ivns", "NS1": "ns1", "RDT": "ns1", "PLT": "plt", "PLATELET": "plt", "PLATELETS": "plt"}


def parse_sms(text: str, codes: list[str]):
    """Returns list of dicts {kind, code, item_id, qty} and list of error strings."""
    out, errors = [], []
    code_set = {c.upper(): c for c in codes}
    for line in [l.strip() for l in text.strip().splitlines() if l.strip()]:
        toks = re.split(r"[\s,:=]+", line.upper())
        if len(toks) < 4 or toks[0] not in ("STK", "USE"):
            errors.append(f"'{line}': expected 'STK <facility-code> <item> <qty> ...'")
            continue
        kind, code = toks[0], toks[1]
        if code not in code_set:
            errors.append(f"'{line}': unknown facility code {code}")
            continue
        pairs = toks[2:]
        if len(pairs) % 2:
            errors.append(f"'{line}': item/quantity pairs are incomplete")
            continue
        for k in range(0, len(pairs), 2):
            item = ALIASES.get(pairs[k])
            if item is None or item not in ITEM_IDS:
                errors.append(f"'{line}': unknown item '{pairs[k]}'")
                continue
            if not pairs[k + 1].isdigit():
                errors.append(f"'{line}': quantity for {pairs[k]} is not a number")
                continue
            out.append(dict(kind="stock" if kind == "STK" else "use", code=code_set[code], item_id=item,
                            qty=int(pairs[k + 1])))
    return out, errors
