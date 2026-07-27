#!/usr/bin/env python3
"""
Fundamental valuation for the FRAX thesis.

Pulls REAL trailing-12m protocol fees (DefiLlama) and token market caps
(CoinGecko) for FRAX and its stablecoin/DeFi peers, computes price-to-fees (P/F)
multiples, and asks two honest questions:

  1. Is FRAX cheap vs the peer group on current fundamentals? (re-rating upside)
  2. What does the $5 thesis target actually require in fee growth?

Data is current/trailing (not June-2025 vintage); the relative-value logic holds
regardless. Reported as found — including where it does NOT support the target.
"""

import statistics
import numpy as np

try:
    import requests
except ImportError:
    requests = None

# DefiLlama fee-protocol names → token (Frax fees are spread across products)
FEE_MAP = {
    "FRAX": ["Frax", "Fraxlend", "Frax Swap", "Frax Ether", "Frax FPI", "Frax USD", "Fraxtal"],
    "CRV": ["Curve DEX", "Curve LlamaLend"],
    "AAVE": ["Aave V2", "Aave V3", "Aave V4"],
    "SKY": ["Sky Lending"],
    "ENA": ["Ethena USDe"],
    "LDO": ["Lido"],
    "PENDLE": ["Pendle"],
    "CVX": ["Convex Finance"],
    "RSR": ["Reserve Protocol"],
    "LQTY": ["Liquity V1", "Liquity V2"],
    "USUAL": ["Usual USD0"],
    "SPELL": ["Abracadabra Spell"],
}
CG_IDS = {"FRAX": "frax-share", "CRV": "curve-dao-token", "AAVE": "aave", "SKY": "sky",
          "ENA": "ethena", "LDO": "lido-dao", "PENDLE": "pendle", "CVX": "convex-finance",
          "RSR": "reserve-rights-token", "LQTY": "liquity", "USUAL": "usual", "SPELL": "spell-token"}
NAMES = {"FRAX": "Frax (frxUSD/Fraxtal)", "CRV": "Curve", "AAVE": "Aave", "SKY": "Sky (Maker)",
         "ENA": "Ethena", "LDO": "Lido", "PENDLE": "Pendle", "CVX": "Convex",
         "RSR": "Reserve", "LQTY": "Liquity", "USUAL": "Usual", "SPELL": "Abracadabra"}


def _fees_map():
    d = requests.get("https://api.llama.fi/overview/fees?excludeTotalDataChart=true"
                     "&excludeTotalDataChartBreakdown=true&dataType=dailyFees", timeout=30).json()
    return {p.get("name", ""): (p.get("total1y") or 0) for p in d.get("protocols", [])}


def _frax_tvl():
    try:
        r = requests.get("https://api.llama.fi/tvl/frax", timeout=15)
        return float(r.json())
    except Exception:
        return None


def _mcaps():
    ids = ",".join(CG_IDS.values())
    d = requests.get("https://api.coingecko.com/api/v3/coins/markets",
                     params={"vs_currency": "usd", "ids": ids, "per_page": 100}, timeout=30).json()
    out = {}
    if isinstance(d, list):
        by_id = {c.get("id"): c for c in d}
        for sym, cid in CG_IDS.items():
            c = by_id.get(cid)
            if c:
                out[sym] = {"mcap": c.get("market_cap"), "fdv": c.get("fully_diluted_valuation"),
                            "price": c.get("current_price")}
    return out


def run(target_price=5.0):
    if requests is None:
        return None
    try:
        fees = _fees_map(); caps = _mcaps()
    except Exception as e:
        return {"error": str(e)[:120]}
    rows = []
    for sym, products in FEE_MAP.items():
        f1y = sum(fees.get(p, 0) or 0 for p in products)
        cap = caps.get(sym, {})
        mc, price = cap.get("mcap"), cap.get("price")
        if not mc or not f1y or f1y <= 0:
            continue
        rows.append({"sym": sym, "name": NAMES.get(sym, sym), "mcap": round(mc, 0),
                     "fees_1y": round(f1y, 0), "price": price, "pf": round(mc / f1y, 2)})
    peers = [r for r in rows if r["sym"] != "FRAX" and r["pf"] > 0]
    med_pf = round(statistics.median(r["pf"] for r in peers), 2) if peers else None
    frax = next((r for r in rows if r["sym"] == "FRAX"), None)
    out = {"rows": sorted(rows, key=lambda r: r["pf"]), "median_pf": med_pf,
           "n_peers": len(peers), "frax_tvl": _frax_tvl(), "target_price": target_price}
    if frax and med_pf:
        circ = frax["mcap"] / frax["price"] if frax["price"] else None
        out["frax"] = frax
        out["rerating_mult"] = round(med_pf / frax["pf"], 2)              # to peer median P/F
        out["implied_price_rerate"] = round(frax["price"] * med_pf / frax["pf"], 4)
        out["implied_mcap_rerate"] = round(frax["mcap"] * med_pf / frax["pf"], 0)
        if circ:
            tgt_mcap = target_price * circ
            fees_needed = tgt_mcap / med_pf
            out["target_mcap"] = round(tgt_mcap, 0)
            out["fees_needed"] = round(fees_needed, 0)
            out["fee_growth_x"] = round(fees_needed / frax["fees_1y"], 1)
            out["target_pf_on_current_fees"] = round(tgt_mcap / frax["fees_1y"], 1)
    return out


if __name__ == "__main__":
    import json
    print(json.dumps(run(), indent=2))
