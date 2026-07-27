#!/usr/bin/env python3
"""
Holder / wallet concentration analysis for FRAX (ex-FXS) on Ethereum.

Uses Ethplorer's public API to pull the real top token holders, classifies each
(locked-staking / protocol-contract / bridge-system / individual EOA), and reads
whether the observable structure leans toward holder CONFIDENCE (locked, committed)
or EXIT (liquid whales distributing).

Honest limits, surfaced on the dashboard:
  • Ethereum only. After the North Star migration most supply moved to Fraxtal,
    which this snapshot does NOT index — so it is a partial view.
  • It is a snapshot, not a time series: it shows structure, not per-wallet
    accumulation/distribution over time (that needs a full indexer / paid key).
"""

import time
import numpy as np

try:
    import requests
except ImportError:
    requests = None

FXS = "0x3432B6A60D23Ca0dFCa7761B7ab56459D9C964D0"
API = "https://api.ethplorer.io"

# verified / high-confidence labels (Etherscan / creation context)
KNOWN = {
    "0xc8418af6358ffdda74e09ca9cc3fe03ca6adc5b0": ("veFXS — vote-escrowed (locked)", "locked"),
    "0x36cb65c1967a0fb0eee11569c51c2f2aa1ca6f6d": ("Frax system contract (Fraxtal-era, ~L2 bridge/lockbox)", "structural"),
    "0x0000000000000000000000000000000000000000": ("null / burn", "structural"),
}


def _get(path, **params):
    params["apiKey"] = "freekey"
    return requests.get(f"{API}/{path}", params=params, timeout=20).json()


def _wallet_flow(addr):
    """Recent net FXS flow for one wallet (transfers, not confirmed buys/sells)."""
    try:
        d = _get(f"getAddressHistory/{addr}", token=FXS, type="transfer", limit=10)
        ops = d.get("operations", [])
        if not ops:
            return None
        a = addr.lower(); net = 0.0; last = 0
        for o in ops:
            v = float(o["value"]) / 1e18
            net += v if o["to"].lower() == a else -v
            last = max(last, int(o.get("timestamp", 0)))
        days = (time.time() - last) / 86400 if last else None
        if days is not None and days > 365:
            direction = "dormant / holding"
        elif net > 0:
            direction = "net inflow (accumulating)"
        elif net < 0:
            direction = "net outflow (distributing)"
        else:
            direction = "flat"
        return {"net": round(net, 0), "last_ts": last,
                "days_since": round(days) if days is not None else None,
                "dir": direction, "n": len(ops)}
    except Exception:
        return None


def run(top=12):
    if requests is None:
        return None
    try:
        ti = _get(f"getTokenInfo/{FXS}")
        th = _get(f"getTopTokenHolders/{FXS}", limit=top)
        holders = th.get("holders", [])
        if not holders:
            return None
        rows = []; flows_done = 0
        for h in holders:
            a = h["address"].lower(); share = float(h.get("share", 0))
            flow = None
            if a in KNOWN:
                label, kind = KNOWN[a]
            else:
                try:
                    info = _get(f"getAddressInfo/{h['address']}")
                    is_c = bool(info.get("contractInfo")) or bool(info.get("tokenInfo"))
                except Exception:
                    is_c = False
                label, kind = ("protocol / contract", "contract") if is_c else ("large wallet — EOA (identity unverified)", "wallet")
                time.sleep(0.2)
                if kind == "wallet" and flows_done < 6:      # recent trade behavior for top EOAs
                    flow = _wallet_flow(h["address"]); flows_done += 1; time.sleep(0.2)
            rows.append({"addr": h["address"], "share": round(share, 2), "label": label,
                         "kind": kind, "flow": flow})
        by = lambda k: round(sum(r["share"] for r in rows if r["kind"] == k), 2)
        locked = by("locked"); structural = by("structural"); contract = by("contract"); wallet = by("wallet")
        return {
            "holders_count": ti.get("holdersCount"),
            "price": (ti.get("price") or {}).get("rate"),
            "rows": rows,
            "top1": rows[0]["share"] if rows else None,
            "top10": round(sum(r["share"] for r in rows[:10]), 2),
            "locked_share": locked, "structural_share": structural,
            "contract_share": contract, "wallet_share": wallet,
            "committed_share": round(locked + structural + contract, 2),  # non-EOA
            "flows": {
                "accumulating": sum(1 for r in rows if r.get("flow") and "accumulating" in r["flow"]["dir"]),
                "distributing": sum(1 for r in rows if r.get("flow") and "distributing" in r["flow"]["dir"]),
                "dormant": sum(1 for r in rows if r.get("flow") and "dormant" in r["flow"]["dir"]),
            },
        }
    except Exception as e:
        return {"error": str(e)[:120]}


if __name__ == "__main__":
    import json
    print(json.dumps(run(), indent=2))
