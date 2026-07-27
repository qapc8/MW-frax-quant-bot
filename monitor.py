#!/usr/bin/env python3
"""
MW-QB position monitor
======================

Daily mark-to-market monitoring + quant analytics + an AI-agent decision log for
the seeded FRAX position, run against REAL daily candles.

The tracked asset is the Frax flagship token, which migrated ticker FXS -> FRAX
in the 2025 North Star rebrand. The price series is stitched: FXS/USDT through
its last candle, then FRAX/USDT onward, giving one continuous real track from
entry to date.

The principal (spot leg) has NO protective stop before leverage — held to the
$4.40 trigger; only the leveraged leg is stopped. So every day the agent's
rational action is HOLD; the per-day note explains why.

Usage:
    python monitor.py                        # print log + write monitor_log.json
    python monitor.py --entry 2025-07-13 --end 2026-07-24
"""

import argparse
import json
import math
from datetime import date, datetime

import numpy as np
import pandas as pd

import ccxt
import diligence
import holders as holders_mod
import valuation
import signals as sig
from quant_bot import load_cfg

ANN = math.sqrt(365)


def _fetch(ex, symbol, limit=1000):
    raw = ex.fetch_ohlcv(symbol, timeframe="1d", limit=limit)
    df = pd.DataFrame(raw, columns=["ts", "open", "high", "low", "close", "volume"])
    df["date"] = pd.to_datetime(df["ts"], unit="ms").dt.date
    return df


# FRAX's TRUE vertical: decentralized stablecoin issuers (protocol token vs its
# own stablecoin). Like-for-like peers, not the whole of DeFi. Each entry is a
# list of ticker segments so a mid-window rename (e.g. MKR -> SKY, as FXS -> FRAX)
# is chained by returns rather than raw price.
PEERS = {
    "ENA":     ["ENA/USDT"],                # Ethena — USDe
    "MKR/SKY": ["MKR/USDT", "SKY/USDT"],    # MakerDAO -> Sky — DAI/USDS (renamed)
    "CRV":     ["CRV/USDT"],                # Curve — crvUSD
    "AAVE":    ["AAVE/USDT"],               # Aave — GHO
    "LQTY":    ["LQTY/USDT"],               # Liquity — LUSD/BOLD
    "SPELL":   ["SPELL/USDT"],              # Abracadabra — MIM
    "RSR":     ["RSR/USDT"],                # Reserve — eUSD/RSV
    "USUAL":   ["USUAL/USDT"],              # Usual — USD0
}


# Real, dated macro/crypto shocks over the window (sourced, not fabricated) to
# contextualize the drawdown legs. Edit/extend with verified dates only.
MACRO_EVENTS = [
    ("2025-10-06", "BTC ATH ~$126k — crypto cycle top"),
    ("2025-10-10", "Trump 100% China-tariff shock — crypto crash (Oct 10–12)"),
    ("2025-11-30", "November selloff — ~$7B BTC-ETF outflows, BTC −17%"),
    ("2025-12-15", "December meltdown — BTC −32% from ATH"),
    ("2026-01-28", "FOMC hawkish hold — BTC −8% in 48h"),
]


def _peer_index(ex, segments, entry_d, end_d):
    """Return {date_str: index_level} rebased to 1.0 at the first date. Multi-segment
    peers (ticker renames) are chained by daily return; the redenomination boundary
    carries the level (0% step) instead of a spurious price jump."""
    frames = []
    for pair in segments:
        try:
            df = _fetch(ex, pair)
        except Exception:
            continue
        df = df[(df["date"] >= entry_d) & (df["date"] <= end_d)][["date", "close"]].copy()
        df["seg"] = pair
        frames.append(df)
    if not frames:
        return None
    allrows = (pd.concat(frames).sort_values("date")
               .drop_duplicates("date", keep="first").reset_index(drop=True))
    if len(allrows) < 40:
        return None
    idx = {}; level = 1.0; prev = None; prev_seg = None
    for _, r in allrows.iterrows():
        if prev is not None and r["seg"] == prev_seg:
            level *= r["close"] / prev
        idx[str(r["date"])] = level
        prev = r["close"]; prev_seg = r["seg"]
    return idx


def _basket(ex, entry_iso, end_iso):
    """Equal-weight stablecoin-issuer basket, each peer rebased to entry."""
    entry_d = date.fromisoformat(entry_iso); end_d = date.fromisoformat(end_iso)
    peer_idx = {}; final = {}; sharpe = {}
    for sym, segs in PEERS.items():
        idx = _peer_index(ex, segs, entry_d, end_d)
        if idx is None:
            continue
        peer_idx[sym] = idx
        final[sym] = round(idx[max(idx)] - 1, 4)
        levels = np.array([idx[d] for d in sorted(idx)])
        rets = np.diff(levels) / levels[:-1]
        sd = rets.std(ddof=1) if len(rets) > 1 else 0.0
        sharpe[sym] = round(float(rets.mean() / sd * ANN), 3) if sd else 0.0
    def basket_cum(date_str):
        vals = [peer_idx[s][date_str] - 1 for s in peer_idx if date_str in peer_idx[s]]
        return float(np.mean(vals)) if vals else None
    return {"basket_cum": basket_cum, "final": final, "sharpe": sharpe, "used": list(peer_idx.keys())}


def _stitch_frax(ex):
    """FXS/USDT then FRAX/USDT (ticker migrated in the North Star rebrand)."""
    fxs = _fetch(ex, "FXS/USDT")
    frax = _fetch(ex, "FRAX/USDT")
    split = fxs["date"].max()
    frax = frax[frax["date"] > split]
    out = pd.concat([fxs, frax], ignore_index=True)
    out["ticker"] = ["FXS" if d <= split else "FRAX" for d in out["date"]]
    return out, split


def _sector_clause(rel):
    if rel is None:
        return ""
    if rel <= -0.10:
        return f" Lagging the DeFi vertical by {abs(rel)*100:.0f}pts — FRAX-specific weakness atop a sector drawdown."
    if rel >= 0.10:
        return f" Outperforming the vertical by {rel*100:.0f}pts — relative strength vs peers."
    return " In line with the DeFi vertical — move is sector-wide, not FRAX-specific."


def _note(px, day_ret, mom, conv, dist_trig, cumret, dd, thresh, trigger, ticker, idx, rel=None):
    reg = ("momentum constructive" if mom >= 0.8 else
           "momentum broken" if mom <= 0.4 else "momentum mixed")
    if day_ret <= -0.08:
        mv = f"sharp {day_ret*100:.0f}% session"
    elif day_ret >= 0.08:
        mv = f"+{day_ret*100:.0f}% pop"
    else:
        mv = f"{day_ret*100:+.1f}% day"
    below = dist_trig * 100  # how far px sits under the arm, %
    addgate = ("signal hot but off-thesis (px<arm) → no add" if conv >= thresh
               else f"add-gate cold (conv {conv:.2f}<{thresh:.2f})")
    t = idx % 4
    if t == 0:
        s = (f"HOLD — {reg}; {mv}. Px ${px:.3f} sits {below:.0f}% below the ${trigger:.2f} "
             f"leverage arm, levered leg disarmed. No stop on principal by mandate → no forced "
             f"exit; {addgate}. Position unchanged.")
    elif t == 1:
        s = (f"HOLD — {mv}, {reg}. Mandate holds the spot leg un-stopped to the ${trigger:.2f} "
             f"trigger ({below:.0f}% away); {addgate}. Drawdown {dd*100:.0f}% from HWM tolerated by "
             f"design. No action.")
    elif t == 2:
        s = (f"HOLD — monitoring only. Trigger not hit ({below:.0f}% under arm) → no leverage. "
             f"{reg}; {addgate}. Cum {cumret*100:+.0f}%. No rebalance, no stop configured pre-leverage.")
    else:
        s = (f"HOLD — {reg}; {mv}. Below arm by {below:.0f}pts, no downside stop; conviction "
             f"{conv:.2f} vs {thresh:.2f}. Thesis unresolved — hold to trigger or invalidation. Unchanged.")
    return s + _sector_clause(rel)


def _analytics(closes, values, cost, btc_ret, dates):
    px = np.asarray(closes, float)
    lr = np.diff(np.log(px)); n = len(lr)
    mean_d = float(lr.mean()) if n else 0.0
    std_d = float(lr.std(ddof=1)) if n > 1 else 0.0
    dstd = float(np.sqrt(np.mean(np.square(np.minimum(lr, 0.0))))) if n else 0.0
    z = lr - mean_d
    skew = float(np.mean(z**3) / std_d**3) if std_d else 0.0
    kurt = float(np.mean(z**4) / std_d**4 - 3.0) if std_d else 0.0
    var95 = float(np.percentile(lr, 5)) if n else 0.0
    cvar95 = float(lr[lr <= var95].mean()) if n and (lr <= var95).any() else var95
    b = np.array([btc_ret.get(d, np.nan) for d in dates[1:]]); mask = ~np.isnan(b)
    beta = corr = float("nan")
    if mask.sum() > 2:
        cov = np.cov(lr[mask], b[mask])
        beta = float(cov[0, 1] / cov[1, 1]) if cov[1, 1] else float("nan")
        corr = float(np.corrcoef(lr[mask], b[mask])[0, 1])
    val = np.asarray(values, float); peak = np.maximum.accumulate(val); dd = val / peak - 1
    longest = cur = 0
    for u in (val < peak):
        cur = cur + 1 if u else 0; longest = max(longest, cur)
    cumret = val / cost - 1
    roll = [None] * len(px); lrf = np.concatenate([[np.nan], lr])
    for i in range(len(px)):
        if i >= 30:
            roll[i] = round(float(np.nanstd(lrf[i - 29:i + 1], ddof=1) * ANN), 4)
    hist, edges = np.histogram(np.clip(lr * 100, -25, 25), bins=15, range=(-25, 25))
    return {
        "vol_ann": round(std_d * ANN, 4), "ret_ann": round(mean_d * 365, 4),
        "sharpe": round(mean_d / std_d * ANN, 3) if std_d else 0.0,
        "sortino": round(mean_d / dstd * ANN, 3) if dstd else 0.0,
        "downside_dev_ann": round(dstd * ANN, 4), "var95_d": round(var95, 4), "cvar95_d": round(cvar95, 4),
        "skew": round(skew, 3), "kurtosis": round(kurt, 3),
        "beta_btc": None if math.isnan(beta) else round(beta, 3),
        "corr_btc": None if math.isnan(corr) else round(corr, 3),
        "max_dd": round(float(dd.min()), 4), "max_dd_date": str(dates[int(dd.argmin())]),
        "max_dd_duration": longest, "time_underwater": round(float((val < peak).mean()), 4),
        "mfe": round(float(cumret.max()), 4), "mae": round(float(cumret.min()), 4),
        "hist_counts": hist.tolist(), "hist_edges": [round(float(e), 1) for e in edges], "roll_vol": roll,
    }


def build_log(cfg, entry_iso, end_iso):
    ex = getattr(ccxt, cfg["exchange"]["id"])({"enableRateLimit": True})
    df, split = _stitch_frax(ex)
    btc = _fetch(ex, "BTC/USDT"); btc["lr"] = np.log(btc["close"]).diff()
    btc_ret = dict(zip(btc["date"].astype(str), btc["lr"]))
    btc_close = dict(zip(btc["date"].astype(str), btc["close"]))
    eth = _fetch(ex, "ETH/USDT")
    eth_close = dict(zip(eth["date"].astype(str), eth["close"]))
    basket = _basket(ex, entry_iso, end_iso)

    end_d = date.fromisoformat(end_iso)
    df = df[df["date"] <= end_d].reset_index(drop=True)
    m = cfg["momentum"]; close = df["close"]
    df["ema_f"] = sig._ema(close, m["ema_fast"]); df["ema_s"] = sig._ema(close, m["ema_slow"])
    df["rsi"] = sig._rsi(close, m["rsi_period"]); df["roc"] = close.pct_change(m["roc_period"]) * 100

    seed = cfg["position"]["seed"]
    entry_px = float(seed["avg_entry"]); amount = float(seed["amount"]); cost = entry_px * amount
    trigger = float(cfg["strategy"].get("leverage_trigger_price", 0) or 0)
    has_stop = cfg["strategy"]["spot_stop_pct"] > 0
    thresh = cfg["signals"]["trigger_threshold"]
    w = cfg["signals"]["weights"]; tw = sum(w.values())

    idx = df.index[df["date"] >= date.fromisoformat(entry_iso)]
    if len(idx) == 0:
        raise SystemExit(f"No candles on/after {entry_iso}")
    start = int(idx[0])

    rows, events = [], []
    prev_close = None; milestones = {-0.25: 0, -0.5: 0, -0.7: 0, -0.85: 0}; hi = False; seen_frax = False
    for i in range(start, len(df)):
        r = df.iloc[i]; px = float(r["close"]); tk = r["ticker"]
        trend = 1.0 if r["ema_f"] > r["ema_s"] else 0.0
        rsi_ok = 1.0 if m["rsi_min"] <= r["rsi"] <= m["rsi_max"] else 0.0
        roc_score = max(0.0, min(1.0, 0.5 + (r["roc"] if pd.notna(r["roc"]) else 0) / 10.0))
        mom = 0.5 * trend + 0.2 * rsi_ok + 0.3 * roc_score
        conv = (mom * w["momentum"] + 0.5 * (w["tvl"] + w["social"] + w["news"])) / tw
        value = amount * px; upnl = value - cost; cumret = value / cost - 1
        day_ret = (px / prev_close - 1) if prev_close else 0.0
        dist_trig = (trigger / px - 1) if trigger else 0.0

        if tk == "FRAX" and not seen_frax:
            seen_frax = True
            events.append({"date": str(r["date"]), "kind": "TICKER",
                           "msg": "ticker migrated FXS → FRAX (North Star rebrand); position carried over, no action"})
        peer_cum = basket["basket_cum"](str(r["date"]))
        rel = round(cumret - peer_cum, 4) if peer_cum is not None else None
        note = _note(px, day_ret, mom, conv, dist_trig, cumret, 0.0, thresh, trigger, tk, i, rel)
        rows.append({"date": str(r["date"]), "ticker": tk, "close": round(px, 4),
                     "day_ret": round(day_ret, 4), "mom": round(mom, 3), "conv": round(conv, 3),
                     "phase": "SPOT", "dist_trigger": round(dist_trig, 4),
                     "value": round(value, 2), "upnl": round(upnl, 2), "cumret": round(cumret, 4),
                     "peer_cum": round(peer_cum, 4) if peer_cum is not None else None,
                     "rel": rel, "vol_base": float(r["volume"]), "note": note})
        if conv >= thresh and not hi:
            hi = True
            events.append({"date": str(r["date"]), "kind": "SIGNAL_HIGH",
                           "msg": f"composite {conv:.2f} >= {thresh:.2f} but px {px:.4f} off-thesis (arm {trigger:.2f}); no leverage"})
        elif conv < thresh:
            hi = False
        for lvl in sorted(milestones, reverse=True):
            if not milestones[lvl] and cumret <= lvl:
                milestones[lvl] = 1
                events.append({"date": str(r["date"]), "kind": "DRAWDOWN",
                               "msg": f"cumulative PnL crossed {lvl*100:.0f}% (px {px:.4f})"})
        prev_close = px

    dates = [x["date"] for x in rows]; closes = [x["close"] for x in rows]; values = [x["value"] for x in rows]
    an = _analytics(closes, values, cost, btc_ret, dates)
    # recompute per-row drawdown for the note's dd field would require a second pass; fill dd + vol30
    vol_ann_o = an["vol_ann"]      # fallback daily vol before the 30d window fills
    peak = -1e18
    for pos, (x, rv) in enumerate(zip(rows, an.pop("roll_vol"))):
        peak = max(peak, x["value"]); x["dd"] = round(x["value"] / peak - 1, 4); x["vol30"] = rv
        # ── exit liquidity / slippage — square-root market-impact law ──
        # slippage ≈ Y · σ_daily · √(position / daily_volume), Y≈0.9
        V = x["vol_base"]
        if V and V > 0:
            sig_d = (rv if rv is not None else vol_ann_o) / ANN
            part = amount / V                              # position as a multiple of daily volume
            slip = min(0.95, 0.9 * sig_d * math.sqrt(part))
            x["adv_usd"] = round(V * x["close"], 2)
            x["exit_part"] = round(part, 3)
            x["exit_slip"] = round(slip, 4)
            x["exit_net"] = round(x["value"] * (1 - slip), 2)
            x["exit_days"] = round(amount / (0.25 * V), 2)  # days to unwind at 25% of ADV
        else:
            x["adv_usd"] = x["exit_part"] = x["exit_slip"] = x["exit_net"] = x["exit_days"] = None
        # refresh note now that the real drawdown is known
        x["note"] = _note(x["close"], x["day_ret"], x["mom"], x["conv"], x["dist_trigger"],
                          x["cumret"], x["dd"], thresh, trigger, x["ticker"], start + pos, x["rel"])

    # ── vertical benchmark analytics ──
    lvl = np.array([1 + x["peer_cum"] if x["peer_cum"] is not None else np.nan for x in rows])
    fr = np.array([1 + x["cumret"] for x in rows])
    b_ret = np.diff(lvl) / lvl[:-1]
    f_ret = np.diff(fr) / fr[:-1]
    mask = ~np.isnan(b_ret)
    beta_vert = corr_vert = None
    if mask.sum() > 2:
        cov = np.cov(f_ret[mask], b_ret[mask])
        beta_vert = round(float(cov[0, 1] / cov[1, 1]), 3) if cov[1, 1] else None
        corr_vert = round(float(np.corrcoef(f_ret[mask], b_ret[mask])[0, 1]), 3)
    bm = b_ret[mask]
    vertical_sharpe = (round(float(bm.mean() / bm.std(ddof=1) * ANN), 3)
                       if mask.sum() > 1 and bm.std(ddof=1) else None)
    peer_final = sorted(({"sym": s, "cum": c, "sharpe": basket["sharpe"].get(s)}
                         for s, c in basket["final"].items()), key=lambda z: z["cum"])
    vert_cum = rows[-1]["peer_cum"]
    rel_final = rows[-1]["rel"]
    # range + FRAX rank across the wider peer universe (incl. FRAX itself)
    n_peers = len(peer_final)
    ranked = sorted(peer_final + [{"sym": "FRAX", "cum": rows[-1]["cumret"]}], key=lambda z: z["cum"])
    frax_rank = next(i for i, z in enumerate(ranked) if z["sym"] == "FRAX") + 1  # 1 = worst
    peer_worst = peer_final[0]
    peer_best = peer_final[-1]
    peer_median = peer_final[n_peers // 2]

    # map macro events to row indices within the window (for chart overlays)
    row_dates = [x["date"] for x in rows]
    macro = []
    for d, lab in MACRO_EVENTS:
        if d < row_dates[0] or d > row_dates[-1]:
            continue
        idx = next((i for i, rd in enumerate(row_dates) if rd >= d), None)
        if idx is not None:
            macro.append({"date": d, "label": lab, "idx": idx, "close": rows[idx]["close"]})

    last = rows[-1]; worst = min(rows, key=lambda x: x["day_ret"]); best = max(rows, key=lambda x: x["day_ret"])
    diligence_res = diligence.run(ex, cfg, last["close"], an["vol_ann"], btc_close)
    valuation_res = valuation.run()
    holders_res = holders_mod.run()
    # cross-denominated returns: position return expressed in BTC / ETH terms
    ed, ld = rows[0]["date"], rows[-1]["date"]
    def _denom(cd):
        e, l = cd.get(ed), cd.get(ld)
        if not e or not l:
            return None, None
        return round((1 + last["cumret"]) * (e / l) - 1, 4), round(l / e - 1, 4)
    cumret_btc, btc_cum = _denom(btc_close)
    cumret_eth, eth_cum = _denom(eth_close)
    # split of the USD loss into sector-beta vs FRAX-specific
    beta_part = vert_cum          # what the vertical average did (market beta)
    idio_part = rel_final         # FRAX-specific excess
    summary = {"symbol": "FXS→FRAX/USDT", "entry_date": entry_iso, "entry_px": entry_px, "amount": amount,
               "cost_basis": round(cost, 2), "trigger": trigger,
               "principal_stop": (round(entry_px * (1 - cfg["strategy"]["spot_stop_pct"]), 4) if has_stop else None),
               "threshold": thresh, "last_date": last["date"], "last_ticker": last["ticker"], "days": len(rows),
               "last_close": last["close"], "value": last["value"], "upnl": last["upnl"], "cumret": last["cumret"],
               "phase": last["phase"], "leverage_armed": False, "dist_trigger": last["dist_trigger"],
               "last_conv": last["conv"], "ticker_split": str(split),
               "vertical_cum": vert_cum, "rel_final": rel_final, "beta_vert": beta_vert,
               "corr_vert": corr_vert, "peers_used": basket["used"], "peer_final": peer_final,
               "n_peers": n_peers, "frax_rank": frax_rank, "peer_worst": peer_worst,
               "peer_best": peer_best, "peer_median": peer_median, "vertical_sharpe": vertical_sharpe,
               "cumret_btc": cumret_btc, "btc_cum": btc_cum, "cumret_eth": cumret_eth, "eth_cum": eth_cum,
               "beta_part": beta_part, "idio_part": idio_part, "unrealized": True, "unlevered": True,
               "exit_adv": last["adv_usd"], "exit_part": last["exit_part"], "exit_slip": last["exit_slip"],
               "exit_net": last["exit_net"], "exit_days": last["exit_days"],
               "leverage": cfg["strategy"]["leverage"], "house_stop_frac": cfg["strategy"]["house_stop_frac"],
               "lev_target_mult": cfg["strategy"].get("leverage_target_mult"), "macro": macro,
               "diligence": diligence_res, "valuation": valuation_res, "holders": holders_res,
               "worst_day": {"date": worst["date"], "ret": worst["day_ret"]},
               "best_day": {"date": best["date"], "ret": best["day_ret"]},
               "generated": datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")}
    summary.update(an)
    events.insert(0, {"date": entry_iso, "kind": "ENTRY",
                      "msg": f"seeded {amount:.0f} @ avg {entry_px:.2f} (cost {cost:,.0f}); NO principal stop; leverage arm {trigger:.2f}"})
    return {"summary": summary, "events": events, "rows": rows}


def print_log(log):
    s = log["summary"]
    print(f"\nMW-QB MONITOR {s['symbol']} entry {s['entry_date']} @ {s['entry_px']} -> {s['last_date']} @ {s['last_close']}")
    for r in log["rows"][::7]:
        print(f"{r['date']}  {r['ticker']:<4} {r['close']:>8.4f} {r['cumret']*100:>7.1f}%  {r['note'][:88]}")
    print(f"\n{s['days']}d | value {s['value']:,.0f} | uPnL {s['upnl']:,.0f} | cum {s['cumret']*100:.1f}% "
          f"| volA {s['vol_ann']*100:.0f}% | Sharpe {s['sharpe']} | maxDD {s['max_dd']*100:.1f}% "
          f"| beta {s['beta_btc']} corr {s['corr_btc']}\n")


def main():
    ap = argparse.ArgumentParser(description="MW-QB daily monitor + AI-agent decision log")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--entry", default="2025-07-13")
    ap.add_argument("--end", default="2026-07-24")
    ap.add_argument("--json", default="monitor_log.json")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()
    log = build_log(load_cfg(args.config), args.entry, args.end)
    with open(args.json, "w") as f:
        json.dump(log, f, indent=2)
    if not args.quiet:
        print_log(log)
    print(f"wrote {args.json}  ({log['summary']['days']} rows)")


if __name__ == "__main__":
    main()
