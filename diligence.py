#!/usr/bin/env python3
"""
Institutional diligence for the MW-QB strategy.

Three tests a professional allocator would demand — run on REAL data, reported
honestly whether or not they support the strategy:

1. Multi-asset signal backtest + statistical significance
   (annualized Sharpe, t-stat, p-value, Probabilistic & Deflated Sharpe,
    Minimum Track Record Length). Tests whether the momentum ENTRY gate has
    edge, pooled across the stablecoin-issuer vertical.

2. Factor regression — is there alpha after controlling for BTC market beta?

3. Monte-Carlo forward EV of the staged-leverage payoff from the current price
   (P(trigger reached), P(profit), expected return vs spot).

All figures are estimates from historical candles + a driftless GBM; not a
forecast. Deflated Sharpe assumes a trial count for the parameter search.
"""

import math
import numpy as np
import pandas as pd

# vertical universe (multi-segment = ticker migration, chained by return)
ASSETS = {
    "FRAX":   ["FXS/USDT", "FRAX/USDT"],
    "MKRSKY": ["MKR/USDT", "SKY/USDT"],
    "CRV":    ["CRV/USDT"], "AAVE": ["AAVE/USDT"], "ENA": ["ENA/USDT"],
    "LQTY":   ["LQTY/USDT"], "SPELL": ["SPELL/USDT"], "RSR": ["RSR/USDT"],
    "USUAL":  ["USUAL/USDT"],
}
TRIALS = 25          # assumed parameter-search breadth for Deflated Sharpe
GAMMA = 0.5772156649  # Euler–Mascheroni


def _ncdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _nppf(p):
    """Acklam's inverse-normal approximation."""
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    pl, ph = 0.02425, 1 - 0.02425
    if p < pl:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p > ph:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    q = p - 0.5; r = q*q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)


def _fetch(ex, sym, limit=1000):
    raw = ex.fetch_ohlcv(sym, timeframe="1d", limit=limit)
    df = pd.DataFrame(raw, columns=["ts", "o", "h", "l", "close", "v"])
    df["date"] = pd.to_datetime(df["ts"], unit="ms").dt.date
    return df[["date", "close"]]


def _asset_index(ex, segs):
    """Chained return-index (1.0 at first date); scale-invariant for signals."""
    frames = []
    for s in segs:
        try:
            d = _fetch(ex, s).copy(); d["seg"] = s; frames.append(d)
        except Exception:
            continue
    if not frames:
        return None
    a = pd.concat(frames).sort_values("date").drop_duplicates("date", keep="first").reset_index(drop=True)
    if len(a) < 120:
        return None
    lvl, prev, pseg, out = 1.0, None, None, {}
    for _, r in a.iterrows():
        if prev is not None and r["seg"] == pseg:
            lvl *= r["close"] / prev
        out[str(r["date"])] = lvl
        prev, pseg = r["close"], r["seg"]
    return out


def _ema(s, n):
    return s.ewm(span=n, adjust=False).mean()


def _rsi(s, n):
    d = s.diff()
    g = d.clip(lower=0).ewm(alpha=1/n, adjust=False).mean()
    l = (-d.clip(upper=0)).ewm(alpha=1/n, adjust=False).mean()
    return (100 - 100/(1 + g/l.replace(0, np.nan))).fillna(100)


def _signal_returns(idx, m, thresh):
    """Momentum long/flat daily strat returns + per-trade returns for one asset."""
    ds = sorted(idx)
    px = pd.Series([idx[d] for d in ds])
    ema_f, ema_s = _ema(px, m["ema_fast"]), _ema(px, m["ema_slow"])
    rsi, roc = _rsi(px, m["rsi_period"]), px.pct_change(m["roc_period"]) * 100
    ret = px.pct_change().fillna(0.0).values
    trend = (ema_f > ema_s).astype(float).values
    rsi_ok = ((rsi >= m["rsi_min"]) & (rsi <= m["rsi_max"])).astype(float).values
    roc_s = np.clip(0.5 + roc.fillna(0).values / 10.0, 0, 1)
    mom = 0.5*trend + 0.2*rsi_ok + 0.3*roc_s
    conv = 0.4*mom + 0.3
    pos = (conv >= thresh).astype(float)          # position held next day
    strat = np.concatenate([[0.0], pos[:-1] * ret[1:]])
    # per-trade returns (contiguous long runs)
    trades, i, n = [], 0, len(pos)
    while i < n:
        if pos[i] >= 1:
            j = i
            while j+1 < n and pos[j+1] >= 1:
                j += 1
            # holding return over [i+1 .. j+1]
            seg = ret[i+1:min(j+2, n)]
            if len(seg):
                trades.append(float(np.prod(1+seg) - 1))
            i = j + 1
        else:
            i += 1
    return dict(zip(ds, strat)), trades


def _significance(rets, trials=TRIALS):
    r = np.asarray(rets, float)
    N = len(r); mu = r.mean(); sd = r.std(ddof=1)
    if sd == 0 or N < 30:
        return None
    sr_d = mu/sd                                   # per-day Sharpe
    sr_a = sr_d * math.sqrt(365)
    t = sr_d * math.sqrt(N)
    p = 2 * (1 - _ncdf(abs(t)))
    z = (r - mu)/sd
    g3 = float(np.mean(z**3)); g4 = float(np.mean(z**4))   # skew, kurtosis (non-excess)
    denom = max(1e-9, 1 - g3*sr_d + (g4-1)/4*sr_d**2)
    psr = _ncdf(sr_d * math.sqrt(N-1) / math.sqrt(denom))
    # Deflated Sharpe: benchmark SR* from the trial count
    var_sr = denom/(N-1)
    sr_star = math.sqrt(var_sr) * ((1-GAMMA)*_nppf(1-1.0/trials) + GAMMA*_nppf(1-1.0/(trials*math.e)))
    dsr = _ncdf((sr_d - sr_star) * math.sqrt(N-1) / math.sqrt(denom))
    # Minimum track record length for PSR>0.95 (vs 0), in years
    z95 = _nppf(0.95)
    mintrl = (denom * (z95/sr_d)**2 + 1) if sr_d > 0 else float("inf")
    return {"N": N, "sharpe_ann": round(sr_a, 3), "t_stat": round(t, 2), "p_value": round(p, 4),
            "skew": round(g3, 3), "kurt": round(g4-3, 2), "psr": round(psr, 3),
            "dsr": round(dsr, 3), "sr_star_ann": round(sr_star*math.sqrt(365), 3),
            "min_trl_years": (round(mintrl/365, 1) if math.isfinite(mintrl) else None)}


def _factor(strat_by_date, btc_idx, m=None):
    """OLS: strat_daily = alpha + beta*btc_daily. Returns alpha(ann), beta, t(alpha), R²."""
    bds = sorted(btc_idx); bpx = pd.Series([btc_idx[d] for d in bds])
    bret = dict(zip(bds, bpx.pct_change().fillna(0.0).values))
    common = [d for d in sorted(strat_by_date) if d in bret]
    if len(common) < 60:
        return None
    y = np.array([strat_by_date[d] for d in common])
    x = np.array([bret[d] for d in common])
    X = np.column_stack([np.ones_like(x), x])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    n, k = len(y), 2
    s2 = (resid @ resid) / (n - k)
    cov = s2 * np.linalg.inv(X.T @ X)
    se_a = math.sqrt(cov[0, 0])
    t_a = beta[0]/se_a if se_a else 0.0
    ss = ((y - y.mean())**2).sum()
    r2 = 1 - (resid @ resid)/ss if ss else 0.0
    return {"alpha_ann": round(beta[0]*365, 4), "beta_btc": round(beta[1], 3),
            "t_alpha": round(t_a, 2), "r2": round(r2, 3), "N": n}


def _montecarlo(S0, K, C, N_units, lev, hsf, tgt_mult, vol_ann, H=365, M=20000, seed=7):
    rng = np.random.default_rng(seed)
    sig = vol_ann/math.sqrt(365)
    Z = rng.standard_normal((M, H))
    logpath = np.cumsum((-0.5*sig*sig) + sig*Z, axis=1)   # driftless (martingale)
    path = S0 * np.exp(logpath)
    ST = path[:, -1]; pmax = path.max(axis=1)
    triggered = pmax >= K
    Hm = N_units*K - C                                     # house money at trigger
    Stp = K*(1 + (tgt_mult-1)/lev)
    spot_eq = N_units*ST
    legP = lev*Hm*(np.clip(ST, None, Stp)/K - 1)
    legP = np.maximum(legP, -hsf*Hm)                       # house-stop floor
    strat_eq = np.where(triggered, C + Hm + legP, N_units*ST)
    spot_pl = spot_eq/C - 1; strat_pl = strat_eq/C - 1
    return {"horizon_days": H, "paths": M, "vol_ann": round(vol_ann, 2),
            "p_trigger": round(float(triggered.mean()), 4),
            "spot_p_profit": round(float((spot_pl > 0).mean()), 4),
            "spot_ev": round(float(spot_pl.mean()), 3),
            "strat_p_profit": round(float((strat_pl > 0).mean()), 4),
            "strat_ev": round(float(strat_pl.mean()), 3),
            "strat_p5": round(float(np.percentile(strat_pl, 5)), 3),
            "strat_p50": round(float(np.percentile(strat_pl, 50)), 3),
            "strat_p95": round(float(np.percentile(strat_pl, 95)), 3)}


def run(ex, cfg, last_close, vol_ann, btc_idx):
    m = cfg["momentum"]; thresh = cfg["signals"]["trigger_threshold"]
    st = cfg["strategy"]
    port_by_date, all_trades, used = {}, [], []
    for name, segs in ASSETS.items():
        idx = _asset_index(ex, segs)
        if not idx:
            continue
        sr, trades = _signal_returns(idx, m, thresh)
        used.append(name); all_trades += trades
        for d, v in sr.items():
            port_by_date.setdefault(d, []).append(v)
    dates = sorted(port_by_date)
    port = {d: float(np.mean(port_by_date[d])) for d in dates}   # equal-weight vertical
    port_arr = [port[d] for d in dates]

    sig = _significance(port_arr)
    fac = _factor(port, btc_idx)
    # equity curve for maxDD / total
    eq = np.cumprod(1 + np.asarray(port_arr))
    peak = np.maximum.accumulate(eq)
    bt = {"assets": used, "n_assets": len(used), "n_days": len(dates),
          "total_return": round(float(eq[-1]-1), 3),
          "max_dd": round(float((eq/peak-1).min()), 3),
          "n_trades": len(all_trades),
          "win_rate": round(float(np.mean([1 if t > 0 else 0 for t in all_trades])), 3) if all_trades else None,
          "avg_trade": round(float(np.mean(all_trades)), 4) if all_trades else None,
          "expectancy": round(float(np.mean(all_trades)), 4) if all_trades else None}
    mc = _montecarlo(last_close, st.get("leverage_trigger_price", 4.4) or 4.4,
                     float(cfg["position"]["seed"]["avg_entry"])*float(cfg["position"]["seed"]["amount"]),
                     float(cfg["position"]["seed"]["amount"]), st["leverage"], st["house_stop_frac"],
                     st.get("leverage_target_mult", 2.0) or 2.0, vol_ann)
    return {"backtest": bt, "significance": sig, "factor": fac, "montecarlo": mc, "trials": TRIALS}
