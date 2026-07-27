"""
Signal layer for the Frax Quant Bot.

Produces a composite conviction score in [0, 1] from four factors:
    momentum · TVL · social · news

Design rule: a factor with no configured data source returns a NEUTRAL 0.5.
We never fabricate market data, headlines, or sentiment. TVL is pulled from the
real DefiLlama API; social/news require your own API keys and otherwise stay
neutral (and say so in the log).
"""

import logging
import os

import numpy as np
import pandas as pd

try:
    import requests
except ImportError:
    requests = None

log = logging.getLogger("fmb.signals")


# ── Momentum factor (from OHLCV) ────────────────────────────────────────────

def _ema(s, n):
    return s.ewm(span=n, adjust=False).mean()


def _rsi(s, n):
    d = s.diff()
    gain = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    loss = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(100.0)


def momentum_score(df: pd.DataFrame, cfg: dict) -> float:
    """0..1 momentum conviction from EMA trend, RSI band, and ROC."""
    close = df["close"]
    ema_f = _ema(close, cfg["ema_fast"]).iloc[-2]
    ema_s = _ema(close, cfg["ema_slow"]).iloc[-2]
    rsi = _rsi(close, cfg["rsi_period"]).iloc[-2]
    roc = close.pct_change(cfg["roc_period"]).iloc[-2] * 100

    trend = 1.0 if ema_f > ema_s else 0.0
    # RSI inside [min, max] is good; center of band scores highest.
    lo, hi = cfg["rsi_min"], cfg["rsi_max"]
    rsi_ok = 1.0 if lo <= rsi <= hi else 0.0
    # ROC mapped through a soft ramp: 0% -> 0.5, +5% -> ~1.0, -5% -> ~0.0
    roc_score = float(np.clip(0.5 + roc / 10.0, 0.0, 1.0))

    score = 0.5 * trend + 0.2 * rsi_ok + 0.3 * roc_score
    return float(np.clip(score, 0.0, 1.0))


# ── TVL factor (real DefiLlama API) ─────────────────────────────────────────

def tvl_score(cfg: dict) -> float:
    """Score protocol TVL trend over `lookback_days`. Neutral 0.5 on any issue."""
    if cfg.get("provider", "defillama") == "none" or requests is None:
        log.info("TVL signal disabled -> neutral 0.5")
        return 0.5
    protocol = cfg.get("protocol", "frax")
    try:
        r = requests.get(f"https://api.llama.fi/protocol/{protocol}", timeout=15)
        r.raise_for_status()
        series = r.json().get("tvl", [])
        if len(series) < 2:
            return 0.5
        df = pd.DataFrame(series)  # columns: date, totalLiquidityUSD
        df = df.tail(cfg.get("lookback_days", 14) + 1)
        first = df["totalLiquidityUSD"].iloc[0]
        last = df["totalLiquidityUSD"].iloc[-1]
        pct = (last / first - 1) * 100 if first else 0.0
        full = cfg.get("up_pct_for_full", 15.0)
        score = float(np.clip(0.5 + pct / (2 * full), 0.0, 1.0))
        log.info("TVL %s: %.0f -> %.0f (%.1f%% over window) score %.2f",
                 protocol, first, last, pct, score)
        return score
    except Exception as exc:
        log.warning("TVL fetch failed (%s) -> neutral 0.5", exc)
        return 0.5


# ── Social factor (pluggable; neutral unless configured) ────────────────────

def social_score(cfg: dict) -> float:
    provider = cfg.get("provider", "none")
    if provider == "none":
        log.info("Social signal disabled -> neutral 0.5")
        return 0.5
    key = os.getenv("FMB_SOCIAL_KEY", "")
    if not key or requests is None:
        log.info("Social provider '%s' has no FMB_SOCIAL_KEY -> neutral 0.5", provider)
        return 0.5
    try:
        if provider == "lunarcrush":
            # LunarCrush v4: galaxy score / sentiment -> normalize to 0..1.
            r = requests.get(
                "https://lunarcrush.com/api4/public/coins/FXS/v1",
                headers={"Authorization": f"Bearer {key}"}, timeout=15)
            r.raise_for_status()
            d = r.json().get("data", {})
            galaxy = d.get("galaxy_score")  # 0..100
            if galaxy is None:
                return 0.5
            score = float(np.clip(galaxy / 100.0, 0.0, 1.0))
            log.info("Social (lunarcrush) galaxy_score %.0f -> %.2f", galaxy, score)
            return score
        log.warning("Unknown social provider '%s' -> neutral 0.5", provider)
        return 0.5
    except Exception as exc:
        log.warning("Social fetch failed (%s) -> neutral 0.5", exc)
        return 0.5


# ── News factor (pluggable; neutral unless configured) ──────────────────────

def news_score(cfg: dict) -> float:
    provider = cfg.get("provider", "none")
    if provider == "none":
        log.info("News signal disabled -> neutral 0.5")
        return 0.5
    key = os.getenv("FMB_NEWS_KEY", "")
    if not key or requests is None:
        log.info("News provider '%s' has no FMB_NEWS_KEY -> neutral 0.5", provider)
        return 0.5
    try:
        if provider == "cryptopanic":
            # Ratio of positive to (positive+negative) votes over recent posts.
            r = requests.get(
                "https://cryptopanic.com/api/v1/posts/",
                params={"auth_token": key, "currencies": "FXS", "public": "true"},
                timeout=15)
            r.raise_for_status()
            posts = r.json().get("results", [])
            pos = sum(p.get("votes", {}).get("positive", 0) for p in posts)
            neg = sum(p.get("votes", {}).get("negative", 0) for p in posts)
            if pos + neg == 0:
                return 0.5
            score = float(np.clip(pos / (pos + neg), 0.0, 1.0))
            log.info("News (cryptopanic) pos %d neg %d -> %.2f", pos, neg, score)
            return score
        log.warning("Unknown news provider '%s' -> neutral 0.5", provider)
        return 0.5
    except Exception as exc:
        log.warning("News fetch failed (%s) -> neutral 0.5", exc)
        return 0.5


# ── Composite ───────────────────────────────────────────────────────────────

def composite(df: pd.DataFrame, signals_cfg: dict, momentum_cfg: dict) -> dict:
    """Return {factors, weights, score} — the weighted conviction in [0,1]."""
    factors = {
        "momentum": momentum_score(df, momentum_cfg),
        "tvl": tvl_score(signals_cfg.get("tvl", {})),
        "social": social_score(signals_cfg.get("social", {})),
        "news": news_score(signals_cfg.get("news", {})),
    }
    weights = signals_cfg["weights"]
    total_w = sum(weights.values()) or 1.0
    score = sum(factors[k] * weights.get(k, 0.0) for k in factors) / total_w
    return {"factors": factors, "weights": weights, "score": float(score)}
