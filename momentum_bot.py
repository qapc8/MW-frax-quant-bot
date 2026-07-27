#!/usr/bin/env python3
"""
Frax Momentum Bot
=================

A momentum trading bot for the Frax ecosystem (defaults to FXS/USDT — see the
note in config.yaml about FRAX being a stablecoin).

Strategy: long-only momentum.
    ENTER long  when  EMA_fast > EMA_slow  AND  RSI in [rsi_min, rsi_max]
                      AND  ROC > roc_min      (all three must agree)
    EXIT        when  EMA_fast < EMA_slow  OR  stop-loss / take-profit /
                      trailing-stop is hit.

Runs in PAPER mode by default (engine.live: false) — no real orders are placed
until you set live: true and provide API keys. Uses ccxt so it works across
most centralized exchanges.

Usage:
    python momentum_bot.py                 # run the live loop (paper by default)
    python momentum_bot.py --once          # evaluate a single time and exit
    python momentum_bot.py --backtest      # backtest the strategy on recent candles
    python momentum_bot.py --config my.yaml
"""

import argparse
import json
import logging
import os
import signal
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

try:
    import ccxt
except ImportError:
    ccxt = None


# ── Indicators ─────────────────────────────────────────────────────────────

def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def rsi(series: pd.Series, period: int) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    out = 100 - (100 / (1 + rs))
    return out.fillna(100.0)  # no losses => maximally overbought


def roc(series: pd.Series, period: int) -> pd.Series:
    """Rate of change in percent."""
    return series.pct_change(periods=period) * 100.0


def add_indicators(df: pd.DataFrame, s: "StrategyCfg") -> pd.DataFrame:
    df = df.copy()
    df["ema_fast"] = ema(df["close"], s.ema_fast)
    df["ema_slow"] = ema(df["close"], s.ema_slow)
    df["rsi"] = rsi(df["close"], s.rsi_period)
    df["roc"] = roc(df["close"], s.roc_period)
    return df


def entry_signal(row: pd.Series, s: "StrategyCfg") -> bool:
    return bool(
        row["ema_fast"] > row["ema_slow"]
        and s.rsi_min <= row["rsi"] <= s.rsi_max
        and row["roc"] > s.roc_min
    )


def trend_broken(row: pd.Series) -> bool:
    """Momentum has flipped — the primary exit condition."""
    return bool(row["ema_fast"] < row["ema_slow"])


# ── Config ─────────────────────────────────────────────────────────────────

@dataclass
class ExchangeCfg:
    id: str = "binance"
    symbol: str = "FXS/USDT"
    timeframe: str = "1h"
    api_key: str = ""
    secret: str = ""


@dataclass
class StrategyCfg:
    ema_fast: int = 12
    ema_slow: int = 26
    rsi_period: int = 14
    rsi_min: float = 45.0
    rsi_max: float = 78.0
    roc_period: int = 10
    roc_min: float = 0.0


@dataclass
class RiskCfg:
    quote_balance: float = 1000.0
    risk_per_trade: float = 0.02
    stop_loss_pct: float = 0.04
    take_profit_pct: float = 0.10
    trailing_stop_pct: float = 0.05
    max_position_pct: float = 0.95


@dataclass
class EngineCfg:
    live: bool = False
    poll_seconds: int = 3600
    state_file: str = "state.json"
    log_file: str = "bot.log"


@dataclass
class Config:
    exchange: ExchangeCfg
    strategy: StrategyCfg
    risk: RiskCfg
    engine: EngineCfg

    @classmethod
    def load(cls, path: str) -> "Config":
        raw = yaml.safe_load(Path(path).read_text()) or {}
        cfg = cls(
            exchange=ExchangeCfg(**(raw.get("exchange") or {})),
            strategy=StrategyCfg(**(raw.get("strategy") or {})),
            risk=RiskCfg(**(raw.get("risk") or {})),
            engine=EngineCfg(**(raw.get("engine") or {})),
        )
        # Environment overrides for secrets — never commit keys.
        cfg.exchange.api_key = os.getenv("FMB_API_KEY", cfg.exchange.api_key)
        cfg.exchange.secret = os.getenv("FMB_SECRET", cfg.exchange.secret)
        return cfg


# ── Position state ─────────────────────────────────────────────────────────

@dataclass
class Position:
    is_open: bool = False
    entry_price: float = 0.0
    amount: float = 0.0          # base-currency size
    stop_price: float = 0.0
    take_profit: float = 0.0
    highest: float = 0.0         # peak price seen, for trailing stop

    def to_dict(self):
        return asdict(self)


class State:
    def __init__(self, path: str, starting_quote: float):
        self.path = Path(path)
        self.quote = starting_quote          # free quote balance (paper)
        self.position = Position()
        self.realized_pnl = 0.0
        self.trades = 0
        if self.path.exists():
            self._load()

    def _load(self):
        d = json.loads(self.path.read_text())
        self.quote = d.get("quote", self.quote)
        self.realized_pnl = d.get("realized_pnl", 0.0)
        self.trades = d.get("trades", 0)
        self.position = Position(**d.get("position", {}))

    def save(self):
        self.path.write_text(json.dumps({
            "quote": self.quote,
            "realized_pnl": self.realized_pnl,
            "trades": self.trades,
            "position": self.position.to_dict(),
        }, indent=2))

    def equity(self, price: float) -> float:
        return self.quote + (self.position.amount * price if self.position.is_open else 0.0)


# ── Bot ────────────────────────────────────────────────────────────────────

class MomentumBot:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.log = logging.getLogger("fmb")
        self.state = State(cfg.engine.state_file, cfg.risk.quote_balance)
        self.exchange = self._make_exchange()
        self._running = True

    def _make_exchange(self):
        if ccxt is None:
            raise SystemExit("ccxt not installed. Run: pip install -r requirements.txt")
        ex_cls = getattr(ccxt, self.cfg.exchange.id, None)
        if ex_cls is None:
            raise SystemExit(f"Unknown exchange id: {self.cfg.exchange.id}")
        params = {"enableRateLimit": True}
        if self.cfg.engine.live:
            params["apiKey"] = self.cfg.exchange.api_key
            params["secret"] = self.cfg.exchange.secret
        return ex_cls(params)

    # ---- market data ----
    def fetch_ohlcv(self, limit: int = 300) -> pd.DataFrame:
        raw = self.exchange.fetch_ohlcv(
            self.cfg.exchange.symbol, timeframe=self.cfg.exchange.timeframe, limit=limit
        )
        df = pd.DataFrame(raw, columns=["ts", "open", "high", "low", "close", "volume"])
        df["dt"] = pd.to_datetime(df["ts"], unit="ms")
        return df

    # ---- sizing ----
    def position_size(self, price: float, stop_price: float) -> float:
        """Risk-based sizing: risk_per_trade of equity over the stop distance,
        capped by max_position_pct of equity."""
        equity = self.state.equity(price)
        risk_amount = equity * self.cfg.risk.risk_per_trade
        stop_dist = max(price - stop_price, 1e-9)
        amount = risk_amount / stop_dist
        max_amount = (equity * self.cfg.risk.max_position_pct) / price
        amount = min(amount, max_amount, self.state.quote / price)
        return max(amount, 0.0)

    # ---- order execution (paper vs live) ----
    def _buy(self, price: float, amount: float):
        cost = price * amount
        if self.cfg.engine.live:
            self.exchange.create_market_buy_order(self.cfg.exchange.symbol, amount)
        self.state.quote -= cost
        self.log.info("BUY  %.6f @ %.4f  (cost %.2f)", amount, price, cost)

    def _sell(self, price: float, amount: float, reason: str):
        proceeds = price * amount
        if self.cfg.engine.live:
            self.exchange.create_market_sell_order(self.cfg.exchange.symbol, amount)
        self.state.quote += proceeds
        pnl = (price - self.state.position.entry_price) * amount
        self.state.realized_pnl += pnl
        self.state.trades += 1
        self.log.info("SELL %.6f @ %.4f  (%s)  pnl %.2f  total %.2f",
                      amount, price, reason, pnl, self.state.realized_pnl)

    # ---- core decision on the latest closed candle ----
    def evaluate(self):
        df = add_indicators(self.fetch_ohlcv(), self.cfg.strategy)
        # Use the last *closed* candle to avoid acting on a forming bar.
        row = df.iloc[-2]
        price = float(row["close"])
        pos = self.state.position

        mode = "LIVE" if self.cfg.engine.live else "PAPER"
        self.log.info(
            "[%s] %s %s | px %.4f ema%d %.4f ema%d %.4f rsi %.1f roc %.2f%% | eq %.2f",
            mode, self.cfg.exchange.symbol, self.cfg.exchange.timeframe, price,
            self.cfg.strategy.ema_fast, row["ema_fast"],
            self.cfg.strategy.ema_slow, row["ema_slow"],
            row["rsi"], row["roc"], self.state.equity(price),
        )

        if pos.is_open:
            self._manage_open_position(row, price)
        elif entry_signal(row, self.cfg.strategy):
            self._open_position(price)

        self.state.save()

    def _open_position(self, price: float):
        stop = price * (1 - self.cfg.risk.stop_loss_pct)
        amount = self.position_size(price, stop)
        if amount <= 0:
            self.log.info("Signal fired but size == 0 (insufficient balance).")
            return
        self._buy(price, amount)
        self.state.position = Position(
            is_open=True,
            entry_price=price,
            amount=amount,
            stop_price=stop,
            take_profit=price * (1 + self.cfg.risk.take_profit_pct),
            highest=price,
        )

    def _manage_open_position(self, row: pd.Series, price: float):
        pos = self.state.position
        pos.highest = max(pos.highest, price)

        # Trailing stop ratchets up as price makes new highs.
        if self.cfg.risk.trailing_stop_pct > 0:
            trail = pos.highest * (1 - self.cfg.risk.trailing_stop_pct)
            pos.stop_price = max(pos.stop_price, trail)

        reason = None
        if price <= pos.stop_price:
            reason = "stop-loss/trailing"
        elif price >= pos.take_profit:
            reason = "take-profit"
        elif trend_broken(row):
            reason = "trend-flip"

        if reason:
            self._sell(price, pos.amount, reason)
            self.state.position = Position()

    # ---- run loop ----
    def run(self):
        signal.signal(signal.SIGINT, self._stop)
        signal.signal(signal.SIGTERM, self._stop)
        self.log.info("Starting loop (poll %ds). Ctrl-C to stop.",
                      self.cfg.engine.poll_seconds)
        while self._running:
            try:
                self.evaluate()
            except Exception as exc:  # keep the loop alive on transient errors
                self.log.exception("evaluate() failed: %s", exc)
            for _ in range(self.cfg.engine.poll_seconds):
                if not self._running:
                    break
                time.sleep(1)
        self.log.info("Stopped. Realized PnL: %.2f over %d trades.",
                      self.state.realized_pnl, self.state.trades)

    def _stop(self, *_):
        self._running = False

    # ---- backtest ----
    def backtest(self, limit: int = 1000):
        df = add_indicators(self.fetch_ohlcv(limit=limit), self.cfg.strategy)
        cash = self.cfg.risk.quote_balance
        amount = 0.0
        entry = stop = tp = highest = 0.0
        trades = wins = 0
        equity_curve = []

        for i in range(1, len(df)):
            row = df.iloc[i]
            price = float(row["close"])
            if amount > 0:
                highest = max(highest, price)
                if self.cfg.risk.trailing_stop_pct > 0:
                    stop = max(stop, highest * (1 - self.cfg.risk.trailing_stop_pct))
                exit_now = price <= stop or price >= tp or trend_broken(row)
                if exit_now:
                    cash += price * amount
                    if price > entry:
                        wins += 1
                    trades += 1
                    amount = 0.0
            elif entry_signal(row, self.cfg.strategy):
                entry = price
                stop = price * (1 - self.cfg.risk.stop_loss_pct)
                tp = price * (1 + self.cfg.risk.take_profit_pct)
                highest = price
                spend = cash * self.cfg.risk.max_position_pct
                amount = spend / price
                cash -= spend
            equity_curve.append(cash + amount * price)

        final = equity_curve[-1] if equity_curve else cash
        start = self.cfg.risk.quote_balance
        ret = (final / start - 1) * 100
        curve = pd.Series(equity_curve)
        peak = curve.cummax()
        max_dd = ((curve - peak) / peak).min() * 100 if len(curve) else 0.0

        print("\n── Backtest ────────────────────────────────")
        print(f"Symbol        : {self.cfg.exchange.symbol} @ {self.cfg.exchange.timeframe}")
        print(f"Candles       : {len(df)}")
        print(f"Start equity  : {start:,.2f}")
        print(f"Final equity  : {final:,.2f}")
        print(f"Total return  : {ret:+.2f}%")
        print(f"Trades        : {trades}  (win rate {100*wins/trades:.1f}%)"
              if trades else "Trades        : 0")
        print(f"Max drawdown  : {max_dd:.2f}%")
        print("────────────────────────────────────────────\n")


# ── CLI ────────────────────────────────────────────────────────────────────

def setup_logging(log_file: str):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(log_file)],
    )


def main():
    ap = argparse.ArgumentParser(description="Frax momentum trading bot")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--once", action="store_true", help="evaluate once and exit")
    ap.add_argument("--backtest", action="store_true", help="backtest on recent candles")
    args = ap.parse_args()

    cfg = Config.load(args.config)
    setup_logging(cfg.engine.log_file)
    bot = MomentumBot(cfg)

    if cfg.engine.live:
        logging.getLogger("fmb").warning(
            "LIVE MODE ENABLED — real orders will be placed on %s.", cfg.exchange.id)

    if args.backtest:
        bot.backtest()
    elif args.once:
        bot.evaluate()
    else:
        bot.run()


if __name__ == "__main__":
    main()
