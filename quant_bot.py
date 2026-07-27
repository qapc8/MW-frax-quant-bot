#!/usr/bin/env python3
"""
Frax Quant Bot
==============

A signal-gated, staged-leverage quant bot for the Frax ecosystem.

Pipeline
--------
1. GATE  — build a weighted composite conviction score from four factors
   (momentum · TVL · social · news, see signals.py). If it does not exceed
   `signals.trigger_threshold`, THE ALGO DOES NOT TRIGGER. Optionally also
   require price to be on the user's thesis trajectory (target path).

2. SPOT  — on trigger, deploy principal into a spot long with a protective stop.

3. HOUSE — once the position reaches `strategy.double_target` (default 2x), BANK
   THE PRINCIPAL (move it to safe cash) and deploy only the PROFIT ("house
   money") into a leveraged long with an adequate stop. Principal is preserved;
   only house money is ever exposed to leverage.

Phases: IDLE -> SPOT -> HOUSE -> IDLE.

Safety: paper trading by default (engine.live: false). No real orders and no
real leverage are placed until you set live: true and provide keys. Leverage in
paper mode is simulated with an explicit liquidation model.

Usage:
    python quant_bot.py                # run the loop (paper by default)
    python quant_bot.py --once         # evaluate once and exit
    python quant_bot.py --backtest     # backtest the full state machine
    python quant_bot.py --config my.yaml
"""

import argparse
import json
import logging
import signal as signal_mod
import sys
import time
from datetime import date
from pathlib import Path

import pandas as pd
import yaml

import signals as sig

try:
    import ccxt
except ImportError:
    ccxt = None

log = logging.getLogger("fmb")


# ── Thesis trajectory ───────────────────────────────────────────────────────

def on_thesis(price: float, thesis: dict, today: date) -> bool:
    """True if `price` is at/above the linearly-interpolated target path.

    The path runs from target_1 (date/price) to target_2. Before target_1 we
    only require price > 0 (no lower anchor); after target_2 we require price
    >= target_2. These are the USER'S assumptions, not a forecast.
    """
    try:
        d1 = date.fromisoformat(thesis["target_1_date"])
        d2 = date.fromisoformat(thesis["target_2_date"])
        p1 = float(thesis["target_1_price"])
        p2 = float(thesis["target_2_price"])
    except Exception:
        return True  # malformed thesis -> don't block

    if today <= d1:
        return price > 0
    if today >= d2:
        return price >= p2
    frac = (today - d1).days / max((d2 - d1).days, 1)
    expected = p1 + (p2 - p1) * frac
    return price >= expected


# ── State ───────────────────────────────────────────────────────────────────

class State:
    """Persistent bot state across restarts."""

    def __init__(self, path: str, starting_quote: float):
        self.path = Path(path)
        self.phase = "IDLE"            # IDLE | SPOT | HOUSE
        self.free_quote = starting_quote
        self.safe_quote = 0.0          # banked principal (never re-risked automatically)
        self.realized_pnl = 0.0
        self.trades = 0

        # Spot leg
        self.spot_amount = 0.0
        self.spot_entry = 0.0
        self.spot_cost = 0.0
        self.spot_stop = 0.0

        # Leveraged (house) leg
        self.lev_margin = 0.0          # house money posted as collateral
        self.lev_notional = 0.0        # margin * leverage
        self.lev_amount = 0.0          # base size of the leveraged position
        self.lev_entry = 0.0
        self.lev_stop = 0.0
        self.lev_liq = 0.0             # liquidation price
        self.lev_target = 0.0          # optional take-profit price
        self.lev_high = 0.0            # peak for trailing stop

        if self.path.exists():
            self.__dict__.update(json.loads(self.path.read_text()))
            self.path = Path(path)

    def save(self):
        d = {k: v for k, v in self.__dict__.items() if k != "path"}
        d["path"] = str(self.path)
        self.path.write_text(json.dumps(d, indent=2))

    def equity(self, price: float) -> float:
        eq = self.free_quote + self.safe_quote
        if self.phase == "SPOT":
            eq += self.spot_amount * price
        elif self.phase == "HOUSE":
            eq += self.lev_margin + (price - self.lev_entry) * self.lev_amount
        return eq


# ── Bot ─────────────────────────────────────────────────────────────────────

class QuantBot:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.live = cfg["engine"]["live"]
        self.symbol = cfg["exchange"]["symbol"]
        self.timeframe = cfg["exchange"]["timeframe"]
        self.state = State(cfg["engine"]["state_file"], cfg["risk"]["quote_balance"])
        self.exchange = self._make_exchange()
        self._running = True

    def _make_exchange(self):
        if ccxt is None:
            raise SystemExit("ccxt not installed. pip install -r requirements.txt")
        ex_cls = getattr(ccxt, self.cfg["exchange"]["id"], None)
        if ex_cls is None:
            raise SystemExit(f"Unknown exchange: {self.cfg['exchange']['id']}")
        params = {"enableRateLimit": True}
        if self.live:
            params["apiKey"] = self.cfg["exchange"]["api_key"]
            params["secret"] = self.cfg["exchange"]["secret"]
        return ex_cls(params)

    def fetch_ohlcv(self, limit=400) -> pd.DataFrame:
        raw = self.exchange.fetch_ohlcv(self.symbol, timeframe=self.timeframe, limit=limit)
        df = pd.DataFrame(raw, columns=["ts", "open", "high", "low", "close", "volume"])
        df["dt"] = pd.to_datetime(df["ts"], unit="ms")
        return df

    # ---- order helpers (paper vs live) ----
    def _market_buy(self, amount, price, tag):
        if self.live:
            self.exchange.create_market_buy_order(self.symbol, amount)
        log.info("BUY  %.6f @ %.4f  (%s)", amount, price, tag)

    def _market_sell(self, amount, price, tag):
        if self.live:
            self.exchange.create_market_sell_order(self.symbol, amount)
        log.info("SELL %.6f @ %.4f  (%s)", amount, price, tag)

    # ---- main evaluation on the latest closed candle ----
    def evaluate(self, today: date | None = None):
        today = today or date.today()
        df = self.fetch_ohlcv()
        price = float(df["close"].iloc[-2])  # last closed candle
        conv = sig.composite(df, self.cfg["signals"], self.cfg["momentum"])
        f = conv["factors"]
        log.info("[%s] %s phase=%s px=%.4f conv=%.2f (mom %.2f tvl %.2f soc %.2f news %.2f) eq=%.2f",
                 "LIVE" if self.live else "PAPER", self.symbol, self.state.phase, price,
                 conv["score"], f["momentum"], f["tvl"], f["social"], f["news"],
                 self.state.equity(price))

        if self.state.phase == "IDLE":
            self._maybe_enter(price, conv["score"], today)
        elif self.state.phase == "SPOT":
            self._manage_spot(price)
        elif self.state.phase == "HOUSE":
            self._manage_house(price)

        self.state.save()
        return {"price": price, "conviction": conv, "phase": self.state.phase}

    # ---- IDLE -> SPOT ----
    def _maybe_enter(self, price, conviction, today):
        threshold = self.cfg["signals"]["trigger_threshold"]
        if conviction < threshold:
            log.info("No trigger: conviction %.2f < threshold %.2f", conviction, threshold)
            return
        th = self.cfg["thesis"]
        if th.get("require_on_thesis") and not on_thesis(price, th, today):
            log.info("No trigger: price %.4f below thesis trajectory.", price)
            return

        alloc = self.state.free_quote * self.cfg["risk"]["entry_alloc_pct"]
        amount = alloc / price
        if amount <= 0:
            log.info("Trigger fired but no free quote to deploy.")
            return
        self._market_buy(amount, price, "spot entry")
        self.state.free_quote -= alloc
        self.state.phase = "SPOT"
        self.state.spot_amount = amount
        self.state.spot_entry = price
        self.state.spot_cost = alloc
        self.state.spot_stop = price * (1 - self.cfg["strategy"]["spot_stop_pct"])
        log.info("ENTER SPOT: %.6f @ %.4f cost %.2f stop %.4f (2x target value %.2f)",
                 amount, price, alloc, self.state.spot_stop,
                 alloc * self.cfg["strategy"]["double_target"])

    # ---- SPOT: stop out, or hit 2x and roll into leverage ----
    def _manage_spot(self, price):
        s = self.state
        value = s.spot_amount * price

        if price <= s.spot_stop:
            self._market_sell(s.spot_amount, price, "spot stop")
            proceeds = value
            pnl = proceeds - s.spot_cost
            s.free_quote += proceeds
            s.realized_pnl += pnl
            s.trades += 1
            log.info("SPOT STOPPED: pnl %.2f", pnl)
            self._reset_spot(); s.phase = "IDLE"
            return

        target_value = s.spot_cost * self.cfg["strategy"]["double_target"]
        if value >= target_value:
            # Bank principal, deploy profit as leveraged house money.
            self._market_sell(s.spot_amount, price, "2x reached -> bank principal")
            proceeds = value
            profit = proceeds - s.spot_cost
            s.safe_quote += s.spot_cost           # principal preserved
            s.realized_pnl += profit
            s.trades += 1
            log.info("2x REACHED: value %.2f. Banked principal %.2f, house money %.2f",
                     value, s.spot_cost, profit)
            self._reset_spot()
            self._open_leverage(price, profit)

    # ---- open the leveraged leg with house money ----
    def _open_leverage(self, price, margin):
        st = self.cfg["strategy"]
        lev = st["leverage"]
        if margin <= 0:
            log.info("No house money to lever; returning to IDLE.")
            self.state.phase = "IDLE"; return

        notional = margin * lev
        amount = notional / price
        # Liquidation ~ price where a full-margin loss occurs (fees ignored).
        liq = price * (1 - 1 / lev)
        # Stop after losing `house_stop_frac` of the margin: move = frac/lev.
        stop_move = st["house_stop_frac"] / lev
        stop = price * (1 - stop_move)
        stop = max(stop, liq * 1.001)  # keep the stop above liquidation
        target = price * (1 + (st["leverage_target_mult"] - 1) / lev) \
            if st.get("leverage_target_mult") else 0.0

        s = self.state
        s.phase = "HOUSE"
        s.lev_margin = margin
        s.lev_notional = notional
        s.lev_amount = amount
        s.lev_entry = price
        s.lev_stop = stop
        s.lev_liq = liq
        s.lev_target = target
        s.lev_high = price
        log.info("OPEN LEVERAGE %sx: margin %.2f notional %.2f amount %.6f @ %.4f "
                 "stop %.4f (risk ~%.2f) liq %.4f target %.4f",
                 lev, margin, notional, amount, price, stop,
                 margin * st["house_stop_frac"], liq, target)
        if self.live:
            log.warning("LIVE leverage requires a margin/futures account & symbol; "
                        "paper mode simulates it. Verify exchange wiring before use.")

    # ---- HOUSE: manage the leveraged leg ----
    def _manage_house(self, price):
        s, st = self.state, self.cfg["strategy"]
        s.lev_high = max(s.lev_high, price)
        if st.get("trailing_stop_pct", 0) > 0:
            trail = s.lev_high * (1 - st["trailing_stop_pct"])
            s.lev_stop = max(s.lev_stop, trail)

        reason = None
        exit_price = price
        if price <= s.lev_liq:
            reason, exit_price = "LIQUIDATION", s.lev_liq
        elif price <= s.lev_stop:
            reason = "leverage stop/trailing"
        elif s.lev_target and price >= s.lev_target:
            reason = "leverage target"

        if not reason:
            return

        pnl = (exit_price - s.lev_entry) * s.lev_amount  # margin PnL (leveraged)
        pnl = max(pnl, -s.lev_margin)                    # can't lose more than margin
        returned = s.lev_margin + pnl                    # collateral returned
        self._market_sell(s.lev_amount, exit_price, f"close leverage ({reason})")
        s.free_quote += max(returned, 0.0)
        s.realized_pnl += pnl
        s.trades += 1
        log.info("CLOSE LEVERAGE (%s): margin pnl %.2f, returned %.2f, total pnl %.2f",
                 reason, pnl, returned, s.realized_pnl)
        self._reset_leverage()
        s.phase = "IDLE"

    def _reset_spot(self):
        s = self.state
        s.spot_amount = s.spot_entry = s.spot_cost = s.spot_stop = 0.0

    def _reset_leverage(self):
        s = self.state
        s.lev_margin = s.lev_notional = s.lev_amount = s.lev_entry = 0.0
        s.lev_stop = s.lev_liq = s.lev_target = s.lev_high = 0.0

    # ---- run loop ----
    def run(self):
        signal_mod.signal(signal_mod.SIGINT, self._stop)
        signal_mod.signal(signal_mod.SIGTERM, self._stop)
        log.info("Starting loop (poll %ds). Ctrl-C to stop.", self.cfg["engine"]["poll_seconds"])
        while self._running:
            try:
                self.evaluate()
            except Exception as exc:
                log.exception("evaluate() failed: %s", exc)
            for _ in range(self.cfg["engine"]["poll_seconds"]):
                if not self._running:
                    break
                time.sleep(1)
        log.info("Stopped. Realized PnL %.2f over %d trades. Safe cash %.2f.",
                 self.state.realized_pnl, self.state.trades, self.state.safe_quote)

    def _stop(self, *_):
        self._running = False

    # ---- backtest the full state machine ----
    def backtest(self, limit=1500):
        df = self.fetch_ohlcv(limit=limit)
        mom = self.cfg["momentum"]
        st = self.cfg["strategy"]
        threshold = self.cfg["signals"]["trigger_threshold"]
        # Backtest gates on momentum only (TVL/social/news are point-in-time
        # live feeds with no historical replay here) blended with neutral 0.5s.
        w = self.cfg["signals"]["weights"]; tw = sum(w.values())

        free = self.cfg["risk"]["quote_balance"]; safe = 0.0
        phase = "IDLE"
        spot_amt = spot_cost = spot_stop = 0.0
        lev_amt = lev_entry = lev_stop = lev_liq = lev_target = lev_high = lev_margin = 0.0
        trades = wins = triggers = 0
        curve = []

        window = df.iloc[:]
        for i in range(mom["ema_slow"] + mom["roc_period"] + 2, len(df)):
            sub = window.iloc[: i + 1]
            price = float(sub["close"].iloc[-1])
            m = sig.momentum_score(sub, mom)
            conv = (m * w["momentum"] + 0.5 * (w["tvl"] + w["social"] + w["news"])) / tw

            if phase == "IDLE":
                if conv >= threshold:
                    triggers += 1
                    alloc = free * self.cfg["risk"]["entry_alloc_pct"]
                    spot_amt = alloc / price
                    spot_cost = alloc
                    spot_stop = price * (1 - st["spot_stop_pct"])
                    free -= alloc
                    phase = "SPOT"
            elif phase == "SPOT":
                value = spot_amt * price
                if price <= spot_stop:
                    free += value
                    trades += 1
                    phase = "IDLE"
                elif value >= spot_cost * st["double_target"]:
                    profit = value - spot_cost
                    safe += spot_cost
                    if profit > 0:
                        wins += 1
                    trades += 1
                    lev = st["leverage"]
                    lev_margin = profit
                    lev_amt = (profit * lev) / price
                    lev_entry = price
                    lev_liq = price * (1 - 1 / lev)
                    lev_stop = max(price * (1 - st["house_stop_frac"] / lev), lev_liq * 1.001)
                    lev_target = price * (1 + (st["leverage_target_mult"] - 1) / lev) \
                        if st.get("leverage_target_mult") else 0.0
                    lev_high = price
                    phase = "HOUSE" if profit > 0 else "IDLE"
            elif phase == "HOUSE":
                lev_high = max(lev_high, price)
                if st.get("trailing_stop_pct", 0) > 0:
                    lev_stop = max(lev_stop, lev_high * (1 - st["trailing_stop_pct"]))
                exit_p = None
                if price <= lev_liq:
                    exit_p = lev_liq
                elif price <= lev_stop:
                    exit_p = price
                elif lev_target and price >= lev_target:
                    exit_p = price
                if exit_p is not None:
                    pnl = max((exit_p - lev_entry) * lev_amt, -lev_margin)
                    free += max(lev_margin + pnl, 0.0)
                    if pnl > 0:
                        wins += 1
                    trades += 1
                    phase = "IDLE"

            eq = free + safe
            if phase == "SPOT":
                eq += spot_amt * price
            elif phase == "HOUSE":
                eq += lev_margin + (price - lev_entry) * lev_amt
            curve.append(eq)

        start = self.cfg["risk"]["quote_balance"]
        final = curve[-1] if curve else start
        c = pd.Series(curve)
        max_dd = ((c - c.cummax()) / c.cummax()).min() * 100 if len(c) else 0.0
        print("\n── Quant backtest (momentum-gated; TVL/social/news neutral) ──")
        print(f"Symbol        : {self.symbol} @ {self.timeframe}")
        print(f"Candles       : {len(df)}")
        print(f"Trigger thresh: {threshold}")
        print(f"Start equity  : {start:,.2f}")
        print(f"Final equity  : {final:,.2f}")
        print(f"Total return  : {(final/start-1)*100:+.2f}%")
        print(f"Triggers      : {triggers}   Trades: {trades}"
              + (f"   Win rate: {100*wins/trades:.1f}%" if trades else ""))
        print(f"Max drawdown  : {max_dd:.2f}%")
        print(f"Banked (safe) : {safe:,.2f}")
        print("──────────────────────────────────────────────────────────────\n")


# ── CLI ─────────────────────────────────────────────────────────────────────

def setup_logging(path):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(path)],
    )


def load_cfg(path):
    import os
    cfg = yaml.safe_load(Path(path).read_text())
    cfg["exchange"]["api_key"] = os.getenv("FMB_API_KEY", cfg["exchange"].get("api_key", ""))
    cfg["exchange"]["secret"] = os.getenv("FMB_SECRET", cfg["exchange"].get("secret", ""))
    return cfg


def main():
    ap = argparse.ArgumentParser(description="Frax staged-leverage quant bot")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--backtest", action="store_true")
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    setup_logging(cfg["engine"]["log_file"])
    bot = QuantBot(cfg)
    if cfg["engine"]["live"]:
        log.warning("LIVE MODE — real orders will be placed on %s.", cfg["exchange"]["id"])

    if args.backtest:
        bot.backtest()
    elif args.once:
        bot.evaluate()
    else:
        bot.run()


if __name__ == "__main__":
    main()
