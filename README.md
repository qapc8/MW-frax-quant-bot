# MW Frax Quant Bot  (MW-QB)

A signal-gated, staged-leverage quant bot for the Frax ecosystem. Logs are
tagged `MW-QB` for clear identification.

> ⚠️ **Reality check.** FRAX is a **collateral-backed stablecoin pegged to $1**.
> A move to $3 or $5 would be a catastrophic *upward* depeg that mint/redeem
> arbitrage prevents — so those are **not** grounded price forecasts. This bot
> treats them as *your configurable thesis assumptions*, nothing more, and does
> **not** ship any fabricated "June 2025" analysis or news. The tradable,
> volatile Frax token is **FXS (Frax Share)** — the default symbol.

> 🛡️ **Paper by default** (`engine.live: false`): no real orders, no real
> leverage until you flip it and add keys. Leverage in paper mode is *simulated*
> with an explicit liquidation model.

## How it works

```
        ┌─────────── GATE (signals.py) ───────────┐
        │ composite = w·momentum + w·TVL           │
        │           + w·social + w·news            │
        │ if composite < trigger_threshold  → skip │
        │ if require_on_thesis & off-path  → skip  │
        └──────────────────┬───────────────────────┘
                           ▼
   IDLE ──trigger──▶ SPOT ──hits 2x──▶ HOUSE ──stop/target──▶ IDLE
                      │                  │
                  spot stop         bank principal,
                  → IDLE            lever only the profit
```

1. **GATE** — a weighted composite of four factors must clear
   `signals.trigger_threshold`, **or the algo does not trigger**. Optionally the
   price must also sit on your thesis trajectory (`require_on_thesis`).
   - **momentum** — EMA trend + RSI band + ROC, from OHLCV.
   - **TVL** — *real* DefiLlama API (`api.llama.fi/protocol/frax`), scored by
     trend over `lookback_days`.
   - **social** — pluggable (LunarCrush); neutral `0.5` unless `FMB_SOCIAL_KEY` set.
   - **news** — pluggable (CryptoPanic); neutral `0.5` unless `FMB_NEWS_KEY` set.

   Unconfigured factors return a neutral `0.5` — **never fabricated data.**

2. **SPOT** — deploy principal into a spot long with a protective stop
   (`spot_stop_pct`).

3. **Trigger → HOUSE** — once the leverage trigger is hit,
   **bank the principal to safe cash** and deploy only the *profit* ("house
   money") into a leveraged long with an adequate stop (`house_stop_frac` of the
   house money) kept above the computed liquidation price. Principal is
   preserved; only house money is ever leveraged. If the trigger is never
   reached, the leverage leg never fires.

   The trigger is either an **absolute price** (`strategy.leverage_trigger_price`,
   e.g. `4.4`) when set, otherwise a **multiple** (`strategy.double_target`, e.g.
   2x cost basis).

## Managing an existing position (seed)

Set `position.seed.enabled: true` to have the bot manage a position you already
hold instead of opening its own:

```yaml
position:
  seed:
    enabled: true
    avg_entry: 2.31     # your average fill
    amount: 124481      # FXS held
    limit_price: 2.41   # informational: original decision price
```

The bot starts in the SPOT phase holding it and watches for
`leverage_trigger_price`. **Forward-scenario stop arming:** the protective stop
stays dormant until price first reaches your `avg_entry`, so a position seeded
above the current market is *held* (not instantly stopped) while it waits for
the thesis to play out. Once price reaches entry, the stop goes live.

## Setup

```bash
cd ~/frax-momentum-bot
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

```bash
python quant_bot.py --backtest   # backtest the full state machine
python quant_bot.py --once       # one live evaluation (real TVL fetch) then exit
python quant_bot.py              # continuous loop (paper by default)
```

Optional live signal feeds:

```bash
export FMB_SOCIAL_KEY=...   # LunarCrush  (set signals.social.provider: lunarcrush)
export FMB_NEWS_KEY=...     # CryptoPanic (set signals.news.provider: cryptopanic)
```

## Going live

1. Backtest and paper-run until you trust it.
2. `export FMB_API_KEY=...  export FMB_SECRET=...`
3. Set `engine.live: true`. **Leverage requires a margin/futures account and the
   matching futures symbol** — wire and verify that before trusting the HOUSE
   leg live. Start small.

## Tuning (`config.yaml`)

- `signals.trigger_threshold`, `signals.weights` — sensitivity of the gate.
- `thesis.*` — your price targets and whether to enforce the trajectory.
- `strategy.double_target` / `leverage` / `house_stop_frac` / `trailing_stop_pct`.

## Disclaimer

Educational software. Crypto trading and especially leverage carry substantial
risk of total loss. No warranty; use at your own risk.
