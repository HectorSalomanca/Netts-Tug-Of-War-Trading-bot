# Tug-Of-War Sovereign Quant Trading System

> A fully automated, institutional-grade paper trading bot that exploits the conflict between smart money (Sovereign) and retail sentiment (Madman) — inspired by the casino house-edge model.

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)
[![Alpaca](https://img.shields.io/badge/broker-Alpaca%20Paper-yellow.svg)](https://alpaca.markets/)
[![Supabase](https://img.shields.io/badge/database-Supabase-green.svg)](https://supabase.com/)
[![Next.js](https://img.shields.io/badge/dashboard-Next.js-black.svg)](https://nextjs.org/)

---

## The Core Idea

Most trading bots follow indicators blindly. This system watches **two opposing forces** and only trades when they *disagree*. That disagreement is the edge — like a casino that doesn't need to win every hand, just win consistently more than it loses.

- **Sovereign Bot** — thinks like a hedge fund (AFD momentum, ORB, SEC 8-K filings, Kelly sizing)
- **Madman Bot** — thinks like Reddit/retail (RSI pumps, OFI contrarian fades, volume spikes)
- **Referee Engine** — only fires when Sovereign has conviction AND Madman is wrong or neutral
- **Position Manager** — auto-exits on stop loss, take profit, signal flip, crowded exit, or EOD

---

## System Architecture

```
Tug-Of-War System/
│
├── scout/                      # Layer A: Data & Intelligence
│   ├── scout_news.py           # SEC 8-K (edgartools) + Yahoo/StockTwits + Bayesian bot-farm filter
│   ├── scout_tape.py           # Alpaca WebSocket → real-time OFI Z-score + Iceberg detection
│   └── crawl_news.py           # V1 legacy scout (kept for reference)
│
├── quant/                      # Layer B: Quant Engine
│   ├── feature_factory.py      # Adaptive Fractional Differencing + SPY/QQQ neutralization
│   ├── regime_hmm.py           # 3-state Gaussian HMM: Chop / Trend / Crisis
│   ├── earnings_filter.py      # Auto-skip symbols with earnings within 48 hours
│   ├── correlation_guard.py    # Prevent sector over-concentration (max 2 per group)
│   └── backtester.py           # 1-year historical replay with full signal logic
│
├── bots/                       # Layer C: The Agents
│   ├── sovereign_agent.py      # V2: AFD + ORB + 8-K sentiment + Kelly Criterion
│   └── madman_agent.py         # V2: RSI + OFI Contrarian Fade + volume spike
│
├── referee/                    # Layer D: Execution
│   ├── engine_v2.py            # Main engine: HMM gating + Limit IOC + shortfall logging
│   └── position_manager.py     # Auto-exit: stop loss, take profit, signal flip, EOD, crisis
│
├── dashboard/                  # Next.js live dashboard (localhost:3000)
│   ├── app/page.tsx            # Regime badge, tug meters, trade history, shortfall stats
│   ├── app/components/
│   │   ├── TugMeter.tsx        # Per-symbol tension meter with OFI + iceberg display
│   │   └── TradeRow.tsx        # Trade row with implementation shortfall column
│   └── lib/supabase.ts         # Supabase client + TypeScript types
│
├── db/                         # Database schema
│   └── schema.sql              # Supabase table definitions + RLS policies
│
├── com.tugofwar.trader.plist   # macOS launchd: engine runs 24/7
├── com.tugofwar.tape.plist     # macOS launchd: OFI tape streams 24/7
├── requirements.txt            # Python dependencies
└── .env.example                # Environment variable template (copy to .env)
```

---

## How It Decides To Trade

### The Conflict Filter (House-Edge Model)

| Scenario | Verdict | Size |
|---|---|---|
| Sovereign ≥75% + Madman actively opposes | **STRONG EXECUTE** | Full Kelly position |
| Sovereign ≥80% + Madman neutral | **WEAK EXECUTE** | 60% of Kelly position |
| Both agree same direction | CROWDED SKIP | No trade — consensus = no edge |
| Both neutral / Sovereign weak | NO SIGNAL | No trade |

### Regime Gating (3-State HMM)

The Hidden Markov Model runs on SPY daily returns and updates every 60 minutes:

| Regime | Behavior |
|---|---|
| **TREND** | Full day-trade mode — stop 2%, take profit 4%, full position size |
| **CHOP** | Scalp mode — stop 1%, take profit 2%, ORB entries only |
| **CRISIS** | All trading halted, all positions force-closed immediately |

### Auto-Exit Rules (Position Manager)

Every 15-minute cycle checks every open position in priority order:

1. **Stop Loss** — down ≥2% (1% in Chop) → close immediately
2. **Take Profit** — up ≥4% (2% in Chop) → lock gains
3. **Signal Flip** — Sovereign reverses direction after 3+ bars → thesis broken, exit
4. **Crowded Exit** — Madman joins our direction → retail piled in, edge gone
5. **EOD Close** — 3:45 PM ET → close everything, no overnight risk
6. **Crisis Halt** — HMM detects Crisis → force-close all positions

---

## Signal Stack

### Sovereign Agent (Institutional Logic)

| Signal | Description |
|---|---|
| **Adaptive Fractional Differencing (AFD)** | Finds minimum `d` to make price series stationary while preserving trend memory |
| **Pure Alpha** | Strips SPY + QQQ beta via OLS regression — isolates ticker-specific momentum |
| **Opening Range Breakout (ORB)** | First 30-min high/low sets the range; breakout = directional signal |
| **SEC 8-K Sentiment** | edgartools scrapes latest filings, scored bullish/bearish |
| **Kelly Criterion** | `f = (b·p − q) / b × 0.5` — half-Kelly, capped at 25% of equity |

### Madman Agent (Retail/Contrarian Logic)

| Signal | Description |
|---|---|
| **RSI** | Overbought (>70) = retail FOMO, oversold (<30) = retail fear |
| **OFI Contrarian Fade** | RSI >85 AND OFI Z-score <−1.5 = retail exuberant but smart money absorbing → SELL |
| **15-min Pump Detection** | Price up >5% in 15 min = retail chasing |
| **Volume Spike** | Today's volume >2× 20-day average |
| **Institutional Iceberg** | Price flat + negative OFI = smart money absorbing retail buy pressure |

### Scout Tape (Real-Time Microstructure)

- Streams Alpaca WebSocket L1 quotes for all 10 tickers
- Calculates **Order Flow Imbalance**: `OFI = (bid_size − ask_size) / (bid_size + ask_size)`
- Z-scores OFI over a 60-tick rolling window
- Detects icebergs: price within 0.1% of 1-min-ago AND OFI Z < −1.5
- Writes to Supabase `ofi_snapshots` every 60 seconds

### Scout News (Fundamental + Adversarial Filter)

- Fetches SEC EDGAR 8-K filings via `edgartools`
- Scrapes Yahoo Finance headlines + StockTwits messages
- **Bayesian Adversarial Filter**: if ≥10 social posts share the same phrase but 8-K is silent → discarded as coordinated bot farm
- Bayesian scores: P(real | 8-K active) = 0.90, P(real | bot farm) = 0.15

---

## Risk Controls

| Control | Rule |
|---|---|
| Max open positions | 4 at any time |
| Base risk per trade | 2% of equity (floor) |
| Max risk per trade | 5% of equity (Kelly-scaled ceiling) |
| Stop loss | 2% (Trend) / 1% (Chop) |
| Take profit | 4% (Trend) / 2% (Chop) |
| Earnings filter | Skip any symbol with earnings within 48 hours |
| Correlation guard | Max 2 positions from same sector group |
| Short selling | Disabled (paper account) |
| EOD close | All positions closed by 3:45 PM ET |
| Crisis halt | HMM Crisis state → force-close everything |
| Order type | Limit IOC (mid ± 0.05%) — tracks implementation shortfall in bps |

### Sector Groups (Correlation Guard)

```
tech_growth:   NVDA, CRWD, PLTR
crypto_proxy:  MSTR, PLTR
healthcare:    LLY
financials:    JPM
industrials:   CAT
utilities:     NEE
international: TSMC, SONY
```

---

## Watchlist

**Trading targets (10):** `NVDA · CRWD · LLY · TSMC · JPM · NEE · CAT · SONY · PLTR · MSTR`

**Silent benchmarks (not traded):** `SPY · QQQ` — used for regime detection and feature neutralization only

---

## Backtester Results (1 Year, Buy-Only Baseline)

```bash
python3 quant/backtester.py --days 365
```

| Metric | Value |
|---|---|
| Total trades | 172 |
| Win rate | 44.2% |
| Total return | +0.20% |
| Sharpe ratio | 0.76 |
| Max drawdown | 0.8% |
| Profit factor | 1.03× |

> **Note:** Backtester uses daily bars only — no intraday ORB, no live news/OFI signals. The live system has 3 additional signal layers on top of this baseline.

---

## Database Schema (Supabase)

| Table | Purpose |
|---|---|
| `signals` | Every signal from both bots — regime state, OFI Z-score, AFD momentum, Bayesian score |
| `tug_results` | Every referee decision — verdict, tug score, conflict flag |
| `trades` | Every order — side, qty, limit price, shortfall in bps, status, P&L |
| `regime_states` | HMM state history — Chop/Trend/Crisis with confidence + SPY vol/momentum |
| `ofi_snapshots` | Per-symbol OFI Z-score + iceberg flag, written every 60 seconds |

---

## Setup

### 1. Clone & Install

```bash
git clone https://github.com/HectorSalomanca/Nettss-Tug-Of-War-Trading-bot.git
cd Nettss-Tug-Of-War-Trading-bot
pip3 install -r requirements.txt
```

### 2. Environment Variables

```bash
cp .env.example .env
# Fill in your keys
```

```env
ALPACA_API_KEY=your_alpaca_api_key
ALPACA_SECRET_KEY=your_alpaca_secret_key
ALPACA_PAPER_TRADE=True
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_ANON_KEY=your_anon_key
SUPABASE_SERVICE_KEY=your_service_key
USER_ID=your-supabase-user-uuid
```

Get Alpaca keys at [alpaca.markets](https://alpaca.markets/) (free paper trading).
Get Supabase keys at [supabase.com](https://supabase.com/) (free tier).

### 3. Initialize Database

Run `db/schema.sql` in your Supabase SQL editor.

### 4. Run

```bash
# Single test cycle
python3 referee/engine_v2.py --once

# 24/7 continuous (every 15 min)
python3 referee/engine_v2.py --interval 15

# OFI tape streamer (separate terminal)
python3 scout/scout_tape.py

# Backtester
python3 quant/backtester.py --symbols NVDA PLTR JPM --days 365
```

### 5. Dashboard

```bash
cd dashboard
npm install
npm run dev
# Open http://localhost:3000
```

### 6. 24/7 macOS Background Service

```bash
cp com.tugofwar.trader.plist ~/Library/LaunchAgents/
cp com.tugofwar.tape.plist ~/Library/LaunchAgents/
launchctl load ~/Library/LaunchAgents/com.tugofwar.trader.plist
launchctl load ~/Library/LaunchAgents/com.tugofwar.tape.plist

# Monitor logs
tail -f logs/engine.log
tail -f logs/tape.log
```

---

## 24/7 Schedule

| Time (ET) | Action |
|---|---|
| Mon–Fri 4:00 AM – 8:00 PM | Engine cycles every 15 min |
| Every 30 min | Scout news refresh (background) |
| Every 60 min | HMM regime re-fit (background) |
| Every 60 sec | OFI tape flush to Supabase |
| 3:45 PM ET daily | EOD close — all positions exited |
| Weekends / nights | Engine idles, no trades |

---

## Dashboard (localhost:3000)

| Section | What It Shows |
|---|---|
| **Header badge** | Current HMM regime (Trend/Chop/Crisis) with confidence % |
| **Stats row** | Total P&L · Win Rate · Executed · Crowded Skips · Avg Shortfall (bps) |
| **Regime bar** | SPY vol, SPY momentum, trading mode description |
| **Tug meters** | Per-symbol: Sovereign vs Madman direction + confidence, OFI Z-score, iceberg alert |
| **Trade history** | Symbol · side · qty · order type · status · fill price · shortfall · P&L · time |

---

## Python Dependencies

```
alpaca-py>=0.20.0       # Alpaca trading + data API
supabase>=2.0.0         # Database client
python-dotenv>=1.0.0    # Environment variables
pandas>=2.0.0           # Data manipulation
numpy>=1.26.0           # Numerical computing
hmmlearn>=0.3.0         # Gaussian HMM for regime detection
statsmodels>=0.13.0     # ADF stationarity test (for AFD)
fracdiff>=0.9.0         # Fractional differencing
scikit-learn>=1.3.0     # Feature scaling for HMM
edgartools>=2.0.0       # SEC EDGAR 8-K filing access
httpx>=0.27.0           # Async HTTP for news scraping
beautifulsoup4>=4.12.0  # HTML parsing
schedule>=1.2.0         # Cycle scheduling
pytz>=2024.1            # Market hours timezone handling
```

---

## What Makes This Different

| Feature | Basic Bot | This System |
|---|---|---|
| Signal source | Single indicator | 5-layer stack (AFD, ORB, OFI, 8-K, HMM) |
| Position sizing | Fixed dollar | Kelly Criterion (mathematically optimal) |
| Market regime | Ignored | 3-state HMM — strategy adapts automatically |
| Entry filter | Any signal fires | Conflict required — Sovereign ≠ Madman |
| Fake news filter | None | Bayesian adversarial filter (bot-farm detection) |
| Microstructure | None | Real-time OFI + Institutional Iceberg detection |
| Risk management | Stop loss only | 6 exit rules including crisis halt + crowded exit |
| Sector risk | None | Correlation guard — max 2 per sector group |
| Earnings risk | None | 48-hour earnings blackout per symbol |
| Execution | Market orders | Limit IOC — tracks implementation shortfall in bps |

---

## Disclaimer

This system trades on a **paper (simulated) account only**. Built for educational and research purposes. Past backtest performance does not guarantee future results. Do not use with real money without extensive additional testing.

---

## License

MIT License
