# Candlestick Trading Strategy — Deployment Guide

A complete beginner's guide to understanding, installing, and running this automated trading bot.

---

## Table of Contents

1. [What Is This Project?](#1-what-is-this-project)
2. [How It Works (The Logic)](#2-how-it-works-the-logic)
3. [Project Structure](#3-project-structure)
4. [Prerequisites](#4-prerequisites)
5. [Step-by-Step Installation](#5-step-by-step-installation)
6. [Configuration](#6-configuration)
7. [Running the Backtest (Start Here!)](#7-running-the-backtest-start-here)
8. [Training the ML Model](#8-training-the-ml-model)
9. [Running Live Trading (Demo Account)](#9-running-live-trading-demo-account)
10. [Understanding the Output](#10-understanding-the-output)
11. [Troubleshooting](#11-troubleshooting)
12. [Important Warnings](#12-important-warnings)

---

## 1. What Is This Project?

This is an **automated trading bot** that connects to MetaTrader 5 (MT5) and trades financial markets on your behalf. It uses a combination of:

- **Candlestick pattern recognition** — The bot reads price charts and detects 12 well-known candle patterns (like Hammer, Engulfing, Morning Star, etc.)
- **Multi-timeframe analysis** — It checks 3 different timeframes (4-hour, 1-hour, 15-minute) to make sure signals align
- **Machine learning** — An XGBoost AI model scores each signal, filtering out low-probability trades
- **Automatic risk management** — It sizes positions, sets stop-losses, and never risks more than 1% per trade

**What it trades:** Forex (EURUSD, GBPUSD, USDJPY, AUDUSD), Indices (US500, US100, GER40), Commodities (XAUUSD gold, XTIUSD oil)

**Trading style:** Day trading — all positions are opened and closed within the same day. No overnight risk.

### Architecture (Simplified)

```
You start the bot (main.py)
        |
        v
Connects to MetaTrader 5
        |
        v
Every 60 seconds, for each symbol:
        |
        v
[1] Fetches price data on 3 timeframes (H4, H1, M15)
        |
        v
[2] Scans for candlestick patterns on the H1 chart
        |
        v
[3] Checks if the H4 trend and M15 entry confirm the signal
    (Confluence Score: 0-100, needs 60+ to trade)
        |
        v
[4] ML model predicts win probability (needs 65%+ to trade)
        |
        v
[5] Risk checks: position limits, drawdown, spread, R:R ratio
        |
        v
[6] If everything passes -> places the trade with SL and TP
        |
        v
[7] Manages open trades: trailing stops, breakeven moves
        |
        v
End of day: closes all remaining positions
```

---

## 2. How It Works (The Logic)

### Candlestick Patterns (12 Patterns Detected)

| Pattern | Direction | What It Looks Like |
|---------|-----------|-------------------|
| Bullish Engulfing | BUY | A small red candle followed by a big green candle that "swallows" it |
| Bearish Engulfing | SELL | A small green candle followed by a big red candle that swallows it |
| Hammer | BUY | Small body at top, very long lower wick — appears after a downtrend |
| Shooting Star | SELL | Small body at bottom, very long upper wick — appears after an uptrend |
| Morning Star | BUY | 3 candles: big red, tiny body, big green — bottom reversal signal |
| Evening Star | SELL | 3 candles: big green, tiny body, big red — top reversal signal |
| Bullish Harami | BUY | Big red candle, then small green candle inside it |
| Bearish Harami | SELL | Big green candle, then small red candle inside it |
| Doji | Either | Tiny body (open = close) — market is undecided |
| Three White Soldiers | BUY | 3 rising green candles in a row |
| Three Black Crows | SELL | 3 falling red candles in a row |
| Inverted Hammer | BUY | Long upper wick, tiny lower wick — after a downtrend |

### Multi-Timeframe Confluence Scoring

The bot doesn't trade on a single pattern alone. It checks 3 timeframes:

| Timeframe | Role | Max Points |
|-----------|------|-----------|
| H4 (4-hour) | Is the overall trend in our direction? | 30 points |
| H1 (1-hour) | Is there a valid pattern + indicator confirmation? | 40 points |
| M15 (15-min) | Is the precise entry timing right? | 30 points |

**Total: 0 to 100 points. Minimum 60 points needed to trade.**

### ML Model

An XGBoost machine learning model analyzes 18 features (pattern type, RSI, MACD, Bollinger Bands, trend direction, time of day, etc.) and predicts the probability that a trade will be profitable. Only trades with **65%+ predicted probability** are executed.

### Risk Management

| Rule | Setting |
|------|---------|
| Risk per trade | 1% of account balance |
| Hard cap per trade | 2% maximum |
| Daily loss limit | 3% — stops trading for the day |
| Max open positions | 5 at once |
| Max correlated positions | 2 in the same currency |
| Stop Loss | 1.5x ATR (adapts to volatility) |
| Take Profit | 2x the stop loss distance |
| Trailing stop | Moves SL to breakeven at 1:1, then trails |

---

## 3. Project Structure

```
Candle-Stick-Strategy/
|
|-- config/
|   |-- settings.yaml          <-- All settings in one file (edit this)
|
|-- src/                       <-- Core strategy code
|   |-- patterns.py            <-- Detects 12 candlestick patterns
|   |-- indicators.py          <-- RSI, MACD, Bollinger, ATR, etc.
|   |-- confluence.py          <-- Multi-timeframe scoring
|   |-- ml_scorer.py           <-- XGBoost ML model
|   |-- signals.py             <-- Combines everything into trade signals
|   |-- risk_manager.py        <-- Position sizing & risk controls
|   |-- executor.py            <-- Sends orders to MT5
|   |-- data_engine.py         <-- Fetches price data from MT5
|   |-- utils.py               <-- Logging & helper functions
|
|-- backtest/                  <-- Backtesting framework
|   |-- engine.py              <-- Simulates trading on historical data
|   |-- data_loader.py         <-- Loads data from CSV, MT5, or synthetic
|   |-- report.py              <-- Creates charts and performance reports
|
|-- models/
|   |-- xgb_model.joblib       <-- Pre-trained ML model (ready to use)
|
|-- backtest_results/          <-- Charts and trade logs from backtests
|-- logs/                      <-- Log files (created automatically)
|
|-- main.py                    <-- START HERE for live trading
|-- run_backtest.py            <-- START HERE to test the strategy
|-- train_model.py             <-- Retrain the ML model
|-- requirements.txt           <-- Python packages needed
```

---

## 4. Prerequisites

You need the following before starting:

### A. Windows PC
MetaTrader 5 only runs natively on Windows. If you're on Mac/Linux, you'll need a Windows VM or Wine.

### B. MetaTrader 5 Terminal
1. Go to your broker's website (e.g., IC Markets, Pepperstone, OANDA, etc.)
2. Sign up for a **Demo Account** (free, uses fake money)
3. Download and install their MetaTrader 5 platform
4. Log in with the credentials they give you

### C. Python 3.10 or newer
1. Go to https://www.python.org/downloads/
2. Download Python 3.10+ for Windows
3. **IMPORTANT:** During installation, check the box **"Add Python to PATH"**
4. Verify installation: open Command Prompt and type:
   ```
   python --version
   ```
   You should see something like `Python 3.10.x` or newer.

### D. TA-Lib C Library (Required for Technical Indicators)
This is the trickiest dependency. TA-Lib is a C library that Python wraps around.

**Windows installation:**
1. Go to https://github.com/cgohlke/talib-build/releases
2. Download the `.whl` file matching your Python version. For example:
   - Python 3.10, 64-bit: `TA_Lib-0.4.32-cp310-cp310-win_amd64.whl`
   - Python 3.11, 64-bit: `TA_Lib-0.4.32-cp311-cp311-win_amd64.whl`
   - Python 3.12, 64-bit: `TA_Lib-0.4.32-cp312-cp312-win_amd64.whl`
3. Open Command Prompt in the folder where you downloaded the file
4. Install it:
   ```
   pip install TA_Lib-0.4.32-cp310-cp310-win_amd64.whl
   ```
   (Replace the filename with the one you downloaded)

---

## 5. Step-by-Step Installation

Open **Command Prompt** (or PowerShell) and follow these steps:

### Step 1: Clone the Repository
```bash
git clone https://github.com/manuham/Candle-Stick-Strategy.git
cd Candle-Stick-Strategy
```

### Step 2: Create a Virtual Environment (Recommended)
This keeps the project's packages separate from your system Python.
```bash
python -m venv venv
```

Activate it:
```bash
# Windows Command Prompt:
venv\Scripts\activate

# Windows PowerShell:
.\venv\Scripts\Activate.ps1
```

You should see `(venv)` appear at the beginning of your command line.

### Step 3: Install TA-Lib (if not already done)
Follow the TA-Lib instructions in the Prerequisites section above.

### Step 4: Install Python Dependencies
```bash
pip install -r requirements.txt
```

This installs: MetaTrader5, pandas, numpy, scikit-learn, xgboost, PyYAML, matplotlib, and other packages.

### Step 5: Verify Installation
```bash
python -c "import MetaTrader5; import talib; import xgboost; print('All packages installed successfully!')"
```

If you see "All packages installed successfully!" you're good to go.

### Step 6: Set Up MetaTrader 5
1. Open MetaTrader 5
2. Log in to your **demo account**
3. Go to **Tools > Options > Expert Advisors**
4. Check **"Allow algorithmic trading"**
5. Click OK
6. Make sure the symbols you want to trade are visible in Market Watch:
   - Right-click in Market Watch panel > **Show All** (or add specific symbols)

---

## 6. Configuration

All settings are in one file: `config/settings.yaml`

### Key Settings You Might Want to Change

**Symbols** (what to trade):
```yaml
symbols:
  forex:
    - EURUSD      # Remove or add symbols here
    - GBPUSD
    - USDJPY
    - AUDUSD
  indices:
    - US500
    - US100
    - GER40
  commodities:
    - XAUUSD
    - XTIUSD
```
> Note: Symbol names must match EXACTLY what your broker uses in MT5. Some brokers add suffixes like "EURUSD.a" or "EURUSDm". Check your MT5 Market Watch panel.

**Risk settings** (how much to risk):
```yaml
risk:
  max_risk_per_trade: 0.01    # 1% per trade (increase to 0.02 for 2%)
  max_daily_drawdown: 0.03    # 3% max daily loss
  max_open_positions: 5       # Max simultaneous trades
  min_risk_reward: 2.0        # Only take trades with 2:1 reward-to-risk
```

**ML settings:**
```yaml
ml:
  enabled: true               # Set to false to trade without ML
  confidence_threshold: 0.65  # Lower = more trades, higher = fewer but better
```

---

## 7. Running the Backtest (Start Here!)

Backtesting lets you test the strategy on historical data **without risking any money**. Always start here.

### Quick Start (No MT5 Needed)

This uses computer-generated (synthetic) price data:

```bash
python run_backtest.py --synthetic --bars 5000
```

### With Real Data (MT5 Must Be Running)

```bash
python run_backtest.py --symbol EURUSD --bars 5000
```

### From a CSV File

If you have OHLCV data in a CSV file:
```bash
python run_backtest.py --csv path/to/your/data.csv
```

### Without ML Model

```bash
python run_backtest.py --synthetic --no-ml
```

### What You'll See

The backtest will print a summary like this:
```
============================================================
           BACKTEST RESULTS
============================================================
  Symbol:          EURUSD
  Period:          2023-01-01 to 2024-12-31
  Total Trades:    156
  Win Rate:        58.3%
  Profit Factor:   1.72
  Total Return:    24.5%
  Max Drawdown:    8.2%
  Sharpe Ratio:    1.45
============================================================
```

It also saves charts to the `backtest_results/` folder:
| File | What It Shows |
|------|--------------|
| `backtest_equity.png` | How your account balance grew over time |
| `backtest_drawdown.png` | The worst losing streaks |
| `backtest_monthly.png` | Returns broken down by month |
| `backtest_distribution.png` | Histogram of wins and losses |
| `backtest_trades.csv` | Every single trade with full details |

---

## 8. Training the ML Model

A pre-trained model is already included (`models/xgb_model.joblib`), but you can retrain it:

### Quick Train (No MT5 Needed)

```bash
python train_model.py --synthetic --bars 10000
```

### Train on Real Data (MT5 Must Be Running)

```bash
python train_model.py --bars 10000
```

This fetches H1 and H4 data for all configured symbols and trains the model.

### What It Produces

- A trained model saved to `models/xgb_model.joblib`
- A report showing:
  - **Accuracy**: How often the model is correct
  - **Precision**: Of the trades it says "take", how many actually win
  - **Recall**: Of all winning trades, how many it catches
  - **F1 Score**: Balance of precision and recall
  - **Top Features**: Which inputs matter most to the model

### When to Retrain

Retrain the model:
- Every 1-3 months to adapt to changing market conditions
- After adding/removing symbols from your configuration
- If backtest performance degrades significantly

---

## 9. Running Live Trading (Demo Account)

**IMPORTANT: Only use a demo account until you fully understand how this works.**

### Step 1: Make Sure MT5 Is Running and Logged In

Open MetaTrader 5 and verify:
- You're connected (green bar at bottom right)
- Algo trading is enabled (the "AutoTrading" button on the toolbar should be active)

### Step 2: Dry Run First (Simulates Without Trading)

```bash
python main.py --dry-run
```

This runs the full strategy but does NOT place real orders. You'll see log output like:
```
[DRY RUN] Would execute: BUY 0.05 EURUSD @ 1.08523
```

Let it run for a few hours to verify it's working correctly.

### Step 3: Live Trading on Demo Account

```bash
python main.py
```

The bot will:
1. Connect to MT5
2. Load the ML model
3. Start scanning all symbols every 60 seconds
4. Place trades automatically when conditions are met
5. Manage open positions with trailing stops
6. Close all positions at end of day

### How to Monitor

- **Terminal output**: Shows real-time signal detection and trade execution
- **Log file**: Full details saved to `logs/strategy.log`
- **MetaTrader 5**: Watch your positions appear in the Trade tab

### How to Stop

Press **Ctrl+C** in the terminal. The bot will:
1. Stop scanning for new signals
2. Close all open positions
3. Disconnect from MT5
4. Exit cleanly

---

## 10. Understanding the Output

### Log File (`logs/strategy.log`)

```
2024-01-15 14:30:01 INFO  Scanning EURUSD...
2024-01-15 14:30:01 INFO  Pattern detected: Bullish Engulfing (strength: +100)
2024-01-15 14:30:01 INFO  Confluence score: 72/100
2024-01-15 14:30:01 INFO  ML probability: 0.71
2024-01-15 14:30:01 INFO  Risk check passed. Lot size: 0.05
2024-01-15 14:30:02 INFO  Trade executed: BUY 0.05 EURUSD @ 1.08523 SL:1.08200 TP:1.09170
```

### Key Terms

| Term | Meaning |
|------|---------|
| **Confluence Score** | 0-100 rating of how well the signal is confirmed across timeframes |
| **ML Probability** | 0-100% confidence from the AI model that this trade will win |
| **SL (Stop Loss)** | Price where the trade auto-closes to limit losses |
| **TP (Take Profit)** | Price where the trade auto-closes to lock in profit |
| **R:R (Risk-Reward)** | Ratio of potential profit to potential loss (2:1 means you gain 2x what you risk) |
| **ATR** | Average True Range — measures how volatile the market is |
| **Trailing Stop** | A stop loss that moves in your favor as the trade profits |

---

## 11. Troubleshooting

### "Failed to connect to MT5"
- Make sure MetaTrader 5 is open and logged in
- Make sure "Allow algorithmic trading" is enabled in MT5 settings
- Try restarting MT5

### "ModuleNotFoundError: No module named 'talib'"
- TA-Lib C library is not installed. Follow the TA-Lib installation steps in Prerequisites.

### "ModuleNotFoundError: No module named 'MetaTrader5'"
- Run `pip install MetaTrader5`
- This package only works on Windows

### "Symbol EURUSD not found"
- Your broker may use different symbol names (e.g., "EURUSDm", "EURUSD.a")
- Open MT5 Market Watch, find the exact symbol name, and update `config/settings.yaml`

### "No trades generated during backtest"
- Try more bars: `--bars 10000`
- Try without ML: `--no-ml`
- This can happen with synthetic data that's too short or not volatile enough

### TA-Lib Installation Fails
- Make sure you download the `.whl` file matching your exact Python version
- Check your Python version: `python --version`
- Check if 64-bit: `python -c "import struct; print(struct.calcsize('P') * 8)"`

---

## 12. Important Warnings

1. **Start with a DEMO account.** Do not use real money until you have thoroughly tested and understand every aspect of this system.

2. **Past performance does not guarantee future results.** Good backtest results on historical data do not mean the strategy will be profitable in the future.

3. **This is educational software.** It is provided as-is for learning purposes. It is NOT financial advice.

4. **Markets can be unpredictable.** Even with risk management, losses are possible. Never trade with money you cannot afford to lose.

5. **Monitor the bot.** Even when running automatically, check on it regularly. Don't leave it running unattended for extended periods.

6. **Broker differences matter.** Spreads, commissions, symbol names, and execution speed vary by broker. What works in backtesting may differ in live trading.

---

## Quick Reference — Commands

| What | Command |
|------|---------|
| Run backtest (no MT5 needed) | `python run_backtest.py --synthetic --bars 5000` |
| Run backtest (real data) | `python run_backtest.py --symbol EURUSD --bars 5000` |
| Run backtest (no ML) | `python run_backtest.py --synthetic --no-ml` |
| Train ML model (no MT5) | `python train_model.py --synthetic --bars 10000` |
| Train ML model (real data) | `python train_model.py --bars 10000` |
| Dry run (no real trades) | `python main.py --dry-run` |
| Live trading | `python main.py` |
| Live trading (custom config) | `python main.py --config path/to/config.yaml` |
| Stop the bot | Press `Ctrl+C` |
