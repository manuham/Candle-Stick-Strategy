# Three Production-Grade Trading Strategy Prompts

These prompts are designed to be copied into a fresh Claude Code chat session.
Each one produces a complete, deployable MetaTrader 5 trading system — different
strategy, different edge, different ML approach.

---
---

## PROMPT 1: Statistical Mean Reversion + Hidden Markov Model Regime Detection

---

```
You are a senior quantitative developer at a systematic hedge fund. I need you to
build a complete, production-grade automated trading system for MetaTrader 5 that
trades a STATISTICAL MEAN REVERSION strategy with HIDDEN MARKOV MODEL (HMM) regime
detection. This is fundamentally different from trend-following or pattern
recognition — we are exploiting the statistical tendency of prices to revert to
their mean, but ONLY when market conditions (regimes) favour mean reversion.

I am a beginner. Build everything from scratch. Every file. Every line. Working
code. No placeholders. No "TODO" comments. No pseudo-code. Real, executable,
production-ready Python.

=============================================================================
STRATEGY PHILOSOPHY
=============================================================================

Most retail traders lose money because they use the same strategy in all market
conditions. Markets alternate between TRENDING regimes (where momentum works and
mean reversion gets destroyed) and RANGING regimes (where mean reversion prints
money and momentum gets chopped to pieces). The entire edge of this system comes
from:

1. DETECTING the current market regime using a Hidden Markov Model
2. ONLY trading mean reversion when the HMM says we're in a ranging/mean-reverting
   regime
3. SITTING OUT (or hedging) when the HMM detects a trending regime
4. Using statistical Z-scores and Bollinger Band deviation to identify stretched
   prices that are likely to snap back

This is how real quant funds operate. Regime awareness is the #1 thing that
separates profitable systematic strategies from unprofitable ones.

=============================================================================
COMPLETE PROJECT STRUCTURE (create every file)
=============================================================================

Mean-Reversion-Strategy/
├── config/
│   └── settings.yaml              # All configurable parameters
├── src/
│   ├── __init__.py
│   ├── data_engine.py             # MT5 data fetching + multi-TF management
│   ├── regime_detector.py         # Hidden Markov Model regime classification
│   ├── mean_reversion.py          # Z-score, half-life, Hurst exponent calculations
│   ├── indicators.py              # Bollinger Bands, Keltner Channels, RSI, ATR, VWAP
│   ├── signals.py                 # Signal generation (only in mean-reverting regimes)
│   ├── pairs_engine.py            # Cointegration testing + spread trading for pairs
│   ├── risk_manager.py            # Kelly Criterion, position sizing, drawdown control
│   ├── executor.py                # MT5 order execution + position management
│   └── utils.py                   # Logging, config loading, helpers
├── backtest/
│   ├── __init__.py
│   ├── engine.py                  # Walk-forward backtesting engine
│   ├── data_loader.py             # Historical data loading (CSV + MT5 + synthetic)
│   ├── monte_carlo.py             # Monte Carlo simulation for robustness testing
│   └── report.py                  # Performance metrics + visualisation
├── models/                        # Saved HMM models (created at runtime)
├── logs/                          # Trading logs
├── main.py                        # Live trading entry point
├── train_model.py                 # HMM + regime model training
├── run_backtest.py                # Backtesting entry point
├── requirements.txt               # Dependencies
└── .gitignore

=============================================================================
DETAILED COMPONENT SPECIFICATIONS
=============================================================================

--- 1. REGIME DETECTOR (regime_detector.py) ---

This is the CORE of the system. Use the hmmlearn library to implement a Gaussian
Hidden Markov Model with 3 hidden states:

  State 0: LOW VOLATILITY MEAN-REVERTING (our bread and butter — trade aggressively)
  State 1: HIGH VOLATILITY MEAN-REVERTING (trade conservatively, tighter stops)
  State 2: TRENDING (DO NOT trade mean reversion — sit on hands or trade momentum)

Observable features for the HMM (calculate from price data):
  - Realised volatility (20-period rolling std of log returns)
  - Absolute returns (magnitude of moves)
  - Volume ratio (current vs 20-period SMA)
  - Hurst exponent (rolling 100-bar, using rescaled range method)
    H < 0.5 = mean reverting, H = 0.5 = random walk, H > 0.5 = trending
  - Autocorrelation of returns (lag-1, rolling 50-bar)
    Negative autocorrelation = mean reverting, Positive = trending

The HMM should:
  - Be trained on 2+ years of historical data
  - Update daily with new observations (online learning / periodic refit)
  - Output: current_regime (0, 1, or 2) + regime_probability (confidence)
  - Only allow trading when regime 0 or 1 with probability > 0.70

Training process:
  1. Calculate all observable features for historical data
  2. Fit GaussianHMM with n_components=3, covariance_type="full"
  3. Use BIC/AIC to validate 3 states is optimal (test 2-5)
  4. Label states by their characteristics (lowest vol = state 0, etc.)
  5. Save model with joblib
  6. Provide regime_confidence = posterior probability of current state

--- 2. MEAN REVERSION ENGINE (mean_reversion.py) ---

Statistical tools for identifying mean-reversion opportunities:

Z-Score Calculator:
  - z_score = (price - rolling_mean) / rolling_std
  - Use multiple lookback windows: 20, 50, 100 bars
  - Entry when |z_score| > 2.0 (price is 2 standard deviations from mean)
  - Exit when |z_score| < 0.5 (price has reverted toward mean)
  - Signal strength proportional to z_score magnitude

Half-Life of Mean Reversion:
  - Use Ornstein-Uhlenbeck process to estimate how fast prices revert
  - Run linear regression: delta_price = alpha + beta * price_lag1
  - half_life = -log(2) / beta
  - Only trade instruments where half_life is between 5 and 100 bars
  - Shorter half-life = faster reversion = better opportunity

Hurst Exponent:
  - Rescaled range (R/S) method on rolling 100-bar windows
  - H < 0.4: Strong mean reversion (increase position size)
  - H 0.4-0.5: Mild mean reversion (normal position size)
  - H > 0.5: Trending (DO NOT TRADE — this confirms the HMM regime)

Cointegration Engine (for pairs trading):
  - Test all symbol pairs using Engle-Granger two-step method
  - Calculate hedge ratio using OLS regression
  - Build spread = price_A - hedge_ratio * price_B
  - Trade the spread when z_score > 2.0 (buy underperformer, sell outperformer)
  - Rebalance hedge ratio weekly
  - Track cointegration stability (reject if p-value > 0.05)

--- 3. SIGNAL GENERATION (signals.py) ---

Signal pipeline (runs on each new H1 bar):

  Step 1: Check regime — if TRENDING regime, emit NO signals. Period.
  Step 2: Calculate z-scores for all symbols on multiple lookbacks (20, 50, 100)
  Step 3: Check Hurst exponent confirms mean reversion (H < 0.5)
  Step 4: Check half-life is in acceptable range (5-100 bars)
  Step 5: If z_score > 2.0 and regime is MEAN_REVERTING:
          → SELL signal (price above mean, expect reversion down)
  Step 6: If z_score < -2.0 and regime is MEAN_REVERTING:
          → BUY signal (price below mean, expect reversion up)
  Step 7: Confirm with RSI (oversold for BUY, overbought for SELL)
  Step 8: Confirm with Bollinger Band position (outside bands)
  Step 9: Score signal: regime_confidence * z_score_magnitude * hurst_score
  Step 10: If total score > threshold → emit signal

Signal object should include:
  - symbol, direction, z_score, hurst_exponent, half_life
  - regime_state, regime_confidence
  - entry_price, mean_target (the price we expect reversion TO)
  - sl_price (beyond the extreme — 3.0 * std from mean)
  - tp_price (the mean itself, or 0.5 * std from mean for conservative)

--- 4. RISK MANAGEMENT (risk_manager.py) ---

  - Max risk per trade: 1% of account
  - Position sizing: Kelly Criterion with Half Kelly (same as reference project)
  - Max daily drawdown: 3% — stop trading for the day
  - Max open positions: 8 (mean reversion has higher win rate, can handle more)
  - Max correlated positions: 2 per currency
  - Dynamic position sizing based on regime confidence:
    * Regime confidence > 0.90: full position (1% risk)
    * Regime confidence 0.80-0.90: 75% position
    * Regime confidence 0.70-0.80: 50% position
    * Regime confidence < 0.70: NO TRADE
  - Stop loss: 3.0 * ATR from entry (wider than trend strategies — need room)
  - Take profit: the rolling mean (dynamic target that moves)
  - Trailing stop: move SL to breakeven at 1:1 R:R, then trail at 1.5*ATR

--- 5. PAIRS TRADING (pairs_engine.py) ---

  This is a UNIQUE feature not found in most retail systems:
  - Test cointegration between all configured symbol pairs
  - Pairs that pass (p-value < 0.05): EURUSD/GBPUSD, AUDUSD/NZDUSD, US500/US100
  - Calculate the spread and its z-score
  - When spread z-score > 2.0: short the outperformer, long the underperformer
  - When spread z-score < -2.0: reverse
  - Exit when spread z-score crosses 0 (mean reversion complete)
  - This is MARKET NEUTRAL — profits regardless of market direction

--- 6. BACKTESTING ENGINE (backtest/engine.py) ---

CRITICAL IMPROVEMENT over basic backtesting — implement WALK-FORWARD OPTIMISATION:

  1. Divide historical data into windows (e.g., 6 months train, 2 months test)
  2. Train HMM on the training window
  3. Test on the out-of-sample window
  4. Slide forward by the test window size
  5. Repeat until all data is consumed
  6. Aggregate out-of-sample results = realistic performance estimate

This prevents overfitting and gives a realistic estimate of live performance.

Also implement MONTE CARLO SIMULATION (backtest/monte_carlo.py):
  1. Take the actual trade results from the backtest
  2. Randomly resample (with replacement) 10,000 times
  3. Build confidence intervals for: final equity, max drawdown, Sharpe ratio
  4. Report: "95% confidence that max drawdown will be less than X%"
  5. Report: "95% confidence that annual return will be between X% and Y%"
  This tells us how robust the strategy is to different orderings of trades.

--- 7. INDICATORS (indicators.py) ---

Use TA-Lib for all calculations:
  - Bollinger Bands (20, 2.0) — primary mean reversion bands
  - Keltner Channels (20, 1.5*ATR) — secondary confirmation
  - RSI (14) — overbought/oversold confirmation
  - ATR (14) — volatility for stops and position sizing
  - VWAP (session-based) — institutional mean price level
  - EMA (50, 200) — only used to FILTER OUT trending conditions
  - ADX (14) — if ADX > 25, market is trending → reduce exposure
  - Standard Deviation (20, 50, 100) — for z-score calculations
  - Volume SMA (20) — volume confirmation

--- 8. SYMBOLS TO TRADE ---

Forex (best for mean reversion — range-bound 70% of the time):
  EURUSD, GBPUSD, USDJPY, AUDUSD, NZDUSD, USDCAD, EURGBP, AUDNZD

Indices (for pairs trading — correlated but not identical):
  US500, US100, GER40

Commodities:
  XAUUSD, XAGUSD

--- 9. CONFIGURATION (settings.yaml) ---

Every single parameter must be configurable via YAML:
  - All z-score thresholds (entry: 2.0, exit: 0.5, extreme: 3.0)
  - All lookback periods (20, 50, 100)
  - HMM parameters (n_states: 3, min_confidence: 0.70)
  - Hurst exponent thresholds
  - Half-life acceptable range
  - Cointegration p-value threshold
  - All risk parameters
  - All indicator periods
  - Walk-forward window sizes
  - Monte Carlo iterations

--- 10. ENTRY POINTS ---

main.py:
  python main.py                    # Live trading
  python main.py --dry-run          # Simulation mode
  python main.py --config path      # Custom config

train_model.py:
  python train_model.py --synthetic --bars 10000   # Train HMM on synthetic data
  python train_model.py --bars 20000               # Train on MT5 historical data

run_backtest.py:
  python run_backtest.py --synthetic --bars 5000   # Quick test with synthetic data
  python run_backtest.py --symbol EURUSD --bars 10000  # Real data backtest
  python run_backtest.py --walk-forward            # Walk-forward optimisation
  python run_backtest.py --monte-carlo --iterations 10000  # Monte Carlo analysis

=============================================================================
REQUIREMENTS.TXT
=============================================================================

MetaTrader5>=5.0.45
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
hmmlearn>=0.3.0
statsmodels>=0.14.0
xgboost>=2.0.0
PyYAML>=6.0
pytz>=2023.3
joblib>=1.3.0
matplotlib>=3.7.0
seaborn>=0.12.0
TA-Lib>=0.4.28
scipy>=1.11.0

=============================================================================
QUALITY REQUIREMENTS
=============================================================================

- Every file must have proper docstrings and type hints
- All numerical operations must be vectorised with NumPy (no Python loops over bars)
- Logging with rotating file handler (10MB, 5 backups)
- Graceful shutdown on Ctrl+C (close all positions first)
- Synthetic data generator for testing without MT5 connection
- The backtest must generate: equity curve, drawdown chart, monthly returns
  heatmap, regime overlay chart, and trade distribution histogram
- Walk-forward results should show in-sample vs out-of-sample performance
- Monte Carlo should produce confidence interval charts

Build the ENTIRE system now. Start with requirements.txt, then config, then each
src module in dependency order, then backtest modules, then entry points. Do not
stop until every file is complete.
```

---
---

## PROMPT 2: Multi-Timeframe Momentum + LSTM Deep Learning Prediction

---

```
You are a senior quantitative developer at a systematic hedge fund. I need you to
build a complete, production-grade automated trading system for MetaTrader 5 that
trades a MOMENTUM / TREND-FOLLOWING strategy enhanced with LSTM (Long Short-Term
Memory) deep learning price prediction. This is fundamentally different from mean
reversion or candlestick patterns — we are exploiting the empirical fact that
assets in motion tend to stay in motion, and using deep learning to predict WHERE
that motion is going.

I am a beginner. Build everything from scratch. Every file. Every line. Working
code. No placeholders. No "TODO" comments. No pseudo-code. Real, executable,
production-ready Python.

=============================================================================
STRATEGY PHILOSOPHY
=============================================================================

Momentum is the most persistent anomaly in financial markets. Academic research
(Jegadeesh & Titman 1993, Asness et al. 2013) proves that assets that have been
going up tend to continue going up, and assets going down tend to continue going
down. This "momentum premium" exists across ALL asset classes and ALL time periods.

However, raw momentum has two fatal flaws:
1. MOMENTUM CRASHES: Sudden, violent reversals that wipe out years of profits
2. LATE ENTRIES: By the time momentum is confirmed, much of the move is over

Our edge comes from:
1. Using an LSTM neural network to PREDICT momentum 5-20 bars into the future,
   entering BEFORE the crowd confirms the trend
2. Using VOLATILITY-ADJUSTED momentum (dividing returns by volatility) to get
   cleaner signals that don't just chase noise
3. Using MULTI-TIMEFRAME momentum alignment (Daily/H4/H1) — only trade when
   all timeframes agree on direction
4. Using BREAKOUT DETECTION with volume confirmation to catch the start of new
   momentum waves
5. Having a CRASH DETECTOR that reduces exposure when momentum reversal risk
   is high (based on momentum dispersion and correlation spikes)

=============================================================================
COMPLETE PROJECT STRUCTURE (create every file)
=============================================================================

Momentum-LSTM-Strategy/
├── config/
│   └── settings.yaml              # All configurable parameters
├── src/
│   ├── __init__.py
│   ├── data_engine.py             # MT5 data fetching + multi-TF management
│   ├── momentum_engine.py         # Momentum calculations (absolute, relative, vol-adjusted)
│   ├── breakout_detector.py       # Support/resistance breakout identification
│   ├── lstm_predictor.py          # LSTM model for price movement prediction
│   ├── crash_detector.py          # Momentum crash early warning system
│   ├── indicators.py              # ADX, ATR, OBV, CMF, Donchian, SuperTrend, MACD
│   ├── signals.py                 # Signal generation combining all components
│   ├── risk_manager.py            # Volatility targeting, position sizing, drawdown control
│   ├── executor.py                # MT5 order execution + position management
│   └── utils.py                   # Logging, config, helpers
├── backtest/
│   ├── __init__.py
│   ├── engine.py                  # Walk-forward backtesting engine
│   ├── data_loader.py             # Historical data loading (CSV + MT5 + synthetic)
│   ├── monte_carlo.py             # Monte Carlo robustness testing
│   └── report.py                  # Performance metrics + visualisation
├── models/                        # Saved LSTM models
├── logs/
├── main.py                        # Live trading entry point
├── train_model.py                 # LSTM training pipeline
├── run_backtest.py                # Backtesting entry point
├── requirements.txt
└── .gitignore

=============================================================================
DETAILED COMPONENT SPECIFICATIONS
=============================================================================

--- 1. MOMENTUM ENGINE (momentum_engine.py) ---

Calculate multiple types of momentum for each symbol:

Absolute Momentum (Time-Series Momentum):
  - mom_N = (close / close_N_bars_ago) - 1
  - Calculate for N = 10, 20, 50, 100, 200 bars
  - If mom > 0: bullish momentum. If mom < 0: bearish momentum.
  - Combine multiple periods: weighted average (shorter periods get more weight)

Relative Momentum (Cross-Sectional):
  - Rank all symbols by their absolute momentum
  - Top quartile = "strong momentum" → go long
  - Bottom quartile = "weak momentum" → go short
  - This creates a LONG-SHORT portfolio that is partially market neutral

Volatility-Adjusted Momentum (the real edge):
  - vol_adj_mom = mom_N / rolling_std(returns, N)
  - This normalises momentum by volatility
  - A 5% move in a low-vol asset is MORE significant than a 5% move in a high-vol asset
  - This is how institutional momentum strategies work (risk-parity momentum)

Rate of Change Acceleration:
  - Not just "is the trend up?" but "is the trend ACCELERATING?"
  - roc = (mom_10 - mom_10_shifted_5) / 5
  - Positive acceleration = momentum strengthening → increase exposure
  - Negative acceleration = momentum fading → tighten stops / reduce size

--- 2. BREAKOUT DETECTOR (breakout_detector.py) ---

Identify key structural breakouts that initiate new momentum waves:

Donchian Channel Breakouts:
  - Upper channel = highest high of last N bars (default 20)
  - Lower channel = lowest low of last N bars
  - Breakout BUY: close > upper channel (new N-bar high)
  - Breakout SELL: close < lower channel (new N-bar low)
  - Use multiple periods: 20, 50, 100 (alignment = stronger breakout)

Support/Resistance Detection:
  - Identify horizontal S/R levels using pivot points and price clustering
  - A breakout THROUGH a level that has been tested 3+ times is powerful
  - Track the number of touches at each level
  - Volume must increase on the breakout bar (volume_ratio > 1.5)

Failed Breakout Detection (crucial for avoiding fakeouts):
  - If price breaks above resistance but closes back below within 3 bars → FAILED
  - Failed breakouts are STRONG reversal signals
  - Track breakout success rate per symbol to adjust confidence

ATR Expansion Filter:
  - Current ATR must be > 1.2x its 50-bar SMA (volatility is expanding)
  - Breakouts during low volatility are more likely to be fakeouts
  - Breakouts during expanding volatility are more likely to be genuine

--- 3. LSTM PREDICTOR (lstm_predictor.py) ---

This is the ML CORE. Build a proper LSTM using PyTorch:

Architecture:
  - Input: sequence of 60 bars, each bar has ~25 features
  - Layer 1: LSTM(input_size=25, hidden_size=128, num_layers=2, dropout=0.3)
  - Layer 2: Fully connected (128 → 64, ReLU, Dropout 0.2)
  - Layer 3: Fully connected (64 → 3, Softmax)
  - Output: 3 classes — UP (price goes up >1 ATR in next 20 bars),
                         DOWN (price goes down >1 ATR in next 20 bars),
                         FLAT (price stays within 1 ATR)

Input Features (per bar, ~25 total):
  - OHLCV normalised (returns-based, not raw prices)
  - Log returns (1-bar, 5-bar, 10-bar, 20-bar)
  - Volatility-adjusted momentum (10, 20, 50 periods)
  - RSI (14), normalised to 0-1
  - MACD histogram, normalised by ATR
  - Bollinger Band %B
  - OBV rate of change
  - ATR percentile rank (rolling 100 bars)
  - ADX, normalised to 0-1
  - Volume ratio (current / 20-bar SMA)
  - Candle features: body_ratio, upper_shadow_ratio, lower_shadow_ratio
  - Hour of day (sin/cos encoded for cyclical nature)
  - Day of week (sin/cos encoded)

Training Pipeline:
  1. Fetch 2+ years of H1 data per symbol
  2. Calculate all features
  3. Create sequences: sliding window of 60 bars → predict next 20 bars direction
  4. Label: UP if max(high) - entry > 1*ATR before min(low) - entry < -1*ATR
           DOWN if the reverse, FLAT otherwise
  5. Train/validation/test split: 70/15/15 (chronological, NO shuffle)
  6. Use class weights to handle imbalanced classes
  7. Train with Adam optimizer, lr=0.001, with ReduceLROnPlateau scheduler
  8. Early stopping on validation loss (patience=15 epochs)
  9. Save best model checkpoint

Prediction:
  - Input the last 60 bars of features
  - Output: probability distribution over [UP, DOWN, FLAT]
  - Only trade when max probability > 0.60
  - Use the UP/DOWN probability as signal strength

IMPORTANT: All features must be NORMALISED using TRAINING SET statistics only.
Save the scaler (StandardScaler) with the model. Never use future data for
normalisation (this is data leakage and the #1 mistake in ML trading).

--- 4. CRASH DETECTOR (crash_detector.py) ---

Early warning system for momentum crashes. This is what separates this strategy
from naive momentum that blows up:

Momentum Dispersion:
  - Calculate momentum for all symbols
  - When the SPREAD between strongest and weakest momentum is extreme (>2 std),
    a crash becomes more likely (winners and losers are too far apart)
  - Reduce position sizes by 50% when dispersion is high

Correlation Spike Detection:
  - Calculate rolling 20-bar correlation between all symbol pairs
  - When average correlation spikes above 0.80, markets are "risk-off"
  - High correlation = everything moving together = diversification failing
  - Reduce exposure by 50-75% during correlation spikes

Momentum Reversal Detector:
  - When the 10-bar momentum flips sign but 50-bar hasn't yet
  - Short-term reversal within long-term trend = potential crash
  - Tighten all trailing stops to 0.5*ATR during reversal warnings

VIX-Like Volatility Monitor (using ATR):
  - Build an internal "fear index" from ATR across all symbols
  - fear_index = average(ATR_percentile_rank) across all symbols
  - If fear_index > 80th percentile: reduce all position sizes by 50%
  - If fear_index > 95th percentile: close all positions immediately

--- 5. SIGNAL GENERATION (signals.py) ---

Multi-layer signal pipeline:

  Step 1: Calculate momentum scores (absolute, relative, vol-adjusted) for all symbols
  Step 2: Check crash detector — if warning active, skip or reduce size
  Step 3: Check LSTM prediction — need >0.60 probability for UP or DOWN
  Step 4: Check breakout detector — is there structural confirmation?
  Step 5: Check multi-timeframe alignment:
          - Daily momentum direction (context)
          - H4 momentum direction (trend)
          - H1 momentum + LSTM prediction (signal)
          All three must agree for a signal
  Step 6: Score signal = momentum_strength * lstm_confidence * breakout_score
  Step 7: Rank all signals across all symbols — only take top N (capital is limited)

--- 6. RISK MANAGEMENT (risk_manager.py) ---

VOLATILITY TARGETING (this is the institutional approach):
  - Target a constant portfolio volatility of 10% annualised
  - position_size = target_vol / (instrument_vol * sqrt(252))
  - When volatility is LOW: take BIGGER positions (more bang for your risk budget)
  - When volatility is HIGH: take SMALLER positions (don't blow up)
  - This automatically reduces exposure before crashes (vol rises → size shrinks)

Additional risk controls:
  - Max risk per trade: 1.5% of account (momentum trades have wider stops)
  - Max daily drawdown: 4%
  - Max open positions: 10 (momentum trades in many instruments for diversification)
  - Stop loss: 2.0 * ATR from entry (momentum needs room to breathe)
  - Take profit: 4.0 * ATR (momentum trades have fat-tail positive skew)
  - Trailing stop: activate at 2:1 R:R, trail by 1.5*ATR
  - CRASH DETECTOR OVERRIDE: if crash warning → halve all position sizes
  - Sector exposure limits: max 40% in any single asset class

--- 7. SYMBOLS ---

Forex: EURUSD, GBPUSD, USDJPY, AUDUSD, NZDUSD, USDCAD, USDCHF
Indices: US500, US100, GER40, UK100, JPN225
Commodities: XAUUSD, XAGUSD, XTIUSD, XNGUSD

(Momentum works best with a LARGE universe — more instruments = more opportunities
and better diversification)

--- 8. BACKTESTING ---

Same as Prompt 1: Walk-forward optimisation + Monte Carlo simulation.
Additionally implement:
  - REGIME-SPECIFIC ANALYSIS: Show strategy performance in each HMM-detected regime
  - TURNOVER ANALYSIS: Calculate annual portfolio turnover and transaction costs
  - FACTOR ATTRIBUTION: Decompose returns into momentum factor, volatility factor,
    and alpha (unique to our LSTM)
  - ROLLING SHARPE: 6-month rolling Sharpe ratio chart to show consistency

=============================================================================
REQUIREMENTS.TXT
=============================================================================

MetaTrader5>=5.0.45
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
torch>=2.0.0
PyYAML>=6.0
pytz>=2023.3
joblib>=1.3.0
matplotlib>=3.7.0
seaborn>=0.12.0
TA-Lib>=0.4.28
scipy>=1.11.0

Note: Use PyTorch (torch) NOT TensorFlow. PyTorch is more Pythonic, easier to
debug, and the industry standard for quantitative finance.

=============================================================================
QUALITY REQUIREMENTS
=============================================================================

Same as the reference project:
- Full docstrings and type hints
- Vectorised NumPy operations
- Rotating log files
- Graceful shutdown
- Synthetic data generator
- Complete backtest reports with charts
- Walk-forward and Monte Carlo analysis

Build the ENTIRE system now. Every file. Complete. No shortcuts.
```

---
---

## PROMPT 3: Reinforcement Learning Multi-Asset Portfolio Allocation

---

```
You are a senior quantitative developer at a systematic hedge fund. I need you to
build a complete, production-grade automated trading system for MetaTrader 5 that
uses REINFORCEMENT LEARNING (PPO algorithm) to learn optimal PORTFOLIO ALLOCATION
across multiple asset classes. This is fundamentally different from single-instrument
signal-based strategies — instead of asking "should I buy or sell EURUSD?", the RL
agent asks "given the current market state, what is the OPTIMAL ALLOCATION of my
capital across ALL instruments to maximise risk-adjusted returns?"

I am a beginner. Build everything from scratch. Every file. Every line. Working
code. No placeholders. No "TODO" comments. No pseudo-code. Real, executable,
production-ready Python.

=============================================================================
STRATEGY PHILOSOPHY
=============================================================================

Traditional trading strategies make binary decisions: buy or sell a single
instrument. But the real question in portfolio management is ALLOCATION — how much
capital to put into each asset, and when to shift between assets.

Consider: when stocks crash, gold usually rallies. When USD strengthens, emerging
market currencies weaken. When oil spikes, airlines suffer but energy companies
benefit. These CROSS-ASSET RELATIONSHIPS are where the real alpha lives.

A reinforcement learning agent can learn these complex, non-linear relationships
that no human could code as rules. It learns through millions of simulated
interactions with the market:

1. STATE: What does the market look like right now? (prices, momentum, volatility,
   correlations across ALL assets simultaneously)
2. ACTION: What allocation should I have? (e.g., 30% EURUSD long, 20% gold long,
   10% S&P short, 40% cash)
3. REWARD: What risk-adjusted return did that allocation produce?

Over millions of episodes, the agent learns the optimal policy: given ANY market
state, the allocation that maximises long-term risk-adjusted returns.

This is how the most sophisticated hedge funds in the world operate (Bridgewater,
Two Sigma, Renaissance Technologies). They don't trade single instruments — they
manage PORTFOLIOS.

=============================================================================
COMPLETE PROJECT STRUCTURE (create every file)
=============================================================================

RL-Portfolio-Strategy/
├── config/
│   └── settings.yaml              # All configurable parameters
├── src/
│   ├── __init__.py
│   ├── data_engine.py             # MT5 data fetching for all instruments
│   ├── feature_engine.py          # State representation / feature engineering
│   ├── environment.py             # Custom Gymnasium environment for portfolio trading
│   ├── agent.py                   # PPO agent implementation
│   ├── reward_functions.py        # Sharpe, Sortino, and custom reward functions
│   ├── correlation_engine.py      # Cross-asset correlation and regime analysis
│   ├── macro_detector.py          # Risk-on/risk-off regime detection
│   ├── risk_manager.py            # Portfolio-level risk controls
│   ├── executor.py                # MT5 order execution for portfolio rebalancing
│   └── utils.py                   # Logging, config, helpers
├── backtest/
│   ├── __init__.py
│   ├── engine.py                  # Portfolio backtest engine
│   ├── data_loader.py             # Historical data loading
│   ├── monte_carlo.py             # Robustness testing
│   └── report.py                  # Portfolio-specific reporting
├── models/                        # Saved RL agent checkpoints
├── logs/
├── main.py                        # Live portfolio management
├── train_agent.py                 # RL agent training
├── run_backtest.py                # Backtesting entry point
├── requirements.txt
└── .gitignore

=============================================================================
DETAILED COMPONENT SPECIFICATIONS
=============================================================================

--- 1. CUSTOM GYMNASIUM ENVIRONMENT (environment.py) ---

This is the CORE. Build a custom Gym environment that simulates portfolio trading:

Class: PortfolioTradingEnv(gymnasium.Env)

Observation Space (STATE):
  For each of the N instruments, provide:
  - Returns: 1-bar, 5-bar, 10-bar, 20-bar, 50-bar (5 features)
  - Volatility: 10-bar, 20-bar realised vol (2 features)
  - Momentum: vol-adjusted momentum 10, 20, 50 (3 features)
  - RSI (14), normalised 0-1 (1 feature)
  - MACD histogram, normalised (1 feature)
  - Bollinger %B (1 feature)
  - ATR percentile (1 feature)
  - Volume ratio (1 feature)
  Total per instrument: 15 features

  Portfolio-level features:
  - Current allocation weights (N features — what we're currently holding)
  - Portfolio return (1-bar, 5-bar, 20-bar) (3 features)
  - Portfolio volatility (20-bar) (1 feature)
  - Cross-asset average correlation (1 feature)
  - Risk-on/risk-off indicator (1 feature)
  - Cash percentage (1 feature)
  - Current drawdown from peak (1 feature)
  Total portfolio features: N + 8

  Observation = concatenation of all instrument features + portfolio features
  For 12 instruments: 12*15 + 12 + 8 = 200 features

Action Space:
  - Continuous: Box(low=-1.0, high=1.0, shape=(N,))
  - Each element represents the target allocation for that instrument
  - Positive = long, Negative = short, 0 = no position
  - Allocations are normalised so that sum(abs(weights)) <= 1.0 (max 100% capital usage)
  - The remainder is held as cash

Step Function:
  1. Receive action (target allocations)
  2. Calculate rebalancing cost (transaction costs for changing positions)
  3. Apply allocations to the portfolio
  4. Advance one time step (one H4 bar — rebalance every 4 hours)
  5. Calculate portfolio return = sum(weight_i * return_i) - transaction_costs
  6. Calculate reward using the reward function
  7. Return: new_observation, reward, done, truncated, info

Reset Function:
  1. Randomly select a starting point in historical data
  2. Initial allocation: 100% cash
  3. Return initial observation

--- 2. REWARD FUNCTIONS (reward_functions.py) ---

The reward function is CRITICAL. A bad reward function = a bad agent. Implement
multiple options and make them configurable:

Differential Sharpe Ratio (RECOMMENDED — this is what works best):
  - DSR_t = (B_{t-1} * delta_A_t - 0.5 * A_{t-1} * delta_B_t) / (B_{t-1} - A_{t-1}^2)^{3/2}
  - Where A_t = exponential moving average of returns
  - And B_t = exponential moving average of squared returns
  - This directly optimises the Sharpe ratio incrementally
  - Reference: Moody & Saffell (2001), "Learning to Trade via Direct RL"

Risk-Adjusted Return:
  - reward = portfolio_return - 0.5 * lambda * portfolio_variance
  - lambda controls risk aversion (default 2.0)
  - Penalises the agent for taking excessive risk

Sortino-Based Reward:
  - reward = portfolio_return / downside_deviation
  - Only penalises DOWNSIDE volatility, not upside
  - The agent learns it's OK to be volatile upward

Drawdown Penalty:
  - Apply additional negative reward proportional to current drawdown
  - reward -= max(0, drawdown - 0.05) * 10
  - Agent gets heavily punished for drawdowns exceeding 5%

Turnover Penalty:
  - reward -= turnover_rate * transaction_cost_rate
  - Prevents the agent from churning the portfolio
  - Forces the agent to learn that trading has costs

--- 3. PPO AGENT (agent.py) ---

Implement Proximal Policy Optimisation using PyTorch. PPO is the standard
algorithm for continuous action spaces in RL:

Actor Network (Policy):
  - Input: state_dim (200 for 12 instruments)
  - Hidden layers: 512 → 256 → 128 (ReLU activations)
  - Output: N allocations (tanh activation for [-1, 1] range)
  - Add a learned log_std parameter for exploration
  - Network outputs: mean allocation + std for each instrument

Critic Network (Value Function):
  - Input: same state_dim
  - Hidden layers: 512 → 256 → 128 (ReLU activations)
  - Output: scalar value estimate (how good is this state?)

PPO Training Loop:
  1. Collect trajectories: run the policy for T steps in the environment
  2. Calculate advantages using GAE (Generalised Advantage Estimation, lambda=0.95)
  3. For K epochs (default 10):
     a. Calculate ratio = new_prob / old_prob
     b. Clip ratio to [1-epsilon, 1+epsilon] (epsilon=0.2)
     c. Policy loss = -min(ratio * advantage, clipped_ratio * advantage)
     d. Value loss = MSE(predicted_value, actual_return)
     e. Entropy bonus to encourage exploration
     f. Total loss = policy_loss + 0.5 * value_loss - 0.01 * entropy
  4. Update networks with Adam optimiser (lr=3e-4)

Training Configuration:
  - Episodes: 50,000 minimum (RL needs LOTS of experience)
  - Episode length: 500 steps (500 H4 bars ≈ 3 months)
  - Batch size: 2048 steps
  - Gamma (discount): 0.99
  - GAE lambda: 0.95
  - Clip epsilon: 0.2
  - Learning rate: 3e-4 with cosine annealing
  - Entropy coefficient: 0.01 (decaying to 0.001)

IMPORTANT: Train on MULTIPLE random windows of historical data, not just one pass.
This prevents the agent from memorising a specific period and teaches it to
generalise. Each episode should start at a random point in the data.

--- 4. FEATURE ENGINE (feature_engine.py) ---

Prepare the state representation for the RL agent:

Per-Instrument Features (15 per instrument):
  - log_return_1, log_return_5, log_return_10, log_return_20, log_return_50
  - realised_vol_10, realised_vol_20
  - vol_adj_momentum_10, vol_adj_momentum_20, vol_adj_momentum_50
  - rsi_normalised (RSI/100)
  - macd_normalised (macd_histogram / ATR)
  - bb_pctb
  - atr_percentile (rolling rank over 100 bars)
  - volume_ratio

Portfolio Features:
  - current_weights (one per instrument)
  - portfolio_return_1, portfolio_return_5, portfolio_return_20
  - portfolio_vol_20
  - avg_cross_correlation
  - risk_on_off_indicator
  - cash_pct
  - current_drawdown

ALL FEATURES MUST BE NORMALISED:
  - Use rolling z-score normalisation (not global — prevents look-ahead bias)
  - z_score = (value - rolling_mean_100) / rolling_std_100
  - Clip to [-3, 3] to prevent extreme outliers from destabilising the network
  - This is critical for neural network training stability

--- 5. CORRELATION ENGINE (correlation_engine.py) ---

Track cross-asset relationships that the RL agent uses for allocation:

Rolling Correlation Matrix:
  - Calculate pairwise correlation between all instruments
  - Use 50-bar rolling window
  - Output: NxN correlation matrix updated each bar

Correlation Regime:
  - Average absolute correlation across all pairs
  - LOW correlation (<0.3): good diversification → agent can spread across assets
  - MEDIUM correlation (0.3-0.6): normal market
  - HIGH correlation (>0.6): crisis/risk-off → agent should concentrate or go cash

Eigenvalue Analysis (Principal Component Analysis):
  - Run PCA on the correlation matrix
  - If the first eigenvalue explains >60% of variance → all assets moving together
  - This is an early warning of market stress

Correlation Breakdowns:
  - Track when historically correlated pairs DECOUPLE
  - This signals regime change → pass to the RL agent as a feature

--- 6. MACRO DETECTOR (macro_detector.py) ---

Detect risk-on vs risk-off market environments:

Risk-On Indicators:
  - Stock indices (US500, US100) trending up
  - Gold (XAUUSD) trending down or flat
  - High-yield currencies (AUD, NZD) strengthening
  - Low average correlation across assets
  - ATR declining across most instruments

Risk-Off Indicators:
  - Stock indices falling
  - Gold rallying strongly
  - Safe-haven currencies (JPY, CHF, USD) strengthening
  - Correlation spiking (everything selling together)
  - ATR expanding across most instruments

Output:
  - risk_score: -1.0 (extreme risk-off) to +1.0 (extreme risk-on)
  - This is fed directly into the RL agent's state representation
  - The agent learns to go defensive during risk-off and aggressive during risk-on

--- 7. RISK MANAGEMENT (risk_manager.py) ---

PORTFOLIO-LEVEL risk management (not per-trade):

Position Limits:
  - Max allocation per instrument: 25% of portfolio
  - Max gross exposure: 150% (allows some leverage but not excessive)
  - Max net exposure: 80% (prevents fully directional bets)
  - Min cash: 10% (always keep some powder dry)

Drawdown Controls:
  - At 5% drawdown: reduce all positions by 25%
  - At 8% drawdown: reduce all positions by 50%
  - At 10% drawdown: close all positions, wait 48 hours
  - These are HARD LIMITS that override the RL agent's decisions
  - The agent is smart but can make mistakes — these are safety rails

Volatility Targeting:
  - Target annualised portfolio volatility: 10%
  - If realised vol > 12%: scale down all positions proportionally
  - If realised vol < 8%: allow agent to scale up (but don't force it)

Concentration Limits:
  - Max allocation to any single asset class: 50%
  - Asset classes: Forex, Indices, Commodities
  - Prevents the agent from putting everything into one basket

Rebalancing Controls:
  - Minimum rebalance threshold: 5% change in any weight
  - Don't rebalance if transaction costs would exceed 0.1% of portfolio
  - Max rebalance frequency: every 4 hours (H4 bars)

--- 8. EXECUTOR (executor.py) ---

Portfolio rebalancing execution for MT5:

  1. Calculate target positions from RL agent weights + risk manager adjustments
  2. Calculate current positions from MT5
  3. Calculate DELTA (what needs to change)
  4. Execute delta trades:
     - Close positions that need to be reduced
     - Open/increase positions that need to be added
     - Use market orders with slippage tolerance
  5. Verify final positions match targets (within tolerance)

Handle partial fills and requotes gracefully.
Track total transaction costs per rebalance.

--- 9. SYMBOLS (12 instruments across 3 asset classes) ---

Forex (5):
  EURUSD, GBPUSD, USDJPY, AUDUSD, USDCHF

Indices (4):
  US500, US100, GER40, JPN225

Commodities (3):
  XAUUSD, XAGUSD, XTIUSD

Why 12: This is enough for meaningful diversification but not so many that the
RL agent's action space becomes unmanageable. The 200-dimensional state space
and 12-dimensional action space are within PPO's comfortable range.

--- 10. BACKTESTING ---

Portfolio-Specific Metrics:
  - Total return, CAGR, annualised volatility
  - Sharpe ratio, Sortino ratio, Calmar ratio
  - Maximum drawdown (% and duration)
  - Portfolio turnover (annual)
  - Transaction cost impact
  - Alpha vs equal-weight benchmark
  - Alpha vs 60/40 stock/bond benchmark
  - Information ratio
  - Tracking error

Portfolio-Specific Charts:
  - Equity curve vs benchmarks (equal-weight, 60/40)
  - Allocation weights over time (stacked area chart)
  - Rolling Sharpe ratio (6-month window)
  - Drawdown chart
  - Correlation heatmap evolution (quarterly snapshots)
  - Risk-on/risk-off regime overlay on equity curve
  - Contribution to return by instrument
  - Monthly returns table

Walk-Forward + Monte Carlo (same as other strategies).

--- 11. TRAINING (train_agent.py) ---

  python train_agent.py --synthetic --episodes 50000    # No MT5 needed
  python train_agent.py --episodes 100000               # Train on real MT5 data
  python train_agent.py --resume models/checkpoint.pt   # Continue training

Training should:
  1. Load or generate historical data for all instruments
  2. Create the Gymnasium environment
  3. Train PPO agent for N episodes
  4. Log training metrics: episode reward, portfolio Sharpe, avg allocation
  5. Save checkpoints every 1000 episodes
  6. Plot training curves (reward over time, Sharpe over time)
  7. Run final evaluation on held-out test data
  8. Save best model based on test Sharpe ratio

--- 12. LIVE TRADING (main.py) ---

  python main.py                    # Live portfolio management
  python main.py --dry-run          # Simulation mode
  python main.py --config path      # Custom config

Main loop:
  1. Every H4 bar (4 hours):
     a. Fetch latest data for all 12 instruments
     b. Build state representation
     c. Get RL agent's target allocation
     d. Apply risk manager constraints
     e. Execute rebalancing trades
     f. Log current portfolio state
  2. Every 1 minute: monitor portfolio value, check drawdown limits
  3. Graceful shutdown: close all positions on Ctrl+C

=============================================================================
REQUIREMENTS.TXT
=============================================================================

MetaTrader5>=5.0.45
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
torch>=2.0.0
gymnasium>=0.29.0
PyYAML>=6.0
pytz>=2023.3
joblib>=1.3.0
matplotlib>=3.7.0
seaborn>=0.12.0
TA-Lib>=0.4.28
scipy>=1.11.0

=============================================================================
QUALITY REQUIREMENTS
=============================================================================

Same baseline as reference project:
- Full docstrings and type hints
- Vectorised NumPy operations
- Rotating log files
- Graceful shutdown
- Synthetic data generator (CRITICAL for RL — agent needs millions of steps)
- Complete portfolio-specific backtest reports with charts

ADDITIONAL RL-SPECIFIC REQUIREMENTS:
- Training must be reproducible (set all random seeds)
- Training curves must be logged and plottable
- Agent must be saveable/loadable (checkpoint system)
- Environment must support vectorised parallel environments for faster training
  (use gymnasium.vector.SyncVectorEnv with 4-8 parallel envs)
- Include a simple RANDOM AGENT baseline for comparison
- Include an EQUAL-WEIGHT baseline for comparison

Build the ENTIRE system now. Every file. Complete. No shortcuts.
```

---
---

## How to Use These Prompts

1. Open a **new Claude Code chat** (fresh session, no prior context)
2. Copy-paste ONE prompt above (they are self-contained)
3. Let Claude build the entire system
4. After it finishes, run the backtest first: `python run_backtest.py --synthetic`
5. If everything works, proceed to MT5 integration

## Strategy Comparison

| Feature | Prompt 1 (Mean Reversion) | Prompt 2 (Momentum LSTM) | Prompt 3 (RL Portfolio) |
|---|---|---|---|
| **Edge** | Price snaps back to mean | Trend continues | Optimal allocation |
| **ML Model** | Hidden Markov Model | LSTM Neural Network | PPO Reinforcement Learning |
| **When it works** | Ranging/choppy markets | Strong trending markets | All conditions (adaptive) |
| **When it fails** | Trending markets | Sudden reversals | Insufficient training data |
| **Complexity** | Medium | High | Very High |
| **Holding period** | Hours to 1-2 days | Days to weeks | Continuous (rebalance 4h) |
| **Risk profile** | High win rate, small wins | Lower win rate, large wins | Steady risk-adjusted returns |
| **Key innovation** | Regime detection + pairs | Crash detector + breakouts | Cross-asset allocation |

## Portfolio Diversification

These three strategies are **intentionally uncorrelated**:
- Mean reversion profits in **ranging** markets
- Momentum profits in **trending** markets
- RL portfolio profits by **allocating between** regimes

Running all three simultaneously on separate accounts creates a
**meta-diversified** trading operation where at least one strategy
is always working regardless of market conditions.
