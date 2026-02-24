"""
Multi-Timeframe Confluence Scoring Engine.

Scores how well a trade signal aligns across three timeframes:
  - H4 (trend)   → Is the higher-TF trend in our favour?       (0–30 pts)
  - H1 (signal)  → Does the signal TF show pattern + indicator? (0–40 pts)
  - M15 (entry)  → Does the entry TF confirm the trade?         (0–30 pts)

Total score range: 0–100.  Minimum threshold to trade is configurable
(default 60).
"""

import logging
from dataclasses import dataclass
from typing import Optional

import pandas as pd

from src.patterns import detect_all as detect_patterns
from src.indicators import calculate_all as calculate_indicators

logger = logging.getLogger("strategy")


@dataclass
class ConfluenceResult:
    """Stores the breakdown of a confluence evaluation."""
    total_score: float
    h4_score: float
    h1_score: float
    m15_score: float
    direction: str           # "BUY", "SELL", or "NONE"
    pattern_name: str        # Dominant pattern that triggered
    pattern_strength: float  # Strength of dominant pattern
    details: dict            # Granular score breakdown


def score_h4_trend(df_h4: pd.DataFrame, direction: str, config: dict) -> tuple:
    """
    Score higher-timeframe trend alignment (0–30 points).

    +30 if EMA50 > EMA200 for BUY (or EMA50 < EMA200 for SELL).
    +10 if flat / crossing.
    +0  if opposing trend.
    """
    max_pts = config.get("weights", {}).get("h4_trend_alignment", 30)

    if df_h4 is None or df_h4.empty:
        return 0.0, {"h4_reason": "no_data"}

    ind_cfg = config.get("_indicator_config", {})
    df_h4 = calculate_indicators(df_h4, ind_cfg)
    last = df_h4.iloc[-1]

    ema_fast = last.get("ema_fast")
    ema_slow = last.get("ema_slow")

    if pd.isna(ema_fast) or pd.isna(ema_slow):
        return 0.0, {"h4_reason": "ema_not_ready"}

    trend_bullish = ema_fast > ema_slow
    trend_bearish = ema_fast < ema_slow

    # How far apart are the EMAs? (trend strength)
    ema_gap_pct = abs(ema_fast - ema_slow) / ema_slow * 100 if ema_slow != 0 else 0

    if direction == "BUY" and trend_bullish:
        score = max_pts
        reason = "aligned_bullish"
    elif direction == "SELL" and trend_bearish:
        score = max_pts
        reason = "aligned_bearish"
    elif ema_gap_pct < 0.1:
        score = max_pts * 0.33
        reason = "flat_trend"
    else:
        score = 0.0
        reason = "opposing_trend"

    return score, {"h4_reason": reason, "h4_ema_gap_pct": round(ema_gap_pct, 4)}


def score_h1_pattern_indicator(
    df_h1: pd.DataFrame,
    direction: str,
    config: dict,
) -> tuple:
    """
    Score H1 pattern + indicator alignment (0–40 points).

    Breakdown:
      Pattern detected:      +15
      RSI confirms:          +10
      MACD aligns:           +10
      Price at BB extreme:   +5
    """
    max_pts = config.get("weights", {}).get("h1_pattern_indicator", 40)
    score = 0.0
    details = {}

    if df_h1 is None or df_h1.empty:
        return 0.0, {"h1_reason": "no_data"}

    ind_cfg = config.get("_indicator_config", {})
    df_h1 = calculate_indicators(df_h1, ind_cfg)
    patterns = detect_patterns(df_h1)
    last_pat = patterns.iloc[-1]
    last_ind = df_h1.iloc[-1]

    dominant_signal = last_pat.get("dominant_signal", 0)
    dominant_pattern = last_pat.get("dominant_pattern", "none")

    details["h1_pattern"] = dominant_pattern
    details["h1_pattern_strength"] = float(dominant_signal)

    # --- Pattern detected (+15) ---
    if direction == "BUY" and dominant_signal > 0:
        score += 15
    elif direction == "SELL" and dominant_signal < 0:
        score += 15

    # --- RSI confirmation (+10) ---
    rsi = last_ind.get("rsi")
    if rsi is not None and not pd.isna(rsi):
        if direction == "BUY" and rsi < 70:
            score += 10
            details["h1_rsi_ok"] = True
        elif direction == "SELL" and rsi > 30:
            score += 10
            details["h1_rsi_ok"] = True
        else:
            details["h1_rsi_ok"] = False
    details["h1_rsi"] = float(rsi) if rsi is not None and not pd.isna(rsi) else None

    # --- MACD direction (+10) ---
    macd_hist = last_ind.get("macd_histogram")
    if macd_hist is not None and not pd.isna(macd_hist):
        if direction == "BUY" and macd_hist > 0:
            score += 10
        elif direction == "SELL" and macd_hist < 0:
            score += 10

    # --- Bollinger Band extreme (+5) ---
    bb_pctb = last_ind.get("bb_pctb")
    if bb_pctb is not None and not pd.isna(bb_pctb):
        if direction == "BUY" and bb_pctb < 0.2:
            score += 5
        elif direction == "SELL" and bb_pctb > 0.8:
            score += 5

    return min(score, max_pts), details


def score_m15_entry(df_m15: pd.DataFrame, direction: str, config: dict) -> tuple:
    """
    Score M15 entry confirmation (0–30 points).

    Breakdown:
      M15 pattern aligns:    +15
      Stochastic confirms:   +10
      Volume above average:  +5
    """
    max_pts = config.get("weights", {}).get("m15_entry_confirmation", 30)
    score = 0.0
    details = {}

    if df_m15 is None or df_m15.empty:
        return 0.0, {"m15_reason": "no_data"}

    ind_cfg = config.get("_indicator_config", {})
    df_m15 = calculate_indicators(df_m15, ind_cfg)
    patterns = detect_patterns(df_m15)
    last_pat = patterns.iloc[-1]
    last_ind = df_m15.iloc[-1]

    m15_signal = last_pat.get("dominant_signal", 0)
    details["m15_pattern"] = last_pat.get("dominant_pattern", "none")

    # --- M15 pattern aligns (+15) ---
    if direction == "BUY" and m15_signal > 0:
        score += 15
    elif direction == "SELL" and m15_signal < 0:
        score += 15

    # --- Stochastic confirmation (+10) ---
    stoch_k = last_ind.get("stoch_k")
    stoch_d = last_ind.get("stoch_d")
    if stoch_k is not None and stoch_d is not None:
        if not pd.isna(stoch_k) and not pd.isna(stoch_d):
            if direction == "BUY" and stoch_k > stoch_d and stoch_k < 80:
                score += 10
            elif direction == "SELL" and stoch_k < stoch_d and stoch_k > 20:
                score += 10

    # --- Volume above average (+5) ---
    vol_ratio = last_ind.get("volume_ratio")
    if vol_ratio is not None and not pd.isna(vol_ratio) and vol_ratio > 1.5:
        score += 5

    return min(score, max_pts), details


def evaluate_confluence(
    df_h4: Optional[pd.DataFrame],
    df_h1: Optional[pd.DataFrame],
    df_m15: Optional[pd.DataFrame],
    config: dict,
) -> ConfluenceResult:
    """
    Full confluence evaluation across all three timeframes.

    First determines direction from H1 pattern signals, then scores
    alignment across H4, H1, and M15.

    Args:
        df_h4: H4 OHLCV DataFrame (trend context).
        df_h1: H1 OHLCV DataFrame (signal generation).
        df_m15: M15 OHLCV DataFrame (entry timing).
        config: Confluence config section from settings.

    Returns:
        ConfluenceResult with total score and breakdown.
    """
    indicator_config = config.get("_indicator_config", {})

    # Step 1: Determine direction from H1 patterns
    if df_h1 is None or df_h1.empty:
        return ConfluenceResult(
            total_score=0, h4_score=0, h1_score=0, m15_score=0,
            direction="NONE", pattern_name="none", pattern_strength=0,
            details={"reason": "no_h1_data"},
        )

    df_h1_ind = calculate_indicators(df_h1.copy(), indicator_config)
    patterns_h1 = detect_patterns(df_h1_ind)
    last_pattern = patterns_h1.iloc[-1]

    dominant_signal = last_pattern.get("dominant_signal", 0)
    dominant_pattern = last_pattern.get("dominant_pattern", "none")

    if dominant_signal == 0:
        return ConfluenceResult(
            total_score=0, h4_score=0, h1_score=0, m15_score=0,
            direction="NONE", pattern_name="none", pattern_strength=0,
            details={"reason": "no_pattern_detected"},
        )

    direction = "BUY" if dominant_signal > 0 else "SELL"

    # Step 2: Score each timeframe
    h4_score, h4_details = score_h4_trend(df_h4, direction, config)
    h1_score, h1_details = score_h1_pattern_indicator(df_h1, direction, config)
    m15_score, m15_details = score_m15_entry(df_m15, direction, config)

    total = h4_score + h1_score + m15_score
    all_details = {**h4_details, **h1_details, **m15_details}

    logger.debug(
        f"Confluence: {direction} | H4={h4_score:.0f} H1={h1_score:.0f} "
        f"M15={m15_score:.0f} | Total={total:.0f} | Pattern={dominant_pattern}"
    )

    return ConfluenceResult(
        total_score=total,
        h4_score=h4_score,
        h1_score=h1_score,
        m15_score=m15_score,
        direction=direction,
        pattern_name=dominant_pattern,
        pattern_strength=float(dominant_signal),
        details=all_details,
    )
