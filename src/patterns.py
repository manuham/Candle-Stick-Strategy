"""
Candlestick Pattern Detector — 12 statistically proven patterns.

Each detector function takes a DataFrame with OHLCV columns and returns a
pandas Series of signal strength values:
    +100  = strong bullish signal
    +50   = moderate bullish signal
    -100  = strong bearish signal
    -50   = moderate bearish signal
    0     = no signal

All calculations are vectorized with numpy for performance.
"""

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Helper: pre-compute candle anatomy
# ---------------------------------------------------------------------------

def candle_properties(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute body, shadow, and ratio properties for every candle.

    Adds columns: body, body_abs, upper_shadow, lower_shadow,
                  candle_range, body_ratio, is_bullish, is_bearish.
    """
    o = df["open"].values
    h = df["high"].values
    l = df["low"].values
    c = df["close"].values

    body = c - o                          # signed: positive = bullish
    body_abs = np.abs(body)
    candle_range = h - l
    # Avoid division by zero for doji-like candles
    safe_range = np.where(candle_range == 0, 1e-10, candle_range)

    upper_shadow = h - np.maximum(o, c)
    lower_shadow = np.minimum(o, c) - l
    body_ratio = body_abs / safe_range

    props = df.copy()
    props["body"] = body
    props["body_abs"] = body_abs
    props["upper_shadow"] = upper_shadow
    props["lower_shadow"] = lower_shadow
    props["candle_range"] = candle_range
    props["body_ratio"] = body_ratio
    props["is_bullish"] = body > 0
    props["is_bearish"] = body < 0
    return props


# ---------------------------------------------------------------------------
# Individual pattern detectors
# ---------------------------------------------------------------------------

def detect_bullish_engulfing(df: pd.DataFrame) -> pd.Series:
    """Bullish Engulfing: bearish candle followed by larger bullish candle."""
    p = candle_properties(df)
    prev_bearish = p["is_bearish"].shift(1).fillna(False)
    curr_bullish = p["is_bullish"]

    prev_body_top = np.maximum(p["open"].shift(1), p["close"].shift(1))
    prev_body_bot = np.minimum(p["open"].shift(1), p["close"].shift(1))
    curr_body_top = np.maximum(p["open"], p["close"])
    curr_body_bot = np.minimum(p["open"], p["close"])

    engulfs = (curr_body_top >= prev_body_top) & (curr_body_bot <= prev_body_bot)
    significant = p["body_abs"] > p["body_abs"].shift(1)

    signal = prev_bearish & curr_bullish & engulfs & significant
    return (signal.astype(int) * 100).rename("bullish_engulfing")


def detect_bearish_engulfing(df: pd.DataFrame) -> pd.Series:
    """Bearish Engulfing: bullish candle followed by larger bearish candle."""
    p = candle_properties(df)
    prev_bullish = p["is_bullish"].shift(1).fillna(False)
    curr_bearish = p["is_bearish"]

    prev_body_top = np.maximum(p["open"].shift(1), p["close"].shift(1))
    prev_body_bot = np.minimum(p["open"].shift(1), p["close"].shift(1))
    curr_body_top = np.maximum(p["open"], p["close"])
    curr_body_bot = np.minimum(p["open"], p["close"])

    engulfs = (curr_body_top >= prev_body_top) & (curr_body_bot <= prev_body_bot)
    significant = p["body_abs"] > p["body_abs"].shift(1)

    signal = prev_bullish & curr_bearish & engulfs & significant
    return (signal.astype(int) * -100).rename("bearish_engulfing")


def detect_hammer(df: pd.DataFrame) -> pd.Series:
    """
    Hammer: small body at top, long lower shadow (>= 2x body),
    tiny upper shadow (<= 30% of body). Bullish reversal.
    """
    p = candle_properties(df)
    safe_body = np.where(p["body_abs"] == 0, 1e-10, p["body_abs"])

    long_lower = p["lower_shadow"] >= 2.0 * safe_body
    small_upper = p["upper_shadow"] <= 0.3 * safe_body
    has_body = p["body_ratio"] > 0.01

    # Should appear after a decline (close trending down over 5 bars)
    downtrend = p["close"] < p["close"].shift(5)

    signal = long_lower & small_upper & has_body & downtrend
    return (signal.astype(int) * 80).rename("hammer")


def detect_shooting_star(df: pd.DataFrame) -> pd.Series:
    """
    Shooting Star: small body at bottom, long upper shadow (>= 2x body),
    tiny lower shadow. Bearish reversal.
    """
    p = candle_properties(df)
    safe_body = np.where(p["body_abs"] == 0, 1e-10, p["body_abs"])

    long_upper = p["upper_shadow"] >= 2.0 * safe_body
    small_lower = p["lower_shadow"] <= 0.3 * safe_body
    has_body = p["body_ratio"] > 0.01

    # Should appear after a rise
    uptrend = p["close"] > p["close"].shift(5)

    signal = long_upper & small_lower & has_body & uptrend
    return (signal.astype(int) * -80).rename("shooting_star")


def detect_morning_star(df: pd.DataFrame) -> pd.Series:
    """
    Morning Star (3-candle bullish reversal):
    1. Large bearish candle
    2. Small-body candle (gap down or small body)
    3. Large bullish candle closing above midpoint of candle 1
    """
    p = candle_properties(df)

    # Candle 1: large bearish (2 bars ago)
    c1_bearish = p["is_bearish"].shift(2).fillna(False)
    c1_large = p["body_ratio"].shift(2) > 0.5

    # Candle 2: small body (1 bar ago)
    c2_small = p["body_ratio"].shift(1) < 0.3

    # Candle 3: large bullish (current)
    c3_bullish = p["is_bullish"]
    c3_large = p["body_ratio"] > 0.5

    # Candle 3 close above midpoint of candle 1
    c1_mid = (p["open"].shift(2) + p["close"].shift(2)) / 2
    c3_above_mid = p["close"] > c1_mid

    signal = c1_bearish & c1_large & c2_small & c3_bullish & c3_large & c3_above_mid
    return (signal.astype(int) * 100).rename("morning_star")


def detect_evening_star(df: pd.DataFrame) -> pd.Series:
    """
    Evening Star (3-candle bearish reversal):
    1. Large bullish candle
    2. Small-body candle
    3. Large bearish candle closing below midpoint of candle 1
    """
    p = candle_properties(df)

    c1_bullish = p["is_bullish"].shift(2).fillna(False)
    c1_large = p["body_ratio"].shift(2) > 0.5

    c2_small = p["body_ratio"].shift(1) < 0.3

    c3_bearish = p["is_bearish"]
    c3_large = p["body_ratio"] > 0.5

    c1_mid = (p["open"].shift(2) + p["close"].shift(2)) / 2
    c3_below_mid = p["close"] < c1_mid

    signal = c1_bullish & c1_large & c2_small & c3_bearish & c3_large & c3_below_mid
    return (signal.astype(int) * -100).rename("evening_star")


def detect_bullish_harami(df: pd.DataFrame) -> pd.Series:
    """Bullish Harami: large bearish candle followed by small bullish candle inside it."""
    p = candle_properties(df)

    prev_bearish = p["is_bearish"].shift(1).fillna(False)
    prev_large = p["body_ratio"].shift(1) > 0.5
    curr_bullish = p["is_bullish"]

    prev_body_top = np.maximum(p["open"].shift(1), p["close"].shift(1))
    prev_body_bot = np.minimum(p["open"].shift(1), p["close"].shift(1))
    curr_body_top = np.maximum(p["open"], p["close"])
    curr_body_bot = np.minimum(p["open"], p["close"])

    contained = (curr_body_top <= prev_body_top) & (curr_body_bot >= prev_body_bot)

    signal = prev_bearish & prev_large & curr_bullish & contained
    return (signal.astype(int) * 60).rename("bullish_harami")


def detect_bearish_harami(df: pd.DataFrame) -> pd.Series:
    """Bearish Harami: large bullish candle followed by small bearish candle inside it."""
    p = candle_properties(df)

    prev_bullish = p["is_bullish"].shift(1).fillna(False)
    prev_large = p["body_ratio"].shift(1) > 0.5
    curr_bearish = p["is_bearish"]

    prev_body_top = np.maximum(p["open"].shift(1), p["close"].shift(1))
    prev_body_bot = np.minimum(p["open"].shift(1), p["close"].shift(1))
    curr_body_top = np.maximum(p["open"], p["close"])
    curr_body_bot = np.minimum(p["open"], p["close"])

    contained = (curr_body_top <= prev_body_top) & (curr_body_bot >= prev_body_bot)

    signal = prev_bullish & prev_large & curr_bearish & contained
    return (signal.astype(int) * -60).rename("bearish_harami")


def detect_doji(df: pd.DataFrame) -> pd.Series:
    """
    Doji: body is extremely small relative to the candle range.
    Returns +50 after downtrend, -50 after uptrend, 0 otherwise.
    """
    p = candle_properties(df)

    is_doji = p["body_ratio"] <= 0.1
    # Directional context
    after_down = p["close"].shift(1) < p["close"].shift(5)
    after_up = p["close"].shift(1) > p["close"].shift(5)

    result = pd.Series(0, index=df.index, dtype=int)
    result[is_doji & after_down] = 50   # potential bullish reversal
    result[is_doji & after_up] = -50    # potential bearish reversal
    return result.rename("doji")


def detect_three_white_soldiers(df: pd.DataFrame) -> pd.Series:
    """
    Three White Soldiers: 3 consecutive bullish candles, each closing higher,
    each opening within the previous candle's body.
    """
    p = candle_properties(df)

    b1 = p["is_bullish"].shift(2).fillna(False)
    b2 = p["is_bullish"].shift(1).fillna(False)
    b3 = p["is_bullish"]

    # Each closes higher
    higher_close = (p["close"].shift(1) > p["close"].shift(2)) & (p["close"] > p["close"].shift(1))

    # Each opens within previous body
    o2_in_b1 = (p["open"].shift(1) >= p["open"].shift(2)) & (p["open"].shift(1) <= p["close"].shift(2))
    o3_in_b2 = (p["open"] >= p["open"].shift(1)) & (p["open"] <= p["close"].shift(1))

    # Substantial bodies
    decent_body = (p["body_ratio"].shift(2) > 0.5) & (p["body_ratio"].shift(1) > 0.5) & (p["body_ratio"] > 0.5)

    signal = b1 & b2 & b3 & higher_close & o2_in_b1 & o3_in_b2 & decent_body
    return (signal.astype(int) * 90).rename("three_white_soldiers")


def detect_three_black_crows(df: pd.DataFrame) -> pd.Series:
    """
    Three Black Crows: 3 consecutive bearish candles, each closing lower,
    each opening within the previous candle's body.
    """
    p = candle_properties(df)

    b1 = p["is_bearish"].shift(2).fillna(False)
    b2 = p["is_bearish"].shift(1).fillna(False)
    b3 = p["is_bearish"]

    lower_close = (p["close"].shift(1) < p["close"].shift(2)) & (p["close"] < p["close"].shift(1))

    # Opens within previous body (bearish: open <= prev_open and open >= prev_close)
    o2_in_b1 = (p["open"].shift(1) <= p["open"].shift(2)) & (p["open"].shift(1) >= p["close"].shift(2))
    o3_in_b2 = (p["open"] <= p["open"].shift(1)) & (p["open"] >= p["close"].shift(1))

    decent_body = (p["body_ratio"].shift(2) > 0.5) & (p["body_ratio"].shift(1) > 0.5) & (p["body_ratio"] > 0.5)

    signal = b1 & b2 & b3 & lower_close & o2_in_b1 & o3_in_b2 & decent_body
    return (signal.astype(int) * -90).rename("three_black_crows")


def detect_inverted_hammer(df: pd.DataFrame) -> pd.Series:
    """
    Inverted Hammer: long upper shadow (>= 2x body), tiny lower shadow.
    Appears after a downtrend. Bullish reversal.
    """
    p = candle_properties(df)
    safe_body = np.where(p["body_abs"] == 0, 1e-10, p["body_abs"])

    long_upper = p["upper_shadow"] >= 2.0 * safe_body
    small_lower = p["lower_shadow"] <= 0.3 * safe_body
    has_body = p["body_ratio"] > 0.01

    downtrend = p["close"] < p["close"].shift(5)

    signal = long_upper & small_lower & has_body & downtrend
    return (signal.astype(int) * 70).rename("inverted_hammer")


# ---------------------------------------------------------------------------
# Aggregate detector
# ---------------------------------------------------------------------------

ALL_DETECTORS = [
    detect_bullish_engulfing,
    detect_bearish_engulfing,
    detect_hammer,
    detect_shooting_star,
    detect_morning_star,
    detect_evening_star,
    detect_bullish_harami,
    detect_bearish_harami,
    detect_doji,
    detect_three_white_soldiers,
    detect_three_black_crows,
    detect_inverted_hammer,
]


def detect_all(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run all pattern detectors and return a DataFrame of signals.

    Each column is a pattern name, values are signal strengths.
    Also adds 'dominant_signal' (strongest absolute signal per bar)
    and 'dominant_pattern' (name of the strongest pattern).
    """
    results = pd.DataFrame(index=df.index)
    for detector in ALL_DETECTORS:
        series = detector(df)
        results[series.name] = series

    # Dominant signal: the pattern with the largest absolute value per row
    abs_vals = results.abs()
    dominant_col = abs_vals.idxmax(axis=1)
    results["dominant_pattern"] = dominant_col

    # Vectorized lookup of the dominant signal value
    idx_arr = np.arange(len(results))
    col_indices = [results.columns.get_loc(c) for c in dominant_col]
    results["dominant_signal"] = results.values[idx_arr, col_indices].astype(float)

    # Zero out dominant when all patterns are zero
    all_zero = abs_vals.sum(axis=1) == 0
    results.loc[all_zero, "dominant_pattern"] = "none"
    results.loc[all_zero, "dominant_signal"] = 0

    return results
