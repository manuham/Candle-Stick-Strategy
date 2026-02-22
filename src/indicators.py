"""
Technical Indicator calculations using TA-Lib.

All indicators are added as new columns to the input DataFrame.
"""

import pandas as pd
import numpy as np
import talib


def calculate_all(df: pd.DataFrame, config: dict = None) -> pd.DataFrame:
    """
    Calculate all technical indicators and add them as columns.

    Args:
        df: OHLCV DataFrame (must have open, high, low, close, volume columns).
        config: Indicator config dict (from settings.yaml 'indicators' section).
                Uses defaults if None.

    Returns:
        DataFrame with all original + indicator columns.
    """
    if config is None:
        config = {}

    result = df.copy()

    o = result["open"].values.astype(float)
    h = result["high"].values.astype(float)
    l = result["low"].values.astype(float)
    c = result["close"].values.astype(float)
    v = result.get("volume", result.get("tick_volume", pd.Series(0, index=result.index))).values.astype(float)

    # --- RSI ---
    rsi_period = config.get("rsi_period", 14)
    result["rsi"] = talib.RSI(c, timeperiod=rsi_period)

    # --- MACD ---
    macd_fast = config.get("macd_fast", 12)
    macd_slow = config.get("macd_slow", 26)
    macd_signal_period = config.get("macd_signal", 9)
    macd_line, macd_signal, macd_hist = talib.MACD(
        c, fastperiod=macd_fast, slowperiod=macd_slow, signalperiod=macd_signal_period,
    )
    result["macd"] = macd_line
    result["macd_signal"] = macd_signal
    result["macd_histogram"] = macd_hist

    # --- Bollinger Bands ---
    bb_period = config.get("bb_period", 20)
    bb_std = config.get("bb_std", 2)
    bb_upper, bb_middle, bb_lower = talib.BBANDS(
        c, timeperiod=bb_period, nbdevup=bb_std, nbdevdn=bb_std, matype=0,
    )
    result["bb_upper"] = bb_upper
    result["bb_middle"] = bb_middle
    result["bb_lower"] = bb_lower
    # %B: where price sits within the bands (0 = lower, 1 = upper)
    bb_width = bb_upper - bb_lower
    bb_width_safe = np.where(bb_width == 0, np.nan, bb_width)
    result["bb_pctb"] = (c - bb_lower) / bb_width_safe

    # --- ATR ---
    atr_period = config.get("atr_period", 14)
    result["atr"] = talib.ATR(h, l, c, timeperiod=atr_period)

    # --- EMA (fast and slow) ---
    ema_fast_period = config.get("ema_fast", 50)
    ema_slow_period = config.get("ema_slow", 200)
    result["ema_fast"] = talib.EMA(c, timeperiod=ema_fast_period)
    result["ema_slow"] = talib.EMA(c, timeperiod=ema_slow_period)

    # --- Volume SMA ---
    vol_period = config.get("volume_sma_period", 20)
    result["volume_sma"] = talib.SMA(v, timeperiod=vol_period)
    vol_sma_safe = np.where(result["volume_sma"] == 0, np.nan, result["volume_sma"])
    result["volume_ratio"] = v / vol_sma_safe

    # --- Stochastic Oscillator ---
    stoch_period = config.get("stoch_period", 14)
    stoch_smooth = config.get("stoch_smooth", 3)
    stoch_k, stoch_d = talib.STOCH(
        h, l, c,
        fastk_period=stoch_period, slowk_period=stoch_smooth,
        slowk_matype=0, slowd_period=stoch_smooth, slowd_matype=0,
    )
    result["stoch_k"] = stoch_k
    result["stoch_d"] = stoch_d

    # --- ADX (trend strength) ---
    adx_period = config.get("adx_period", 14)
    result["adx"] = talib.ADX(h, l, c, timeperiod=adx_period)

    # --- Derived features ---
    # Trend direction based on EMAs
    result["trend_bullish"] = result["ema_fast"] > result["ema_slow"]
    result["trend_bearish"] = result["ema_fast"] < result["ema_slow"]

    # ATR percentile rank over last 100 bars (volatility regime)
    atr_series = result["atr"]
    result["atr_rank"] = atr_series.rolling(100).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False,
    )

    return result


def get_indicator_summary(df: pd.DataFrame) -> dict:
    """
    Return a snapshot of the latest indicator values for quick reference.
    """
    if df.empty:
        return {}

    last = df.iloc[-1]
    return {
        "rsi": last.get("rsi"),
        "macd_histogram": last.get("macd_histogram"),
        "bb_pctb": last.get("bb_pctb"),
        "atr": last.get("atr"),
        "ema_fast": last.get("ema_fast"),
        "ema_slow": last.get("ema_slow"),
        "trend_bullish": last.get("trend_bullish"),
        "stoch_k": last.get("stoch_k"),
        "adx": last.get("adx"),
        "volume_ratio": last.get("volume_ratio"),
    }
