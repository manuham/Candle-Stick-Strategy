"""
Backtest Data Loader — Load historical data from CSV files or MT5.
"""

import logging
import os
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger("strategy")


def load_csv(filepath: str) -> Optional[pd.DataFrame]:
    """
    Load OHLCV data from a CSV file.

    Expected columns: time (or date/datetime), open, high, low, close, volume.
    """
    path = Path(filepath)
    if not path.exists():
        logger.error(f"CSV file not found: {filepath}")
        return None

    df = pd.read_csv(filepath)

    # Normalise column names
    df.columns = [c.strip().lower() for c in df.columns]

    # Identify time column
    time_col = None
    for candidate in ["time", "date", "datetime", "timestamp"]:
        if candidate in df.columns:
            time_col = candidate
            break

    if time_col is None:
        logger.error("No time/date column found in CSV")
        return None

    df[time_col] = pd.to_datetime(df[time_col], utc=True)
    df.set_index(time_col, inplace=True)
    df.index.name = "time"

    # Ensure required columns
    required = ["open", "high", "low", "close"]
    for col in required:
        if col not in df.columns:
            logger.error(f"Missing required column: {col}")
            return None

    if "volume" not in df.columns:
        if "tick_volume" in df.columns:
            df.rename(columns={"tick_volume": "volume"}, inplace=True)
        else:
            df["volume"] = 0

    df.sort_index(inplace=True)
    logger.info(f"Loaded {len(df)} bars from {filepath}")
    return df


def fetch_and_cache(
    data_engine,
    symbol: str,
    timeframe: str,
    count: int,
    cache_dir: str = "data",
) -> Optional[pd.DataFrame]:
    """
    Fetch data from MT5 and cache to CSV for future backtests.

    Returns DataFrame or None.
    """
    cache_file = os.path.join(cache_dir, f"{symbol}_{timeframe}_{count}.csv")

    # Try cache first
    if os.path.exists(cache_file):
        df = load_csv(cache_file)
        if df is not None and len(df) >= count * 0.9:
            logger.info(f"Using cached data: {cache_file}")
            return df

    # Fetch from MT5
    df = data_engine.get_rates(symbol, timeframe, count)
    if df is None:
        return None

    # Save to cache
    os.makedirs(cache_dir, exist_ok=True)
    df.to_csv(cache_file)
    logger.info(f"Cached {len(df)} bars to {cache_file}")

    return df


def load_multi_tf(
    data_engine,
    symbol: str,
    config: dict,
    cache_dir: str = "data",
) -> dict:
    """
    Load multi-timeframe data for backtesting.

    Returns dict with keys 'trend', 'signal', 'entry'.
    """
    tf_config = config.get("timeframes", {})
    bar_counts = config.get("bar_counts", {})

    result = {}
    for role, tf_str in tf_config.items():
        count = bar_counts.get(role, 500)

        if data_engine is not None and data_engine.connected:
            result[role] = fetch_and_cache(
                data_engine, symbol, tf_str, count, cache_dir,
            )
        else:
            # Try loading from cache
            cache_file = os.path.join(cache_dir, f"{symbol}_{tf_str}_{count}.csv")
            result[role] = load_csv(cache_file)

    return result


def generate_synthetic_data(
    bars: int = 5000,
    start_price: float = 1.1000,
    volatility: float = 0.0005,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate synthetic OHLCV data for testing when no real data is available.
    Uses a random walk with realistic candle structure.
    """
    import numpy as np

    rng = np.random.RandomState(seed)
    prices = [start_price]

    for _ in range(bars - 1):
        change = rng.normal(0, volatility)
        prices.append(prices[-1] + change)

    prices = np.array(prices)

    # Create OHLCV from price path
    opens = prices.copy()
    # Add intra-bar noise
    noise = rng.normal(0, volatility * 0.5, bars)
    closes = opens + noise

    highs = np.maximum(opens, closes) + np.abs(rng.normal(0, volatility * 0.3, bars))
    lows = np.minimum(opens, closes) - np.abs(rng.normal(0, volatility * 0.3, bars))
    volume = rng.poisson(1000, bars).astype(float)

    # Generate hourly timestamps
    time_index = pd.date_range(
        start="2023-01-01", periods=bars, freq="1h", tz="UTC",
    )

    df = pd.DataFrame({
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": volume,
    }, index=time_index)
    df.index.name = "time"

    return df
