"""
Shared fixtures for the test suite.
"""

import pytest
import numpy as np
import pandas as pd

from src.utils import load_config


@pytest.fixture
def config():
    """Load the project config."""
    return load_config("config/settings.yaml")


@pytest.fixture
def sample_ohlcv():
    """Generate a small OHLCV DataFrame for unit tests (100 bars)."""
    rng = np.random.RandomState(42)
    n = 100
    prices = np.cumsum(rng.normal(0, 0.0005, n)) + 1.1

    opens = prices.copy()
    noise = rng.normal(0, 0.00025, n)
    closes = opens + noise
    highs = np.maximum(opens, closes) + np.abs(rng.normal(0, 0.00015, n))
    lows = np.minimum(opens, closes) - np.abs(rng.normal(0, 0.00015, n))
    volume = rng.poisson(1000, n).astype(float)

    idx = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    return pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes, "volume": volume},
        index=idx,
    )


@pytest.fixture
def large_ohlcv():
    """Generate a larger OHLCV DataFrame (5000 bars) for backtest tests."""
    rng = np.random.RandomState(42)
    n = 5000
    prices = np.cumsum(rng.normal(0, 0.0005, n)) + 1.1

    opens = prices.copy()
    noise = rng.normal(0, 0.00025, n)
    closes = opens + noise
    highs = np.maximum(opens, closes) + np.abs(rng.normal(0, 0.00015, n))
    lows = np.minimum(opens, closes) - np.abs(rng.normal(0, 0.00015, n))
    volume = rng.poisson(1000, n).astype(float)

    idx = pd.date_range("2023-01-01", periods=n, freq="1h", tz="UTC")
    return pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes, "volume": volume},
        index=idx,
    )


@pytest.fixture
def bullish_engulfing_df():
    """DataFrame with a clear bullish engulfing pattern at the last bar."""
    idx = pd.date_range("2024-01-01", periods=10, freq="1h", tz="UTC")
    data = {
        "open":  [1.10, 1.10, 1.10, 1.10, 1.10, 1.10, 1.09, 1.08, 1.075, 1.065],
        "high":  [1.11, 1.11, 1.11, 1.11, 1.11, 1.11, 1.10, 1.09, 1.080, 1.085],
        "low":   [1.09, 1.09, 1.09, 1.09, 1.09, 1.09, 1.08, 1.07, 1.065, 1.060],
        "close": [1.10, 1.10, 1.10, 1.10, 1.10, 1.10, 1.08, 1.07, 1.068, 1.082],
        "volume": [100] * 10,
    }
    return pd.DataFrame(data, index=idx)


@pytest.fixture
def bearish_engulfing_df():
    """DataFrame with a clear bearish engulfing pattern at the last bar."""
    idx = pd.date_range("2024-01-01", periods=10, freq="1h", tz="UTC")
    data = {
        "open":  [1.10, 1.10, 1.10, 1.10, 1.10, 1.10, 1.11, 1.12, 1.125, 1.135],
        "high":  [1.11, 1.11, 1.11, 1.11, 1.11, 1.11, 1.12, 1.13, 1.135, 1.140],
        "low":   [1.09, 1.09, 1.09, 1.09, 1.09, 1.09, 1.10, 1.11, 1.120, 1.118],
        "close": [1.10, 1.10, 1.10, 1.10, 1.10, 1.10, 1.12, 1.13, 1.132, 1.120],
        "volume": [100] * 10,
    }
    return pd.DataFrame(data, index=idx)
