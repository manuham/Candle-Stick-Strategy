"""
Unit tests for technical indicator calculations.
"""

import numpy as np
import pandas as pd
import pytest

from src.indicators import calculate_all, get_indicator_summary


class TestCalculateAll:
    def test_adds_rsi_column(self, sample_ohlcv):
        result = calculate_all(sample_ohlcv)
        assert "rsi" in result.columns

    def test_adds_macd_columns(self, sample_ohlcv):
        result = calculate_all(sample_ohlcv)
        assert "macd" in result.columns
        assert "macd_signal" in result.columns
        assert "macd_histogram" in result.columns

    def test_adds_bollinger_bands(self, sample_ohlcv):
        result = calculate_all(sample_ohlcv)
        assert "bb_upper" in result.columns
        assert "bb_middle" in result.columns
        assert "bb_lower" in result.columns
        assert "bb_pctb" in result.columns

    def test_adds_atr(self, sample_ohlcv):
        result = calculate_all(sample_ohlcv)
        assert "atr" in result.columns
        # ATR should be non-negative where it's not NaN
        valid = result["atr"].dropna()
        assert (valid >= 0).all()

    def test_adds_ema(self, sample_ohlcv):
        result = calculate_all(sample_ohlcv)
        assert "ema_fast" in result.columns
        assert "ema_slow" in result.columns

    def test_adds_stochastic(self, sample_ohlcv):
        result = calculate_all(sample_ohlcv)
        assert "stoch_k" in result.columns
        assert "stoch_d" in result.columns

    def test_adds_adx(self, sample_ohlcv):
        result = calculate_all(sample_ohlcv)
        assert "adx" in result.columns

    def test_adds_volume_ratio(self, sample_ohlcv):
        result = calculate_all(sample_ohlcv)
        assert "volume_sma" in result.columns
        assert "volume_ratio" in result.columns

    def test_adds_trend_flags(self, sample_ohlcv):
        result = calculate_all(sample_ohlcv)
        assert "trend_bullish" in result.columns
        assert "trend_bearish" in result.columns

    def test_rsi_range(self, sample_ohlcv):
        result = calculate_all(sample_ohlcv)
        valid_rsi = result["rsi"].dropna()
        assert (valid_rsi >= 0).all()
        assert (valid_rsi <= 100).all()

    def test_bb_pctb_upper_above_lower(self, sample_ohlcv):
        result = calculate_all(sample_ohlcv)
        valid = result[["bb_upper", "bb_lower"]].dropna()
        assert (valid["bb_upper"] >= valid["bb_lower"]).all()

    def test_preserves_original_columns(self, sample_ohlcv):
        result = calculate_all(sample_ohlcv)
        for col in ["open", "high", "low", "close", "volume"]:
            assert col in result.columns

    def test_custom_config(self, sample_ohlcv):
        cfg = {"rsi_period": 7, "atr_period": 7}
        result = calculate_all(sample_ohlcv, cfg)
        # With shorter period, fewer NaN values at the start
        assert result["rsi"].notna().sum() > 0

    def test_output_length(self, sample_ohlcv):
        result = calculate_all(sample_ohlcv)
        assert len(result) == len(sample_ohlcv)


class TestGetIndicatorSummary:
    def test_returns_dict(self, sample_ohlcv):
        df = calculate_all(sample_ohlcv)
        summary = get_indicator_summary(df)
        assert isinstance(summary, dict)
        assert "rsi" in summary
        assert "atr" in summary

    def test_empty_dataframe(self):
        df = pd.DataFrame()
        summary = get_indicator_summary(df)
        assert summary == {}
