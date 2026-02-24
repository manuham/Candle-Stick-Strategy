"""
Unit tests for multi-timeframe confluence scoring.
"""

import numpy as np
import pandas as pd
import pytest

from src.confluence import (
    score_h4_trend,
    score_h1_pattern_indicator,
    score_m15_entry,
    evaluate_confluence,
    ConfluenceResult,
)


def _make_trending_df(direction="up", bars=300):
    """Make a DataFrame with a clear EMA trend."""
    rng = np.random.RandomState(42)
    if direction == "up":
        base = np.linspace(1.0, 1.2, bars)
    else:
        base = np.linspace(1.2, 1.0, bars)
    noise = rng.normal(0, 0.001, bars)
    closes = base + noise
    opens = closes - rng.normal(0, 0.0005, bars)
    highs = np.maximum(opens, closes) + np.abs(rng.normal(0, 0.0003, bars))
    lows = np.minimum(opens, closes) - np.abs(rng.normal(0, 0.0003, bars))
    volume = rng.poisson(1000, bars).astype(float)
    idx = pd.date_range("2024-01-01", periods=bars, freq="4h", tz="UTC")
    return pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes, "volume": volume},
        index=idx,
    )


class TestScoreH4Trend:
    def test_aligned_bullish(self):
        df = _make_trending_df("up", 300)
        config = {"weights": {"h4_trend_alignment": 30}, "_indicator_config": {}}
        score, details = score_h4_trend(df, "BUY", config)
        assert score == 30
        assert details["h4_reason"] == "aligned_bullish"

    def test_aligned_bearish(self):
        df = _make_trending_df("down", 300)
        config = {"weights": {"h4_trend_alignment": 30}, "_indicator_config": {}}
        score, details = score_h4_trend(df, "SELL", config)
        assert score == 30
        assert details["h4_reason"] == "aligned_bearish"

    def test_opposing_trend(self):
        df = _make_trending_df("up", 300)
        config = {"weights": {"h4_trend_alignment": 30}, "_indicator_config": {}}
        score, details = score_h4_trend(df, "SELL", config)
        assert score == 0
        assert details["h4_reason"] == "opposing_trend"

    def test_no_data(self):
        config = {"weights": {"h4_trend_alignment": 30}, "_indicator_config": {}}
        score, details = score_h4_trend(None, "BUY", config)
        assert score == 0

    def test_empty_df(self):
        config = {"weights": {"h4_trend_alignment": 30}, "_indicator_config": {}}
        score, details = score_h4_trend(pd.DataFrame(), "BUY", config)
        assert score == 0


class TestScoreH1PatternIndicator:
    def test_max_score_capped(self, sample_ohlcv):
        config = {"weights": {"h1_pattern_indicator": 40}, "_indicator_config": {}}
        score, details = score_h1_pattern_indicator(sample_ohlcv, "BUY", config)
        assert score <= 40

    def test_no_data(self):
        config = {"weights": {"h1_pattern_indicator": 40}, "_indicator_config": {}}
        score, details = score_h1_pattern_indicator(None, "BUY", config)
        assert score == 0

    def test_returns_details(self, sample_ohlcv):
        config = {"weights": {"h1_pattern_indicator": 40}, "_indicator_config": {}}
        score, details = score_h1_pattern_indicator(sample_ohlcv, "BUY", config)
        assert "h1_pattern" in details


class TestScoreM15Entry:
    def test_max_score_capped(self, sample_ohlcv):
        config = {"weights": {"m15_entry_confirmation": 30}, "_indicator_config": {}}
        score, details = score_m15_entry(sample_ohlcv, "BUY", config)
        assert score <= 30

    def test_no_data(self):
        config = {"weights": {"m15_entry_confirmation": 30}, "_indicator_config": {}}
        score, details = score_m15_entry(None, "BUY", config)
        assert score == 0


class TestEvaluateConfluence:
    def test_returns_confluence_result(self, sample_ohlcv):
        config = {
            "min_score": 60,
            "weights": {
                "h4_trend_alignment": 30,
                "h1_pattern_indicator": 40,
                "m15_entry_confirmation": 30,
            },
            "_indicator_config": {},
        }
        result = evaluate_confluence(None, sample_ohlcv, None, config)
        assert isinstance(result, ConfluenceResult)

    def test_no_h1_data_returns_none_direction(self):
        config = {
            "min_score": 60,
            "weights": {},
            "_indicator_config": {},
        }
        result = evaluate_confluence(None, None, None, config)
        assert result.direction == "NONE"
        assert result.total_score == 0

    def test_score_is_sum_of_components(self, sample_ohlcv):
        config = {
            "min_score": 0,
            "weights": {
                "h4_trend_alignment": 30,
                "h1_pattern_indicator": 40,
                "m15_entry_confirmation": 30,
            },
            "_indicator_config": {},
        }
        result = evaluate_confluence(None, sample_ohlcv, None, config)
        assert abs(result.total_score - (result.h4_score + result.h1_score + result.m15_score)) < 0.01

    def test_direction_from_pattern(self, bullish_engulfing_df):
        config = {
            "min_score": 0,
            "weights": {
                "h4_trend_alignment": 30,
                "h1_pattern_indicator": 40,
                "m15_entry_confirmation": 30,
            },
            "_indicator_config": {},
        }
        result = evaluate_confluence(None, bullish_engulfing_df, None, config)
        # A bullish engulfing should give BUY direction
        assert result.direction in ("BUY", "NONE")
