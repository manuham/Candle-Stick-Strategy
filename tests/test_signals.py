"""
Unit tests for signal generation.
"""

import numpy as np
import pandas as pd
import pytest

from src.signals import Signal, SignalGenerator


def _make_h1(bars=300, seed=42):
    rng = np.random.RandomState(seed)
    base = np.linspace(1.1, 1.15, bars)
    noise = rng.normal(0, 0.001, bars)
    closes = base + noise
    opens = closes - rng.normal(0, 0.0005, bars)
    highs = np.maximum(opens, closes) + np.abs(rng.normal(0, 0.0003, bars))
    lows = np.minimum(opens, closes) - np.abs(rng.normal(0, 0.0003, bars))
    volume = rng.poisson(1000, bars).astype(float)
    idx = pd.date_range("2024-01-01", periods=bars, freq="1h", tz="UTC")
    return pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes, "volume": volume},
        index=idx,
    )


class TestSignalDataclass:
    def test_sl_distance(self):
        sig = Signal(
            symbol="EURUSD", direction="BUY", timeframe="H1",
            pattern_name="hammer", pattern_strength=80.0,
            confluence_score=75.0, ml_probability=0.7,
            suggested_sl=1.0985, suggested_tp=1.1030,
            atr_value=0.001, entry_price=1.1000,
        )
        assert abs(sig.sl_distance - 0.0015) < 1e-6

    def test_risk_reward_ratio(self):
        sig = Signal(
            symbol="EURUSD", direction="BUY", timeframe="H1",
            pattern_name="hammer", pattern_strength=80.0,
            confluence_score=75.0, ml_probability=0.7,
            suggested_sl=1.0985, suggested_tp=1.1030,
            atr_value=0.001, entry_price=1.1000,
        )
        assert sig.risk_reward_ratio == pytest.approx(2.0, abs=0.01)

    def test_to_dict(self):
        sig = Signal(
            symbol="EURUSD", direction="BUY", timeframe="H1",
            pattern_name="hammer", pattern_strength=80.0,
            confluence_score=75.0, ml_probability=0.7,
            suggested_sl=1.0985, suggested_tp=1.1030,
            atr_value=0.001, entry_price=1.1000,
        )
        d = sig.to_dict()
        assert d["symbol"] == "EURUSD"
        assert d["direction"] == "BUY"
        assert "timestamp" in d


class TestSignalGenerator:
    def test_generate_returns_none_or_signal(self, config):
        gen = SignalGenerator(config, ml_scorer=None)
        df_h1 = _make_h1(300)
        result = gen.generate("EURUSD", None, df_h1, None)
        assert result is None or isinstance(result, Signal)

    def test_generate_with_all_timeframes(self, config):
        gen = SignalGenerator(config, ml_scorer=None)
        df_h1 = _make_h1(300, seed=42)
        df_h4 = _make_h1(300, seed=43)
        df_m15 = _make_h1(300, seed=44)
        result = gen.generate("EURUSD", df_h4, df_h1, df_m15)
        assert result is None or isinstance(result, Signal)

    def test_scan_all_symbols(self, config):
        gen = SignalGenerator(config, ml_scorer=None)
        data = {
            "EURUSD": {
                "trend": _make_h1(300, seed=1),
                "signal": _make_h1(300, seed=2),
                "entry": _make_h1(300, seed=3),
            },
            "GBPUSD": {
                "trend": _make_h1(300, seed=4),
                "signal": _make_h1(300, seed=5),
                "entry": _make_h1(300, seed=6),
            },
        }
        signals = gen.scan_all_symbols(data)
        assert isinstance(signals, list)
        for s in signals:
            assert isinstance(s, Signal)
