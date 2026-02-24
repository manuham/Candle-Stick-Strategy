"""
Unit tests for the backtesting engine.
"""

import numpy as np
import pandas as pd
import pytest

from backtest.engine import BacktestEngine, BacktestResult, BacktestTrade
from backtest.data_loader import generate_synthetic_data


class TestBacktestEngine:
    def test_run_returns_result(self, config):
        engine = BacktestEngine(config)
        df_h1 = generate_synthetic_data(bars=2000, seed=42)
        df_h4 = generate_synthetic_data(bars=500, seed=43)
        result = engine.run(df_h1, df_h4, symbol="TEST")
        assert isinstance(result, BacktestResult)

    def test_equity_curve_exists(self, config):
        engine = BacktestEngine(config)
        df_h1 = generate_synthetic_data(bars=2000, seed=42)
        result = engine.run(df_h1, symbol="TEST")
        assert isinstance(result.equity_curve, pd.Series)
        assert len(result.equity_curve) > 0

    def test_initial_balance(self, config):
        engine = BacktestEngine(config)
        df_h1 = generate_synthetic_data(bars=2000, seed=42)
        result = engine.run(df_h1, symbol="TEST")
        # First equity value should be near initial balance
        assert abs(result.equity_curve.iloc[0] - config["backtest"]["initial_balance"]) < 100

    def test_metrics_computed(self, config):
        engine = BacktestEngine(config)
        df_h1 = generate_synthetic_data(bars=2000, seed=42)
        result = engine.run(df_h1, symbol="TEST")
        m = result.metrics
        assert "total_trades" in m
        assert "win_rate" in m
        assert "sharpe_ratio" in m
        assert "max_drawdown_pct" in m
        assert "profit_factor" in m

    def test_no_trades_metrics(self, config):
        """If no patterns fire, we should still get valid metrics."""
        engine = BacktestEngine(config)
        # Very short data — may not produce any patterns after warmup
        df_h1 = generate_synthetic_data(bars=250, seed=42)
        result = engine.run(df_h1, symbol="TEST")
        assert result.metrics["total_trades"] >= 0

    def test_trades_have_required_fields(self, config):
        engine = BacktestEngine(config)
        df_h1 = generate_synthetic_data(bars=3000, seed=42)
        result = engine.run(df_h1, symbol="TEST")
        for trade in result.trades:
            assert isinstance(trade, BacktestTrade)
            assert trade.symbol == "TEST"
            assert trade.direction in ("BUY", "SELL")
            assert trade.exit_reason in ("TP", "SL", "TRAILING", "EOD", "SIGNAL")

    def test_with_h4_context(self, config):
        engine = BacktestEngine(config)
        df_h1 = generate_synthetic_data(bars=2000, seed=42)
        df_h4 = generate_synthetic_data(bars=500, seed=43)
        result = engine.run(df_h1, df_h4, symbol="TEST")
        assert isinstance(result, BacktestResult)


class TestMetricsCalculation:
    def test_empty_trades(self):
        metrics = BacktestEngine._calculate_metrics(
            [], pd.Series([10000, 10000], name="equity")
        )
        assert metrics["total_trades"] == 0
        assert metrics["win_rate"] == 0

    def test_win_rate_correct(self):
        trades = [
            BacktestTrade(
                symbol="T", direction="BUY", entry_price=1.0, exit_price=1.01,
                entry_time=pd.Timestamp("2024-01-01"), exit_time=pd.Timestamp("2024-01-02"),
                lot_size=0.1, sl=0.99, tp=1.01, pnl_pips=0.01,
                pnl_dollars=100.0, exit_reason="TP", pattern_name="hammer",
                confluence_score=70, ml_probability=0.7,
            ),
            BacktestTrade(
                symbol="T", direction="BUY", entry_price=1.0, exit_price=0.99,
                entry_time=pd.Timestamp("2024-01-03"), exit_time=pd.Timestamp("2024-01-04"),
                lot_size=0.1, sl=0.99, tp=1.01, pnl_pips=-0.01,
                pnl_dollars=-100.0, exit_reason="SL", pattern_name="hammer",
                confluence_score=65, ml_probability=0.6,
            ),
        ]
        eq = pd.Series([10000, 10100, 10000], name="equity")
        metrics = BacktestEngine._calculate_metrics(trades, eq)
        assert metrics["win_rate"] == 0.5
        assert metrics["total_trades"] == 2
