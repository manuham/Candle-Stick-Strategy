"""
Unit tests for walk-forward optimization.
"""

import pytest
from backtest.walk_forward import WalkForwardOptimizer, WalkForwardResult
from backtest.data_loader import generate_synthetic_data


class TestWalkForwardOptimizer:
    def test_run_returns_result(self, config):
        wf = WalkForwardOptimizer(config, train_bars=1500, test_bars=500, retrain_ml=False)
        df_h1 = generate_synthetic_data(bars=5000, seed=42)
        result = wf.run(df_h1, symbol="WF_TEST")
        assert isinstance(result, WalkForwardResult)
        assert len(result.windows) >= 1

    def test_multiple_windows(self, config):
        wf = WalkForwardOptimizer(config, train_bars=1000, test_bars=500, retrain_ml=False)
        df_h1 = generate_synthetic_data(bars=5000, seed=42)
        result = wf.run(df_h1, symbol="WF_TEST")
        # 5000 bars with 1000 train + 500 test = should get multiple windows
        assert len(result.windows) >= 2

    def test_aggregated_trades_from_all_windows(self, config):
        wf = WalkForwardOptimizer(config, train_bars=1000, test_bars=500, retrain_ml=False)
        df_h1 = generate_synthetic_data(bars=5000, seed=42)
        result = wf.run(df_h1, symbol="WF_TEST")
        # Aggregated trades should be sum of all window trades
        total = sum(len(w.result.trades) for w in result.windows if w.result)
        assert len(result.aggregated_trades) == total

    def test_insufficient_data_fallback(self, config):
        wf = WalkForwardOptimizer(config, train_bars=4000, test_bars=2000, retrain_ml=False)
        df_h1 = generate_synthetic_data(bars=500, seed=42)
        result = wf.run(df_h1, symbol="WF_TEST")
        # Should fallback to single backtest
        assert len(result.windows) == 1

    def test_with_h4_data(self, config):
        wf = WalkForwardOptimizer(config, train_bars=1500, test_bars=500, retrain_ml=False)
        df_h1 = generate_synthetic_data(bars=5000, seed=42)
        df_h4 = generate_synthetic_data(bars=1250, seed=43)
        result = wf.run(df_h1, df_h4, symbol="WF_TEST")
        assert isinstance(result, WalkForwardResult)

    def test_aggregated_metrics_present(self, config):
        wf = WalkForwardOptimizer(config, train_bars=1000, test_bars=500, retrain_ml=False)
        df_h1 = generate_synthetic_data(bars=5000, seed=42)
        result = wf.run(df_h1, symbol="WF_TEST")
        assert "total_trades" in result.aggregated_metrics
        assert "sharpe_ratio" in result.aggregated_metrics
