"""
Unit tests for Monte Carlo simulation.
"""

import os
import tempfile
import numpy as np
import pytest

from backtest.monte_carlo import MonteCarloSimulator, MonteCarloResult


class TestMonteCarloSimulator:
    def test_run_returns_result(self):
        mc = MonteCarloSimulator(initial_balance=10000)
        pnls = [100, -50, 75, -30, 200, -80, 150, -40, 60, -20]
        result = mc.run(pnls, iterations=1000)
        assert isinstance(result, MonteCarloResult)
        assert result.iterations == 1000

    def test_arrays_correct_length(self):
        mc = MonteCarloSimulator(initial_balance=10000)
        pnls = [100, -50, 75, -30, 200]
        result = mc.run(pnls, iterations=500)
        assert len(result.final_equities) == 500
        assert len(result.max_drawdowns) == 500
        assert len(result.sharpe_ratios) == 500

    def test_confidence_intervals_structure(self):
        mc = MonteCarloSimulator(initial_balance=10000)
        pnls = [100, -50, 75, -30, 200, -80]
        result = mc.run(pnls, iterations=1000)
        ci = result.confidence_intervals
        assert "final_equity" in ci
        assert "max_drawdown" in ci
        assert "sharpe_ratio" in ci
        assert "5th" in ci["final_equity"]
        assert "95th" in ci["final_equity"]
        assert "50th" in ci["final_equity"]

    def test_profitable_trades_give_positive_median(self):
        mc = MonteCarloSimulator(initial_balance=10000)
        pnls = [100, 200, 150, 50, 300, 100]  # All positive
        result = mc.run(pnls, iterations=1000)
        assert result.confidence_intervals["final_equity"]["50th"] > 10000

    def test_losing_trades_give_negative_median(self):
        mc = MonteCarloSimulator(initial_balance=10000)
        pnls = [-100, -200, -150, -50, -300, -100]  # All negative
        result = mc.run(pnls, iterations=1000)
        assert result.confidence_intervals["final_equity"]["50th"] < 10000

    def test_empty_trades(self):
        mc = MonteCarloSimulator(initial_balance=10000)
        result = mc.run([], iterations=1000)
        assert result.iterations == 0
        assert len(result.final_equities) == 0

    def test_percentiles_present(self):
        mc = MonteCarloSimulator(initial_balance=10000)
        pnls = [100, -50, 75, -30, 200]
        result = mc.run(pnls, iterations=1000)
        pct = result.percentiles
        assert "return_5th" in pct
        assert "return_95th" in pct
        assert "prob_profitable" in pct

    def test_reproducible_with_seed(self):
        mc1 = MonteCarloSimulator(initial_balance=10000, seed=42)
        mc2 = MonteCarloSimulator(initial_balance=10000, seed=42)
        pnls = [100, -50, 75, -30, 200]
        r1 = mc1.run(pnls, iterations=100)
        r2 = mc2.run(pnls, iterations=100)
        np.testing.assert_array_equal(r1.final_equities, r2.final_equities)

    def test_plot_does_not_crash(self):
        mc = MonteCarloSimulator(initial_balance=10000)
        pnls = [100, -50, 75, -30, 200, -80]
        result = mc.run(pnls, iterations=100)
        path = os.path.join(tempfile.mkdtemp(), "mc_test.png")
        MonteCarloSimulator.plot_equity_distribution(result, save_path=path)
        assert os.path.exists(path)

    def test_drawdown_non_negative(self):
        mc = MonteCarloSimulator(initial_balance=10000)
        pnls = [100, -50, 75, -30, 200]
        result = mc.run(pnls, iterations=1000)
        assert (result.max_drawdowns >= 0).all()
