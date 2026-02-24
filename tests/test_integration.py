"""
Integration tests — end-to-end pipeline validation.

Tests the full flow: data → patterns → indicators → confluence → signals → backtest → report.
"""

import os
import tempfile
import numpy as np
import pandas as pd
import pytest

from backtest.data_loader import generate_synthetic_data
from backtest.engine import BacktestEngine, BacktestResult
from backtest.report import generate_full_report
from src.indicators import calculate_all as calculate_indicators
from src.patterns import detect_all as detect_patterns
from src.confluence import evaluate_confluence
from src.signals import SignalGenerator, Signal
from src.ml_scorer import MLScorer
from src.risk_manager import RiskManager, TradeRecord
from src.utils import load_config, get_all_symbols


class TestFullPipeline:
    """Test the complete signal generation pipeline end-to-end."""

    def test_data_to_patterns_to_indicators(self):
        """Data → Indicators → Patterns all chain correctly."""
        df = generate_synthetic_data(bars=500, seed=42)
        df_ind = calculate_indicators(df)
        patterns = detect_patterns(df)

        assert len(df_ind) == 500
        assert len(patterns) == 500
        assert "rsi" in df_ind.columns
        assert "dominant_signal" in patterns.columns

    def test_confluence_scoring_end_to_end(self):
        """Full confluence scoring with all 3 timeframes."""
        df_h4 = generate_synthetic_data(bars=300, seed=42)
        df_h1 = generate_synthetic_data(bars=500, seed=43)
        df_m15 = generate_synthetic_data(bars=500, seed=44)

        config = {
            "min_score": 0,
            "weights": {
                "h4_trend_alignment": 30,
                "h1_pattern_indicator": 40,
                "m15_entry_confirmation": 30,
            },
            "_indicator_config": {},
        }
        result = evaluate_confluence(df_h4, df_h1, df_m15, config)
        assert result.total_score >= 0
        assert result.total_score <= 100
        assert result.direction in ("BUY", "SELL", "NONE")

    def test_signal_generator_end_to_end(self, config):
        """SignalGenerator produces valid signals or None."""
        gen = SignalGenerator(config, ml_scorer=None)
        df_h4 = generate_synthetic_data(bars=300, seed=42)
        df_h1 = generate_synthetic_data(bars=500, seed=43)
        df_m15 = generate_synthetic_data(bars=500, seed=44)

        result = gen.generate("EURUSD", df_h4, df_h1, df_m15)
        if result is not None:
            assert isinstance(result, Signal)
            assert result.symbol == "EURUSD"
            assert result.direction in ("BUY", "SELL")
            assert result.atr_value > 0
            assert result.sl_distance > 0

    def test_ml_train_predict_cycle(self):
        """Train model → predict → verify probabilities."""
        ml_config = {
            "ml": {
                "enabled": True,
                "confidence_threshold": 0.5,
                "lookforward_bars": 24,
                "rr_target": 2.0,
                "train_test_split": 0.8,
                "model_path": os.path.join(tempfile.mkdtemp(), "test_int_model.joblib"),
                "xgboost_params": {"n_estimators": 50, "max_depth": 3},
            }
        }
        scorer = MLScorer(ml_config)
        df_h1 = generate_synthetic_data(bars=2000, seed=42)

        metrics = scorer.train(df_h1, None, {})
        if scorer.model is not None:
            features = MLScorer.build_features(df_h1)
            prob = scorer.predict_proba(features)
            assert 0.0 <= prob <= 1.0


class TestBacktestIntegration:
    """Full backtest pipeline: data → engine → report."""

    def test_full_backtest_synthetic(self, config):
        """Run a complete backtest on synthetic data."""
        df_h1 = generate_synthetic_data(bars=3000, seed=42)
        df_h4 = generate_synthetic_data(bars=750, seed=43)

        engine = BacktestEngine(config)
        result = engine.run(df_h1, df_h4, symbol="SYNTHETIC")

        assert isinstance(result, BacktestResult)
        assert len(result.equity_curve) > 0
        assert "total_trades" in result.metrics
        assert "sharpe_ratio" in result.metrics
        assert result.metrics["max_drawdown_pct"] >= 0

    def test_backtest_report_generation(self, config):
        """Report generation creates all expected files."""
        df_h1 = generate_synthetic_data(bars=3000, seed=42)
        engine = BacktestEngine(config)
        result = engine.run(df_h1, symbol="SYNTHETIC")

        output_dir = tempfile.mkdtemp()
        generate_full_report(result, output_dir=output_dir)

        # Check that report files were created
        expected_files = [
            "backtest_equity.png",
            "backtest_drawdown.png",
        ]
        for fname in expected_files:
            path = os.path.join(output_dir, fname)
            assert os.path.exists(path), f"Missing report file: {fname}"

    def test_backtest_with_ml_scorer(self, config):
        """Backtest with ML scorer enabled."""
        ml_config = {**config}
        ml_config["ml"] = {
            "enabled": True,
            "confidence_threshold": 0.3,
            "lookforward_bars": 24,
            "rr_target": 2.0,
            "train_test_split": 0.8,
            "model_path": os.path.join(tempfile.mkdtemp(), "bt_model.joblib"),
            "xgboost_params": {"n_estimators": 50, "max_depth": 3},
        }

        # Train model first
        scorer = MLScorer(ml_config)
        df_train = generate_synthetic_data(bars=2000, seed=42)
        scorer.train(df_train, None, {})

        if scorer.model is not None:
            engine = BacktestEngine(ml_config, ml_scorer=scorer)
            df_h1 = generate_synthetic_data(bars=2000, seed=99)
            result = engine.run(df_h1, symbol="ML_TEST")
            assert isinstance(result, BacktestResult)


class TestRiskManagementIntegration:
    """Risk management integration with full trade lifecycle."""

    def test_kelly_adapts_to_trade_history(self, config):
        """Kelly fraction changes as trade history accumulates."""
        rm = RiskManager(config)
        initial_frac = rm.get_risk_fraction()

        # Simulate a profitable track record
        for _ in range(50):
            rm.record_trade(TradeRecord("EURUSD", "BUY", profit=200.0, risk_amount=100.0))

        adapted_frac = rm.get_risk_fraction()
        # After 50 winning trades, Kelly should suggest higher risk than default
        assert adapted_frac >= initial_frac

    def test_daily_drawdown_stops_trading(self, config):
        """Daily drawdown should halt trading when exceeded."""
        rm = RiskManager(config)
        rm.reset_daily(10000)

        assert rm.check_daily_drawdown(9800) is True   # 2% loss OK
        assert rm.check_daily_drawdown(9600) is False   # 4% loss > 3% limit
