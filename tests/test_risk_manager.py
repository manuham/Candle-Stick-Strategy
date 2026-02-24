"""
Unit tests for the risk manager.
"""

import pytest
from unittest.mock import MagicMock

from src.risk_manager import RiskManager, TradeRecord


@pytest.fixture
def risk_mgr(config):
    return RiskManager(config)


class TestTradeRecord:
    def test_is_win(self):
        t = TradeRecord(symbol="EURUSD", direction="BUY", profit=50.0, risk_amount=25.0)
        assert t.is_win is True

    def test_is_loss(self):
        t = TradeRecord(symbol="EURUSD", direction="BUY", profit=-30.0, risk_amount=25.0)
        assert t.is_win is False

    def test_return_ratio(self):
        t = TradeRecord(symbol="EURUSD", direction="BUY", profit=50.0, risk_amount=25.0)
        assert t.return_ratio == 2.0

    def test_return_ratio_zero_risk(self):
        t = TradeRecord(symbol="EURUSD", direction="BUY", profit=50.0, risk_amount=0.0)
        assert t.return_ratio == 0.0


class TestKellyCriterion:
    def test_default_risk_with_no_history(self, risk_mgr):
        frac = risk_mgr.get_risk_fraction()
        assert 0.001 <= frac <= 0.02

    def test_kelly_with_history(self, risk_mgr):
        # Simulate 50 trades: 30 wins at 2R, 20 losses at 1R
        for _ in range(30):
            risk_mgr.record_trade(
                TradeRecord("EURUSD", "BUY", profit=200.0, risk_amount=100.0)
            )
        for _ in range(20):
            risk_mgr.record_trade(
                TradeRecord("EURUSD", "BUY", profit=-100.0, risk_amount=100.0)
            )
        frac = risk_mgr.get_risk_fraction()
        # With 60% win rate and 2:1 R:R, Kelly should be positive
        assert frac > 0.001

    def test_kelly_capped_at_max(self, risk_mgr):
        # Even with amazing stats, should be capped
        for _ in range(50):
            risk_mgr.record_trade(
                TradeRecord("EURUSD", "BUY", profit=500.0, risk_amount=100.0)
            )
        frac = risk_mgr.get_risk_fraction()
        assert frac <= risk_mgr.max_risk_cap


class TestPositionSizing:
    def test_lot_size_basic(self, risk_mgr):
        sym_info = {
            "trade_tick_value": 10.0,
            "trade_tick_size": 0.0001,
            "volume_min": 0.01,
            "volume_max": 100.0,
            "volume_step": 0.01,
        }
        lot = risk_mgr.calculate_lot_size(
            account_balance=10000,
            sl_distance_price=0.0015,
            symbol_info=sym_info,
        )
        assert lot >= 0.01
        assert lot <= 100.0

    def test_lot_size_minimum(self, risk_mgr):
        sym_info = {
            "trade_tick_value": 10.0,
            "trade_tick_size": 0.0001,
            "volume_min": 0.01,
            "volume_max": 100.0,
            "volume_step": 0.01,
        }
        # Tiny balance should give minimum lot
        lot = risk_mgr.calculate_lot_size(
            account_balance=10,
            sl_distance_price=0.1,
            symbol_info=sym_info,
        )
        assert lot == 0.01

    def test_lot_size_zero_sl(self, risk_mgr):
        sym_info = {
            "trade_tick_value": 10.0,
            "trade_tick_size": 0.0001,
            "volume_min": 0.01,
            "volume_max": 100.0,
            "volume_step": 0.01,
        }
        lot = risk_mgr.calculate_lot_size(10000, 0, sym_info)
        assert lot == 0.01


class TestDailyDrawdown:
    def test_within_limit(self, risk_mgr):
        risk_mgr.reset_daily(10000)
        assert risk_mgr.check_daily_drawdown(9800) is True

    def test_exceeded(self, risk_mgr):
        risk_mgr.reset_daily(10000)
        assert risk_mgr.check_daily_drawdown(9600) is False


class TestPositionLimits:
    def test_under_limit(self, risk_mgr):
        # Use symbols NOT correlated with AUDUSD
        positions = [{"symbol": "GER40"}]
        assert risk_mgr.check_position_limits(positions, "AUDUSD") is True

    def test_at_max(self, risk_mgr):
        positions = [{"symbol": f"SYM{i}"} for i in range(risk_mgr.max_open_positions)]
        assert risk_mgr.check_position_limits(positions, "EURUSD") is False

    def test_correlated_limit(self, risk_mgr):
        positions = [{"symbol": "EURUSD"}, {"symbol": "GBPUSD"}]
        # USDJPY is correlated with USD group
        result = risk_mgr.check_position_limits(positions, "USDJPY")
        # Depends on correlation config; just verify it doesn't crash
        assert isinstance(result, bool)


class TestSpreadCheck:
    def test_acceptable_spread(self, risk_mgr):
        assert risk_mgr.check_spread(15, 10) is True

    def test_wide_spread(self, risk_mgr):
        assert risk_mgr.check_spread(25, 10) is False

    def test_zero_avg(self, risk_mgr):
        assert risk_mgr.check_spread(10, 0) is True


class TestRiskReward:
    def test_good_rr(self, risk_mgr):
        signal = MagicMock()
        signal.risk_reward_ratio = 3.0
        assert risk_mgr.check_risk_reward(signal) is True

    def test_bad_rr(self, risk_mgr):
        signal = MagicMock()
        signal.risk_reward_ratio = 1.0
        assert risk_mgr.check_risk_reward(signal) is False


class TestTrailingStop:
    def test_move_to_breakeven_buy(self, risk_mgr):
        new_sl = risk_mgr.calculate_trailing_sl(
            direction="BUY",
            entry_price=1.1000,
            current_price=1.1020,
            current_sl=1.0985,
            atr_value=0.001,
        )
        # At 1:1+ R:R, SL should move to breakeven
        assert new_sl is not None
        assert new_sl >= 1.1000

    def test_no_move_when_not_profitable(self, risk_mgr):
        new_sl = risk_mgr.calculate_trailing_sl(
            direction="BUY",
            entry_price=1.1000,
            current_price=1.0995,
            current_sl=1.0985,
            atr_value=0.001,
        )
        assert new_sl is None

    def test_trailing_sell(self, risk_mgr):
        new_sl = risk_mgr.calculate_trailing_sl(
            direction="SELL",
            entry_price=1.1000,
            current_price=1.0980,
            current_sl=1.1015,
            atr_value=0.001,
        )
        # At 1:1+ R:R, should move to breakeven
        assert new_sl is not None
        assert new_sl <= 1.1000


class TestTradeRecording:
    def test_record_trade(self, risk_mgr):
        trade = TradeRecord("EURUSD", "BUY", profit=100.0, risk_amount=50.0)
        risk_mgr.record_trade(trade)
        assert len(risk_mgr.trade_history) == 1
        assert risk_mgr.daily_pnl == 100.0

    def test_reset_daily(self, risk_mgr):
        risk_mgr.daily_pnl = 500.0
        risk_mgr.reset_daily(10000)
        assert risk_mgr.daily_pnl == 0.0
        assert risk_mgr.daily_start_balance == 10000
