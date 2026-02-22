"""
Risk Manager — Position sizing (Kelly Criterion), drawdown control,
exposure limits, and trailing stop logic.
"""

import logging
from dataclasses import dataclass
from typing import List, Optional

from src.utils import get_correlated_symbols

logger = logging.getLogger("strategy")


@dataclass
class TradeRecord:
    """Lightweight record of a closed trade for Kelly calculation."""
    symbol: str
    direction: str
    profit: float
    risk_amount: float

    @property
    def is_win(self) -> bool:
        return self.profit > 0

    @property
    def return_ratio(self) -> float:
        """Return as a multiple of risk (e.g., 2.0 means 2R)."""
        if self.risk_amount == 0:
            return 0.0
        return self.profit / self.risk_amount


class RiskManager:
    """Manages position sizing, risk limits, and trailing stops."""

    def __init__(self, config: dict):
        risk_cfg = config.get("risk", {})
        kelly_cfg = config.get("kelly", {})

        self.max_risk_per_trade = risk_cfg.get("max_risk_per_trade", 0.01)
        self.max_risk_cap = risk_cfg.get("max_risk_cap", 0.02)
        self.max_daily_drawdown = risk_cfg.get("max_daily_drawdown", 0.03)
        self.max_open_positions = risk_cfg.get("max_open_positions", 5)
        self.max_correlated = risk_cfg.get("max_correlated_positions", 2)
        self.min_risk_reward = risk_cfg.get("min_risk_reward", 2.0)
        self.sl_atr_mult = risk_cfg.get("sl_atr_multiplier", 1.5)
        self.tp_rr_ratio = risk_cfg.get("tp_rr_ratio", 2.0)
        self.spread_filter_mult = risk_cfg.get("spread_filter_multiplier", 2.0)
        self.close_before_eod = risk_cfg.get("close_before_eod_minutes", 30)

        # Trailing stop config
        trail_cfg = risk_cfg.get("trailing", {})
        self.trailing_enabled = trail_cfg.get("enabled", True)
        self.breakeven_rr = trail_cfg.get("breakeven_at_rr", 1.0)
        self.trail_atr_mult = trail_cfg.get("trail_atr_multiplier", 1.0)
        self.trail_activate_rr = trail_cfg.get("trail_activate_rr", 1.5)

        # Kelly settings
        self.kelly_enabled = kelly_cfg.get("enabled", True)
        self.kelly_fraction = kelly_cfg.get("fraction", 0.5)
        self.kelly_min_trades = kelly_cfg.get("min_trades_required", 30)
        self.kelly_lookback = kelly_cfg.get("lookback_trades", 100)
        self.kelly_default_risk = kelly_cfg.get("default_risk", 0.01)

        # State
        self.trade_history: List[TradeRecord] = []
        self.daily_pnl: float = 0.0
        self.daily_start_balance: float = 0.0

    # ------------------------------------------------------------------
    # Kelly Criterion
    # ------------------------------------------------------------------

    def _calculate_kelly(self) -> float:
        """
        Calculate Half Kelly fraction from recent trade history.

        Kelly% = (W * R - L) / R
        Where:
            W = win rate
            R = average win / average loss (reward/risk ratio)
            L = 1 - W

        Returns:
            Recommended risk fraction (as decimal, e.g., 0.015 = 1.5%).
        """
        recent = self.trade_history[-self.kelly_lookback:]

        if len(recent) < self.kelly_min_trades:
            return self.kelly_default_risk

        wins = [t for t in recent if t.is_win]
        losses = [t for t in recent if not t.is_win]

        if not wins or not losses:
            return self.kelly_default_risk

        win_rate = len(wins) / len(recent)
        avg_win = sum(t.profit for t in wins) / len(wins)
        avg_loss = abs(sum(t.profit for t in losses) / len(losses))

        if avg_loss == 0:
            return self.kelly_default_risk

        reward_risk = avg_win / avg_loss
        kelly_full = (win_rate * reward_risk - (1 - win_rate)) / reward_risk

        if kelly_full <= 0:
            logger.warning(f"Kelly is negative ({kelly_full:.4f}) — no edge detected, using minimum risk")
            return self.kelly_default_risk * 0.5

        kelly_adjusted = kelly_full * self.kelly_fraction  # Half Kelly
        return kelly_adjusted

    def get_risk_fraction(self) -> float:
        """Get the current risk fraction per trade, capped at max."""
        if self.kelly_enabled:
            kelly = self._calculate_kelly()
            risk = min(kelly, self.max_risk_cap)
        else:
            risk = self.max_risk_per_trade

        return max(0.001, risk)  # Floor at 0.1%

    # ------------------------------------------------------------------
    # Position Sizing
    # ------------------------------------------------------------------

    def calculate_lot_size(
        self,
        account_balance: float,
        sl_distance_price: float,
        symbol_info: dict,
    ) -> float:
        """
        Calculate lot size based on risk fraction and SL distance.

        Args:
            account_balance: Current account balance.
            sl_distance_price: Absolute price distance to stop loss.
            symbol_info: Dict with trade_tick_value, trade_tick_size, volume_min, etc.

        Returns:
            Lot size rounded to the nearest allowed step.
        """
        risk_fraction = self.get_risk_fraction()
        risk_amount = account_balance * risk_fraction

        tick_value = symbol_info.get("trade_tick_value", 1.0)
        tick_size = symbol_info.get("trade_tick_size", 0.00001)
        volume_min = symbol_info.get("volume_min", 0.01)
        volume_max = symbol_info.get("volume_max", 100.0)
        volume_step = symbol_info.get("volume_step", 0.01)

        if tick_value == 0 or tick_size == 0 or sl_distance_price == 0:
            logger.warning("Invalid symbol info or SL distance for lot calculation")
            return volume_min

        # lot_size = risk_amount / (sl_in_ticks * tick_value)
        sl_in_ticks = sl_distance_price / tick_size
        lot_size = risk_amount / (sl_in_ticks * tick_value)

        # Round to nearest volume step
        lot_size = round(lot_size / volume_step) * volume_step
        lot_size = max(volume_min, min(lot_size, volume_max))

        logger.debug(
            f"Position sizing: balance={account_balance:.2f}, "
            f"risk%={risk_fraction*100:.2f}%, risk$={risk_amount:.2f}, "
            f"SL_dist={sl_distance_price:.5f}, lots={lot_size:.2f}"
        )

        return lot_size

    # ------------------------------------------------------------------
    # Risk Checks
    # ------------------------------------------------------------------

    def check_daily_drawdown(self, current_balance: float) -> bool:
        """
        Check if daily loss limit has been breached.
        Returns True if trading is still allowed, False to stop.
        """
        if self.daily_start_balance == 0:
            self.daily_start_balance = current_balance
            return True

        daily_loss = self.daily_start_balance - current_balance
        daily_loss_pct = daily_loss / self.daily_start_balance

        if daily_loss_pct >= self.max_daily_drawdown:
            logger.warning(
                f"Daily drawdown limit hit: {daily_loss_pct*100:.2f}% "
                f"(limit: {self.max_daily_drawdown*100:.1f}%)"
            )
            return False
        return True

    def check_position_limits(
        self,
        open_positions: list,
        new_symbol: str,
    ) -> bool:
        """
        Check if opening a new position is allowed given current exposure.

        Returns True if the trade is allowed, False otherwise.
        """
        # Max open positions
        if len(open_positions) >= self.max_open_positions:
            logger.info(f"Max open positions ({self.max_open_positions}) reached")
            return False

        # Correlation check
        correlated = get_correlated_symbols(new_symbol)
        correlated_count = sum(
            1 for pos in open_positions
            if pos.get("symbol") in correlated or pos.get("symbol") == new_symbol
        )

        if correlated_count >= self.max_correlated:
            logger.info(
                f"Correlated exposure limit ({self.max_correlated}) "
                f"reached for {new_symbol}"
            )
            return False

        return True

    def check_spread(
        self,
        current_spread: float,
        average_spread: float,
    ) -> bool:
        """Check if the current spread is acceptable."""
        if average_spread == 0:
            return True
        if current_spread > self.spread_filter_mult * average_spread:
            logger.info(
                f"Spread too wide: {current_spread:.1f} > "
                f"{self.spread_filter_mult}x avg ({average_spread:.1f})"
            )
            return False
        return True

    def check_risk_reward(self, signal) -> bool:
        """Verify the signal meets minimum R:R requirements."""
        rr = signal.risk_reward_ratio
        if rr < self.min_risk_reward:
            logger.info(
                f"R:R too low: {rr:.2f} < {self.min_risk_reward}"
            )
            return False
        return True

    # ------------------------------------------------------------------
    # Trailing Stop
    # ------------------------------------------------------------------

    def calculate_trailing_sl(
        self,
        direction: str,
        entry_price: float,
        current_price: float,
        current_sl: float,
        atr_value: float,
    ) -> Optional[float]:
        """
        Calculate new SL level based on trailing stop rules.

        Returns:
            New SL price if it should be moved, None if no change.
        """
        if not self.trailing_enabled or atr_value == 0:
            return None

        sl_distance = abs(entry_price - current_sl)
        if sl_distance == 0:
            return None

        if direction == "BUY":
            current_rr = (current_price - entry_price) / sl_distance
            # Move to breakeven at 1:1
            if current_rr >= self.breakeven_rr and current_sl < entry_price:
                new_sl = entry_price
                logger.debug(f"Trailing: moving SL to breakeven at {new_sl:.5f}")
                return new_sl

            # Trail after activation
            if current_rr >= self.trail_activate_rr:
                trail_sl = current_price - (self.trail_atr_mult * atr_value)
                if trail_sl > current_sl:
                    logger.debug(f"Trailing: moving SL to {trail_sl:.5f}")
                    return trail_sl

        elif direction == "SELL":
            current_rr = (entry_price - current_price) / sl_distance
            if current_rr >= self.breakeven_rr and current_sl > entry_price:
                new_sl = entry_price
                logger.debug(f"Trailing: moving SL to breakeven at {new_sl:.5f}")
                return new_sl

            if current_rr >= self.trail_activate_rr:
                trail_sl = current_price + (self.trail_atr_mult * atr_value)
                if trail_sl < current_sl:
                    logger.debug(f"Trailing: moving SL to {trail_sl:.5f}")
                    return trail_sl

        return None

    # ------------------------------------------------------------------
    # Trade Recording
    # ------------------------------------------------------------------

    def record_trade(self, trade: TradeRecord):
        """Record a completed trade for Kelly calculation."""
        self.trade_history.append(trade)
        self.daily_pnl += trade.profit
        logger.info(
            f"Trade recorded: {trade.symbol} {trade.direction} "
            f"P&L={trade.profit:.2f} ({'WIN' if trade.is_win else 'LOSS'})"
        )

    def reset_daily(self, current_balance: float):
        """Reset daily tracking (call at start of each trading day)."""
        self.daily_pnl = 0.0
        self.daily_start_balance = current_balance
        logger.info(f"Daily reset — starting balance: {current_balance:.2f}")
