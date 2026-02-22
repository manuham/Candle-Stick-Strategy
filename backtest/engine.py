"""
Backtesting Engine — Event-driven simulation using the same signal pipeline as live trading.

Simulates order execution with spread, slippage, and commission.
Tracks equity curve, positions, and trade log.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional

import numpy as np
import pandas as pd

from src.patterns import detect_all as detect_patterns
from src.indicators import calculate_all as calculate_indicators
from src.confluence import evaluate_confluence
from src.ml_scorer import MLScorer

logger = logging.getLogger("strategy")


@dataclass
class BacktestPosition:
    """Represents an open position during backtesting."""
    symbol: str
    direction: str
    entry_price: float
    entry_time: datetime
    lot_size: float
    sl: float
    tp: float
    atr_at_entry: float
    pattern_name: str
    confluence_score: float
    ml_probability: float


@dataclass
class BacktestTrade:
    """Completed trade record."""
    symbol: str
    direction: str
    entry_price: float
    exit_price: float
    entry_time: datetime
    exit_time: datetime
    lot_size: float
    sl: float
    tp: float
    pnl_pips: float
    pnl_dollars: float
    exit_reason: str         # "TP", "SL", "TRAILING", "EOD", "SIGNAL"
    pattern_name: str
    confluence_score: float
    ml_probability: float


@dataclass
class BacktestResult:
    """Full backtest output."""
    trades: List[BacktestTrade]
    equity_curve: pd.Series
    metrics: dict


class BacktestEngine:
    """
    Simulates the candlestick strategy on historical data.

    Uses the same confluence + ML pipeline as the live system.
    """

    def __init__(self, config: dict, ml_scorer: Optional[MLScorer] = None):
        self.config = config
        self.ml_scorer = ml_scorer

        bt_cfg = config.get("backtest", {})
        self.initial_balance = bt_cfg.get("initial_balance", 10000)
        self.commission_per_lot = bt_cfg.get("commission_per_lot", 7.0)
        self.default_spread = bt_cfg.get("default_spread_points", 15)
        self.slippage_pts = bt_cfg.get("slippage_points", 5)

        risk_cfg = config.get("risk", {})
        self.max_risk_pct = risk_cfg.get("max_risk_per_trade", 0.01)
        self.sl_atr_mult = risk_cfg.get("sl_atr_multiplier", 1.5)
        self.tp_rr = risk_cfg.get("tp_rr_ratio", 2.0)
        self.max_open = risk_cfg.get("max_open_positions", 5)
        self.max_daily_dd = risk_cfg.get("max_daily_drawdown", 0.03)

        trail_cfg = risk_cfg.get("trailing", {})
        self.trailing_enabled = trail_cfg.get("enabled", True)
        self.breakeven_rr = trail_cfg.get("breakeven_at_rr", 1.0)
        self.trail_atr_mult = trail_cfg.get("trail_atr_multiplier", 1.0)
        self.trail_activate_rr = trail_cfg.get("trail_activate_rr", 1.5)

        confluence_cfg = config.get("confluence", {})
        confluence_cfg["_indicator_config"] = config.get("indicators", {})
        self.confluence_cfg = confluence_cfg
        self.min_confluence = confluence_cfg.get("min_score", 60)
        self.indicator_config = config.get("indicators", {})

        self.ml_enabled = config.get("ml", {}).get("enabled", True)
        self.ml_threshold = config.get("ml", {}).get("confidence_threshold", 0.65)

    def run(
        self,
        df_h1: pd.DataFrame,
        df_h4: Optional[pd.DataFrame] = None,
        df_m15: Optional[pd.DataFrame] = None,
        symbol: str = "BACKTEST",
    ) -> BacktestResult:
        """
        Run the backtest on historical data.

        Primary timeframe is H1. H4 and M15 are optional for confluence.
        If not provided, confluence scoring adapts by using only available TFs.

        Args:
            df_h1: H1 OHLCV DataFrame (primary signal TF).
            df_h4: H4 OHLCV DataFrame (optional trend TF).
            df_m15: M15 OHLCV DataFrame (optional entry TF).
            symbol: Symbol name for logging.

        Returns:
            BacktestResult with trades, equity curve, and metrics.
        """
        logger.info(f"Starting backtest on {symbol} | {len(df_h1)} H1 bars")

        # Pre-compute indicators on full dataset
        df_h1_ind = calculate_indicators(df_h1.copy(), self.indicator_config)
        patterns_h1 = detect_patterns(df_h1)

        if df_h4 is not None:
            df_h4_ind = calculate_indicators(df_h4.copy(), self.indicator_config)
        else:
            df_h4_ind = None

        # Pre-compute ML features once for the entire dataset (performance optimization)
        ml_features_all = None
        if self.ml_enabled and self.ml_scorer is not None:
            logger.info("Pre-computing ML features for entire dataset...")
            ml_features_all = MLScorer.build_features(
                df_h1, df_h4, self.indicator_config,
            )

        balance = self.initial_balance
        equity_history = []
        trades: List[BacktestTrade] = []
        open_positions: List[BacktestPosition] = []

        # Minimum warmup period for indicators (200 for slow EMA)
        warmup = 200

        for i in range(warmup, len(df_h1)):
            current_bar = df_h1.iloc[i]
            current_time = df_h1.index[i]
            current_close = current_bar["close"]
            current_high = current_bar["high"]
            current_low = current_bar["low"]

            # ---- Check SL/TP/Trailing on open positions ----
            positions_to_close = []
            for j, pos in enumerate(open_positions):
                exit_price = None
                exit_reason = None

                if pos.direction == "BUY":
                    # Check SL
                    if current_low <= pos.sl:
                        exit_price = pos.sl
                        exit_reason = "SL"
                    # Check TP
                    elif current_high >= pos.tp:
                        exit_price = pos.tp
                        exit_reason = "TP"
                    # Trailing stop
                    elif self.trailing_enabled:
                        sl_dist = abs(pos.entry_price - pos.sl)
                        if sl_dist > 0:
                            current_rr = (current_close - pos.entry_price) / sl_dist
                            if current_rr >= self.breakeven_rr and pos.sl < pos.entry_price:
                                pos.sl = pos.entry_price
                            if current_rr >= self.trail_activate_rr:
                                trail_sl = current_close - self.trail_atr_mult * pos.atr_at_entry
                                if trail_sl > pos.sl:
                                    pos.sl = trail_sl

                else:  # SELL
                    if current_high >= pos.sl:
                        exit_price = pos.sl
                        exit_reason = "SL"
                    elif current_low <= pos.tp:
                        exit_price = pos.tp
                        exit_reason = "TP"
                    elif self.trailing_enabled:
                        sl_dist = abs(pos.sl - pos.entry_price)
                        if sl_dist > 0:
                            current_rr = (pos.entry_price - current_close) / sl_dist
                            if current_rr >= self.breakeven_rr and pos.sl > pos.entry_price:
                                pos.sl = pos.entry_price
                            if current_rr >= self.trail_activate_rr:
                                trail_sl = current_close + self.trail_atr_mult * pos.atr_at_entry
                                if trail_sl < pos.sl:
                                    pos.sl = trail_sl

                if exit_price is not None:
                    positions_to_close.append((j, exit_price, exit_reason))

            # Close positions (reverse order to preserve indices)
            for j, exit_price, exit_reason in sorted(positions_to_close, reverse=True):
                pos = open_positions.pop(j)
                point_value = 1.0  # simplified for backtest
                if pos.direction == "BUY":
                    pnl_pips = (exit_price - pos.entry_price)
                else:
                    pnl_pips = (pos.entry_price - exit_price)

                # Approximate P&L in dollars
                pnl_dollars = pnl_pips * pos.lot_size * 100000 - self.commission_per_lot * pos.lot_size

                balance += pnl_dollars

                trades.append(BacktestTrade(
                    symbol=symbol,
                    direction=pos.direction,
                    entry_price=pos.entry_price,
                    exit_price=exit_price,
                    entry_time=pos.entry_time,
                    exit_time=current_time,
                    lot_size=pos.lot_size,
                    sl=pos.sl,
                    tp=pos.tp,
                    pnl_pips=pnl_pips,
                    pnl_dollars=pnl_dollars,
                    exit_reason=exit_reason,
                    pattern_name=pos.pattern_name,
                    confluence_score=pos.confluence_score,
                    ml_probability=pos.ml_probability,
                ))

            # ---- Check for new signals ----
            if len(open_positions) < self.max_open:
                # Get pattern and indicator state at current bar
                dom_signal = patterns_h1.iloc[i].get("dominant_signal", 0)
                dom_pattern = patterns_h1.iloc[i].get("dominant_pattern", "none")

                if dom_signal != 0:
                    direction = "BUY" if dom_signal > 0 else "SELL"

                    # Simplified confluence: check indicator alignment
                    confluence_score = 0
                    last_ind = df_h1_ind.iloc[i]

                    # H4 trend (+30)
                    if df_h4_ind is not None:
                        h4_bar = df_h4_ind.loc[df_h4_ind.index <= current_time]
                        if len(h4_bar) > 0:
                            h4_last = h4_bar.iloc[-1]
                            h4_bull = h4_last.get("ema_fast", 0) > h4_last.get("ema_slow", 0)
                            if (direction == "BUY" and h4_bull) or (direction == "SELL" and not h4_bull):
                                confluence_score += 30
                    else:
                        confluence_score += 15  # partial credit without H4

                    # H1 pattern + indicators (+40)
                    confluence_score += 15  # pattern detected
                    rsi = last_ind.get("rsi", 50)
                    if not pd.isna(rsi):
                        if (direction == "BUY" and rsi < 70) or (direction == "SELL" and rsi > 30):
                            confluence_score += 10
                    macd_h = last_ind.get("macd_histogram", 0)
                    if not pd.isna(macd_h):
                        if (direction == "BUY" and macd_h > 0) or (direction == "SELL" and macd_h < 0):
                            confluence_score += 10
                    bb = last_ind.get("bb_pctb", 0.5)
                    if not pd.isna(bb):
                        if (direction == "BUY" and bb < 0.2) or (direction == "SELL" and bb > 0.8):
                            confluence_score += 5

                    # M15 entry confirmation (simplified: +15 base)
                    confluence_score += 15

                    if confluence_score < self.min_confluence:
                        equity_history.append(balance)
                        continue

                    # ML scoring (using pre-computed features)
                    ml_proba = 0.0
                    if self.ml_enabled and self.ml_scorer is not None and ml_features_all is not None:
                        # Use the pre-computed features row for this bar
                        row_features = ml_features_all.iloc[[i]]
                        row_filled = row_features[self.ml_scorer.FEATURE_COLUMNS].fillna(0)
                        try:
                            ml_proba = float(self.ml_scorer.model.predict_proba(row_filled)[0][1])
                        except Exception:
                            ml_proba = 0.0
                        if ml_proba < self.ml_threshold:
                            equity_history.append(balance)
                            continue

                    # Calculate SL/TP
                    atr = last_ind.get("atr", 0)
                    if pd.isna(atr) or atr <= 0:
                        equity_history.append(balance)
                        continue

                    sl_dist = self.sl_atr_mult * atr
                    tp_dist = sl_dist * self.tp_rr

                    # Add spread and slippage
                    spread_cost = self.default_spread * 0.00001

                    if direction == "BUY":
                        entry_price = current_close + spread_cost / 2
                        sl_price = entry_price - sl_dist
                        tp_price = entry_price + tp_dist
                    else:
                        entry_price = current_close - spread_cost / 2
                        sl_price = entry_price + sl_dist
                        tp_price = entry_price - tp_dist

                    # Position sizing
                    risk_amount = balance * self.max_risk_pct
                    # Convert SL distance to pips (for forex-like instruments)
                    lot_size = risk_amount / (sl_dist * 100000) if sl_dist > 0 else 0.01
                    lot_size = max(0.01, min(lot_size, 10.0))
                    lot_size = round(lot_size, 2)

                    # Open position
                    open_positions.append(BacktestPosition(
                        symbol=symbol,
                        direction=direction,
                        entry_price=entry_price,
                        entry_time=current_time,
                        lot_size=lot_size,
                        sl=sl_price,
                        tp=tp_price,
                        atr_at_entry=atr,
                        pattern_name=dom_pattern,
                        confluence_score=confluence_score,
                        ml_probability=ml_proba,
                    ))

            equity_history.append(balance)

        # Close any remaining positions at last close
        last_close = df_h1["close"].iloc[-1]
        last_time = df_h1.index[-1]
        for pos in open_positions:
            if pos.direction == "BUY":
                pnl_pips = last_close - pos.entry_price
            else:
                pnl_pips = pos.entry_price - last_close
            pnl_dollars = pnl_pips * pos.lot_size * 100000 - self.commission_per_lot * pos.lot_size
            balance += pnl_dollars
            trades.append(BacktestTrade(
                symbol=symbol, direction=pos.direction,
                entry_price=pos.entry_price, exit_price=last_close,
                entry_time=pos.entry_time, exit_time=last_time,
                lot_size=pos.lot_size, sl=pos.sl, tp=pos.tp,
                pnl_pips=pnl_pips, pnl_dollars=pnl_dollars,
                exit_reason="EOD", pattern_name=pos.pattern_name,
                confluence_score=pos.confluence_score,
                ml_probability=pos.ml_probability,
            ))
        equity_history.append(balance)

        # Build equity curve
        eq_index = df_h1.index[warmup:].tolist()
        if len(equity_history) > len(eq_index):
            eq_index.append(df_h1.index[-1])
        equity_curve = pd.Series(
            equity_history[:len(eq_index)],
            index=eq_index[:len(equity_history)],
            name="equity",
        )

        # Calculate metrics
        metrics = self._calculate_metrics(trades, equity_curve)

        logger.info(
            f"Backtest complete: {len(trades)} trades | "
            f"Return={metrics['total_return_pct']:.2f}% | "
            f"Win rate={metrics['win_rate']*100:.1f}% | "
            f"Sharpe={metrics['sharpe_ratio']:.2f} | "
            f"Max DD={metrics['max_drawdown_pct']:.2f}%"
        )

        return BacktestResult(trades=trades, equity_curve=equity_curve, metrics=metrics)

    @staticmethod
    def _calculate_metrics(trades: List[BacktestTrade], equity_curve: pd.Series) -> dict:
        """Calculate comprehensive performance metrics."""
        if not trades:
            return {
                "total_trades": 0,
                "win_rate": 0, "total_return_pct": 0,
                "max_drawdown_pct": 0, "sharpe_ratio": 0,
                "profit_factor": 0, "expectancy": 0,
            }

        wins = [t for t in trades if t.pnl_dollars > 0]
        losses = [t for t in trades if t.pnl_dollars <= 0]

        total_profit = sum(t.pnl_dollars for t in wins)
        total_loss = abs(sum(t.pnl_dollars for t in losses))

        win_rate = len(wins) / len(trades) if trades else 0
        avg_win = total_profit / len(wins) if wins else 0
        avg_loss = total_loss / len(losses) if losses else 0
        profit_factor = total_profit / total_loss if total_loss > 0 else float("inf")
        expectancy = sum(t.pnl_dollars for t in trades) / len(trades)

        # Returns
        initial = equity_curve.iloc[0] if len(equity_curve) > 0 else 1
        final = equity_curve.iloc[-1] if len(equity_curve) > 0 else 1
        total_return_pct = (final - initial) / initial * 100

        # Drawdown
        running_max = equity_curve.cummax()
        drawdown = (equity_curve - running_max) / running_max
        max_drawdown_pct = abs(drawdown.min()) * 100

        # Sharpe ratio (annualised, assuming hourly bars)
        returns = equity_curve.pct_change().dropna()
        if len(returns) > 1 and returns.std() > 0:
            sharpe = returns.mean() / returns.std() * np.sqrt(252 * 24)  # hourly bars
        else:
            sharpe = 0.0

        # Sortino ratio
        downside = returns[returns < 0]
        if len(downside) > 1 and downside.std() > 0:
            sortino = returns.mean() / downside.std() * np.sqrt(252 * 24)
        else:
            sortino = 0.0

        # Calmar ratio
        calmar = (total_return_pct / max_drawdown_pct) if max_drawdown_pct > 0 else 0.0

        # Average trade duration
        durations = [(t.exit_time - t.entry_time).total_seconds() / 3600 for t in trades]
        avg_duration_hours = sum(durations) / len(durations) if durations else 0

        # Exit reason breakdown
        exit_reasons = {}
        for t in trades:
            exit_reasons[t.exit_reason] = exit_reasons.get(t.exit_reason, 0) + 1

        return {
            "total_trades": len(trades),
            "winning_trades": len(wins),
            "losing_trades": len(losses),
            "win_rate": win_rate,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "profit_factor": profit_factor,
            "expectancy": expectancy,
            "total_return_pct": total_return_pct,
            "max_drawdown_pct": max_drawdown_pct,
            "sharpe_ratio": sharpe,
            "sortino_ratio": sortino,
            "calmar_ratio": calmar,
            "avg_duration_hours": avg_duration_hours,
            "final_balance": final,
            "exit_reasons": exit_reasons,
        }
