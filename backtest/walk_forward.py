"""
Walk-Forward Optimization — Prevents overfitting by testing on rolling
out-of-sample windows.

Process:
  1. Divide data into windows (e.g., 6 months train, 2 months test)
  2. Train ML model on training window
  3. Run backtest on out-of-sample test window
  4. Slide forward by test window size
  5. Repeat until all data consumed
  6. Aggregate out-of-sample results for realistic performance estimate
"""

import logging
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd

from backtest.engine import BacktestEngine, BacktestResult, BacktestTrade
from src.ml_scorer import MLScorer

logger = logging.getLogger("strategy")


@dataclass
class WalkForwardWindow:
    """One train/test window in the walk-forward process."""
    window_id: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    train_bars: int
    test_bars: int
    result: Optional[BacktestResult]
    ml_metrics: Optional[dict]


@dataclass
class WalkForwardResult:
    """Aggregated walk-forward optimization result."""
    windows: List[WalkForwardWindow]
    aggregated_trades: List[BacktestTrade]
    aggregated_equity: pd.Series
    aggregated_metrics: dict
    in_sample_metrics: dict
    out_of_sample_metrics: dict


class WalkForwardOptimizer:
    """
    Implements walk-forward optimization with optional ML model retraining
    at each window.
    """

    def __init__(
        self,
        config: dict,
        train_bars: int = 4320,    # ~6 months of H1 bars
        test_bars: int = 1440,     # ~2 months of H1 bars
        retrain_ml: bool = True,
    ):
        self.config = config
        self.train_bars = train_bars
        self.test_bars = test_bars
        self.retrain_ml = retrain_ml

    def run(
        self,
        df_h1: pd.DataFrame,
        df_h4: Optional[pd.DataFrame] = None,
        symbol: str = "WALKFORWARD",
    ) -> WalkForwardResult:
        """
        Run walk-forward optimization on historical data.

        Args:
            df_h1: Full H1 OHLCV dataset.
            df_h4: Full H4 OHLCV dataset (optional).
            symbol: Symbol name for logging.

        Returns:
            WalkForwardResult with per-window and aggregated results.
        """
        total_bars = len(df_h1)
        min_required = self.train_bars + self.test_bars
        if total_bars < min_required:
            logger.warning(
                f"Walk-forward requires at least {min_required} bars, got {total_bars}. "
                f"Running single backtest instead."
            )
            engine = BacktestEngine(self.config)
            result = engine.run(df_h1, df_h4, symbol=symbol)
            window = WalkForwardWindow(
                window_id=0,
                train_start=df_h1.index[0],
                train_end=df_h1.index[-1],
                test_start=df_h1.index[0],
                test_end=df_h1.index[-1],
                train_bars=total_bars,
                test_bars=0,
                result=result,
                ml_metrics=None,
            )
            return WalkForwardResult(
                windows=[window],
                aggregated_trades=result.trades,
                aggregated_equity=result.equity_curve,
                aggregated_metrics=result.metrics,
                in_sample_metrics=result.metrics,
                out_of_sample_metrics=result.metrics,
            )

        windows: List[WalkForwardWindow] = []
        all_oos_trades: List[BacktestTrade] = []
        oos_equity_pieces: List[pd.Series] = []
        window_id = 0
        start_idx = 0

        logger.info(
            f"Walk-forward: {total_bars} bars, "
            f"train={self.train_bars}, test={self.test_bars}"
        )

        while start_idx + self.train_bars + self.test_bars <= total_bars:
            train_end_idx = start_idx + self.train_bars
            test_end_idx = min(train_end_idx + self.test_bars, total_bars)

            train_h1 = df_h1.iloc[start_idx:train_end_idx]
            test_h1 = df_h1.iloc[train_end_idx:test_end_idx]

            # Align H4 data to the same time ranges
            train_h4 = None
            test_h4 = None
            if df_h4 is not None and not df_h4.empty:
                train_h4 = df_h4[
                    (df_h4.index >= train_h1.index[0])
                    & (df_h4.index <= train_h1.index[-1])
                ]
                test_h4 = df_h4[
                    (df_h4.index >= test_h1.index[0])
                    & (df_h4.index <= test_h1.index[-1])
                ]
                if train_h4.empty:
                    train_h4 = None
                if test_h4 is not None and test_h4.empty:
                    test_h4 = None

            logger.info(
                f"Window {window_id}: "
                f"Train {train_h1.index[0].date()} → {train_h1.index[-1].date()} "
                f"({len(train_h1)} bars) | "
                f"Test {test_h1.index[0].date()} → {test_h1.index[-1].date()} "
                f"({len(test_h1)} bars)"
            )

            # Optionally retrain ML model on the training window
            ml_scorer = None
            ml_metrics = None
            if self.retrain_ml:
                ml_scorer = MLScorer(self.config)
                indicator_config = self.config.get("indicators", {})
                ml_metrics = ml_scorer.train(train_h1, train_h4, indicator_config)
                if ml_scorer.model is None:
                    ml_scorer = None
                    logger.info(f"  Window {window_id}: ML training failed, running without ML")
                else:
                    acc = ml_metrics.get("accuracy", 0)
                    logger.info(f"  Window {window_id}: ML accuracy = {acc:.3f}")

            # Run out-of-sample backtest
            engine = BacktestEngine(self.config, ml_scorer=ml_scorer)
            result = engine.run(test_h1, test_h4, symbol=f"{symbol}_W{window_id}")

            windows.append(WalkForwardWindow(
                window_id=window_id,
                train_start=train_h1.index[0],
                train_end=train_h1.index[-1],
                test_start=test_h1.index[0],
                test_end=test_h1.index[-1],
                train_bars=len(train_h1),
                test_bars=len(test_h1),
                result=result,
                ml_metrics=ml_metrics,
            ))

            all_oos_trades.extend(result.trades)
            oos_equity_pieces.append(result.equity_curve)

            logger.info(
                f"  Window {window_id} OOS: {len(result.trades)} trades, "
                f"Return={result.metrics['total_return_pct']:.2f}%, "
                f"Sharpe={result.metrics['sharpe_ratio']:.2f}"
            )

            # Slide forward by test window size
            start_idx = train_end_idx
            window_id += 1

        # Aggregate out-of-sample equity curve
        aggregated_equity = self._chain_equity_curves(oos_equity_pieces)

        # Calculate aggregated metrics
        aggregated_metrics = BacktestEngine._calculate_metrics(
            all_oos_trades, aggregated_equity,
        )

        # In-sample summary (average across windows)
        is_metrics = self._average_window_metrics(windows, "result")

        logger.info(
            f"\nWalk-forward complete: {len(windows)} windows, "
            f"{len(all_oos_trades)} total OOS trades"
        )
        logger.info(
            f"  OOS Return={aggregated_metrics['total_return_pct']:.2f}%, "
            f"Sharpe={aggregated_metrics['sharpe_ratio']:.2f}, "
            f"Max DD={aggregated_metrics['max_drawdown_pct']:.2f}%"
        )

        return WalkForwardResult(
            windows=windows,
            aggregated_trades=all_oos_trades,
            aggregated_equity=aggregated_equity,
            aggregated_metrics=aggregated_metrics,
            in_sample_metrics=is_metrics,
            out_of_sample_metrics=aggregated_metrics,
        )

    def _chain_equity_curves(self, pieces: List[pd.Series]) -> pd.Series:
        """Chain multiple equity curves, adjusting each to start where the previous ended."""
        if not pieces:
            return pd.Series(dtype=float)

        chained = pieces[0].copy()
        for piece in pieces[1:]:
            if piece.empty:
                continue
            offset = chained.iloc[-1] - piece.iloc[0]
            adjusted = piece + offset
            chained = pd.concat([chained, adjusted.iloc[1:]])

        chained.name = "equity"
        return chained

    @staticmethod
    def _average_window_metrics(windows: List[WalkForwardWindow], attr: str) -> dict:
        """Average metrics across all windows."""
        metric_keys = [
            "total_return_pct", "win_rate", "sharpe_ratio",
            "max_drawdown_pct", "profit_factor", "expectancy",
        ]
        averages = {}
        for key in metric_keys:
            values = []
            for w in windows:
                result = getattr(w, attr, None)
                if result is not None and hasattr(result, "metrics"):
                    val = result.metrics.get(key, 0)
                    if val is not None:
                        values.append(val)
            averages[key] = np.mean(values) if values else 0.0

        return averages
