"""
Monte Carlo Simulation — Robustness testing by randomly resampling trades.

Takes actual trade results from a backtest, randomly resamples (with replacement)
N times, and builds confidence intervals for key performance metrics.

This tells us how robust the strategy is to different orderings of trades
and helps estimate the range of expected performance.
"""

import logging
from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

logger = logging.getLogger("strategy")


@dataclass
class MonteCarloResult:
    """Result of a Monte Carlo simulation."""
    iterations: int
    final_equities: np.ndarray
    max_drawdowns: np.ndarray
    sharpe_ratios: np.ndarray
    confidence_intervals: dict
    percentiles: dict


class MonteCarloSimulator:
    """
    Performs Monte Carlo simulation on backtest trade results.

    Randomly resamples trade P&L values (with replacement) to build
    a distribution of possible outcomes.
    """

    def __init__(self, initial_balance: float = 10000, seed: int = 42):
        self.initial_balance = initial_balance
        self.seed = seed

    def run(
        self,
        trade_pnls: List[float],
        iterations: int = 10000,
    ) -> MonteCarloResult:
        """
        Run Monte Carlo simulation.

        Args:
            trade_pnls: List of trade P&L values in dollars.
            iterations: Number of random resamplings.

        Returns:
            MonteCarloResult with distributions and confidence intervals.
        """
        if not trade_pnls:
            logger.warning("No trades to simulate")
            return MonteCarloResult(
                iterations=0,
                final_equities=np.array([]),
                max_drawdowns=np.array([]),
                sharpe_ratios=np.array([]),
                confidence_intervals={},
                percentiles={},
            )

        rng = np.random.RandomState(self.seed)
        n_trades = len(trade_pnls)
        pnl_array = np.array(trade_pnls)

        final_equities = np.zeros(iterations)
        max_drawdowns = np.zeros(iterations)
        sharpe_ratios = np.zeros(iterations)

        logger.info(
            f"Monte Carlo: {iterations} iterations, {n_trades} trades, "
            f"initial balance={self.initial_balance}"
        )

        for i in range(iterations):
            # Resample trades with replacement
            sampled = rng.choice(pnl_array, size=n_trades, replace=True)

            # Build equity curve
            equity = self.initial_balance + np.cumsum(sampled)
            equity = np.insert(equity, 0, self.initial_balance)

            final_equities[i] = equity[-1]

            # Max drawdown
            running_max = np.maximum.accumulate(equity)
            drawdown_pct = (equity - running_max) / running_max
            max_drawdowns[i] = abs(np.min(drawdown_pct)) * 100

            # Sharpe ratio (simplified — based on trade returns)
            if len(sampled) > 1:
                trade_returns = sampled / self.initial_balance
                if trade_returns.std() > 0:
                    sharpe_ratios[i] = (
                        trade_returns.mean() / trade_returns.std() * np.sqrt(252)
                    )

        # Confidence intervals
        confidence_intervals = {
            "final_equity": {
                "5th": np.percentile(final_equities, 5),
                "25th": np.percentile(final_equities, 25),
                "50th": np.percentile(final_equities, 50),
                "75th": np.percentile(final_equities, 75),
                "95th": np.percentile(final_equities, 95),
                "mean": np.mean(final_equities),
                "std": np.std(final_equities),
            },
            "max_drawdown": {
                "5th": np.percentile(max_drawdowns, 5),
                "25th": np.percentile(max_drawdowns, 25),
                "50th": np.percentile(max_drawdowns, 50),
                "75th": np.percentile(max_drawdowns, 75),
                "95th": np.percentile(max_drawdowns, 95),
                "mean": np.mean(max_drawdowns),
            },
            "sharpe_ratio": {
                "5th": np.percentile(sharpe_ratios, 5),
                "25th": np.percentile(sharpe_ratios, 25),
                "50th": np.percentile(sharpe_ratios, 50),
                "75th": np.percentile(sharpe_ratios, 75),
                "95th": np.percentile(sharpe_ratios, 95),
                "mean": np.mean(sharpe_ratios),
            },
        }

        total_return_pct = (final_equities - self.initial_balance) / self.initial_balance * 100

        percentiles = {
            "return_5th": np.percentile(total_return_pct, 5),
            "return_95th": np.percentile(total_return_pct, 95),
            "dd_95th": np.percentile(max_drawdowns, 95),
            "prob_profitable": np.mean(final_equities > self.initial_balance) * 100,
            "prob_loss_gt_10pct": np.mean(total_return_pct < -10) * 100,
        }

        logger.info(
            f"Monte Carlo results:\n"
            f"  95% CI Return: {percentiles['return_5th']:.1f}% to {percentiles['return_95th']:.1f}%\n"
            f"  95% CI Max DD: < {percentiles['dd_95th']:.1f}%\n"
            f"  Probability profitable: {percentiles['prob_profitable']:.1f}%"
        )

        return MonteCarloResult(
            iterations=iterations,
            final_equities=final_equities,
            max_drawdowns=max_drawdowns,
            sharpe_ratios=sharpe_ratios,
            confidence_intervals=confidence_intervals,
            percentiles=percentiles,
        )

    @staticmethod
    def plot_equity_distribution(
        result: MonteCarloResult,
        save_path: str = "monte_carlo_equity.png",
    ):
        """Plot distribution of final equity values."""
        if result.iterations == 0:
            return

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # Final equity distribution
        axes[0].hist(result.final_equities, bins=100, color="#2196F3", alpha=0.7, edgecolor="white")
        axes[0].axvline(
            result.confidence_intervals["final_equity"]["50th"],
            color="red", linestyle="--", label="Median",
        )
        axes[0].axvline(
            result.confidence_intervals["final_equity"]["5th"],
            color="orange", linestyle=":", label="5th percentile",
        )
        axes[0].axvline(
            result.confidence_intervals["final_equity"]["95th"],
            color="green", linestyle=":", label="95th percentile",
        )
        axes[0].set_title("Final Equity Distribution", fontweight="bold")
        axes[0].set_xlabel("Final Equity ($)")
        axes[0].set_ylabel("Frequency")
        axes[0].legend(fontsize=8)
        axes[0].grid(True, alpha=0.3)

        # Max drawdown distribution
        axes[1].hist(result.max_drawdowns, bins=100, color="#F44336", alpha=0.7, edgecolor="white")
        axes[1].axvline(
            result.confidence_intervals["max_drawdown"]["95th"],
            color="darkred", linestyle="--", label="95th percentile",
        )
        axes[1].set_title("Max Drawdown Distribution", fontweight="bold")
        axes[1].set_xlabel("Max Drawdown (%)")
        axes[1].set_ylabel("Frequency")
        axes[1].legend(fontsize=8)
        axes[1].grid(True, alpha=0.3)

        # Sharpe ratio distribution
        axes[2].hist(result.sharpe_ratios, bins=100, color="#4CAF50", alpha=0.7, edgecolor="white")
        axes[2].axvline(
            result.confidence_intervals["sharpe_ratio"]["50th"],
            color="darkgreen", linestyle="--", label="Median",
        )
        axes[2].set_title("Sharpe Ratio Distribution", fontweight="bold")
        axes[2].set_xlabel("Sharpe Ratio")
        axes[2].set_ylabel("Frequency")
        axes[2].legend(fontsize=8)
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()
        logger.info(f"Monte Carlo charts saved to {save_path}")

    @staticmethod
    def print_report(result: MonteCarloResult):
        """Print Monte Carlo confidence intervals."""
        if result.iterations == 0:
            print("No Monte Carlo results (no trades)")
            return

        ci = result.confidence_intervals
        pct = result.percentiles

        print("\n" + "=" * 60)
        print("         MONTE CARLO SIMULATION RESULTS")
        print(f"         ({result.iterations:,} iterations)")
        print("=" * 60)

        print(f"\n  Final Equity:")
        print(f"    5th percentile:   ${ci['final_equity']['5th']:>12,.2f}")
        print(f"    25th percentile:  ${ci['final_equity']['25th']:>12,.2f}")
        print(f"    Median:           ${ci['final_equity']['50th']:>12,.2f}")
        print(f"    75th percentile:  ${ci['final_equity']['75th']:>12,.2f}")
        print(f"    95th percentile:  ${ci['final_equity']['95th']:>12,.2f}")

        print(f"\n  Max Drawdown:")
        print(f"    Median:           {ci['max_drawdown']['50th']:>10.2f}%")
        print(f"    95% confidence:   < {ci['max_drawdown']['95th']:.2f}%")

        print(f"\n  Sharpe Ratio:")
        print(f"    Median:           {ci['sharpe_ratio']['50th']:>10.2f}")
        print(f"    95% range:        {ci['sharpe_ratio']['5th']:.2f} to {ci['sharpe_ratio']['95th']:.2f}")

        print(f"\n  Return (95% CI):    {pct['return_5th']:.1f}% to {pct['return_95th']:.1f}%")
        print(f"  Prob. profitable:   {pct['prob_profitable']:.1f}%")
        print(f"  Prob. loss > 10%:   {pct['prob_loss_gt_10pct']:.1f}%")
        print("=" * 60)
