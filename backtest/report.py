"""
Backtest Reporting — Performance visualization and summary statistics.
"""

import logging
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.dates as mdates  # noqa: E402

logger = logging.getLogger("strategy")


def print_summary(metrics: dict):
    """Print a formatted summary table of backtest metrics."""
    print("\n" + "=" * 60)
    print("            BACKTEST PERFORMANCE SUMMARY")
    print("=" * 60)

    rows = [
        ("Total Trades", f"{metrics.get('total_trades', 0)}"),
        ("Winning Trades", f"{metrics.get('winning_trades', 0)}"),
        ("Losing Trades", f"{metrics.get('losing_trades', 0)}"),
        ("Win Rate", f"{metrics.get('win_rate', 0) * 100:.1f}%"),
        ("", ""),
        ("Total Return", f"{metrics.get('total_return_pct', 0):.2f}%"),
        ("Final Balance", f"${metrics.get('final_balance', 0):,.2f}"),
        ("Max Drawdown", f"{metrics.get('max_drawdown_pct', 0):.2f}%"),
        ("", ""),
        ("Profit Factor", f"{metrics.get('profit_factor', 0):.2f}"),
        ("Expectancy / Trade", f"${metrics.get('expectancy', 0):.2f}"),
        ("Avg Win", f"${metrics.get('avg_win', 0):.2f}"),
        ("Avg Loss", f"${metrics.get('avg_loss', 0):.2f}"),
        ("", ""),
        ("Sharpe Ratio", f"{metrics.get('sharpe_ratio', 0):.2f}"),
        ("Sortino Ratio", f"{metrics.get('sortino_ratio', 0):.2f}"),
        ("Calmar Ratio", f"{metrics.get('calmar_ratio', 0):.2f}"),
        ("", ""),
        ("Avg Trade Duration", f"{metrics.get('avg_duration_hours', 0):.1f} hours"),
    ]

    for label, value in rows:
        if label == "":
            print("-" * 60)
        else:
            print(f"  {label:<25s} {value:>30s}")

    # Exit reasons
    exit_reasons = metrics.get("exit_reasons", {})
    if exit_reasons:
        print("-" * 60)
        print("  Exit Reasons:")
        for reason, count in sorted(exit_reasons.items()):
            print(f"    {reason:<20s} {count:>5d}")

    print("=" * 60)


def plot_equity_curve(
    equity_curve: pd.Series,
    save_path: str = "backtest_equity.png",
):
    """Plot the equity curve and save to file."""
    fig, ax = plt.subplots(figsize=(14, 6))

    ax.plot(equity_curve.index, equity_curve.values, linewidth=1.2, color="#2196F3")
    ax.fill_between(
        equity_curve.index, equity_curve.iloc[0], equity_curve.values,
        alpha=0.1, color="#2196F3",
    )

    ax.set_title("Equity Curve", fontsize=14, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Account Balance ($)")
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.autofmt_xdate()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    logger.info(f"Equity curve saved to {save_path}")


def plot_drawdown(
    equity_curve: pd.Series,
    save_path: str = "backtest_drawdown.png",
):
    """Plot drawdown chart."""
    running_max = equity_curve.cummax()
    drawdown_pct = (equity_curve - running_max) / running_max * 100

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.fill_between(drawdown_pct.index, 0, drawdown_pct.values, color="#F44336", alpha=0.5)
    ax.plot(drawdown_pct.index, drawdown_pct.values, color="#F44336", linewidth=0.8)

    ax.set_title("Drawdown (%)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown %")
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.autofmt_xdate()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    logger.info(f"Drawdown chart saved to {save_path}")


def plot_monthly_returns(
    equity_curve: pd.Series,
    save_path: str = "backtest_monthly.png",
):
    """Plot monthly returns heatmap."""
    # Calculate monthly returns
    monthly = equity_curve.resample("ME").last().pct_change().dropna() * 100

    if monthly.empty:
        logger.warning("Not enough data for monthly returns plot")
        return

    # Reshape into year x month matrix
    monthly_df = pd.DataFrame({
        "year": monthly.index.year,
        "month": monthly.index.month,
        "return": monthly.values,
    })
    pivot = monthly_df.pivot_table(index="year", columns="month", values="return")
    pivot.columns = [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun",
        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
    ][:len(pivot.columns)]

    fig, ax = plt.subplots(figsize=(14, max(3, len(pivot) * 0.6)))
    cmap = plt.cm.RdYlGn
    im = ax.imshow(pivot.values, cmap=cmap, aspect="auto", vmin=-5, vmax=5)

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)

    # Add text annotations
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.1f}%", ha="center", va="center", fontsize=8)

    ax.set_title("Monthly Returns (%)", fontsize=14, fontweight="bold")
    fig.colorbar(im, ax=ax, label="Return %")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    logger.info(f"Monthly returns heatmap saved to {save_path}")


def plot_trade_distribution(
    trades: list,
    save_path: str = "backtest_distribution.png",
):
    """Plot trade P&L distribution."""
    if not trades:
        return

    pnls = [t.pnl_dollars for t in trades]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Histogram
    axes[0].hist(pnls, bins=50, color="#2196F3", alpha=0.7, edgecolor="white")
    axes[0].axvline(x=0, color="red", linestyle="--", linewidth=1)
    axes[0].set_title("Trade P&L Distribution", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("P&L ($)")
    axes[0].set_ylabel("Frequency")
    axes[0].grid(True, alpha=0.3)

    # Cumulative P&L
    cumulative = np.cumsum(pnls)
    axes[1].plot(range(len(cumulative)), cumulative, color="#4CAF50", linewidth=1.2)
    axes[1].set_title("Cumulative P&L by Trade", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Trade #")
    axes[1].set_ylabel("Cumulative P&L ($)")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    logger.info(f"Trade distribution chart saved to {save_path}")


def generate_full_report(
    result,
    output_dir: str = ".",
):
    """Generate all report outputs."""
    import os
    os.makedirs(output_dir, exist_ok=True)

    print_summary(result.metrics)

    plot_equity_curve(
        result.equity_curve,
        os.path.join(output_dir, "backtest_equity.png"),
    )
    plot_drawdown(
        result.equity_curve,
        os.path.join(output_dir, "backtest_drawdown.png"),
    )
    plot_monthly_returns(
        result.equity_curve,
        os.path.join(output_dir, "backtest_monthly.png"),
    )
    plot_trade_distribution(
        result.trades,
        os.path.join(output_dir, "backtest_distribution.png"),
    )

    # Save trade log to CSV
    if result.trades:
        trade_data = []
        for t in result.trades:
            trade_data.append({
                "symbol": t.symbol,
                "direction": t.direction,
                "entry_time": t.entry_time,
                "exit_time": t.exit_time,
                "entry_price": t.entry_price,
                "exit_price": t.exit_price,
                "lot_size": t.lot_size,
                "pnl_pips": t.pnl_pips,
                "pnl_dollars": t.pnl_dollars,
                "exit_reason": t.exit_reason,
                "pattern": t.pattern_name,
                "confluence": t.confluence_score,
                "ml_prob": t.ml_probability,
            })
        trade_df = pd.DataFrame(trade_data)
        csv_path = os.path.join(output_dir, "backtest_trades.csv")
        trade_df.to_csv(csv_path, index=False)
        logger.info(f"Trade log saved to {csv_path}")
