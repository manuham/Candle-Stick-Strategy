"""
Backtesting Entry Point — Validate the strategy on historical data.

Usage:
    python run_backtest.py                             # Default: synthetic data
    python run_backtest.py --symbol EURUSD             # Specific symbol from MT5
    python run_backtest.py --csv data/EURUSD_H1.csv    # From CSV file
    python run_backtest.py --synthetic --bars 10000    # Synthetic data
    python run_backtest.py --walk-forward              # Walk-forward optimization
    python run_backtest.py --monte-carlo --iterations 10000  # Monte Carlo analysis
"""

import argparse
import os
import sys

from src.utils import load_config, setup_logging
from src.data_engine import DataEngine
from src.ml_scorer import MLScorer
from backtest.engine import BacktestEngine
from backtest.data_loader import (
    load_csv,
    fetch_and_cache,
    generate_synthetic_data,
)
from backtest.report import generate_full_report
from backtest.walk_forward import WalkForwardOptimizer
from backtest.monte_carlo import MonteCarloSimulator


def parse_args():
    parser = argparse.ArgumentParser(description="Candlestick Strategy — Backtesting")
    parser.add_argument("--config", default="config/settings.yaml", help="Path to config file")
    parser.add_argument("--symbol", default=None, help="Symbol to backtest (requires MT5)")
    parser.add_argument("--csv", default=None, help="Path to CSV file with H1 OHLCV data")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic data")
    parser.add_argument("--bars", type=int, default=5000, help="Number of bars for synthetic data")
    parser.add_argument("--output", default=".", help="Output directory for reports")
    parser.add_argument("--no-ml", action="store_true", help="Disable ML scoring")
    parser.add_argument("--walk-forward", action="store_true", help="Run walk-forward optimization")
    parser.add_argument("--wf-train", type=int, default=4320, help="Walk-forward training window (bars)")
    parser.add_argument("--wf-test", type=int, default=1440, help="Walk-forward test window (bars)")
    parser.add_argument("--monte-carlo", action="store_true", help="Run Monte Carlo simulation")
    parser.add_argument("--iterations", type=int, default=10000, help="Monte Carlo iterations")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    logger = setup_logging(config)

    logger.info("=" * 60)
    logger.info("  Candlestick Strategy — Backtesting")
    logger.info("=" * 60)

    # Load ML model if enabled
    ml_scorer = None
    if not args.no_ml and config.get("ml", {}).get("enabled", True):
        ml_scorer = MLScorer(config)
        if ml_scorer.load_model():
            logger.info("ML model loaded for backtesting")
        else:
            logger.info("No ML model found — backtesting without ML scoring")
            ml_scorer = None

    # Determine data source
    symbol = "SYNTHETIC"
    df_h1 = None
    df_h4 = None

    if args.csv:
        # Load from CSV
        logger.info(f"Loading data from CSV: {args.csv}")
        df_h1 = load_csv(args.csv)
        if df_h1 is None:
            logger.error("Failed to load CSV data")
            sys.exit(1)
        symbol = args.csv.split("/")[-1].split("_")[0]

    elif args.symbol:
        # Fetch from MT5
        symbol = args.symbol
        logger.info(f"Fetching {symbol} data from MT5...")

        data_engine = DataEngine(config)
        if not data_engine.connect():
            logger.error("Cannot connect to MT5. Use --synthetic or --csv instead.")
            sys.exit(1)

        df_h1 = fetch_and_cache(data_engine, symbol, "H1", args.bars, "data")
        df_h4 = fetch_and_cache(data_engine, symbol, "H4", args.bars // 4, "data")
        data_engine.disconnect()

        if df_h1 is None:
            logger.error(f"Failed to fetch data for {symbol}")
            sys.exit(1)

    else:
        # Default: synthetic data
        args.synthetic = True

    if args.synthetic:
        logger.info(f"Generating synthetic data ({args.bars} bars)...")
        df_h1 = generate_synthetic_data(bars=args.bars, seed=42)
        df_h4 = generate_synthetic_data(bars=args.bars // 4, seed=43)
        symbol = "SYNTHETIC"

    logger.info(f"Backtesting {symbol} with {len(df_h1)} H1 bars")
    if df_h4 is not None:
        logger.info(f"H4 context: {len(df_h4)} bars")

    os.makedirs(args.output, exist_ok=True)

    # --- Walk-Forward Optimization ---
    if args.walk_forward:
        logger.info("\n--- Walk-Forward Optimization ---")
        wf = WalkForwardOptimizer(
            config,
            train_bars=args.wf_train,
            test_bars=args.wf_test,
            retrain_ml=not args.no_ml,
        )
        wf_result = wf.run(df_h1, df_h4, symbol=symbol)

        # Generate report from aggregated OOS results
        from backtest.engine import BacktestResult
        oos_result = BacktestResult(
            trades=wf_result.aggregated_trades,
            equity_curve=wf_result.aggregated_equity,
            metrics=wf_result.aggregated_metrics,
        )
        generate_full_report(oos_result, output_dir=args.output)

        # Print per-window summary
        print("\n" + "=" * 70)
        print("  WALK-FORWARD OPTIMIZATION — PER-WINDOW RESULTS")
        print("=" * 70)
        for w in wf_result.windows:
            m = w.result.metrics if w.result else {}
            print(
                f"  Window {w.window_id}: "
                f"Test {w.test_start.date()} → {w.test_end.date()} | "
                f"Trades={m.get('total_trades', 0):>3d} | "
                f"Return={m.get('total_return_pct', 0):>7.2f}% | "
                f"Sharpe={m.get('sharpe_ratio', 0):>6.2f} | "
                f"MaxDD={m.get('max_drawdown_pct', 0):>6.2f}%"
            )
        print("-" * 70)
        am = wf_result.aggregated_metrics
        print(
            f"  AGGREGATED OOS: "
            f"Trades={am.get('total_trades', 0)} | "
            f"Return={am.get('total_return_pct', 0):.2f}% | "
            f"Sharpe={am.get('sharpe_ratio', 0):.2f} | "
            f"MaxDD={am.get('max_drawdown_pct', 0):.2f}%"
        )
        print("=" * 70)

        result = oos_result
    else:
        # Standard single-pass backtest
        engine = BacktestEngine(config, ml_scorer)
        result = engine.run(
            df_h1=df_h1,
            df_h4=df_h4,
            df_m15=None,
            symbol=symbol,
        )
        generate_full_report(result, output_dir=args.output)

    # --- Monte Carlo Simulation ---
    if args.monte_carlo and result.trades:
        logger.info(f"\n--- Monte Carlo Simulation ({args.iterations} iterations) ---")
        trade_pnls = [t.pnl_dollars for t in result.trades]
        mc = MonteCarloSimulator(
            initial_balance=config.get("backtest", {}).get("initial_balance", 10000),
        )
        mc_result = mc.run(trade_pnls, iterations=args.iterations)
        MonteCarloSimulator.print_report(mc_result)
        MonteCarloSimulator.plot_equity_distribution(
            mc_result,
            save_path=os.path.join(args.output, "monte_carlo.png"),
        )

    # Return exit code based on results
    if result.metrics.get("total_trades", 0) == 0:
        logger.warning("No trades were generated during backtest")
        return 1

    logger.info(f"\nBacktest complete. Reports saved to {args.output}/")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
