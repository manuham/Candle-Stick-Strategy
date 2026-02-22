"""
Backtesting Entry Point — Validate the strategy on historical data.

Usage:
    python run_backtest.py                             # Default: synthetic data
    python run_backtest.py --symbol EURUSD             # Specific symbol from MT5
    python run_backtest.py --csv data/EURUSD_H1.csv    # From CSV file
    python run_backtest.py --synthetic --bars 10000    # Synthetic data
"""

import argparse
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


def parse_args():
    parser = argparse.ArgumentParser(description="Candlestick Strategy — Backtesting")
    parser.add_argument("--config", default="config/settings.yaml", help="Path to config file")
    parser.add_argument("--symbol", default=None, help="Symbol to backtest (requires MT5)")
    parser.add_argument("--csv", default=None, help="Path to CSV file with H1 OHLCV data")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic data")
    parser.add_argument("--bars", type=int, default=5000, help="Number of bars for synthetic data")
    parser.add_argument("--output", default=".", help="Output directory for reports")
    parser.add_argument("--no-ml", action="store_true", help="Disable ML scoring")
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

    # Run backtest
    engine = BacktestEngine(config, ml_scorer)
    result = engine.run(
        df_h1=df_h1,
        df_h4=df_h4,
        df_m15=None,  # M15 not available in simplified backtest mode
        symbol=symbol,
    )

    # Generate report
    generate_full_report(result, output_dir=args.output)

    # Return exit code based on results
    if result.metrics.get("total_trades", 0) == 0:
        logger.warning("No trades were generated during backtest")
        return 1

    logger.info(f"\nBacktest complete. Reports saved to {args.output}/")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
