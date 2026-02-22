"""
ML Model Training Entry Point — Train the XGBoost pattern scorer.

Fetches historical data from MT5 (or loads from cache), engineers features,
trains the model, and saves it for use in live trading.

Usage:
    python train_model.py                           # Default config
    python train_model.py --config path/to/config   # Custom config
    python train_model.py --synthetic               # Use synthetic data (no MT5)
"""

import argparse
import sys
import json

from src.utils import load_config, setup_logging, get_all_symbols
from src.data_engine import DataEngine
from src.ml_scorer import MLScorer
from backtest.data_loader import fetch_and_cache, generate_synthetic_data


def parse_args():
    parser = argparse.ArgumentParser(description="Candlestick Strategy — ML Model Training")
    parser.add_argument("--config", default="config/settings.yaml", help="Path to config file")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic data instead of MT5")
    parser.add_argument("--bars", type=int, default=10000, help="Number of H1 bars for training")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    logger = setup_logging(config)

    logger.info("=" * 60)
    logger.info("  Candlestick Strategy — ML Model Training")
    logger.info("=" * 60)

    ml_scorer = MLScorer(config)
    indicator_config = config.get("indicators", {})
    all_metrics = {}

    if args.synthetic:
        # Use synthetic data for testing
        logger.info("Using synthetic data for training")
        df_h1 = generate_synthetic_data(bars=args.bars, seed=42)
        df_h4 = generate_synthetic_data(bars=args.bars // 4, seed=43)

        logger.info(f"Synthetic H1 data: {len(df_h1)} bars")
        logger.info(f"Synthetic H4 data: {len(df_h4)} bars")

        metrics = ml_scorer.train(df_h1, df_h4, indicator_config)
        all_metrics["SYNTHETIC"] = metrics
    else:
        # Fetch real data from MT5
        data_engine = DataEngine(config)
        if not data_engine.connect():
            logger.error("Cannot connect to MT5. Use --synthetic for testing.")
            sys.exit(1)

        symbols = get_all_symbols(config)
        h1_frames = []
        df_h4_combined = None

        for symbol in symbols:
            logger.info(f"Fetching data for {symbol}...")

            # Fetch H1 data
            df_h1 = fetch_and_cache(data_engine, symbol, "H1", args.bars, "data")
            if df_h1 is None:
                logger.warning(f"Could not fetch H1 data for {symbol} — skipping")
                continue

            # Fetch H4 data
            df_h4 = fetch_and_cache(data_engine, symbol, "H4", args.bars // 4, "data")

            h1_frames.append(df_h1)
            if df_h4 is not None and df_h4_combined is None:
                df_h4_combined = df_h4

            logger.info(f"  {symbol}: {len(df_h1)} H1 bars")

        data_engine.disconnect()

        if not h1_frames:
            logger.error("No data fetched. Cannot train model.")
            sys.exit(1)

        # Train on each symbol's data or combined
        logger.info(f"\nTraining on {len(h1_frames)} symbol dataset(s)...")

        # For simplicity, train on the first symbol's data
        # A production system would combine all symbols
        for i, df_h1 in enumerate(h1_frames):
            symbol = symbols[i] if i < len(symbols) else f"SYMBOL_{i}"
            logger.info(f"\nTraining on {symbol} ({len(df_h1)} bars)...")
            metrics = ml_scorer.train(df_h1, df_h4_combined, indicator_config)
            all_metrics[symbol] = metrics

    # Save model
    if ml_scorer.model is not None:
        ml_scorer.save_model()

    # Print results
    print("\n" + "=" * 60)
    print("            ML TRAINING RESULTS")
    print("=" * 60)

    for symbol, metrics in all_metrics.items():
        if "error" in metrics:
            print(f"\n  {symbol}: ERROR — {metrics['error']}")
            continue

        print(f"\n  {symbol}:")
        print(f"    Accuracy:        {metrics.get('accuracy', 0):.3f}")
        print(f"    Precision:       {metrics.get('precision', 0):.3f}")
        print(f"    Recall:          {metrics.get('recall', 0):.3f}")
        print(f"    F1 Score:        {metrics.get('f1', 0):.3f}")
        print(f"    Train Samples:   {metrics.get('train_samples', 0)}")
        print(f"    Test Samples:    {metrics.get('test_samples', 0)}")
        print(f"    Positive Rate:   {metrics.get('positive_rate_test', 0):.3f}")

        importances = metrics.get("feature_importances", {})
        if importances:
            print(f"\n    Top Features:")
            for feat, imp in list(importances.items())[:10]:
                print(f"      {feat:<30s} {imp:.4f}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
