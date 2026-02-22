"""
Live Trading Entry Point — Runs the candlestick strategy on MetaTrader 5.

Usage:
    python main.py                          # Normal mode
    python main.py --config path/to/config  # Custom config
    python main.py --dry-run                # Simulate without executing trades
"""

import argparse
import signal
import sys
import time
import logging
from datetime import datetime, timezone

from src.utils import load_config, setup_logging, get_all_symbols, utc_now
from src.data_engine import DataEngine
from src.signals import SignalGenerator
from src.ml_scorer import MLScorer
from src.risk_manager import RiskManager, TradeRecord
from src.executor import Executor


def parse_args():
    parser = argparse.ArgumentParser(description="Candlestick Strategy — Live Trading")
    parser.add_argument("--config", default="config/settings.yaml", help="Path to config file")
    parser.add_argument("--dry-run", action="store_true", help="Run without executing trades")
    return parser.parse_args()


def main():
    args = parse_args()

    # Load configuration
    config = load_config(args.config)
    logger = setup_logging(config)
    logger.info("=" * 60)
    logger.info("  Candlestick Strategy — Live Trading")
    logger.info("=" * 60)

    if args.dry_run:
        logger.info("DRY RUN MODE — no trades will be executed")

    # Initialize components
    data_engine = DataEngine(config)
    if not data_engine.connect():
        logger.error("Failed to connect to MT5. Exiting.")
        sys.exit(1)

    # Load ML model
    ml_scorer = MLScorer(config)
    if config.get("ml", {}).get("enabled", True):
        if ml_scorer.load_model():
            logger.info("ML model loaded successfully")
        else:
            logger.warning("ML model not found — running with confluence-only mode")
            ml_scorer = None

    signal_gen = SignalGenerator(config, ml_scorer)
    risk_manager = RiskManager(config)
    executor = Executor(config)

    symbols = get_all_symbols(config)
    logger.info(f"Trading symbols: {symbols}")

    # Get initial account info
    account = data_engine.get_account_info()
    if account:
        risk_manager.reset_daily(account["balance"])
        logger.info(f"Account balance: {account['balance']:.2f} {account['currency']}")

    # Graceful shutdown handler
    running = True

    def shutdown_handler(signum, frame):
        nonlocal running
        logger.info("Shutdown signal received — closing positions and exiting")
        running = False

    signal.signal(signal.SIGINT, shutdown_handler)
    signal.signal(signal.SIGTERM, shutdown_handler)

    # Track the last processed bar to avoid duplicate signals
    last_bar_time = {}
    current_day = utc_now().date()

    logger.info("Entering main trading loop...")

    try:
        while running:
            now = utc_now()

            # Daily reset
            if now.date() != current_day:
                current_day = now.date()
                account = data_engine.get_account_info()
                if account:
                    risk_manager.reset_daily(account["balance"])
                logger.info(f"New trading day: {current_day}")

            # Check daily drawdown
            account = data_engine.get_account_info()
            if account and not risk_manager.check_daily_drawdown(account["balance"]):
                logger.warning("Daily drawdown limit reached — waiting for next day")
                time.sleep(300)
                continue

            # Scan each symbol
            for symbol in symbols:
                try:
                    # Fetch multi-timeframe data
                    tf_data = data_engine.get_multi_tf_data(symbol)

                    signal_df = tf_data.get("signal")
                    if signal_df is None or signal_df.empty:
                        continue

                    # Check if we've already processed this bar
                    latest_bar = signal_df.index[-1]
                    if symbol in last_bar_time and last_bar_time[symbol] == latest_bar:
                        continue
                    last_bar_time[symbol] = latest_bar

                    # Generate signal
                    trade_signal = signal_gen.generate(
                        symbol=symbol,
                        df_h4=tf_data.get("trend"),
                        df_h1=tf_data.get("signal"),
                        df_m15=tf_data.get("entry"),
                    )

                    if trade_signal is None:
                        continue

                    # Risk checks
                    open_positions = executor.get_open_positions()

                    if not risk_manager.check_position_limits(open_positions, symbol):
                        continue

                    if not risk_manager.check_risk_reward(trade_signal):
                        continue

                    # Spread check
                    tick = data_engine.get_tick(symbol)
                    sym_info = data_engine.get_symbol_info(symbol)
                    if tick and sym_info:
                        current_spread = (tick["ask"] - tick["bid"]) / sym_info["point"]
                        avg_spread = sym_info["spread"]
                        if not risk_manager.check_spread(current_spread, avg_spread):
                            continue

                    # Close before EOD check
                    close_before = config.get("risk", {}).get("close_before_eod_minutes", 30)
                    # Simplified: skip new trades after 20:30 UTC for forex
                    if now.hour >= 21 or (now.hour == 20 and now.minute >= 30):
                        logger.debug(f"Near EOD — skipping new trade for {symbol}")
                        continue

                    # Calculate position size
                    if account and sym_info:
                        lot_size = risk_manager.calculate_lot_size(
                            account_balance=account["balance"],
                            sl_distance_price=trade_signal.sl_distance,
                            symbol_info=sym_info,
                        )
                    else:
                        lot_size = 0.01  # Minimum fallback

                    # Execute trade
                    if not args.dry_run:
                        result = executor.open_position(
                            symbol=trade_signal.symbol,
                            direction=trade_signal.direction,
                            lot_size=lot_size,
                            sl_price=trade_signal.suggested_sl,
                            tp_price=trade_signal.suggested_tp,
                            comment=f"{trade_signal.pattern_name}_{trade_signal.confluence_score:.0f}",
                        )
                        if result:
                            logger.info(f"Trade executed: {result}")
                    else:
                        logger.info(
                            f"[DRY RUN] Would execute: {trade_signal.direction} "
                            f"{lot_size} {symbol} @ {trade_signal.entry_price}"
                        )

                except Exception as e:
                    logger.error(f"Error processing {symbol}: {e}", exc_info=True)

            # Check and apply trailing stops
            if not args.dry_run:
                executor.check_and_trail(
                    risk_manager, data_engine,
                    config.get("indicators", {}),
                )

            # Sleep until next check (aligned to M15 bar boundaries)
            # Check every 60 seconds, but only process on new bars
            time.sleep(60)

    except Exception as e:
        logger.error(f"Unhandled error in main loop: {e}", exc_info=True)
    finally:
        # End-of-day: close all positions (day trading mode)
        if not args.dry_run:
            open_pos = executor.get_open_positions()
            if open_pos:
                logger.info(f"Closing {len(open_pos)} remaining positions (day trading mode)")
                executor.close_all_positions()

        data_engine.disconnect()
        logger.info("Strategy shutdown complete")


if __name__ == "__main__":
    main()
