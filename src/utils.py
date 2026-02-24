"""
Utility functions: logging setup, config loading, time helpers.
"""

import os
import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime, timezone
from pathlib import Path

import yaml
import pytz


UTC = pytz.timezone("Etc/UTC")

# Market session boundaries (UTC hours)
SESSIONS = {
    "ASIAN": (0, 8),
    "LONDON": (8, 16),
    "NEW_YORK": (13, 21),
}

# Currency groupings for correlation checks
CURRENCY_GROUPS = {
    "USD": ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "US500", "US100", "XAUUSD", "XTIUSD"],
    "EUR": ["EURUSD", "GER40"],
    "GBP": ["GBPUSD"],
    "JPY": ["USDJPY"],
    "AUD": ["AUDUSD"],
}

# MT5 timeframe string mapping
TIMEFRAME_MAP = {
    "M1": 1, "M5": 5, "M15": 15, "M30": 30,
    "H1": 16385, "H4": 16388, "H12": 16396,
    "D1": 16408, "W1": 32769, "MN1": 49153,
}


def load_config(path: str = "config/settings.yaml") -> dict:
    """Load YAML configuration file."""
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def setup_logging(config: dict) -> logging.Logger:
    """Configure rotating file + console logging."""
    log_cfg = config.get("logging", {})
    log_file = log_cfg.get("file", "logs/strategy.log")
    log_level = getattr(logging, log_cfg.get("level", "INFO"))
    max_bytes = log_cfg.get("max_bytes", 10_485_760)
    backup_count = log_cfg.get("backup_count", 5)

    # Ensure log directory exists
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    logger = logging.getLogger("strategy")
    logger.setLevel(log_level)

    # Avoid duplicate handlers on repeated calls
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # File handler with rotation
    fh = RotatingFileHandler(log_file, maxBytes=max_bytes, backupCount=backup_count)
    fh.setLevel(log_level)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(log_level)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    return logger


def utc_now() -> datetime:
    """Return current UTC datetime (timezone-aware)."""
    return datetime.now(timezone.utc)


def get_session(hour: int) -> str:
    """Return the trading session name for a given UTC hour."""
    for session, (start, end) in SESSIONS.items():
        if start <= hour < end:
            return session
    return "OFF_HOURS"


def get_all_symbols(config: dict) -> list:
    """Flatten all symbol lists from config into a single list."""
    symbols = []
    for group in config.get("symbols", {}).values():
        symbols.extend(group)
    return symbols


def resolve_timeframe(tf_string: str) -> int:
    """Convert timeframe string (e.g., 'H1') to MT5 integer constant."""
    val = TIMEFRAME_MAP.get(tf_string.upper())
    if val is None:
        raise ValueError(f"Unknown timeframe: {tf_string}")
    return val


def get_correlated_symbols(symbol: str) -> list:
    """Return symbols that share a currency with the given symbol."""
    related = set()
    for currency, syms in CURRENCY_GROUPS.items():
        if symbol in syms:
            related.update(syms)
    related.discard(symbol)
    return list(related)
