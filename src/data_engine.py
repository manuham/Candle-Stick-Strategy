"""
Data Engine: MT5 connection management and multi-timeframe OHLCV data fetching.

Provides both live MT5 connectivity and a fallback mode for backtesting
where data is loaded from pandas DataFrames directly.
"""

import logging
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd
import pytz

from src.utils import resolve_timeframe, UTC

logger = logging.getLogger("strategy")

# Try to import MetaTrader5; if unavailable, the engine runs in offline mode.
try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    mt5 = None
    MT5_AVAILABLE = False
    logger.warning("MetaTrader5 package not available — running in offline/backtest mode only")


class DataEngine:
    """Manages MT5 connection and provides OHLCV data."""

    def __init__(self, config: dict):
        self.config = config
        self.connected = False
        self._tf_config = config.get("timeframes", {})
        self._bar_counts = config.get("bar_counts", {})

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    def connect(self) -> bool:
        """Initialize MT5 terminal connection."""
        if not MT5_AVAILABLE:
            logger.error("MetaTrader5 package is not installed")
            return False

        if not mt5.initialize():
            logger.error(f"MT5 initialization failed: {mt5.last_error()}")
            return False

        account_info = mt5.account_info()
        if account_info is None:
            logger.error("Failed to get account info")
            mt5.shutdown()
            return False

        self.connected = True
        logger.info(
            f"Connected to MT5 — Account: {account_info.login}, "
            f"Balance: {account_info.balance:.2f} {account_info.currency}"
        )
        return True

    def disconnect(self):
        """Shutdown MT5 connection."""
        if MT5_AVAILABLE and self.connected:
            mt5.shutdown()
            self.connected = False
            logger.info("Disconnected from MT5")

    # ------------------------------------------------------------------
    # Data Fetching
    # ------------------------------------------------------------------

    def get_rates(
        self,
        symbol: str,
        timeframe: str,
        count: int = 500,
    ) -> Optional[pd.DataFrame]:
        """
        Fetch OHLCV bars from MT5.

        Args:
            symbol: Trading instrument (e.g., 'EURUSD').
            timeframe: Timeframe string (e.g., 'H1', 'M15', 'D1').
            count: Number of bars to retrieve.

        Returns:
            DataFrame with columns: time, open, high, low, close, tick_volume, spread, real_volume.
            Index is UTC datetime. Returns None on failure.
        """
        if not MT5_AVAILABLE or not self.connected:
            logger.error("MT5 not connected — cannot fetch rates")
            return None

        tf_int = resolve_timeframe(timeframe)

        # Ensure symbol is visible in Market Watch
        if not mt5.symbol_select(symbol, True):
            logger.warning(f"Failed to select symbol {symbol}")
            return None

        rates = mt5.copy_rates_from_pos(symbol, tf_int, 0, count)
        if rates is None or len(rates) == 0:
            err = mt5.last_error()
            logger.error(f"Failed to fetch {symbol} {timeframe}: {err}")
            return None

        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
        df.set_index("time", inplace=True)
        df.rename(
            columns={"tick_volume": "volume"},
            inplace=True,
        )
        return df

    def get_multi_tf_data(self, symbol: str) -> dict:
        """
        Fetch data for all configured timeframes (trend, signal, entry).

        Returns:
            Dict with keys 'trend', 'signal', 'entry' mapping to DataFrames.
            Missing timeframes map to None.
        """
        result = {}
        for role, tf_str in self._tf_config.items():
            count = self._bar_counts.get(role, 500)
            result[role] = self.get_rates(symbol, tf_str, count)
        return result

    def get_tick(self, symbol: str) -> Optional[dict]:
        """Get the latest bid/ask tick for a symbol."""
        if not MT5_AVAILABLE or not self.connected:
            return None
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            return None
        return {
            "bid": tick.bid,
            "ask": tick.ask,
            "last": tick.last,
            "time": datetime.fromtimestamp(tick.time, tz=timezone.utc),
        }

    def get_symbol_info(self, symbol: str) -> Optional[dict]:
        """Get symbol trading specifications."""
        if not MT5_AVAILABLE or not self.connected:
            return None
        info = mt5.symbol_info(symbol)
        if info is None:
            return None
        return {
            "name": info.name,
            "digits": info.digits,
            "point": info.point,
            "trade_tick_value": info.trade_tick_value,
            "trade_tick_size": info.trade_tick_size,
            "volume_min": info.volume_min,
            "volume_max": info.volume_max,
            "volume_step": info.volume_step,
            "spread": info.spread,
            "trade_contract_size": info.trade_contract_size,
        }

    def get_account_info(self) -> Optional[dict]:
        """Get current account information."""
        if not MT5_AVAILABLE or not self.connected:
            return None
        info = mt5.account_info()
        if info is None:
            return None
        return {
            "balance": info.balance,
            "equity": info.equity,
            "margin": info.margin,
            "free_margin": info.margin_free,
            "profit": info.profit,
            "currency": info.currency,
            "login": info.login,
        }

    # ------------------------------------------------------------------
    # Backtest helpers — accept DataFrames directly
    # ------------------------------------------------------------------

    @staticmethod
    def dataframe_from_arrays(
        time: np.ndarray,
        open_: np.ndarray,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        volume: np.ndarray,
    ) -> pd.DataFrame:
        """Build a standard OHLCV DataFrame from numpy arrays."""
        df = pd.DataFrame({
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        })
        df.index = pd.to_datetime(time, unit="s", utc=True)
        df.index.name = "time"
        return df
