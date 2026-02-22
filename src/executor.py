"""
Order Executor — MT5 trade execution with error handling, retries, and position management.
"""

import logging
import time
from typing import List, Optional

logger = logging.getLogger("strategy")

try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    mt5 = None
    MT5_AVAILABLE = False


class Executor:
    """Handles all MT5 order operations."""

    def __init__(self, config: dict):
        exec_cfg = config.get("execution", {})
        self.magic = exec_cfg.get("magic_number", 234000)
        self.comment_prefix = exec_cfg.get("comment_prefix", "CS_")
        self.slippage = exec_cfg.get("slippage_points", 10)
        self.max_retries = exec_cfg.get("max_retries", 3)
        self.retry_delay = exec_cfg.get("retry_delay_seconds", 1)

    # ------------------------------------------------------------------
    # Order Execution
    # ------------------------------------------------------------------

    def open_position(
        self,
        symbol: str,
        direction: str,
        lot_size: float,
        sl_price: float,
        tp_price: float,
        comment: str = "",
    ) -> Optional[dict]:
        """
        Open a market position on MT5.

        Args:
            symbol: Trading instrument.
            direction: "BUY" or "SELL".
            lot_size: Volume in lots.
            sl_price: Stop loss price.
            tp_price: Take profit price.
            comment: Order comment.

        Returns:
            Dict with order result or None on failure.
        """
        if not MT5_AVAILABLE:
            logger.error("MT5 not available — cannot execute orders")
            return None

        # Get current price
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            logger.error(f"Cannot get tick for {symbol}")
            return None

        if direction == "BUY":
            order_type = mt5.ORDER_TYPE_BUY
            price = tick.ask
        elif direction == "SELL":
            order_type = mt5.ORDER_TYPE_SELL
            price = tick.bid
        else:
            logger.error(f"Invalid direction: {direction}")
            return None

        full_comment = f"{self.comment_prefix}{comment}"

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": lot_size,
            "type": order_type,
            "price": price,
            "sl": sl_price,
            "tp": tp_price,
            "deviation": self.slippage,
            "magic": self.magic,
            "comment": full_comment,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        # Retry loop for requotes
        for attempt in range(1, self.max_retries + 1):
            # Pre-check
            check = mt5.order_check(request)
            if check is None:
                logger.error(f"Order check returned None for {symbol}")
                return None

            if check.retcode != 0:
                logger.warning(
                    f"Order check failed (attempt {attempt}): "
                    f"retcode={check.retcode}, comment={check.comment}"
                )
                if attempt < self.max_retries:
                    time.sleep(self.retry_delay)
                    # Refresh price
                    tick = mt5.symbol_info_tick(symbol)
                    if tick:
                        request["price"] = tick.ask if direction == "BUY" else tick.bid
                    continue
                return None

            result = mt5.order_send(request)
            if result is None:
                logger.error(f"order_send returned None for {symbol}")
                return None

            if result.retcode == mt5.TRADE_RETCODE_DONE:
                order_result = {
                    "ticket": result.order,
                    "deal": result.deal,
                    "symbol": symbol,
                    "direction": direction,
                    "volume": lot_size,
                    "price": result.price,
                    "sl": sl_price,
                    "tp": tp_price,
                    "comment": full_comment,
                }
                logger.info(
                    f"ORDER OPENED: {direction} {lot_size} {symbol} @ {result.price} "
                    f"| SL={sl_price} TP={tp_price} | Ticket={result.order}"
                )
                return order_result

            elif result.retcode == mt5.TRADE_RETCODE_REQUOTE:
                logger.warning(f"Requote on {symbol} (attempt {attempt})")
                if attempt < self.max_retries:
                    time.sleep(self.retry_delay)
                    tick = mt5.symbol_info_tick(symbol)
                    if tick:
                        request["price"] = tick.ask if direction == "BUY" else tick.bid
                    continue

            else:
                logger.error(
                    f"Order failed: retcode={result.retcode}, "
                    f"comment={result.comment}"
                )
                return None

        logger.error(f"Max retries reached for {symbol} {direction}")
        return None

    def close_position(self, ticket: int, symbol: str = None) -> bool:
        """
        Close a specific position by ticket.

        Returns True if successfully closed.
        """
        if not MT5_AVAILABLE:
            return False

        # Get position info
        position = mt5.positions_get(ticket=ticket)
        if not position or len(position) == 0:
            logger.warning(f"Position {ticket} not found")
            return False

        pos = position[0]
        symbol = symbol or pos.symbol

        # Determine close direction (opposite of position)
        if pos.type == mt5.ORDER_TYPE_BUY:
            close_type = mt5.ORDER_TYPE_SELL
            tick = mt5.symbol_info_tick(symbol)
            price = tick.bid if tick else 0
        else:
            close_type = mt5.ORDER_TYPE_BUY
            tick = mt5.symbol_info_tick(symbol)
            price = tick.ask if tick else 0

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": pos.volume,
            "type": close_type,
            "position": ticket,
            "price": price,
            "deviation": self.slippage,
            "magic": self.magic,
            "comment": f"{self.comment_prefix}close",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        result = mt5.order_send(request)
        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
            logger.info(f"Position {ticket} closed @ {result.price}")
            return True

        logger.error(
            f"Failed to close position {ticket}: "
            f"retcode={result.retcode if result else 'None'}"
        )
        return False

    def modify_position(
        self,
        ticket: int,
        symbol: str,
        new_sl: Optional[float] = None,
        new_tp: Optional[float] = None,
    ) -> bool:
        """
        Modify SL/TP of an existing position.

        Returns True if successfully modified.
        """
        if not MT5_AVAILABLE:
            return False

        position = mt5.positions_get(ticket=ticket)
        if not position:
            logger.warning(f"Position {ticket} not found for modification")
            return False

        pos = position[0]

        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "symbol": symbol,
            "position": ticket,
            "sl": new_sl if new_sl is not None else pos.sl,
            "tp": new_tp if new_tp is not None else pos.tp,
        }

        result = mt5.order_send(request)
        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
            logger.debug(
                f"Position {ticket} modified: SL={request['sl']}, TP={request['tp']}"
            )
            return True

        logger.error(
            f"Failed to modify position {ticket}: "
            f"retcode={result.retcode if result else 'None'}"
        )
        return False

    def close_all_positions(self) -> int:
        """
        Emergency close all positions managed by this strategy.

        Returns number of positions closed.
        """
        if not MT5_AVAILABLE:
            return 0

        positions = mt5.positions_get()
        if not positions:
            return 0

        closed = 0
        for pos in positions:
            if pos.magic == self.magic:
                if self.close_position(pos.ticket, pos.symbol):
                    closed += 1

        logger.info(f"Closed {closed} positions (emergency close all)")
        return closed

    def get_open_positions(self) -> List[dict]:
        """Get all open positions managed by this strategy."""
        if not MT5_AVAILABLE:
            return []

        positions = mt5.positions_get()
        if not positions:
            return []

        result = []
        for pos in positions:
            if pos.magic == self.magic:
                result.append({
                    "ticket": pos.ticket,
                    "symbol": pos.symbol,
                    "direction": "BUY" if pos.type == mt5.ORDER_TYPE_BUY else "SELL",
                    "volume": pos.volume,
                    "open_price": pos.price_open,
                    "current_price": pos.price_current,
                    "sl": pos.sl,
                    "tp": pos.tp,
                    "profit": pos.profit,
                    "swap": pos.swap,
                    "comment": pos.comment,
                    "time": pos.time,
                })

        return result

    def check_and_trail(
        self,
        risk_manager,
        data_engine,
        indicator_config: dict,
    ) -> int:
        """
        Check all open positions and apply trailing stop logic.

        Returns number of positions modified.
        """
        from src.indicators import calculate_all

        positions = self.get_open_positions()
        modified = 0

        for pos in positions:
            symbol = pos["symbol"]
            direction = pos["direction"]
            entry_price = pos["open_price"]
            current_price = pos["current_price"]
            current_sl = pos["sl"]

            # Get current ATR
            df = data_engine.get_rates(symbol, "H1", 50)
            if df is None:
                continue

            df = calculate_all(df, indicator_config)
            atr_value = df["atr"].iloc[-1]

            if atr_value is None or atr_value == 0:
                continue

            new_sl = risk_manager.calculate_trailing_sl(
                direction=direction,
                entry_price=entry_price,
                current_price=current_price,
                current_sl=current_sl,
                atr_value=atr_value,
            )

            if new_sl is not None:
                if self.modify_position(pos["ticket"], symbol, new_sl=new_sl):
                    modified += 1

        if modified:
            logger.info(f"Trailing stops updated for {modified} position(s)")

        return modified
