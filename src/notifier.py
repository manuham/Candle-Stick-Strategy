"""
Notification System — Telegram and Discord alerts for trade signals, errors, and daily summaries.

Supports:
  - Telegram Bot API (via HTTP POST)
  - Discord Webhooks (via HTTP POST)

Configuration via settings.yaml:
  notifications:
    enabled: true
    telegram:
      enabled: true
      bot_token: "YOUR_BOT_TOKEN"
      chat_id: "YOUR_CHAT_ID"
    discord:
      enabled: true
      webhook_url: "YOUR_WEBHOOK_URL"
"""

import json
import logging
from typing import Optional
from urllib.request import Request, urlopen
from urllib.error import URLError

logger = logging.getLogger("strategy")

# Timeout for HTTP requests (seconds)
REQUEST_TIMEOUT = 10


class Notifier:
    """Sends trade notifications via Telegram and/or Discord."""

    def __init__(self, config: dict):
        notif_cfg = config.get("notifications", {})
        self.enabled = notif_cfg.get("enabled", False)

        # Telegram
        tg_cfg = notif_cfg.get("telegram", {})
        self.tg_enabled = tg_cfg.get("enabled", False) and self.enabled
        self.tg_bot_token = tg_cfg.get("bot_token", "")
        self.tg_chat_id = tg_cfg.get("chat_id", "")

        # Discord
        dc_cfg = notif_cfg.get("discord", {})
        self.dc_enabled = dc_cfg.get("enabled", False) and self.enabled
        self.dc_webhook_url = dc_cfg.get("webhook_url", "")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def notify_signal(self, signal) -> None:
        """Send a trade signal notification."""
        direction_emoji = "BUY" if signal.direction == "BUY" else "SELL"
        message = (
            f"SIGNAL: {direction_emoji} {signal.symbol}\n"
            f"Pattern: {signal.pattern_name}\n"
            f"Confluence: {signal.confluence_score:.0f}/100\n"
            f"ML Prob: {signal.ml_probability:.1%}\n"
            f"Entry: {signal.entry_price}\n"
            f"SL: {signal.suggested_sl} | TP: {signal.suggested_tp}\n"
            f"R:R: {signal.risk_reward_ratio:.1f}"
        )
        self._send(message)

    def notify_order_opened(self, order_result: dict) -> None:
        """Send notification when an order is opened."""
        message = (
            f"ORDER OPENED: {order_result['direction']} "
            f"{order_result['volume']} {order_result['symbol']}\n"
            f"Price: {order_result['price']}\n"
            f"SL: {order_result['sl']} | TP: {order_result['tp']}\n"
            f"Ticket: {order_result['ticket']}"
        )
        self._send(message)

    def notify_order_closed(
        self,
        symbol: str,
        direction: str,
        pnl: float,
        reason: str,
    ) -> None:
        """Send notification when an order is closed."""
        result_text = "WIN" if pnl > 0 else "LOSS"
        message = (
            f"ORDER CLOSED: {direction} {symbol}\n"
            f"P&L: ${pnl:+.2f} ({result_text})\n"
            f"Reason: {reason}"
        )
        self._send(message)

    def notify_daily_summary(
        self,
        date: str,
        trades_today: int,
        daily_pnl: float,
        balance: float,
        win_rate: float,
    ) -> None:
        """Send end-of-day summary."""
        message = (
            f"DAILY SUMMARY — {date}\n"
            f"Trades: {trades_today}\n"
            f"P&L: ${daily_pnl:+.2f}\n"
            f"Balance: ${balance:,.2f}\n"
            f"Win Rate: {win_rate:.0%}"
        )
        self._send(message)

    def notify_error(self, error_message: str) -> None:
        """Send error/alert notification."""
        message = f"ALERT: {error_message}"
        self._send(message)

    def notify_drawdown_warning(self, current_dd_pct: float, limit_pct: float) -> None:
        """Send drawdown warning."""
        message = (
            f"DRAWDOWN WARNING\n"
            f"Current: {current_dd_pct:.1%}\n"
            f"Limit: {limit_pct:.1%}\n"
            f"Trading may be halted."
        )
        self._send(message)

    # ------------------------------------------------------------------
    # Transport
    # ------------------------------------------------------------------

    def _send(self, message: str) -> None:
        """Send message to all enabled channels."""
        if not self.enabled:
            return

        if self.tg_enabled:
            self._send_telegram(message)
        if self.dc_enabled:
            self._send_discord(message)

    def _send_telegram(self, message: str) -> bool:
        """Send message via Telegram Bot API."""
        if not self.tg_bot_token or not self.tg_chat_id:
            logger.warning("Telegram not configured (missing bot_token or chat_id)")
            return False

        url = f"https://api.telegram.org/bot{self.tg_bot_token}/sendMessage"
        payload = {
            "chat_id": self.tg_chat_id,
            "text": message,
            "parse_mode": "HTML",
        }
        return self._http_post(url, payload, "Telegram")

    def _send_discord(self, message: str) -> bool:
        """Send message via Discord Webhook."""
        if not self.dc_webhook_url:
            logger.warning("Discord webhook URL not configured")
            return False

        payload = {"content": message}
        return self._http_post(self.dc_webhook_url, payload, "Discord")

    @staticmethod
    def _http_post(url: str, payload: dict, channel_name: str) -> bool:
        """Execute an HTTP POST request with JSON payload."""
        data = json.dumps(payload).encode("utf-8")
        req = Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
                if resp.status < 300:
                    return True
                logger.warning(f"{channel_name} returned status {resp.status}")
                return False
        except URLError as e:
            logger.error(f"{channel_name} notification failed: {e}")
            return False
        except Exception as e:
            logger.error(f"{channel_name} notification error: {e}")
            return False
