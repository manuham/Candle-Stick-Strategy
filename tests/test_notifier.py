"""
Unit tests for the notification system.
"""

import json
from unittest.mock import patch, MagicMock
import pytest

from src.notifier import Notifier


@pytest.fixture
def notifier_disabled():
    config = {"notifications": {"enabled": False}}
    return Notifier(config)


@pytest.fixture
def notifier_telegram():
    config = {
        "notifications": {
            "enabled": True,
            "telegram": {
                "enabled": True,
                "bot_token": "test_token",
                "chat_id": "12345",
            },
            "discord": {"enabled": False},
        }
    }
    return Notifier(config)


@pytest.fixture
def notifier_discord():
    config = {
        "notifications": {
            "enabled": True,
            "telegram": {"enabled": False},
            "discord": {
                "enabled": True,
                "webhook_url": "https://discord.com/api/webhooks/test",
            },
        }
    }
    return Notifier(config)


class TestNotifierInit:
    def test_disabled_by_default(self):
        n = Notifier({})
        assert n.enabled is False

    def test_telegram_config(self, notifier_telegram):
        assert notifier_telegram.tg_enabled is True
        assert notifier_telegram.tg_bot_token == "test_token"

    def test_discord_config(self, notifier_discord):
        assert notifier_discord.dc_enabled is True


class TestNotifySignal:
    @patch("src.notifier.Notifier._http_post", return_value=True)
    def test_sends_telegram(self, mock_post, notifier_telegram):
        signal = MagicMock()
        signal.direction = "BUY"
        signal.symbol = "EURUSD"
        signal.pattern_name = "hammer"
        signal.confluence_score = 75.0
        signal.ml_probability = 0.72
        signal.entry_price = 1.1000
        signal.suggested_sl = 1.0985
        signal.suggested_tp = 1.1030
        signal.risk_reward_ratio = 2.0

        notifier_telegram.notify_signal(signal)
        mock_post.assert_called_once()
        args = mock_post.call_args
        assert "EURUSD" in args[0][1]["text"]

    def test_does_nothing_when_disabled(self, notifier_disabled):
        signal = MagicMock()
        signal.direction = "BUY"
        signal.symbol = "EURUSD"
        signal.pattern_name = "hammer"
        signal.confluence_score = 75.0
        signal.ml_probability = 0.7
        signal.entry_price = 1.1000
        signal.suggested_sl = 1.0985
        signal.suggested_tp = 1.1030
        signal.risk_reward_ratio = 2.0
        # Should not raise or send anything
        notifier_disabled.notify_signal(signal)


class TestNotifyOrderOpened:
    @patch("src.notifier.Notifier._http_post", return_value=True)
    def test_sends_message(self, mock_post, notifier_telegram):
        order = {
            "direction": "BUY", "volume": 0.1, "symbol": "EURUSD",
            "price": 1.1000, "sl": 1.0985, "tp": 1.1030, "ticket": 12345,
        }
        notifier_telegram.notify_order_opened(order)
        mock_post.assert_called_once()


class TestNotifyOrderClosed:
    @patch("src.notifier.Notifier._http_post", return_value=True)
    def test_win(self, mock_post, notifier_telegram):
        notifier_telegram.notify_order_closed("EURUSD", "BUY", 150.0, "TP")
        mock_post.assert_called_once()
        assert "WIN" in mock_post.call_args[0][1]["text"]

    @patch("src.notifier.Notifier._http_post", return_value=True)
    def test_loss(self, mock_post, notifier_telegram):
        notifier_telegram.notify_order_closed("EURUSD", "BUY", -100.0, "SL")
        mock_post.assert_called_once()
        assert "LOSS" in mock_post.call_args[0][1]["text"]


class TestNotifyDailySummary:
    @patch("src.notifier.Notifier._http_post", return_value=True)
    def test_sends_summary(self, mock_post, notifier_telegram):
        notifier_telegram.notify_daily_summary("2024-01-15", 5, 250.0, 10250.0, 0.6)
        mock_post.assert_called_once()


class TestNotifyError:
    @patch("src.notifier.Notifier._http_post", return_value=True)
    def test_sends_error(self, mock_post, notifier_telegram):
        notifier_telegram.notify_error("MT5 connection lost")
        mock_post.assert_called_once()


class TestDiscordNotification:
    @patch("src.notifier.Notifier._http_post", return_value=True)
    def test_sends_to_discord(self, mock_post, notifier_discord):
        notifier_discord.notify_error("Test error")
        mock_post.assert_called_once()
        args = mock_post.call_args
        assert "content" in args[0][1]


class TestHttpPost:
    @patch("src.notifier.urlopen")
    def test_success(self, mock_urlopen):
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        result = Notifier._http_post("https://example.com", {"test": True}, "Test")
        assert result is True

    def test_invalid_url_fails_gracefully(self):
        result = Notifier._http_post("https://invalid.invalid.invalid", {"test": True}, "Test")
        assert result is False
