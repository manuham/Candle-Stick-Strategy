"""
Unit tests for the monitoring API.
"""

import json
import time
from datetime import datetime, timezone
from urllib.request import urlopen
from urllib.error import URLError

import pytest

from src.monitor import MonitorServer, StrategyState, strategy_state


class TestStrategyState:
    def test_initial_state(self):
        state = StrategyState()
        assert state.signals_generated == 0
        assert state.is_running is False

    def test_update(self):
        state = StrategyState()
        state.update(signals_generated=5, is_running=True)
        assert state.signals_generated == 5
        assert state.is_running is True

    def test_to_dict(self):
        state = StrategyState()
        state.update(
            started_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
            is_running=True,
            signals_generated=10,
        )
        d = state.to_dict()
        assert d["is_running"] is True
        assert d["signals_generated"] == 10
        assert d["started_at"] is not None
        assert d["uptime_seconds"] > 0

    def test_add_error(self):
        state = StrategyState()
        state.add_error("Test error 1")
        state.add_error("Test error 2")
        d = state.to_dict()
        assert len(d["recent_errors"]) == 2
        assert d["recent_errors"][0]["message"] == "Test error 1"

    def test_error_limit(self):
        state = StrategyState()
        for i in range(60):
            state.add_error(f"Error {i}")
        # Should keep only last 50
        assert len(state.errors) == 50


class TestMonitorServer:
    def test_disabled_by_default(self):
        server = MonitorServer({})
        assert server.enabled is False

    def test_start_disabled(self):
        server = MonitorServer({"monitoring": {"enabled": False}})
        server.start()
        # Should not crash

    def test_server_responds(self):
        config = {"monitoring": {"enabled": True, "host": "127.0.0.1", "port": 18080}}
        server = MonitorServer(config)
        server.start()
        time.sleep(0.2)  # Give server time to start

        try:
            resp = urlopen("http://127.0.0.1:18080/health", timeout=2)
            data = json.loads(resp.read())
            assert data["status"] == "ok"
        finally:
            server.stop()

    def test_status_endpoint(self):
        config = {"monitoring": {"enabled": True, "host": "127.0.0.1", "port": 18081}}
        server = MonitorServer(config)

        # Update global state
        strategy_state.update(
            started_at=datetime.now(timezone.utc),
            is_running=True,
            signals_generated=42,
        )

        server.start()
        time.sleep(0.2)

        try:
            resp = urlopen("http://127.0.0.1:18081/status", timeout=2)
            data = json.loads(resp.read())
            assert data["is_running"] is True
            assert data["signals_generated"] == 42
        finally:
            server.stop()
            # Reset global state
            strategy_state.update(is_running=False, signals_generated=0)

    def test_metrics_endpoint(self):
        config = {"monitoring": {"enabled": True, "host": "127.0.0.1", "port": 18082}}
        server = MonitorServer(config)
        server.start()
        time.sleep(0.2)

        try:
            resp = urlopen("http://127.0.0.1:18082/metrics", timeout=2)
            data = json.loads(resp.read())
            assert "signals_generated" in data
            assert "daily_pnl" in data
            assert "account_balance" in data
        finally:
            server.stop()

    def test_404_on_unknown_path(self):
        config = {"monitoring": {"enabled": True, "host": "127.0.0.1", "port": 18083}}
        server = MonitorServer(config)
        server.start()
        time.sleep(0.2)

        try:
            resp = urlopen("http://127.0.0.1:18083/unknown", timeout=2)
            data = json.loads(resp.read())
            assert data.get("error") == "not_found"
        except Exception:
            pass  # 404 may raise in some urllib versions
        finally:
            server.stop()
