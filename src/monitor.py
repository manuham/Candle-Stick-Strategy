"""
Monitoring API — Lightweight HTTP server for health checks and status queries.

Endpoints:
  GET /health          → 200 OK with uptime
  GET /status          → Strategy status (positions, P&L, signals)
  GET /metrics         → Performance metrics (daily, cumulative)

Uses only the stdlib http.server — no external dependencies required.

Configuration via settings.yaml:
  monitoring:
    enabled: true
    host: "0.0.0.0"
    port: 8080
"""

import json
import logging
import threading
from datetime import datetime, timezone
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Optional

logger = logging.getLogger("strategy")


class StrategyState:
    """
    Shared state container that the main loop updates
    and the monitoring server reads.
    """

    def __init__(self):
        self.started_at: Optional[datetime] = None
        self.last_scan_time: Optional[datetime] = None
        self.signals_generated: int = 0
        self.trades_today: int = 0
        self.daily_pnl: float = 0.0
        self.account_balance: float = 0.0
        self.account_equity: float = 0.0
        self.open_positions: list = []
        self.last_signal: Optional[dict] = None
        self.errors: list = []
        self.is_running: bool = False
        self._lock = threading.Lock()

    def update(self, **kwargs):
        with self._lock:
            for key, value in kwargs.items():
                if hasattr(self, key):
                    setattr(self, key, value)

    def add_error(self, error: str):
        with self._lock:
            self.errors.append({
                "time": datetime.now(timezone.utc).isoformat(),
                "message": error,
            })
            # Keep only last 50 errors
            self.errors = self.errors[-50:]

    def to_dict(self) -> dict:
        with self._lock:
            uptime = None
            if self.started_at:
                uptime = (datetime.now(timezone.utc) - self.started_at).total_seconds()
            return {
                "started_at": self.started_at.isoformat() if self.started_at else None,
                "uptime_seconds": uptime,
                "is_running": self.is_running,
                "last_scan_time": self.last_scan_time.isoformat() if self.last_scan_time else None,
                "signals_generated": self.signals_generated,
                "trades_today": self.trades_today,
                "daily_pnl": self.daily_pnl,
                "account_balance": self.account_balance,
                "account_equity": self.account_equity,
                "open_positions": len(self.open_positions),
                "open_position_details": self.open_positions,
                "last_signal": self.last_signal,
                "recent_errors": self.errors[-10:],
            }


# Module-level state instance (shared between main loop and server)
strategy_state = StrategyState()


class MonitorHandler(BaseHTTPRequestHandler):
    """HTTP request handler for monitoring endpoints."""

    def do_GET(self):
        if self.path == "/health":
            self._respond_json(200, {
                "status": "ok",
                "uptime_seconds": self._uptime(),
            })
        elif self.path == "/status":
            self._respond_json(200, strategy_state.to_dict())
        elif self.path == "/metrics":
            self._respond_json(200, {
                "signals_generated": strategy_state.signals_generated,
                "trades_today": strategy_state.trades_today,
                "daily_pnl": strategy_state.daily_pnl,
                "account_balance": strategy_state.account_balance,
                "open_positions": len(strategy_state.open_positions),
            })
        else:
            self._respond_json(404, {"error": "not_found"})

    def _respond_json(self, status: int, data: dict):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(data, indent=2).encode("utf-8"))

    def _uptime(self) -> float:
        if strategy_state.started_at:
            return (datetime.now(timezone.utc) - strategy_state.started_at).total_seconds()
        return 0.0

    def log_message(self, format, *args):
        # Suppress default HTTP server logging to avoid noise
        pass


class MonitorServer:
    """Runs the monitoring HTTP server in a background thread."""

    def __init__(self, config: dict):
        mon_cfg = config.get("monitoring", {})
        self.enabled = mon_cfg.get("enabled", False)
        self.host = mon_cfg.get("host", "0.0.0.0")
        self.port = mon_cfg.get("port", 8080)
        self._server: Optional[HTTPServer] = None
        self._thread: Optional[threading.Thread] = None

    def start(self):
        """Start the monitoring server in a background thread."""
        if not self.enabled:
            logger.debug("Monitoring server disabled")
            return

        self._server = HTTPServer((self.host, self.port), MonitorHandler)
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            daemon=True,
            name="monitor-server",
        )
        self._thread.start()
        logger.info(f"Monitoring server started on {self.host}:{self.port}")

    def stop(self):
        """Stop the monitoring server."""
        if self._server:
            self._server.shutdown()
            logger.info("Monitoring server stopped")
