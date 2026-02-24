"""
Unit tests for utility functions.
"""

import pytest
from datetime import datetime, timezone

from src.utils import (
    load_config,
    utc_now,
    get_session,
    get_all_symbols,
    resolve_timeframe,
    get_correlated_symbols,
)


class TestLoadConfig:
    def test_loads_valid_config(self):
        config = load_config("config/settings.yaml")
        assert isinstance(config, dict)
        assert "symbols" in config
        assert "indicators" in config

    def test_raises_on_missing_file(self):
        with pytest.raises(FileNotFoundError):
            load_config("nonexistent.yaml")


class TestUtcNow:
    def test_returns_utc(self):
        now = utc_now()
        assert now.tzinfo is not None
        assert isinstance(now, datetime)


class TestGetSession:
    def test_asian_session(self):
        assert get_session(3) == "ASIAN"

    def test_london_session(self):
        assert get_session(10) == "LONDON"

    def test_new_york_session(self):
        # NY session is 13-21 UTC; hour 15 overlaps with London (8-16)
        # get_session returns the first match, which is LONDON at hour 15
        assert get_session(17) == "NEW_YORK"

    def test_off_hours(self):
        assert get_session(23) == "OFF_HOURS"


class TestGetAllSymbols:
    def test_returns_flat_list(self, config):
        symbols = get_all_symbols(config)
        assert isinstance(symbols, list)
        assert len(symbols) > 0
        assert "EURUSD" in symbols

    def test_includes_all_groups(self, config):
        symbols = get_all_symbols(config)
        # Should include forex, indices, and commodities
        assert "US500" in symbols
        assert "XAUUSD" in symbols


class TestResolveTimeframe:
    def test_h1(self):
        assert resolve_timeframe("H1") == 16385

    def test_h4(self):
        assert resolve_timeframe("H4") == 16388

    def test_m15(self):
        assert resolve_timeframe("M15") == 15

    def test_d1(self):
        assert resolve_timeframe("D1") == 16408

    def test_case_insensitive(self):
        assert resolve_timeframe("h1") == 16385

    def test_unknown_raises(self):
        with pytest.raises(ValueError):
            resolve_timeframe("X99")


class TestGetCorrelatedSymbols:
    def test_eurusd_correlations(self):
        related = get_correlated_symbols("EURUSD")
        assert isinstance(related, list)
        assert "EURUSD" not in related  # Should not include itself
        # EURUSD is in USD and EUR groups, so should find some correlated symbols
        assert len(related) > 0

    def test_unknown_symbol(self):
        related = get_correlated_symbols("UNKNOWN_SYMBOL")
        assert related == []
