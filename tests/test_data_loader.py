"""
Unit tests for backtest data loader.
"""

import os
import tempfile
import numpy as np
import pandas as pd
import pytest

from backtest.data_loader import load_csv, generate_synthetic_data


class TestGenerateSyntheticData:
    def test_returns_dataframe(self):
        df = generate_synthetic_data(bars=100, seed=42)
        assert isinstance(df, pd.DataFrame)

    def test_has_ohlcv_columns(self):
        df = generate_synthetic_data(bars=100)
        for col in ["open", "high", "low", "close", "volume"]:
            assert col in df.columns

    def test_correct_length(self):
        df = generate_synthetic_data(bars=500)
        assert len(df) == 500

    def test_high_above_low(self):
        df = generate_synthetic_data(bars=1000, seed=42)
        assert (df["high"] >= df["low"]).all()

    def test_high_above_open_close(self):
        df = generate_synthetic_data(bars=1000, seed=42)
        assert (df["high"] >= df["open"]).all()
        assert (df["high"] >= df["close"]).all()

    def test_low_below_open_close(self):
        df = generate_synthetic_data(bars=1000, seed=42)
        assert (df["low"] <= df["open"]).all()
        assert (df["low"] <= df["close"]).all()

    def test_reproducible_with_seed(self):
        df1 = generate_synthetic_data(bars=100, seed=42)
        df2 = generate_synthetic_data(bars=100, seed=42)
        pd.testing.assert_frame_equal(df1, df2)

    def test_different_seeds_differ(self):
        df1 = generate_synthetic_data(bars=100, seed=42)
        df2 = generate_synthetic_data(bars=100, seed=99)
        assert not df1["close"].equals(df2["close"])

    def test_utc_index(self):
        df = generate_synthetic_data(bars=100)
        assert df.index.tz is not None
        assert str(df.index.tz) == "UTC"

    def test_volume_positive(self):
        df = generate_synthetic_data(bars=100)
        assert (df["volume"] >= 0).all()


class TestLoadCsv:
    def test_load_valid_csv(self):
        df = generate_synthetic_data(bars=100, seed=42)
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
            df.to_csv(f.name)
            path = f.name

        loaded = load_csv(path)
        assert loaded is not None
        assert len(loaded) == 100
        os.unlink(path)

    def test_load_missing_file(self):
        result = load_csv("/nonexistent/file.csv")
        assert result is None

    def test_loads_with_correct_columns(self):
        df = generate_synthetic_data(bars=50, seed=42)
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
            df.to_csv(f.name)
            path = f.name

        loaded = load_csv(path)
        for col in ["open", "high", "low", "close"]:
            assert col in loaded.columns
        os.unlink(path)
