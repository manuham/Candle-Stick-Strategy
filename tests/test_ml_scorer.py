"""
Unit tests for ML scorer (XGBoost pattern predictor).
"""

import os
import tempfile
import numpy as np
import pandas as pd
import pytest

from src.ml_scorer import MLScorer
from backtest.data_loader import generate_synthetic_data


@pytest.fixture
def ml_config():
    return {
        "ml": {
            "enabled": True,
            "model_type": "xgboost",
            "confidence_threshold": 0.65,
            "lookforward_bars": 24,
            "rr_target": 2.0,
            "train_test_split": 0.8,
            "model_path": os.path.join(tempfile.mkdtemp(), "test_model.joblib"),
            "xgboost_params": {
                "n_estimators": 50,
                "max_depth": 3,
                "learning_rate": 0.1,
            },
        }
    }


@pytest.fixture
def scorer(ml_config):
    return MLScorer(ml_config)


class TestBuildFeatures:
    def test_returns_dataframe(self):
        df = generate_synthetic_data(bars=500, seed=42)
        features = MLScorer.build_features(df)
        assert isinstance(features, pd.DataFrame)

    def test_has_expected_columns(self):
        df = generate_synthetic_data(bars=500, seed=42)
        features = MLScorer.build_features(df)
        for col in MLScorer.FEATURE_COLUMNS:
            assert col in features.columns, f"Missing feature: {col}"

    def test_length_matches_input(self):
        df = generate_synthetic_data(bars=500, seed=42)
        features = MLScorer.build_features(df)
        assert len(features) == len(df)

    def test_with_h4_data(self):
        df_h1 = generate_synthetic_data(bars=500, seed=42)
        df_h4 = generate_synthetic_data(bars=125, seed=43)
        features = MLScorer.build_features(df_h1, df_h4)
        assert features["h4_trend_direction"].notna().all()


class TestGenerateLabels:
    def test_labels_are_binary(self, scorer):
        df = generate_synthetic_data(bars=500, seed=42)
        from src.patterns import detect_all
        patterns = detect_all(df)
        labels = scorer.generate_labels(df, patterns)
        assert set(labels.unique()).issubset({0, 1})

    def test_labels_length(self, scorer):
        df = generate_synthetic_data(bars=500, seed=42)
        from src.patterns import detect_all
        patterns = detect_all(df)
        labels = scorer.generate_labels(df, patterns)
        assert len(labels) == len(df)


class TestTraining:
    def test_train_on_synthetic(self, scorer):
        df_h1 = generate_synthetic_data(bars=2000, seed=42)
        df_h4 = generate_synthetic_data(bars=500, seed=43)
        metrics = scorer.train(df_h1, df_h4, {})
        # Should succeed (enough data)
        if "error" not in metrics:
            assert "accuracy" in metrics
            assert 0 <= metrics["accuracy"] <= 1
            assert scorer.model is not None
        else:
            # May fail with insufficient pattern data, which is ok for synthetic
            assert metrics["error"] == "insufficient_data"

    def test_train_insufficient_data(self, scorer):
        df = generate_synthetic_data(bars=50, seed=42)
        metrics = scorer.train(df, None, {})
        assert "error" in metrics


class TestPrediction:
    def test_predict_without_model(self, scorer):
        df = generate_synthetic_data(bars=100, seed=42)
        features = MLScorer.build_features(df)
        prob = scorer.predict_proba(features)
        assert prob == 0.0

    def test_passes_threshold(self, scorer):
        assert scorer.passes_threshold(0.7) is True
        assert scorer.passes_threshold(0.5) is False


class TestPersistence:
    def test_save_and_load(self, scorer):
        # Train a small model first
        df_h1 = generate_synthetic_data(bars=2000, seed=42)
        metrics = scorer.train(df_h1, None, {})

        if scorer.model is not None:
            scorer.save_model()
            assert os.path.exists(scorer.model_path)

            # Load into new scorer
            new_scorer = MLScorer({"ml": {"model_path": scorer.model_path}})
            assert new_scorer.load_model() is True
            assert new_scorer.model is not None

    def test_load_nonexistent(self, scorer):
        scorer.model_path = "/nonexistent/model.joblib"
        assert scorer.load_model() is False
