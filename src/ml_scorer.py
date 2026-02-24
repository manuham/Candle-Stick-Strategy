"""
ML Pattern Scorer — XGBoost model for predicting candlestick pattern outcomes.

The model is trained on historical pattern occurrences with engineered features
and predicts the probability of a profitable trade (achieving target R:R within
a given number of forward bars).
"""

import os
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report,
)

from src.patterns import detect_all as detect_patterns, candle_properties
from src.indicators import calculate_all as calculate_indicators

logger = logging.getLogger("strategy")


class MLScorer:
    """XGBoost-based pattern outcome predictor."""

    FEATURE_COLUMNS = [
        # Pattern features
        "pattern_strength",
        "pattern_is_bullish",
        # Candle morphology
        "body_ratio",
        "upper_shadow_ratio",
        "lower_shadow_ratio",
        "range_vs_atr",
        # Indicator state
        "rsi",
        "macd_histogram",
        "bb_pctb",
        "stoch_k",
        "adx",
        # Context
        "trend_direction",        # +1 bullish, -1 bearish
        "atr_rank",
        "volume_ratio",
        # Multi-TF features (passed in when available)
        "h4_trend_direction",
        "h4_rsi",
        "h4_ema_distance_pct",
        # Time features
        "hour_of_day",
        "day_of_week",
    ]

    def __init__(self, config: dict):
        self.config = config.get("ml", {})
        self.model = None
        self.model_path = self.config.get("model_path", "models/xgb_model.joblib")
        self.confidence_threshold = self.config.get("confidence_threshold", 0.65)
        self.lookforward_bars = self.config.get("lookforward_bars", 24)
        self.rr_target = self.config.get("rr_target", 2.0)

    # ------------------------------------------------------------------
    # Feature Engineering
    # ------------------------------------------------------------------

    @staticmethod
    def build_features(
        df: pd.DataFrame,
        h4_df: Optional[pd.DataFrame] = None,
        indicator_config: dict = None,
    ) -> pd.DataFrame:
        """
        Build ML feature matrix from OHLCV data with patterns and indicators.

        Args:
            df: H1 OHLCV DataFrame (signal timeframe).
            h4_df: H4 OHLCV DataFrame for multi-TF features (optional).
            indicator_config: Indicator settings dict.

        Returns:
            DataFrame with one row per bar, containing all feature columns.
        """
        if indicator_config is None:
            indicator_config = {}

        # Calculate indicators
        df_ind = calculate_indicators(df.copy(), indicator_config)

        # Calculate patterns
        patterns = detect_patterns(df)

        # Candle properties
        props = candle_properties(df)

        features = pd.DataFrame(index=df.index)

        # Pattern features
        features["pattern_strength"] = patterns["dominant_signal"].abs()
        features["pattern_is_bullish"] = (patterns["dominant_signal"] > 0).astype(int)

        # Candle morphology
        features["body_ratio"] = props["body_ratio"]
        safe_range = props["candle_range"].replace(0, np.nan)
        features["upper_shadow_ratio"] = props["upper_shadow"] / safe_range
        features["lower_shadow_ratio"] = props["lower_shadow"] / safe_range

        atr = df_ind.get("atr")
        if atr is not None:
            safe_atr = atr.replace(0, np.nan)
            features["range_vs_atr"] = props["candle_range"] / safe_atr
        else:
            features["range_vs_atr"] = 1.0

        # Indicator state
        features["rsi"] = df_ind.get("rsi", 50)
        features["macd_histogram"] = df_ind.get("macd_histogram", 0)
        features["bb_pctb"] = df_ind.get("bb_pctb", 0.5)
        features["stoch_k"] = df_ind.get("stoch_k", 50)
        features["adx"] = df_ind.get("adx", 25)

        # Context
        ema_fast = df_ind.get("ema_fast")
        ema_slow = df_ind.get("ema_slow")
        if ema_fast is not None and ema_slow is not None:
            features["trend_direction"] = np.where(ema_fast > ema_slow, 1, -1)
        else:
            features["trend_direction"] = 0

        features["atr_rank"] = df_ind.get("atr_rank", 0.5)
        features["volume_ratio"] = df_ind.get("volume_ratio", 1.0)

        # Multi-TF features from H4
        if h4_df is not None and not h4_df.empty:
            h4_ind = calculate_indicators(h4_df.copy(), indicator_config)
            h4_last = h4_ind.iloc[-1]

            h4_ema_fast = h4_last.get("ema_fast")
            h4_ema_slow = h4_last.get("ema_slow")
            if h4_ema_fast is not None and h4_ema_slow is not None:
                features["h4_trend_direction"] = 1 if h4_ema_fast > h4_ema_slow else -1
            else:
                features["h4_trend_direction"] = 0

            features["h4_rsi"] = h4_last.get("rsi", 50)

            h4_ema50 = h4_last.get("ema_fast", 0)
            h4_close = h4_last.get("close", h4_df["close"].iloc[-1])
            if h4_ema50 != 0:
                features["h4_ema_distance_pct"] = (h4_close - h4_ema50) / h4_ema50 * 100
            else:
                features["h4_ema_distance_pct"] = 0
        else:
            features["h4_trend_direction"] = 0
            features["h4_rsi"] = 50
            features["h4_ema_distance_pct"] = 0

        # Time features
        if hasattr(df.index, "hour"):
            features["hour_of_day"] = df.index.hour
            features["day_of_week"] = df.index.dayofweek
        else:
            features["hour_of_day"] = 12
            features["day_of_week"] = 2

        return features

    # ------------------------------------------------------------------
    # Labeling (for training)
    # ------------------------------------------------------------------

    def generate_labels(
        self,
        df: pd.DataFrame,
        patterns: pd.DataFrame,
    ) -> pd.Series:
        """
        Label each bar: 1 if a trade entered at this bar's close would
        reach the R:R target within lookforward_bars, 0 otherwise.

        Uses ATR-based SL and R:R target for TP.
        """
        df_ind = calculate_indicators(df.copy())
        atr = df_ind["atr"]
        close = df["close"]
        dominant_signal = patterns["dominant_signal"]

        labels = pd.Series(0, index=df.index, dtype=int)

        for i in range(len(df) - self.lookforward_bars):
            sig = dominant_signal.iloc[i]
            if sig == 0:
                continue

            entry_price = close.iloc[i]
            current_atr = atr.iloc[i]

            if pd.isna(current_atr) or current_atr == 0:
                continue

            sl_distance = 1.5 * current_atr
            tp_distance = sl_distance * self.rr_target

            future_bars = df.iloc[i + 1: i + 1 + self.lookforward_bars]

            if sig > 0:  # BUY signal
                tp_price = entry_price + tp_distance
                sl_price = entry_price - sl_distance
                # Check if TP is hit before SL
                hit_tp = (future_bars["high"] >= tp_price).any()
                hit_sl = (future_bars["low"] <= sl_price).any()

                if hit_tp:
                    # Check which came first
                    tp_idx = (future_bars["high"] >= tp_price).idxmax()
                    sl_idx = (future_bars["low"] <= sl_price).idxmax() if hit_sl else None
                    if sl_idx is None or tp_idx <= sl_idx:
                        labels.iloc[i] = 1

            else:  # SELL signal
                tp_price = entry_price - tp_distance
                sl_price = entry_price + sl_distance
                hit_tp = (future_bars["low"] <= tp_price).any()
                hit_sl = (future_bars["high"] >= sl_price).any()

                if hit_tp:
                    tp_idx = (future_bars["low"] <= tp_price).idxmax()
                    sl_idx = (future_bars["high"] >= sl_price).idxmax() if hit_sl else None
                    if sl_idx is None or tp_idx <= sl_idx:
                        labels.iloc[i] = 1

        return labels

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        df_h1: pd.DataFrame,
        df_h4: Optional[pd.DataFrame] = None,
        indicator_config: dict = None,
    ) -> dict:
        """
        Train the XGBoost model on historical data.

        Returns:
            Dict with training metrics (accuracy, precision, recall, etc.).
        """
        from xgboost import XGBClassifier

        logger.info("Building features for ML training...")
        features = self.build_features(df_h1, df_h4, indicator_config)
        patterns = detect_patterns(df_h1)

        logger.info("Generating labels...")
        labels = self.generate_labels(df_h1, patterns)

        # Only train on bars where a pattern was detected
        has_pattern = patterns["dominant_signal"].abs() > 0
        mask = has_pattern & features.notna().all(axis=1)

        X = features.loc[mask, self.FEATURE_COLUMNS].copy()
        y = labels.loc[mask].copy()

        # Fill any remaining NaN with defaults
        X = X.fillna(0)

        if len(X) < 50:
            logger.warning(f"Too few training samples ({len(X)}). Need at least 50.")
            return {"error": "insufficient_data", "samples": len(X)}

        logger.info(f"Training on {len(X)} samples (positive: {y.sum()}, negative: {(~y.astype(bool)).sum()})")

        split = self.config.get("train_test_split", 0.8)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=1 - split, shuffle=False,  # Time-series: no shuffle
        )

        xgb_params = self.config.get("xgboost_params", {})
        self.model = XGBClassifier(
            n_estimators=xgb_params.get("n_estimators", 300),
            max_depth=xgb_params.get("max_depth", 6),
            learning_rate=xgb_params.get("learning_rate", 0.05),
            subsample=xgb_params.get("subsample", 0.8),
            colsample_bytree=xgb_params.get("colsample_bytree", 0.8),
            min_child_weight=xgb_params.get("min_child_weight", 3),
            reg_alpha=xgb_params.get("reg_alpha", 0.1),
            reg_lambda=xgb_params.get("reg_lambda", 1.0),
            eval_metric="logloss",
            random_state=42,
        )

        self.model.fit(X_train, y_train)

        # Evaluate
        y_pred = self.model.predict(X_test)

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1": f1_score(y_test, y_pred, zero_division=0),
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "positive_rate_train": float(y_train.mean()),
            "positive_rate_test": float(y_test.mean()),
        }

        # Feature importances
        importances = dict(zip(self.FEATURE_COLUMNS, self.model.feature_importances_))
        metrics["feature_importances"] = dict(
            sorted(importances.items(), key=lambda x: x[1], reverse=True)
        )

        logger.info(
            f"ML Training complete — Accuracy: {metrics['accuracy']:.3f}, "
            f"Precision: {metrics['precision']:.3f}, Recall: {metrics['recall']:.3f}"
        )
        logger.info(f"Classification report:\n{classification_report(y_test, y_pred, zero_division=0)}")

        return metrics

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict_proba(self, features: pd.DataFrame) -> float:
        """
        Predict probability of profitable outcome for the latest bar.

        Returns:
            Float probability (0.0 to 1.0), or 0.0 if model not loaded.
        """
        if self.model is None:
            logger.warning("ML model not loaded — returning 0.0")
            return 0.0

        row = features.iloc[[-1]][self.FEATURE_COLUMNS].fillna(0)
        proba = self.model.predict_proba(row)[0][1]
        return float(proba)

    def passes_threshold(self, probability: float) -> bool:
        """Check if the ML probability exceeds the confidence threshold."""
        return probability >= self.confidence_threshold

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_model(self):
        """Save trained model to disk."""
        if self.model is None:
            logger.error("No model to save")
            return
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        joblib.dump(self.model, self.model_path)
        logger.info(f"Model saved to {self.model_path}")

    def load_model(self) -> bool:
        """Load model from disk. Returns True if successful."""
        if not Path(self.model_path).exists():
            logger.warning(f"Model file not found: {self.model_path}")
            return False
        self.model = joblib.load(self.model_path)
        logger.info(f"Model loaded from {self.model_path}")
        return True
