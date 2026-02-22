"""
Signal Generator — combines pattern detection, indicator confluence,
and ML scoring into actionable trade signals.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional

import pandas as pd

from src.confluence import evaluate_confluence
from src.indicators import calculate_all as calculate_indicators
from src.ml_scorer import MLScorer
from src.utils import utc_now

logger = logging.getLogger("strategy")


@dataclass
class Signal:
    """Represents a fully qualified trade signal."""
    symbol: str
    direction: str              # "BUY" or "SELL"
    timeframe: str              # Signal TF (e.g., "H1")
    pattern_name: str           # Which pattern triggered
    pattern_strength: float     # -100 to +100
    confluence_score: float     # 0 to 100
    ml_probability: float       # 0.0 to 1.0
    suggested_sl: float         # Price level
    suggested_tp: float         # Price level
    atr_value: float            # Current ATR (for position sizing)
    entry_price: float          # Current close price at signal
    timestamp: datetime = field(default_factory=utc_now)

    @property
    def sl_distance(self) -> float:
        """Absolute distance from entry to stop loss."""
        return abs(self.entry_price - self.suggested_sl)

    @property
    def risk_reward_ratio(self) -> float:
        """Actual risk-reward ratio of this signal."""
        if self.sl_distance == 0:
            return 0.0
        return abs(self.entry_price - self.suggested_tp) / self.sl_distance

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "direction": self.direction,
            "timeframe": self.timeframe,
            "pattern_name": self.pattern_name,
            "pattern_strength": self.pattern_strength,
            "confluence_score": self.confluence_score,
            "ml_probability": self.ml_probability,
            "suggested_sl": self.suggested_sl,
            "suggested_tp": self.suggested_tp,
            "atr_value": self.atr_value,
            "entry_price": self.entry_price,
            "timestamp": self.timestamp.isoformat(),
        }


class SignalGenerator:
    """
    Orchestrates the full signal generation pipeline:
    1. Detect patterns on H1 and M15
    2. Calculate indicators on all timeframes
    3. Evaluate multi-timeframe confluence
    4. Score with ML model
    5. Emit qualified signals
    """

    def __init__(self, config: dict, ml_scorer: Optional[MLScorer] = None):
        self.config = config
        self.confluence_config = config.get("confluence", {})
        self.risk_config = config.get("risk", {})
        self.indicator_config = config.get("indicators", {})
        self.ml_config = config.get("ml", {})

        # Inject indicator config into confluence config for sub-calls
        self.confluence_config["_indicator_config"] = self.indicator_config

        self.min_confluence = self.confluence_config.get("min_score", 60)
        self.ml_enabled = self.ml_config.get("enabled", True)
        self.ml_scorer = ml_scorer

        self.sl_atr_mult = self.risk_config.get("sl_atr_multiplier", 1.5)
        self.tp_rr_ratio = self.risk_config.get("tp_rr_ratio", 2.0)

        self.signal_tf = config.get("timeframes", {}).get("signal", "H1")

    def generate(
        self,
        symbol: str,
        df_h4: Optional[pd.DataFrame],
        df_h1: Optional[pd.DataFrame],
        df_m15: Optional[pd.DataFrame],
    ) -> Optional[Signal]:
        """
        Run the full signal generation pipeline for one symbol.

        Returns:
            A Signal object if all criteria are met, None otherwise.
        """
        # Step 1–4: Evaluate confluence
        confluence = evaluate_confluence(
            df_h4, df_h1, df_m15, self.confluence_config,
        )

        if confluence.direction == "NONE":
            return None

        if confluence.total_score < self.min_confluence:
            logger.debug(
                f"{symbol}: Confluence {confluence.total_score:.0f} < {self.min_confluence} — skipping"
            )
            return None

        # Step 5: ML scoring
        ml_probability = 0.0
        if self.ml_enabled and self.ml_scorer is not None:
            features = MLScorer.build_features(
                df_h1, df_h4, self.indicator_config,
            )
            ml_probability = self.ml_scorer.predict_proba(features)

            if not self.ml_scorer.passes_threshold(ml_probability):
                logger.debug(
                    f"{symbol}: ML probability {ml_probability:.3f} "
                    f"< {self.ml_scorer.confidence_threshold} — skipping"
                )
                return None
        elif self.ml_enabled and self.ml_scorer is None:
            # ML enabled but no model loaded — use confluence only
            logger.debug(f"{symbol}: ML enabled but no model — proceeding with confluence only")
            ml_probability = 0.0

        # Step 6: Calculate SL/TP
        df_h1_ind = calculate_indicators(df_h1.copy(), self.indicator_config)
        last = df_h1_ind.iloc[-1]
        entry_price = last["close"]
        atr = last.get("atr", 0)

        if pd.isna(atr) or atr == 0:
            logger.warning(f"{symbol}: ATR is zero or NaN — cannot set SL/TP")
            return None

        sl_distance = self.sl_atr_mult * atr
        tp_distance = sl_distance * self.tp_rr_ratio

        if confluence.direction == "BUY":
            sl_price = entry_price - sl_distance
            tp_price = entry_price + tp_distance
        else:
            sl_price = entry_price + sl_distance
            tp_price = entry_price - tp_distance

        signal = Signal(
            symbol=symbol,
            direction=confluence.direction,
            timeframe=self.signal_tf,
            pattern_name=confluence.pattern_name,
            pattern_strength=confluence.pattern_strength,
            confluence_score=confluence.total_score,
            ml_probability=ml_probability,
            suggested_sl=round(sl_price, 5),
            suggested_tp=round(tp_price, 5),
            atr_value=atr,
            entry_price=round(entry_price, 5),
        )

        logger.info(
            f"SIGNAL: {signal.direction} {signal.symbol} | "
            f"Pattern={signal.pattern_name} | Confluence={signal.confluence_score:.0f} | "
            f"ML={signal.ml_probability:.3f} | Entry={signal.entry_price} | "
            f"SL={signal.suggested_sl} | TP={signal.suggested_tp}"
        )

        return signal

    def scan_all_symbols(
        self,
        multi_tf_data: dict,
    ) -> List[Signal]:
        """
        Scan all symbols and return a list of qualified signals.

        Args:
            multi_tf_data: Dict mapping symbol -> {'trend': df, 'signal': df, 'entry': df}

        Returns:
            List of Signal objects.
        """
        signals = []
        for symbol, tf_data in multi_tf_data.items():
            signal = self.generate(
                symbol=symbol,
                df_h4=tf_data.get("trend"),
                df_h1=tf_data.get("signal"),
                df_m15=tf_data.get("entry"),
            )
            if signal is not None:
                signals.append(signal)

        if signals:
            logger.info(f"Scan complete: {len(signals)} signal(s) generated")
        else:
            logger.debug("Scan complete: no signals")

        return signals
