"""
Model ensemble with dynamic weighting based on rolling backtest accuracy.
Combines LightGBM, LSTM, and Prophet predictions into a unified signal.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from dataclasses import dataclass, field

from models.lightgbm_model import LightGBMModel
from models.lstm_model import LSTMModel
from models.prophet_model import ProphetModel


@dataclass
class HorizonResult:
    """Prediction result for a single horizon from a single model."""
    horizon: int
    model_name: str
    pred_return: float
    direction: str
    confidence: float
    prob_up: float
    pred_price: Optional[float] = None
    ci_lower: Optional[float] = None
    ci_upper: Optional[float] = None


@dataclass
class EnsembleResult:
    """Final ensemble result for a single horizon."""
    horizon: int
    # Ensemble prediction
    ensemble_return: float
    ensemble_direction: str
    ensemble_confidence: float
    ensemble_prob_up: float
    ensemble_price: Optional[float] = None
    ci_lower: Optional[float] = None
    ci_upper: Optional[float] = None
    # Per-model breakdown
    model_predictions: Dict[str, dict] = field(default_factory=dict)
    # Weights
    weights: Dict[str, float] = field(default_factory=dict)
    # Regime info
    fundamental_available: bool = False
    data_regime: str = "price_only"


# ──────────────────────────────────────────────────────────────────────────────
# Dynamic weight calculator
# ──────────────────────────────────────────────────────────────────────────────

def compute_dynamic_weights(
    rolling_errors: Dict[str, List[float]],
    decay_factor: float = 0.95,
) -> Dict[str, float]:
    """
    Compute model weights inversely proportional to recent MAE.

    Recent errors count more (exponential decay).
    """
    weights = {}
    total_inv = 0.0

    for name, errors in rolling_errors.items():
        if not errors:
            weights[name] = 1.0 / len(rolling_errors)
            total_inv = 1.0
            continue

        # Weighted MAE (recent = more important)
        weights_arr = np.array(errors)
        decay = np.array([decay_factor ** i for i in range(len(errors) - 1, -1, -1)])
        weighted_mae = np.average(np.abs(weights_arr), weights=decay)

        inv_mae = 1.0 / (weighted_mae + 1e-9)
        weights[name] = inv_mae
        total_inv += inv_mae

    # Normalize
    for name in weights:
        weights[name] /= total_inv

    return weights


# ──────────────────────────────────────────────────────────────────────────────
# Ensemble
# ──────────────────────────────────────────────────────────────────────────────

class Ensemble:
    """
    Ensemble of LightGBM, LSTM, and Prophet models.
    Supports dynamic reweighting based on rolling backtest performance.
    """

    MODEL_NAMES = ["lightgbm", "lstm", "prophet"]

    def __init__(
        self,
        horizons: List[int] = [1, 5, 20],
        lookback_for_weights: int = 120,
    ):
        self.horizons = horizons
        self.lookback_for_weights = lookback_for_weights

        self.lgb = LightGBMModel(horizons=horizons)
        self.lstm = LSTMModel(horizons=horizons)
        self.prophet = ProphetModel(horizons=horizons)

        # Rolling prediction errors for weight calculation
        self._rolling_errors: Dict[str, List[float]] = {name: [] for name in self.MODEL_NAMES}

        # Current weights
        self._weights: Dict[str, float] = {name: 1.0 / len(self.MODEL_NAMES) for name in self.MODEL_NAMES}

        # Feature columns (set after first fit)
        self._feature_cols: List[str] = []

    def fit(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        fund_df: Optional[pd.DataFrame] = None,
    ) -> Dict[str, dict]:
        """
        Fit all models on the latest data window.

        Returns:
            Dict of model_name -> fit metrics
        """
        self._feature_cols = feature_cols

        metrics = {}

        # LightGBM
        try:
            metrics["lightgbm"] = self.lgb.fit(df, feature_cols, binary_target=True)
        except Exception as e:
            metrics["lightgbm"] = {"error": str(e)}

        # LSTM
        try:
            metrics["lstm"] = self.lstm.fit(df)
        except Exception as e:
            metrics["lstm"] = {"error": str(e)}

        # Prophet
        try:
            metrics["prophet"] = self.prophet.fit(df, fund_df, n_history=800)
        except Exception as e:
            metrics["prophet"] = {"error": str(e)}

        return metrics

    def predict(
        self,
        df: pd.DataFrame,
    ) -> Dict[int, EnsembleResult]:
        """
        Generate ensemble predictions for all horizons.

        Args:
            df: Feature DataFrame (latest row used for prediction)

        Returns:
            Dict[horizon, EnsembleResult]
        """
        last_close = float(df["close"].iloc[-1])
        last_date = df["date"].iloc[-1]

        # Check if fundamental data is available for current date
        fundamental_available = (
            "north_port_price" in df.columns
            and pd.notna(df["north_port_price"].iloc[-1])
        )
        data_regime = "full" if fundamental_available else "price_only"

        # Collect predictions from each model
        lgb_preds = self.lgb.predict(df)
        lstm_preds = self.lstm.predict(df)
        prophet_preds = self.prophet.predict(df)

        results = {}

        for h in self.horizons:
            # Gather individual model predictions
            model_preds = {}

            # LightGBM
            lgb_p = lgb_preds.get(h, {})
            model_preds["lightgbm"] = {
                "pred_return": lgb_p.get("pred_return", np.nan),
                "direction": lgb_p.get("direction", "中性"),
                "confidence": lgb_p.get("confidence", 0.0),
                "prob_up": lgb_p.get("prob_up", 0.5),
                "pred_price": None,
                "ci_lower": None,
                "ci_upper": None,
            }

            # LSTM
            lstm_p = lstm_preds.get(h, {})
            model_preds["lstm"] = {
                "pred_return": lstm_p.get("pred_return", np.nan),
                "direction": lstm_p.get("direction", "中性"),
                "confidence": lstm_p.get("confidence", 0.0),
                "prob_up": lstm_p.get("prob_up", 0.5),
                "pred_price": None,
                "ci_lower": None,
                "ci_upper": None,
            }

            # Prophet
            prophet_p = prophet_preds.get(h, {})
            model_preds["prophet"] = {
                "pred_return": prophet_p.get("pred_return", np.nan),
                "direction": prophet_p.get("direction", "中性"),
                "confidence": prophet_p.get("confidence", 0.0),
                "prob_up": prophet_p.get("prob_up", 0.5),
                "pred_price": prophet_p.get("pred_price"),
                "ci_lower": prophet_p.get("ci_lower"),
                "ci_upper": prophet_p.get("ci_upper"),
            }

            # Weighted ensemble
            valid_models = [n for n in self.MODEL_NAMES
                           if not np.isnan(model_preds[n]["pred_return"])]

            if valid_models:
                ens_return = sum(
                    self._weights[n] * model_preds[n]["pred_return"]
                    for n in valid_models
                )
                ens_prob_up = sum(
                    self._weights[n] * model_preds[n]["prob_up"]
                    for n in valid_models
                )
                ens_confidence = sum(
                    self._weights[n] * model_preds[n]["confidence"]
                    for n in valid_models
                )
                ens_price = last_close * (1 + ens_return)

                # CI from Prophet if available
                if model_preds["prophet"]["ci_lower"] is not None:
                    ci_lower = model_preds["prophet"]["ci_lower"]
                    ci_upper = model_preds["prophet"]["ci_upper"]
                else:
                    ci_lower = None
                    ci_upper = None

                # Direction
                if ens_return > 0.005 and ens_prob_up > 0.55:
                    ens_direction = "看多"
                elif ens_return < -0.005 and ens_prob_up < 0.45:
                    ens_direction = "看空"
                else:
                    ens_direction = "中性"
            else:
                ens_return = 0.0
                ens_prob_up = 0.5
                ens_confidence = 0.0
                ens_direction = "中性"
                ens_price = last_close
                ci_lower = None
                ci_upper = None

            results[h] = EnsembleResult(
                horizon=h,
                ensemble_return=round(ens_return, 6),
                ensemble_direction=ens_direction,
                ensemble_confidence=round(ens_confidence, 3),
                ensemble_prob_up=round(ens_prob_up, 4),
                ensemble_price=round(ens_price, 2),
                ci_lower=round(ci_lower, 2) if ci_lower else None,
                ci_upper=round(ci_upper, 2) if ci_upper else None,
                model_predictions=model_preds,
                weights=dict(self._weights),
                fundamental_available=fundamental_available,
                data_regime=data_regime,
            )

        return results

    def update_weights(
        self,
        horizon: int,
        actual_return: float,
    ) -> Dict[str, float]:
        """
        Update rolling error history with a new actual return.
        Recomputes weights based on recent performance.
        """
        for name in self.MODEL_NAMES:
            model = getattr(self, name.replace(".", "_"))
            # Get the most recent prediction for this horizon from this model
            # (Simplified: store last prediction)
            last_pred = getattr(self, f"_last_pred_{name}", np.nan)
            error = actual_return - last_pred
            self._rolling_errors[name].append(error)
            # Keep only recent history
            if len(self._rolling_errors[name]) > self.lookback_for_weights:
                self._rolling_errors[name] = self._rolling_errors[name][-self.lookback_for_weights:]

        self._weights = compute_dynamic_weights(self._rolling_errors)
        return self._weights

    def get_weights(self) -> Dict[str, float]:
        """Return current model weights."""
        return dict(self._weights)

    def get_latest_predictions(self) -> dict:
        """Return latest predictions from each model (for weight tracking)."""
        return {
            name: getattr(self, f"_last_pred_{name}", np.nan)
            for name in self.MODEL_NAMES
        }
