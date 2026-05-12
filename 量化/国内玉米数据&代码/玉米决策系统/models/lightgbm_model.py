"""
LightGBM model for corn futures prediction.
Trained with rolling window backtesting; supports multi-horizon prediction.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import warnings

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False
    warnings.warn("LightGBM not installed. Using fallback gradient boosting.")


# ──────────────────────────────────────────────────────────────────────────────
# Model configuration
# ──────────────────────────────────────────────────────────────────────────────

HORIZONS = [1, 5, 20]          # prediction horizons in days
LOOKBACK = 500                  # training window (trading days)
ROLLOUT = 20                    # retrain every N days

LGB_PARAMS = {
    "objective": "regression",
    "metric": "mae",
    "boosting_type": "gbdt",
    "num_leaves": 31,
    "learning_rate": 0.05,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "min_child_samples": 20,
    "verbose": -1,
    "n_jobs": -1,
}


class LightGBMModel:
    """LightGBM wrapper with rolling window training and multi-horizon support."""

    def __init__(
        self,
        horizons: List[int] = HORIZONS,
        lookback: int = LOOKBACK,
        rollout: int = ROLLOUT,
    ):
        self.horizons = horizons
        self.lookback = lookback
        self.rollout = rollout
        self._models: Dict[int, object] = {}       # horizon -> trained model
        self._feature_cols: List[str] = []
        self._train_history: List[dict] = []

    # ── Target construction ─────────────────────────────────────────────────────

    @staticmethod
    def make_target(df: pd.DataFrame, horizon: int) -> pd.Series:
        """
        Create regression target: forward return over `horizon` days.
        target[i] = (close[i+horizon] - close[i]) / close[i]
        """
        future_close = df["close"].shift(-horizon)
        target = (future_close - df["close"]) / df["close"]
        return target.rename(f"target_h{horizon}")

    @staticmethod
    def make_binary_target(df: pd.DataFrame, horizon: int) -> pd.Series:
        """Create binary target: 1 if price goes up in `horizon` days, else 0."""
        future_close = df["close"].shift(-horizon)
        target = (future_close > df["close"]).astype(float)
        return target.rename(f"binary_target_h{horizon}")

    # ── Training ────────────────────────────────────────────────────────────────

    def _get_train_mask(self, df: pd.DataFrame) -> np.ndarray:
        """Return boolean mask of rows with enough history for all horizons."""
        max_h = max(self.horizons)
        return np.arange(len(df)) < (len(df) - max_h)

    def fit(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        binary_target: bool = True,
    ) -> Dict[int, dict]:
        """
        Train one model per horizon on the most recent lookback window.

        Args:
            df: Full feature DataFrame
            feature_cols: List of feature column names to use
            binary_target: If True, also train a classification head

        Returns:
            Dict of horizon -> training metrics dict
        """
        self._feature_cols = feature_cols

        if not HAS_LIGHTGBM:
            return {h: {"mae": np.nan, "auc": np.nan} for h in self.horizons}

        # Use last lookback rows for training
        train_start = max(0, len(df) - self.lookback)
        train_df = df.iloc[train_start:].copy()

        # Drop rows without enough future data
        max_h = max(self.horizons)
        train_df = train_df.iloc[:-(max_h) if max_h < len(train_df) else len(train_df)]

        metrics = {}
        for h in self.horizons:
            target = self.make_binary_target(train_df, h) if binary_target else self.make_target(train_df, h)
            reg_target = self.make_target(train_df, h)

            # Align
            valid_idx = target.dropna().index
            X = train_df.loc[valid_idx, feature_cols].fillna(-999)
            y_cls = target.loc[valid_idx]
            y_reg = reg_target.loc[valid_idx]

            if len(X) < 50:
                metrics[h] = {"mae": np.nan, "auc": np.nan, "n_samples": len(X)}
                continue

            # Classification
            train_data = lgb.Dataset(X, label=y_cls)
            params_cls = {**LGB_PARAMS, "objective": "binary", "metric": "auc"}
            model_cls = lgb.train(params_cls, train_data, num_boost_round=200)

            # Regression
            train_data_reg = lgb.Dataset(X, label=y_reg)
            params_reg = {**LGB_PARAMS, "objective": "regression", "metric": "mae"}
            model_reg = lgb.train(params_reg, train_data_reg, num_boost_round=200)

            # Store both
            self._models[h] = {"cls": model_cls, "reg": model_reg}

            metrics[h] = {
                "mae": float(model_reg.best_score["valid_0"]["l1"]),
                "auc": float(model_cls.best_score["valid_0"]["auc"]),
                "n_samples": len(X),
            }

        self._train_history.append({"date": df["date"].iloc[-1], "metrics": metrics})
        return metrics

    # ── Prediction ──────────────────────────────────────────────────────────────

    def predict(self, df: pd.DataFrame) -> Dict[int, dict]:
        """
        Generate predictions for all horizons on the latest row.

        Returns:
            Dict[horizon, {pred_return, direction, confidence, prob_up}]
            pred_return: predicted return (e.g. 0.02 = 2%)
            direction: "bullish" / "neutral" / "bearish"
            confidence: 0-1 confidence score
            prob_up: probability of price going up
        """
        if not self._models:
            return {h: self._empty_pred() for h in self.horizons}

        row = df.iloc[-1:]
        X = row[self._feature_cols].fillna(-999)

        results = {}
        for h in self.horizons:
            if h not in self._models:
                results[h] = self._empty_pred()
                continue

            model_cls = self._models[h]["cls"]
            model_reg = self._models[h]["reg"]

            prob_up = float(model_cls.predict(X)[0])
            pred_return = float(model_reg.predict(X)[0])

            # Direction based on return and probability
            if pred_return > 0.005 and prob_up > 0.55:
                direction = "看多"
                confidence = min(abs(pred_return) * 10 + (prob_up - 0.5) * 0.5, 1.0)
            elif pred_return < -0.005 and prob_up < 0.45:
                direction = "看空"
                confidence = min(abs(pred_return) * 10 + (0.5 - prob_up) * 0.5, 1.0)
            else:
                direction = "中性"
                confidence = 0.3 + (1 - abs(prob_up - 0.5) * 2) * 0.2

            results[h] = {
                "pred_return": round(pred_return, 6),
                "direction": direction,
                "confidence": round(confidence, 3),
                "prob_up": round(prob_up, 4),
            }

        return results

    def rolling_predict(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        n_test: int = 120,
    ) -> pd.DataFrame:
        """
        Rolling prediction over the last n_test rows for evaluation.

        Returns:
            DataFrame with columns: date, horizon, actual_return, pred_return, direction
        """
        records = []
        max_h = max(self.horizons)

        for i in range(len(df) - n_test - max_h, len(df) - max_h, self.rollout):
            window = df.iloc[max(0, i - self.lookback):i]
            target_row = df.iloc[i]

            if len(window) < self.lookback * 0.8:
                continue

            self.fit(window, feature_cols)

            for h in self.horizons:
                actual = (df["close"].iloc[i + h] - df["close"].iloc[i]) / df["close"].iloc[i]
                X = df.iloc[i:i+1][feature_cols].fillna(-999)
                if h in self._models:
                    pred = float(self._models[h]["reg"].predict(X)[0])
                    prob = float(self._models[h]["cls"].predict(X)[0])
                else:
                    pred = np.nan
                    prob = np.nan

                records.append({
                    "date": target_row["date"],
                    "horizon": h,
                    "actual_return": actual,
                    "pred_return": pred,
                    "prob_up": prob,
                })

        return pd.DataFrame(records)

    @staticmethod
    def _empty_pred() -> dict:
        return {"pred_return": np.nan, "direction": "中性", "confidence": 0.0, "prob_up": 0.5}

    # ── Feature importance ──────────────────────────────────────────────────────

    def get_feature_importance(self, horizon: int = 1) -> Optional[pd.DataFrame]:
        """Return feature importance for a given horizon."""
        if horizon not in self._models:
            return None
        model = self._models[horizon]["cls"]
        importance = model.feature_importance(importance_type="gain")
        return pd.DataFrame({
            "feature": self._feature_cols,
            "importance": importance,
        }).sort_values("importance", ascending=False)
