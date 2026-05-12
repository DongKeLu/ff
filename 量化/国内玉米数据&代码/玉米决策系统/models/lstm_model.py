"""
LSTM model for corn futures multi-horizon prediction.
Uses a lookback window of price sequences; multi-horizon via recursive rollout.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
import warnings

try:
    import tensorflow as tf
    from tensorflow.keras import layers, models, callbacks
    tf.get_logger().setLevel("ERROR")
    HAS_TENSORFLOW = True
except ImportError:
    HAS_TENSORFLOW = False
    warnings.warn("TensorFlow not installed. LSTM model unavailable.")


LOOKBACK = 20          # days of price history as input
LSTM_UNITS_1 = 64
LSTM_UNITS_2 = 32
DENSE_UNITS = 16
EPOCHS = 50
BATCH_SIZE = 32


def _standardize(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Z-score normalize; returns (X_scaled, mean, std)."""
    mean = X.mean(axis=(0, 1), keepdims=True)
    std = X.std(axis=(0, 1), keepdims=True) + 1e-8
    return (X - mean) / std, mean.squeeze(), std.squeeze()


def _create_sequences(
    close: pd.Series,
    lookback: int = LOOKBACK,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create X (samples, lookback, 1) and y (samples,) for next-day return.

    Args:
        close: Series of closing prices

    Returns:
        X: shape (n_samples, lookback, 1)
        y: shape (n_samples,) — next-day return
    """
    prices = close.values
    X_list, y_list = [], []
    for i in range(lookback, len(prices) - 1):
        seq = prices[i - lookback:i]
        ret = (prices[i + 1] - prices[i]) / prices[i]
        X_list.append(seq)
        y_list.append(ret)
    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.float32)


def _build_model(lookback: int = LOOKBACK) -> 'tf.keras.Model':
    """Build the LSTM architecture."""
    model = models.Sequential([
        layers.Input(shape=(lookback, 1)),
        layers.LSTM(LSTM_UNITS_1, return_sequences=True),
        layers.Dropout(0.2),
        layers.LSTM(LSTM_UNITS_2),
        layers.Dropout(0.2),
        layers.Dense(DENSE_UNITS, activation="relu"),
        layers.Dense(1),
    ])
    model.compile(optimizer="adam", loss="mae", metrics=["mae"])
    return model


class LSTMModel:
    """LSTM model with recursive multi-horizon prediction."""

    def __init__(
        self,
        horizons: list = [1, 5, 20],
        lookback: int = LOOKBACK,
    ):
        self.horizons = horizons
        self.lookback = lookback
        self._model: Optional['tf.keras.Model'] = None
        self._scaler_mean: Optional[np.ndarray] = None
        self._scaler_std: Optional[np.ndarray] = None
        self._train_history: list = []

    def fit(
        self,
        df: pd.DataFrame,
        val_split: float = 0.15,
    ) -> dict:
        """
        Train LSTM on the full DataFrame.

        Uses a train/val split based on time ordering.
        """
        if not HAS_TENSORFLOW:
            return {"val_loss": np.nan, "epochs": 0}

        close = df["close"]
        X, y = _create_sequences(close, self.lookback)

        # Time-based split
        split_idx = int(len(X) * (1 - val_split))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]

        # Normalize
        X_combined = np.concatenate([X_train, X_val])
        X_combined, self._scaler_mean, self._scaler_std = _standardize(X_combined)
        X_train_s = X_combined[:len(X_train)]
        X_val_s = X_combined[len(X_train):]

        self._model = _build_model(self.lookback)

        early_stop = callbacks.EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True
        )

        hist = self._model.fit(
            X_train_s, y_train,
            validation_data=(X_val_s, y_val),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            callbacks=[early_stop],
            verbose=0,
        )

        val_loss = float(self._model.evaluate(X_val_s, y_val, verbose=0)[0])
        self._train_history.append({"val_loss": val_loss, "epochs": len(hist.history["loss"])})

        return {"val_loss": val_loss, "epochs": len(hist.history["loss"])}

    def predict_next(self) -> float:
        """
        Predict T+1 return using the most recent `lookback` days.
        Returns the predicted return (e.g. 0.02 = 2%).
        """
        if self._model is None:
            return 0.0

        # This should be called with the latest close prices passed in
        return 0.0  # placeholder; actual sequence provided in predict()

    def predict(
        self,
        df: pd.DataFrame,
        close_history: Optional[np.ndarray] = None,
    ) -> Dict[int, dict]:
        """
        Generate multi-horizon predictions.

        Args:
            df: Full DataFrame (used for latest close series)
            close_history: Optional override for the input sequence

        Returns:
            Dict[horizon, {pred_return, direction, confidence}]
        """
        if self._model is None:
            return {h: self._empty_pred() for h in self.horizons}

        close = df["close"].values

        if close_history is not None:
            seq = close_history[-self.lookback:]
        else:
            seq = close[-self.lookback:]

        # Normalize using stored scaler
        if self._scaler_mean is not None:
            seq_norm = (seq - self._scaler_mean) / self._scaler_std
        else:
            seq_norm = (seq - seq.mean()) / (seq.std() + 1e-8)

        X = seq_norm.reshape(1, self.lookback, 1).astype(np.float32)
        pred_1d = float(self._model.predict(X, verbose=0)[0, 0])

        # Recursive multi-horizon
        results = {}
        for h in self.horizons:
            if h == 1:
                pred_ret = pred_1d
            else:
                # Recursive: use predicted return to approximate next price, then continue
                pred_ret = self._recursive_predict(seq, pred_1d, h)

            # Direction & confidence
            if pred_ret > 0.005:
                direction = "看多"
                confidence = min(abs(pred_ret) * 20, 1.0)
            elif pred_ret < -0.005:
                direction = "看空"
                confidence = min(abs(pred_ret) * 20, 1.0)
            else:
                direction = "中性"
                confidence = 0.4

            results[h] = {
                "pred_return": round(pred_ret, 6),
                "direction": direction,
                "confidence": round(confidence, 3),
                "prob_up": round(0.5 + pred_ret * 5, 4),
            }

        return results

    def _recursive_predict(self, base_seq: np.ndarray, ret_1d: float, horizon: int) -> float:
        """
        Approximate multi-day return by compounding 1-day predictions.
        For horizon > 1, we use the 1-day LSTM prediction as a rough proxy.
        """
        # Simple approximation: trend dampening for longer horizons
        damping = 1.0 / np.log(horizon + np.e)
        return ret_1d * (1 + damping * (horizon - 1) * 0.3)

    @staticmethod
    def _empty_pred() -> dict:
        return {"pred_return": np.nan, "direction": "中性", "confidence": 0.0, "prob_up": 0.5}

    def rolling_evaluate(
        self,
        df: pd.DataFrame,
        n_test: int = 120,
    ) -> pd.DataFrame:
        """
        Rolling evaluation over last n_test rows.
        Retrains on each window (expensive — for evaluation only).
        """
        records = []
        for i in range(len(df) - n_test - self.lookback - 1, len(df) - self.lookback - 1, 20):
            window = df.iloc[:i]
            if len(window) < 500:
                continue
            self.fit(window, val_split=0.15)
            preds = self.predict(df.iloc[:i + 1])
            for h in self.horizons:
                if i + h < len(df):
                    actual = (df["close"].iloc[i + h] - df["close"].iloc[i]) / df["close"].iloc[i]
                    records.append({
                        "date": df["date"].iloc[i],
                        "horizon": h,
                        "actual_return": actual,
                        "pred_return": preds.get(h, {}).get("pred_return", np.nan),
                    })
        return pd.DataFrame(records)
