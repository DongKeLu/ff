"""
Prophet model for corn futures trend forecasting.
Decomposes price into trend + yearly seasonality + weekly seasonality.
Also models fundamental series (basis, warehouse receipts) from 2020+.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional
import warnings

try:
    from prophet import Prophet
    HAS_PROPHET = True
except ImportError:
    HAS_PROPHET = False
    warnings.warn("Prophet not installed. Prophet model unavailable.")


class ProphetModel:
    """
    Prophet-based time series forecasting for corn futures.
    Supports price series + fundamental series from 2020+.
    """

    def __init__(self, horizons: list = [1, 5, 20]):
        self.horizons = horizons
        self._price_model: Optional['Prophet'] = None
        self._fund_models: Dict[str, 'Prophet'] = {}
        self._last_date: Optional[pd.Timestamp] = None
        self._last_close: Optional[float] = None

    def _to_prophet_df(self, df: pd.DataFrame, value_col: str = "close") -> pd.DataFrame:
        """Convert DataFrame to Prophet's required format (ds, y)."""
        return df[["date", value_col]].rename(columns={"date": "ds", value_col: "y"}).dropna()

    def fit(
        self,
        df: pd.DataFrame,
        fund_df: Optional[pd.DataFrame] = None,
        n_history: int = 1000,
    ) -> dict:
        """
        Fit Prophet models on the most recent n_history rows.

        Args:
            df: Full feature DataFrame
            fund_df: Fundamental DataFrame (optional)
            n_history: Number of recent rows to use for training

        Returns:
            Dict with fit status for each model
        """
        if not HAS_PROPHET:
            return {"price": "not_available", "fundamentals": "not_available"}

        use_df = df.tail(n_history).copy()
        self._last_date = use_df["date"].iloc[-1]
        self._last_close = float(use_df["close"].iloc[-1])

        results = {}

        # Price model
        try:
            price_prophet_df = self._to_prophet_df(use_df, "close")
            self._price_model = Prophet(
                yearly_seasonality=True,
                weekly_seasonality=False,  # futures don't trade on weekends
                daily_seasonality=False,
                changepoint_prior_scale=0.05,
            )
            self._price_model.fit(price_prophet_df)
            results["price"] = "ok"
        except Exception as e:
            results["price"] = f"error: {e}"
            self._price_model = None

        # Fundamental models (2020+)
        if fund_df is not None:
            fund_fitted = {}
            cutoff = pd.Timestamp("2020-01-01")
            for col_name, prophet_col in [
                ("port_basis", "port_basis"),
                ("warehouse_receipts", "warehouse_receipts"),
            ]:
                if col_name not in use_df.columns:
                    continue
                fund_series = use_df[["date", col_name]].rename(columns={"date": "ds", col_name: "y"}).dropna()
                fund_series = fund_series[fund_series["ds"] >= cutoff]
                if len(fund_series) < 30:
                    continue
                try:
                    m = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
                    m.fit(fund_series)
                    self._fund_models[col_name] = m
                    fund_fitted[col_name] = "ok"
                except Exception:
                    fund_fitted[col_name] = "error"
            results["fundamentals"] = fund_fitted

        return results

    def predict(self, df: pd.DataFrame) -> Dict[int, dict]:
        """
        Generate multi-horizon forecasts.

        Returns:
            Dict[horizon, {pred_price, pred_return, direction, confidence, ci_lower, ci_upper}]
        """
        if self._price_model is None:
            return {h: self._empty_pred() for h in self.horizons}

        results = {}
        last_close = float(df["close"].iloc[-1])
        last_date = df["date"].iloc[-1]

        for h in self.horizons:
            future = self._price_model.make_future_dataframe(periods=h)
            forecast = self._price_model.predict(future)

            # Get the forecast for h days ahead
            if h <= len(forecast):
                row = forecast.iloc[-h]
                pred_price = float(row["yhat"])
                ci_lower = float(row["yhat_lower"])
                ci_upper = float(row["yhat_upper"])
            else:
                pred_price = last_close
                ci_lower = last_close
                ci_upper = last_close

            pred_return = (pred_price - last_close) / last_close
            ci_width = (ci_upper - ci_lower) / last_close

            if pred_return > 0.005:
                direction = "看多"
                confidence = min(abs(pred_return) * 15, 1.0)
            elif pred_return < -0.005:
                direction = "看空"
                confidence = min(abs(pred_return) * 15, 1.0)
            else:
                direction = "中性"
                confidence = 0.35

            # Reduce confidence if CI is very wide
            if ci_width > 0.1:
                confidence *= 0.7

            results[h] = {
                "pred_price": round(pred_price, 2),
                "pred_return": round(pred_return, 6),
                "direction": direction,
                "confidence": round(confidence, 3),
                "prob_up": round(0.5 + pred_return * 5, 4),
                "ci_lower": round(ci_lower, 2),
                "ci_upper": round(ci_upper, 2),
            }

        return results

    def get_components(self) -> Optional[pd.DataFrame]:
        """Return Prophet trend and seasonality components for the latest forecast."""
        if self._price_model is None:
            return None
        future = self._price_model.make_future_dataframe(periods=30)
        return self._price_model.predict(future)

    @staticmethod
    def _empty_pred() -> dict:
        return {
            "pred_price": np.nan,
            "pred_return": np.nan,
            "direction": "中性",
            "confidence": 0.0,
            "prob_up": 0.5,
            "ci_lower": np.nan,
            "ci_upper": np.nan,
        }
