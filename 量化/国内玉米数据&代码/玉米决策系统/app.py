"""
FastAPI backend for corn futures quantitative decision system.
Serves model predictions, price data, and fundamental data via REST API.
"""

import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from datetime import datetime
from typing import Optional, Dict, Any, List

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from data_loader import load_all, merge_price_fundamental, get_data_summary
from feature_engineering import build_all_features, get_feature_columns
from ensemble import Ensemble
from signals import evaluate_all_rules, get_rule_summary, detect_divergence


# ──────────────────────────────────────────────────────────────────────────────
# Application setup
# ──────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="玉米期货量化决策系统",
    description="DCE 玉米期货多模型量化决策辅助系统",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# ──────────────────────────────────────────────────────────────────────────────
# Global state (loaded at startup)
# ──────────────────────────────────────────────────────────────────────────────

_price_df: Optional[pd.DataFrame] = None
_fund_df: Optional[pd.DataFrame] = None
_feature_df: Optional[pd.DataFrame] = None
_ensemble: Optional[Ensemble] = None
_feature_cols: Optional[List[str]] = None
_is_ready: bool = False


def initialize():
    """Load all data and fit all models at startup."""
    global _price_df, _fund_df, _feature_df, _ensemble, _feature_cols, _is_ready

    print("[系统] 正在加载数据...")
    _price_df, _fund_df = load_all()

    print("[系统] 正在构建特征工程...")
    _feature_df = build_all_features(_price_df, _fund_df)

    feature_info = get_feature_columns(_feature_df)
    _feature_cols = feature_info["price_features"] + feature_info["fundamental_features"]

    print("[系统] 正在训练模型（首次加载需要数分钟）...")
    _ensemble = Ensemble(horizons=[1, 5, 20])
    _ensemble.fit(_feature_df, _feature_cols, _fund_df)

    _is_ready = True
    print(f"[系统] 初始化完成！最新日期: {_feature_df['date'].iloc[-1]}")


# ──────────────────────────────────────────────────────────────────────────────
# Pydantic models
# ──────────────────────────────────────────────────────────────────────────────

class PredictionResponse(BaseModel):
    latest_date: str
    latest_close: float
    latest_volume: float
    fundamental_available: bool
    data_regime: str
    predictions: Dict[int, dict]
    model_weights: Dict[str, float]
    logic_signals: Dict[str, Any]


class PriceDataResponse(BaseModel):
    dates: List[str]
    open: List[float]
    high: List[float]
    low: List[float]
    close: List[float]
    volume: List[float]
    features: Dict[str, List[float]]


class FundamentalDataResponse(BaseModel):
    dates: List[str]
    data: Dict[str, List[Optional[float]]]


class HealthResponse(BaseModel):
    status: str
    is_ready: bool
    latest_date: Optional[str]
    total_rows: int
    n_features: int


# ──────────────────────────────────────────────────────────────────────────────
# Routes
# ──────────────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the frontend UI."""
    index_path = Path(__file__).parent / "static" / "index.html"
    if index_path.exists():
        return index_path.read_text()
    return """
    <html><body>
    <h1>玉米期货量化决策系统</h1>
    <p>前端文件未找到，请确保 static/index.html 存在。</p>
    <p><a href="/docs">API 文档</a></p>
    </body></html>
    """


@app.get("/api/health", response_model=HealthResponse)
async def health():
    """Health check endpoint."""
    if not _is_ready:
        raise HTTPException(status_code=503, detail="System not initialized")
    return HealthResponse(
        status="ok",
        is_ready=_is_ready,
        latest_date=str(_feature_df["date"].iloc[-1].date()),
        total_rows=len(_feature_df),
        n_features=len(_feature_cols),
    )


@app.get("/api/prediction", response_model=PredictionResponse)
async def get_prediction():
    """Get the latest multi-model ensemble prediction for all horizons."""
    if not _is_ready:
        raise HTTPException(status_code=503, detail="System not initialized")

    df = _feature_df
    fund = _fund_df

    # Re-fit models with latest data (lightweight update)
    try:
        _ensemble.fit(df, _feature_cols, fund)
    except Exception as e:
        print(f"[警告] 模型更新失败，使用上次结果: {e}")

    predictions = _ensemble.predict(df)

    # Evaluate financial logic rules
    logic_signals_list = evaluate_all_rules(df, horizon=1)
    logic_summary = get_rule_summary(logic_signals_list)

    # Check divergence
    ensemble_dir = list(predictions.values())[0].ensemble_direction
    logic_dir = logic_summary["overall_direction"]
    divergence = detect_divergence(ensemble_dir, logic_dir)

    # Build response
    pred_dict = {}
    for h, res in predictions.items():
        pred_dict[h] = {
            "model_predictions": {
                name: {
                    "pred_return": float(v["pred_return"]) if v["pred_return"] is not None else None,
                    "direction": v["direction"],
                    "confidence": float(v["confidence"]),
                    "prob_up": float(v["prob_up"]),
                }
                for name, v in res.model_predictions.items()
            },
            "ensemble": {
                "return": res.ensemble_return,
                "direction": res.ensemble_direction,
                "confidence": res.ensemble_confidence,
                "prob_up": res.ensemble_prob_up,
                "pred_price": res.ensemble_price,
                "ci_lower": res.ci_lower,
                "ci_upper": res.ci_upper,
            },
            "weights": {k: float(v) for k, v in res.weights.items()},
            "fundamental_available": res.fundamental_available,
            "data_regime": res.data_regime,
        }

    return PredictionResponse(
        latest_date=str(df["date"].iloc[-1].date()),
        latest_close=float(df["close"].iloc[-1]),
        latest_volume=float(df["volume"].iloc[-1]),
        fundamental_available=predictions[1].fundamental_available,
        data_regime=predictions[1].data_regime,
        predictions=pred_dict,
        model_weights={k: float(v) for k, v in _ensemble.get_weights().items()},
        logic_signals={
            **logic_summary,
            "divergence_warning": divergence,
        },
    )


@app.get("/api/price", response_model=PriceDataResponse)
async def get_price_data(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    limit: int = 2000,
):
    """Get price data with features for charting."""
    if not _is_ready:
        raise HTTPException(status_code=503, detail="System not initialized")

    df = _feature_df.tail(limit)

    if start_date:
        df = df[df["date"] >= pd.Timestamp(start_date)]
    if end_date:
        df = df[df["date"] <= pd.Timestamp(end_date)]

    # Key features for overlay (must go under the "features" key)
    features: Dict[str, List[float]] = {}
    feature_cols = [
        "ma5", "ma20", "ma60", "boll_upper", "boll_middle", "boll_lower",
        "macd_dif", "macd_dea", "macd_hist", "rsi_14", "cci_14",
        "atr_14", "kdj_k", "kdj_d",
    ]
    for col in feature_cols:
        if col in df.columns:
            features[col] = [float(x) if pd.notna(x) else None for x in df[col]]

    return PriceDataResponse(
        dates=[str(d.date()) for d in df["date"]],
        open=[float(x) for x in df["open"]],
        high=[float(x) for x in df["high"]],
        low=[float(x) for x in df["low"]],
        close=[float(x) for x in df["close"]],
        volume=[float(x) for x in df["volume"]],
        features=features,
    )


@app.get("/api/fundamental", response_model=FundamentalDataResponse)
async def get_fundamental_data(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    limit: int = 1500,
):
    """Get fundamental data time series."""
    if not _is_ready:
        raise HTTPException(status_code=503, detail="System not initialized")

    df = _fund_df.tail(limit)

    if start_date:
        df = df[df["date"] >= pd.Timestamp(start_date)]
    if end_date:
        df = df[df["date"] <= pd.Timestamp(end_date)]

    # Flatten multi-index columns into "data" nested dict
    dates_list = [str(d.date()) for d in df["date"]]
    data_dict: Dict[str, List[float]] = {}
    for col in df.columns:
        if col == "date":
            continue
        clean_name = col.replace("|", "_").replace(" ", "_")
        data_dict[clean_name] = [float(x) if pd.notna(x) else None for x in df[col]]

    return FundamentalDataResponse(dates=dates_list, data=data_dict)


@app.get("/api/summary")
async def get_summary():
    """Get data quality summary."""
    if not _is_ready:
        raise HTTPException(status_code=503, detail="System not initialized")
    return get_data_summary(_price_df, _fund_df)


@app.get("/api/features")
async def get_features():
    """Get feature column names by category."""
    if not _is_ready:
        raise HTTPException(status_code=503, detail="System not initialized")
    return get_feature_columns(_feature_df)


# ──────────────────────────────────────────────────────────────────────────────
# Startup
# ──────────────────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup_event():
    initialize()


# ──────────────────────────────────────────────────────────────────────────────
# Run
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
