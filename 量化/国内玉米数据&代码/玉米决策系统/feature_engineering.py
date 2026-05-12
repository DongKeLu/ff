"""
Feature engineering for corn futures data.
Builds two feature pools:
  - Price features: available for the full 2004-2026 range
  - Fundamental features: available only from 2020+ (NaN before)
"""

import pandas as pd
import numpy as np
from typing import List, Optional


# Regime boundaries
REGIME_2020 = pd.Timestamp("2020-01-01")


# ──────────────────────────────────────────────────────────────────────────────
# Price-based features (full range)
# ──────────────────────────────────────────────────────────────────────────────

def build_price_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all price/volume based features.
    Works on the full 2004-2026 range.

    Args:
        df: DataFrame with columns date, open, high, low, close, settle, volume

    Returns:
        DataFrame with original columns + new feature columns
    """
    df = df.copy()

    # ── Price returns ──────────────────────────────────────────────────────────
    for d in [1, 3, 5, 10, 20, 60]:
        df[f"return_{d}d"] = df["close"].pct_change(d)

    # ── Moving averages ────────────────────────────────────────────────────────
    for w in [5, 10, 20, 60, 120]:
        df[f"ma{w}"] = df["close"].rolling(w).mean()
        df[f"ma{w}_ratio"] = df["close"] / df[f"ma{w}"]

    # ── Exponential moving averages ───────────────────────────────────────────
    for span in [12, 26]:
        df[f"ema{span}"] = df["close"].ewm(span=span, adjust=False).mean()
        df[f"ema{span}_ratio"] = df["close"] / df[f"ema{span}"]

    # ── MACD ───────────────────────────────────────────────────────────────────
    ema12 = df["close"].ewm(span=12, adjust=False).mean()
    ema26 = df["close"].ewm(span=26, adjust=False).mean()
    df["macd_dif"] = ema12 - ema26
    df["macd_dea"] = df["macd_dif"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = (df["macd_dif"] - df["macd_dea"]) * 2

    # ── RSI(14) ────────────────────────────────────────────────────────────────
    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["rsi_14"] = 100 - (100 / (1 + rs))

    # ── CCI(14) ──────────────────────────────────────────────────────────────
    typical = (df["high"] + df["low"] + df["close"]) / 3
    sma_typical = typical.rolling(14).mean()
    mad = typical.rolling(14).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
    df["cci_14"] = (typical - sma_typical) / (0.015 * mad + 1e-9)

    # ── KDJ ──────────────────────────────────────────────────────────────────
    low_n = df["low"].rolling(9).min()
    high_n = df["high"].rolling(9).max()
    rsv = (df["close"] - low_n) / (high_n - low_n + 1e-9) * 100
    df["kdj_k"] = rsv.ewm(com=2, adjust=False).mean()
    df["kdj_d"] = df["kdj_k"].ewm(com=2, adjust=False).mean()
    df["kdj_j"] = 3 * df["kdj_k"] - 2 * df["kdj_d"]

    # ── Bollinger Bands ──────────────────────────────────────────────────────
    ma20 = df["close"].rolling(20).mean()
    std20 = df["close"].rolling(20).std()
    df["boll_upper"] = ma20 + 2 * std20
    df["boll_middle"] = ma20
    df["boll_lower"] = ma20 - 2 * std20
    df["boll_width"] = (df["boll_upper"] - df["boll_lower"]) / ma20
    df["boll_position"] = (df["close"] - df["boll_lower"]) / (df["boll_upper"] - df["boll_lower"] + 1e-9)

    # ── ATR(14) ──────────────────────────────────────────────────────────────
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift()).abs(),
        (df["low"] - df["close"].shift()).abs(),
    ], axis=1).max(axis=1)
    df["atr_14"] = tr.rolling(14).mean()

    # ── Realized volatility (20-day) ─────────────────────────────────────────
    df["realized_vol_20"] = df["return_1d"].rolling(20).std() * np.sqrt(242)

    # ── Volume features ───────────────────────────────────────────────────────
    df["volume_ma5"] = df["volume"].rolling(5).mean()
    df["volume_ratio"] = df["volume"] / df["volume_ma5"]
    df["volume_change_1d"] = df["volume"].pct_change()

    # ── Price-range features ─────────────────────────────────────────────────
    df["intraday_range"] = (df["high"] - df["low"]) / df["close"]
    df["close_open_ratio"] = df["close"] / df["open"]
    df["upper_shadow"] = (df["high"] - df[["close", "open"]].max(axis=1)) / df["close"]
    df["lower_shadow"] = (df[["close", "open"]].min(axis=1) - df["low"]) / df["close"]

    # ── Open interest proxy (from futures.csv if available) ─────────────────
    # Note: corn_en_no_oi.txt has "no_oi" so we skip OI features here

    # ── Trend detection ───────────────────────────────────────────────────────
    df["ma_bull排列"] = ((df["ma5"] > df["ma10"]) & (df["ma10"] > df["ma20"]) & (df["ma20"] > df["ma60"])).astype(float)
    df["ma_bear排列"] = ((df["ma5"] < df["ma10"]) & (df["ma10"] < df["ma20"]) & (df["ma20"] < df["ma60"])).astype(float)

    # ── MACD cross signal ─────────────────────────────────────────────────────
    df["macd_golden_cross"] = ((df["macd_dif"] > df["macd_dea"]) & (df["macd_dif"].shift() <= df["macd_dea"].shift())).astype(float)
    df["macd_dead_cross"] = ((df["macd_dif"] < df["macd_dea"]) & (df["macd_dif"].shift() >= df["macd_dea"].shift())).astype(float)

    # ── Time features ─────────────────────────────────────────────────────────
    df["month"] = df["date"].dt.month
    df["quarter"] = df["date"].dt.quarter
    for m in range(1, 13):
        df[f"month_{m}"] = (df["month"] == m).astype(float)
    for q in range(1, 5):
        df[f"quarter_{q}"] = (df["quarter"] == q).astype(float)

    # ── Lag features (useful for tree models) ─────────────────────────────────
    for d in [1, 2, 3, 5]:
        df[f"close_lag{d}"] = df["close"].shift(d)
        df[f"volume_lag{d}"] = df["volume"].shift(d)

    # ── Rolling statistics ───────────────────────────────────────────────────
    df["close_roll_mean_10"] = df["close"].rolling(10).mean()
    df["close_roll_std_10"] = df["close"].rolling(10).std()
    df["close_roll_mean_20"] = df["close"].rolling(20).mean()
    df["close_roll_std_20"] = df["close"].rolling(20).std()

    return df


# ──────────────────────────────────────────────────────────────────────────────
# Fundamental features (regime-aware)
# ──────────────────────────────────────────────────────────────────────────────

def build_fundamental_features(df: pd.DataFrame, fund_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute fundamental features from the fundamental DataFrame.
    These are only meaningful from 2020+; NaN before that.

    High-quality columns (100% coverage from 2020):
      - 仓单
      - 华北深加工均价

    Medium-quality (90-99% from 2020):
      - 北港价格
      - 珠三角价格
      - 小麦替代成本优势
      - 东北深加工均价
    """
    df = df.copy()

    # Merge fundamental data on date (left join preserves all price rows)
    fund = fund_df.copy()
    fund["date"] = pd.to_datetime(fund["date"])

    # Flatten multi-index column names for easier access
    flat_cols = {}
    for col in fund.columns:
        if col == "date":
            flat_cols[col] = col
        else:
            parts = col.split("|")
            if len(parts) == 2:
                cat, name = parts
                if "Unnamed" in name:
                    flat_cols[col] = cat.strip()
                else:
                    flat_cols[col] = f"{cat.strip()}_{name.strip()}"
            else:
                flat_cols[col] = col.strip()
    fund = fund.rename(columns=flat_cols)

    merged = df.merge(fund, on="date", how="left")

    # ── Port price features ───────────────────────────────────────────────────
    if "北港价格" in merged.columns:
        merged["north_port_price"] = merged["北港价格"]
        merged["port_basis"] = merged["北港价格"] - merged["close"]  # 基差
        merged["port_basis_ma5"] = merged["port_basis"].rolling(5).mean()
        merged["port_basis_change"] = merged["port_basis"].pct_change()

    if "珠三角价格" in merged.columns:
        merged["south_port_price"] = merged["珠三角价格"]
        merged["north_south_spread"] = merged["北港价格"] - merged["珠三角价格"]

    if {"北港价格", "珠三角价格"}.issubset(merged.columns):
        merged["port_spread"] = merged["北港价格"] - merged["珠三角价格"]

    # ── Warehouse receipts (仓单) ─────────────────────────────────────────────
    if "仓单" in merged.columns:
        merged["warehouse_receipts"] = merged["仓单"]
        merged["wr_ma5"] = merged["仓单"].rolling(5).mean()
        merged["wr_ma20"] = merged["仓单"].rolling(20).mean()
        merged["wr_change_rate"] = merged["仓单"].pct_change()
        merged["wr_ratio_to_ma5"] = merged["仓单"] / merged["wr_ma5"]
        merged["wr_ratio_to_ma20"] = merged["仓单"] / merged["wr_ma20"]
        # Historical percentile of current level
        merged["wr_pct_rank"] = merged["仓单"].rolling(252 * 2, min_periods=60).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 0 else np.nan, raw=False
        )

    # ── Processing price features ────────────────────────────────────────────
    if "华北深加工均价" in merged.columns:
        merged["nc_processing_avg"] = merged["华北深加工均价"]
        merged["processing_bid_spread"] = merged["北港价格"] - merged["华北深加工均价"]  # 利润 proxy

    if "东北深加工均价" in merged.columns:
        merged["ne_processing_avg"] = merged["东北深加工均价"]

    if {"北港价格", "华北深加工均价"}.issubset(merged.columns):
        merged["processing_profit_proxy"] = merged["北港价格"] - merged["华北深加工均价"]

    # ── Wheat substitution advantage ─────────────────────────────────────────
    if "小麦替代成本优势" in merged.columns:
        merged["wheat_sub_advantage"] = merged["小麦替代成本优势"]
        merged["wheat_sub_ma5"] = merged["小麦替代成本优势"].rolling(5).mean()
        merged["wheat_sub_change"] = merged["小麦替代成本优势"].diff()

    # ── Shandong delivery vehicles ────────────────────────────────────────────
    vehicle_cols = [c for c in merged.columns if "山东到车辆" in c or "到车辆" in c.lower()]
    if vehicle_cols:
        merged["shandong_vehicles"] = merged[vehicle_cols[-1]]  # use total column
        merged["shandong_vehicles_ma5"] = merged["shandong_vehicles"].rolling(5).mean()
        merged["shandong_vehicles_ratio"] = merged["shandong_vehicles"] / merged["shandong_vehicles_ma5"]

    # ── Processing volume features ────────────────────────────────────────────
    total_vol_col = None
    for col in merged.columns:
        if "总计" in col and "收购量" in col:
            total_vol_col = col
            break
    if total_vol_col:
        merged["total_processing_volume"] = merged[total_vol_col]
        merged["total_pv_change"] = merged[total_vol_col].pct_change()

    return merged


# ──────────────────────────────────────────────────────────────────────────────
# Full feature pipeline
# ──────────────────────────────────────────────────────────────────────────────

def build_all_features(
    price_df: pd.DataFrame,
    fund_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Run the full feature engineering pipeline.

    Args:
        price_df: Output from load_price_data
        fund_df: Output from load_fundamental_data (optional)

    Returns:
        DataFrame with all features. Rows with insufficient history (first 120 rows)
        are dropped.
    """
    df = build_price_features(price_df)
    if fund_df is not None:
        df = build_fundamental_features(df, fund_df)

    # Drop rows with no features (warm-up period)
    df = df.dropna(subset=["ma5", "ma20", "ma60", "rsi_14", "macd_dif"]).reset_index(drop=True)
    return df


def get_feature_columns(df: pd.DataFrame) -> dict:
    """
    Return the list of feature columns by category.

    Returns:
        Dict with keys: price_features, fundamental_features, all_features
    """
    exclude = {"date", "open", "high", "low", "close", "settle", "volume",
               "return_1d", "month", "quarter"}

    all_feat = [c for c in df.columns if c not in exclude and not c.startswith("month_") and not c.startswith("quarter_")]

    fund_feat = [c for c in all_feat if c in {
        "north_port_price", "south_port_price", "port_basis", "port_basis_ma5",
        "port_basis_change", "north_south_spread", "warehouse_receipts",
        "wr_ma5", "wr_ma20", "wr_change_rate", "wr_ratio_to_ma5", "wr_ratio_to_ma20",
        "wr_pct_rank", "nc_processing_avg", "ne_processing_avg",
        "processing_bid_spread", "processing_profit_proxy",
        "wheat_sub_advantage", "wheat_sub_ma5", "wheat_sub_change",
        "shandong_vehicles", "shandong_vehicles_ma5", "shandong_vehicles_ratio",
        "total_processing_volume", "total_pv_change",
    }]

    price_feat = [c for c in all_feat if c not in fund_feat]

    return {
        "price_features": price_feat,
        "fundamental_features": fund_feat,
        "all_features": all_feat,
    }
