"""
数据加载工具
从 Excel 文件读取 DCE 玉米、DCE 淀粉、CBOT 玉米的日线数据，并进行预处理。
支持 CFTC 基金净持仓数据的融合（前向填充）。
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional


def load_all_daily_data(excel_path: str) -> Dict[str, pd.DataFrame]:
    """
    读取 Excel 中所有日线 sheet，返回 {sheet_name: DataFrame} 字典。

    处理逻辑：
    - 统一列名：小写 + 下划线
    - 日期列转 datetime
    - CBOT 价格单位统一为 美分/蒲式耳 → 转换为与 DCE 相同的 'price' 列（close）
    - 向前填充 CBOT 的交易日空白（CBOT 交易日比 DCE 少）
    """
    excel_path = Path(excel_path)
    raw = pd.read_excel(excel_path, sheet_name=None)

    daily_sheets = {
        "DCE玉米加权日线": "dce_corn",
        "DCE淀粉加权日线": "dce_starch",
        "CBOT玉米加权日线": "cbot_corn",
    }

    result = {}

    for sheet_name, key in daily_sheets.items():
        df = raw[sheet_name].copy()

        # 统一列名
        col_map = {
            "日期": "date",
            "开盘价": "open",
            "最高价": "high",
            "最低价": "low",
            "收盘价": "close",
            "成交量": "volume",
            "持仓量": "open_interest",
            "结算价": "settlement",
        }
        df.rename(columns=col_map, inplace=True)
        df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)

        # CBOT 原始单位是 美分/蒲式耳，保持不变，后续用日收益率
        result[key] = df[["date", "open", "high", "low", "close", "volume", "open_interest", "settlement"]].copy()

    return result


def build_merged_dataset(
    data: Dict[str, pd.DataFrame],
    start_date="2015-01-01",
    end_date="2026-05-12",
) -> pd.DataFrame:
    """
    将 CBOT、DCE玉米、DCE淀粉 三张表按日期对齐，生成每日特征矩阵。

    输出列：
        date, cbot_close, cbot_ret, dce_corn_close, dce_corn_ret,
        dce_starch_close, dce_starch_ret, spread, spread_zscore
    """
    cbot = data["cbot_corn"][["date", "close"]].rename(columns={"close": "cbot_close"})
    corn = data["dce_corn"][["date", "close"]].rename(columns={"close": "dce_corn_close"})
    starch = data["dce_starch"][["date", "close"]].rename(columns={"close": "dce_starch_close"})

    # 以 DCE 玉米交易日为基准（更严格的国内交易日过滤）
    merged = corn.merge(cbot, on="date", how="left")
    merged = merged.merge(starch, on="date", how="left")

    # 向前填充 CBOT 休市日的收盘价
    merged["cbot_close"] = merged["cbot_close"].ffill()

    # 计算日收益率
    merged["cbot_ret"] = merged["cbot_close"].pct_change()
    merged["dce_corn_ret"] = merged["dce_corn_close"].pct_change()
    merged["dce_starch_ret"] = merged["dce_starch_close"].pct_change()

    # 淀粉-玉米价差
    merged["spread"] = merged["dce_starch_close"] - merged["dce_corn_close"]

    # 价差 Z-score（滚动 60 天）
    spread_mean = merged["spread"].rolling(60, min_periods=30).mean()
    spread_std = merged["spread"].rolling(60, min_periods=30).std()
    merged["spread_zscore"] = (merged["spread"] - spread_mean) / spread_std

    # 过滤时间范围
    merged = merged[
        (merged["date"] >= pd.to_datetime(start_date))
        & (merged["date"] <= pd.to_datetime(end_date))
    ].copy()

    merged.reset_index(drop=True, inplace=True)
    return merged


def compute_rolling_ic(
    df: pd.DataFrame,
    ic_window: int = 60,
    ic_mean_window: int = 20,
) -> pd.DataFrame:
    """
    计算滚动 IC（CBOT 收益与 DCE 玉米收益的 Pearson 相关系数）。

    ic_window     : 计算单个 IC 的滚动天数
    ic_mean_window: 过去 IC 均值的天数

    输出新增列：
        rolling_ic      : 每日 IC 值（60天滚动）
        ic_mean_20d     : 过去20天 IC 均值
        ic_active       : bool，IC均值 > 0.03 时为 True
    """
    df = df.copy()

    # 滚动 IC
    df["rolling_ic"] = (
        df["cbot_ret"]
        .rolling(ic_window, min_periods=int(ic_window * 0.6))
        .corr(df["dce_corn_ret"])
    )

    # IC 均值（过去20天）
    df["ic_mean_20d"] = (
        df["rolling_ic"]
        .rolling(ic_mean_window, min_periods=int(ic_mean_window * 0.6))
        .mean()
    )

    # IC 自适应开关：阈值 0.03
    df["ic_active"] = df["ic_mean_20d"] > 0.03

    return df


def get_latest_signals(df: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    """返回最近 n 天的信号详情，用于每日跟踪报告。"""
    return df.tail(n)[
        [
            "date",
            "cbot_close",
            "cbot_ret",
            "spread",
            "spread_zscore",
            "rolling_ic",
            "ic_mean_20d",
            "ic_active",
            "cbot_signal",
            "spread_signal",
            "combined_signal",
            "position",
        ]
    ].to_string(index=False)


def load_and_merge_cftc(
    merged_df: pd.DataFrame,
    cftc_path: str,
) -> pd.DataFrame:
    """
    将 CFTC 基金净持仓数据融入日线数据。

    处理流程：
    1. 调用 cftc_loader 的 load_and_resample_cftc 将周频数据前向填充到日频
    2. 以 date 为键 left join 到 merged_df
    3. 对早期无 CFTC 数据的日期进行前向填充

    新增列：
        cftc_date           — 最近一次 CFTC 报告日期
        fund_net_pct        — 基金净持仓占比（核心信号）
        fund_concentration  — 基金拥挤度
        fund_net            — 基金净持仓绝对手数
        producer_net_pct    — 实体企业净持仓占比
        swap_net_pct        — 掉期商净持仓占比
    """
    try:
        from cftc_loader import load_and_resample_cftc
    except ImportError:
        print("Warning: cftc_loader not found, skipping CFTC merge.")
        return merged_df

    cftc_path = Path(cftc_path)
    if not cftc_path.exists():
        print(f"Warning: CFTC data file not found at {cftc_path}, skipping CFTC merge.")
        return merged_df

    price_dates = merged_df["date"]
    daily_cftc = load_and_resample_cftc(str(cftc_path), price_dates)
    daily_cftc = daily_cftc.reset_index().rename(columns={"index": "date"})

    if daily_cftc.empty:
        print("Warning: CFTC data loaded empty, skipping merge.")
        return merged_df

    merged_df = merged_df.merge(daily_cftc, on="date", how="left")

    ffill_cols = [
        "cftc_date", "fund_net_pct", "fund_concentration",
        "fund_net", "producer_net_pct", "swap_net_pct",
    ]
    for col in ffill_cols:
        if col in merged_df.columns:
            merged_df[col] = merged_df[col].ffill()

    return merged_df
