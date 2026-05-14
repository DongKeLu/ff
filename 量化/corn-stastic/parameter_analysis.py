"""
参数敏感性分析

对以下关键参数进行网格搜索，评估组合绩效（总收益、夏普、最大回撤）：
  1. CBOT 阈值：0.5%, 0.6%, 0.7%, 0.8%, 0.9%, 1.0%
  2. 价差 Z-score 阈值：0.8, 1.0, 1.2, 1.5
  3. IC 阈值：0.01, 0.02, 0.03, 0.05, 0.07
  4. IC 均值窗口：10, 20, 30, 40

输出：
  - 热力图：两两参数组合下的夏普比率
  - 汇总表：所有参数组合的绩效排名
  - 最优参数建议
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path
from itertools import product
from typing import Optional, List

from data_loader import load_all_daily_data, build_merged_dataset, compute_rolling_ic
from dual_factor_ic_strategy import DualFactorICStrategy, StrategyParams
from backtest_engine import BacktestEngine


def evaluate_params(
    df: pd.DataFrame,
    params: StrategyParams,
) -> dict:
    """
    用给定参数运行完整回测，返回绩效指标。
    """
    strat = DualFactorICStrategy(params)
    signals = strat.generate_signals(df.copy())
    engine = BacktestEngine(commission_rate=0.0003)
    result = engine.run(signals.dropna(subset=["dce_corn_ret"]).copy())
    stats = engine.summary(result)

    def parse_pct(s: str) -> float:
        return float(s.rstrip("%")) / 100

    total_ret = parse_pct(stats["总收益率"])
    sharpe = stats["夏普比率"]
    max_dd = parse_pct(stats["最大回撤"])

    return {
        "total_ret": total_ret,
        "annual_ret": parse_pct(stats["年化收益率"]),
        "sharpe": sharpe,
        "max_dd": max_dd,
        "ic_active_rate": (
            float(stats["IC激活率"].rstrip("%")) / 100
            if stats["IC激活率"] != "N/A"
            else np.nan
        ),
    }


def sensitivity_cbot_threshold(
    df: pd.DataFrame,
    thresholds: Optional[List[float]] = None,
) -> pd.DataFrame:
    """单一参数：CBOT 阈值 vs 绩效。"""
    if thresholds is None:
        thresholds = [0.005, 0.006, 0.007, 0.008, 0.009, 0.01]
    results = []
    for t in thresholds:
        p = StrategyParams(cbot_threshold=t)
        r = evaluate_params(df, p)
        r["cbot_threshold"] = t
        results.append(r)
    return pd.DataFrame(results)


def sensitivity_spread_threshold(
    df: pd.DataFrame,
    thresholds: Optional[List[float]] = None,
) -> pd.DataFrame:
    """单一参数：价差 Z-score 阈值 vs 绩效。"""
    if thresholds is None:
        thresholds = [0.8, 1.0, 1.2, 1.5]
    results = []
    for t in thresholds:
        p = StrategyParams(spread_zscore_threshold=t)
        r = evaluate_params(df, p)
        r["spread_threshold"] = t
        results.append(r)
    return pd.DataFrame(results)


def sensitivity_ic_threshold(
    df: pd.DataFrame,
    thresholds: Optional[List[float]] = None,
) -> pd.DataFrame:
    """单一参数：IC 阈值 vs 绩效。"""
    if thresholds is None:
        thresholds = [0.01, 0.02, 0.03, 0.05, 0.07]
    results = []
    for t in thresholds:
        p = StrategyParams(ic_threshold=t)
        r = evaluate_params(df, p)
        r["ic_threshold"] = t
        results.append(r)
    return pd.DataFrame(results)


def sensitivity_ic_window(
    df: pd.DataFrame,
    windows: Optional[List[int]] = None,
) -> pd.DataFrame:
    """单一参数：IC 均值窗口 vs 绩效。"""
    if windows is None:
        windows = [10, 15, 20, 30, 40]
    results = []
    for w in windows:
        p = StrategyParams(ic_mean_window=w)
        r = evaluate_params(df, p)
        r["ic_mean_window"] = w
        results.append(r)
    return pd.DataFrame(results)


def grid_search(
    df: pd.DataFrame,
    cbot_thresholds: Optional[List[float]] = None,
    spread_thresholds: Optional[List[float]] = None,
    ic_thresholds: Optional[List[float]] = None,
) -> pd.DataFrame:
    """三维网格搜索，返回所有组合的绩效表。"""
    if cbot_thresholds is None:
        cbot_thresholds = [0.005, 0.006, 0.007, 0.008, 0.009, 0.01]
    if spread_thresholds is None:
        spread_thresholds = [0.8, 1.0, 1.2, 1.5]
    if ic_thresholds is None:
        ic_thresholds = [0.01, 0.02, 0.03, 0.05]

    results = []
    total = len(cbot_thresholds) * len(spread_thresholds) * len(ic_thresholds)
    print(f"Grid search: {total} parameter combinations ...")

    for i, (cbt, spt, ict) in enumerate(
        product(cbot_thresholds, spread_thresholds, ic_thresholds)
    ):
        p = StrategyParams(
            cbot_threshold=cbt,
            spread_zscore_threshold=spt,
            ic_threshold=ict,
        )
        r = evaluate_params(df, p)
        r["cbot_threshold"] = cbt
        r["spread_threshold"] = spt
        r["ic_threshold"] = ict
        results.append(r)
        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{total} done")

    df_out = pd.DataFrame(results)
    # 综合得分：夏普权重0.4，总收益权重0.3，最大回撤权重0.3
    df_out["score"] = (
        df_out["sharpe"] / df_out["sharpe"].max()
        + df_out["total_ret"] / df_out["total_ret"].max()
        + (1 - df_out["max_dd"].abs() / df_out["max_dd"].abs().min())
    ) / 3

    df_out.sort_values("score", ascending=False, inplace=True)
    df_out.reset_index(drop=True, inplace=True)
    return df_out


def plot_sensitivity_line(
    df_sens: pd.DataFrame,
    param_col: str,
    metric: str = "sharpe",
    save_path: Optional[str] = None,
):
    """绘制单参数敏感性折线图。"""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df_sens[param_col], df_sens[metric], "o-", linewidth=2, markersize=6, color="#1565C0")
    ax.set_xlabel(param_col, fontsize=11)
    ax.set_ylabel(metric.replace("_", " ").title(), fontsize=11)
    ax.set_title(f"Sensitivity: {param_col} vs {metric}", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_heatmap(
    df_grid: pd.DataFrame,
    x_col: str,
    y_col: str,
    z_col: str = "sharpe",
    save_path: Optional[str] = None,
):
    """绘制两参数组合热力图。"""
    pivot = df_grid.pivot_table(values=z_col, index=y_col, columns=x_col)
    fig, ax = plt.subplots(figsize=(10, 7))
    cmap = plt.cm.RdYlGn
    im = ax.imshow(pivot.values, cmap=cmap, aspect="auto", origin="lower")
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_xticklabels([f"{v:.3f}" if isinstance(v, float) else str(v) for v in pivot.columns], rotation=45)
    ax.set_yticklabels([f"{v:.3f}" if isinstance(v, float) else str(v) for v in pivot.index])
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(f"Heatmap: {z_col} — {x_col} × {y_col}", fontsize=13, fontweight="bold")
    plt.colorbar(im, ax=ax, label=z_col.replace("_", " ").title())
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def find_robust_params(df: pd.DataFrame) -> dict:
    """
    寻找对参数扰动最不敏感的稳健参数区域。
    通过标准差最小的组合来评估稳健性。
    """
    grid = grid_search(df)
    # 按夏普排名前20的组合，看参数的集中度
    top20 = grid.head(20)
    robust = {
        "cbot_threshold": top20["cbot_threshold"].mode()[0],
        "spread_threshold": top20["spread_threshold"].mode()[0],
        "ic_threshold": top20["ic_threshold"].mode()[0],
    }
    return robust
