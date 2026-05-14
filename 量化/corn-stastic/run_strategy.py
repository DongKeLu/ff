#!/usr/bin/env python3
"""
双因子 + IC 自适应择时策略 — 主运行脚本

用法：
    python run_strategy.py              # 默认运行完整回测（不含 CFTC）
    python run_strategy.py --cftc       # 启用 CFTC 第三因子
    python run_strategy.py --latest     # 仅输出最新信号
    python run_strategy.py --sensitivity # 运行参数敏感性分析

输出：
    results/ 目录下所有图表和结果文件
"""

from __future__ import annotations

import sys
import argparse
from pathlib import Path
from typing import Tuple, Dict

import pandas as pd
import numpy as np

from data_loader import load_all_daily_data, build_merged_dataset, compute_rolling_ic, load_and_merge_cftc
from dual_factor_ic_strategy import DualFactorICStrategy, StrategyParams
from backtest_engine import BacktestEngine
from parameter_analysis import (
    sensitivity_cbot_threshold,
    sensitivity_ic_threshold,
    sensitivity_spread_threshold,
    grid_search,
    plot_sensitivity_line,
    plot_heatmap,
    find_robust_params,
)


def setup_output_dir() -> Path:
    out = Path(__file__).parent / "results"
    out.mkdir(exist_ok=True)
    return out


def run_full_backtest(
    df: pd.DataFrame,
    params: StrategyParams,
    out_dir: Path,
    use_cftc: bool = False,
) -> Tuple[pd.DataFrame, dict]:
    """运行完整回测并生成所有图表。"""
    prefix = "[CFTC] " if use_cftc else ""
    print("\n" + "=" * 60)
    print(f"Running: {prefix}Dual-Factor + IC Adaptive Strategy")
    print("=" * 60)

    strat = DualFactorICStrategy(params)
    signals = strat.generate_signals(df.copy())

    # 过滤掉 NaN 收益率的交易日
    signals_clean = signals.dropna(subset=["dce_corn_ret"]).copy()

    # 回测
    engine = BacktestEngine(commission_rate=0.0003)
    result = engine.run(signals_clean)

    # 绩效汇总
    stats = engine.summary(result)

    print("\n--- Performance Summary ---")
    for k, v in stats.items():
        print(f"  {k}: {v}")

    # 年度收益
    annual = engine.annual_returns(result)
    print("\n--- Annual Returns ---")
    print(annual.to_string())

    # 生成图表
    suffix = "_cftc" if use_cftc else ""
    print("\nGenerating charts ...")
    engine.plot_equity_curve(result, save_path=out_dir / f"equity_curve{suffix}.png")
    engine.plot_ic_analysis(result, save_path=out_dir / f"ic_analysis{suffix}.png")
    engine.plot_annual_returns(result, save_path=out_dir / f"annual_returns{suffix}.png")
    engine.plot_factor_signals(result, save_path=out_dir / f"factor_signals{suffix}.png")

    # 保存每日结果到 CSV
    export_cols = [
        "date", "cbot_close", "cbot_ret",
        "dce_corn_close", "dce_corn_ret",
        "spread", "spread_zscore",
        "rolling_ic", "ic_mean_20d", "ic_active",
        "cbot_signal", "spread_signal",
        "combined_signal", "position",
        "position_tomorrow", "position_actual",
        "daily_ret", "daily_ret_net",
        "commission", "equity", "drawdown",
    ]
    if use_cftc:
        cftc_cols = ["cftc_signal", "fund_net_pct", "fund_concentration"]
        for col in cftc_cols:
            if col in result.columns:
                export_cols.append(col)
    result[export_cols].to_csv(out_dir / f"daily_results{suffix}.csv", index=False)
    print(f"  Daily results saved to {out_dir / f'daily_results{suffix}.csv'}")

    return result, stats


def run_parameter_analysis(df: pd.DataFrame, out_dir: Path):
    """运行参数敏感性分析。"""
    print("\n" + "=" * 60)
    print("Running: Parameter Sensitivity Analysis")
    print("=" * 60)

    # 1. CBOT 阈值敏感性
    print("\n1. CBOT threshold sensitivity ...")
    cbot_sens = sensitivity_cbot_threshold(df)
    cbot_sens.to_csv(out_dir / "sens_cbot_threshold.csv", index=False)
    plot_sensitivity_line(
        cbot_sens, "cbot_threshold", "sharpe",
        save_path=out_dir / "sens_cbot_threshold.png",
    )

    # 2. 价差阈值敏感性
    print("2. Spread threshold sensitivity ...")
    spread_sens = sensitivity_spread_threshold(df)
    spread_sens.to_csv(out_dir / "sens_spread_threshold.csv", index=False)
    plot_sensitivity_line(
        spread_sens, "spread_threshold", "sharpe",
        save_path=out_dir / "sens_spread_threshold.png",
    )

    # 3. IC 阈值敏感性
    print("3. IC threshold sensitivity ...")
    ic_sens = sensitivity_ic_threshold(df)
    ic_sens.to_csv(out_dir / "sens_ic_threshold.csv", index=False)
    plot_sensitivity_line(
        ic_sens, "ic_threshold", "sharpe",
        save_path=out_dir / "sens_ic_threshold.png",
    )

    # 4. 三维网格搜索
    print("4. Full grid search (this may take ~1 minute) ...")
    grid = grid_search(df)
    grid.to_csv(out_dir / "grid_search_results.csv", index=False)

    # 热力图：IC阈值 × CBOT阈值（夏普）
    plot_heatmap(
        grid, "cbot_threshold", "ic_threshold", z_col="sharpe",
        save_path=out_dir / "heatmap_sharpe.png",
    )
    # 热力图：IC阈值 × CBOT阈值（最大回撤）
    plot_heatmap(
        grid, "cbot_threshold", "ic_threshold", z_col="max_dd",
        save_path=out_dir / "heatmap_maxdd.png",
    )

    # 稳健参数
    robust = find_robust_params(df)
    print("\n--- Robust Parameters (top-20 mode) ---")
    for k, v in robust.items():
        print(f"  {k}: {v}")

    # Top-10 参数组合
    print("\n--- Top-10 Parameter Combinations ---")
    print(grid.head(10)[
        ["cbot_threshold", "spread_threshold", "ic_threshold",
         "total_ret", "sharpe", "max_dd", "ic_active_rate", "score"]
    ].to_string(index=False))


def show_latest_signals(df: pd.DataFrame, params: StrategyParams, n: int = 10):
    """输出最近N日的交易信号。"""
    strat = DualFactorICStrategy(params)
    signals = strat.generate_signals(df.copy())
    report = strat.daily_report(signals.tail(n))

    print("\n" + "=" * 60)
    print("Latest Signals (most recent first)")
    print("=" * 60)

    # Human-readable signal labels
    def signal_label(s):
        return {1: "Long", -1: "Short", 0: "Flat"}.get(int(s), str(s))

    report = report.copy()
    report["cbot_signal"] = report["cbot_signal"].apply(signal_label)
    report["spread_signal"] = report["spread_signal"].apply(signal_label)
    report["position"] = report["position"].apply(signal_label)
    report["ic_active"] = report["ic_active"].map({True: "ON", False: "OFF"})

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 20)
    print(report.to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description="Dual-Factor + IC Adaptive Strategy")
    parser.add_argument(
        "--mode",
        choices=["full", "latest", "sensitivity"],
        default="full",
        help="'full': run backtest + charts (default); "
             "'latest': show latest signals; "
             "'sensitivity': run parameter sensitivity analysis",
    )
    parser.add_argument(
        "--excel",
        type=str,
        default="玉米期货数据.xlsx",
        help="Path to the data Excel file (default: 玉米期货数据.xlsx)",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="results",
        help="Output directory for results (default: results/)",
    )
    parser.add_argument(
        "--cftc",
        action="store_true",
        help="Enable CFTC fund net-position as the third factor",
    )
    args = parser.parse_args()

    excel_path = Path(__file__).parent / args.excel
    if not excel_path.exists():
        print(f"Error: Excel file not found: {excel_path}")
        sys.exit(1)

    out_dir = Path(__file__).parent / args.out_dir
    out_dir.mkdir(exist_ok=True)

    # 加载数据
    print("Loading data ...")
    data = load_all_daily_data(excel_path)
    df = build_merged_dataset(data, start_date="2015-01-01", end_date="2026-05-12")
    df = compute_rolling_ic(df)

    # CFTC 数据（可选）
    use_cftc = args.cftc
    if use_cftc:
        cftc_path = Path(__file__).parent / "data" / "cftc_corn_disaggregated.csv"
        if cftc_path.exists():
            df = load_and_merge_cftc(df, str(cftc_path))
            print(f"  CFTC data merged: {df['fund_net_pct'].notna().sum()} rows with CFTC data")
        else:
            print(f"  Warning: CFTC data not found at {cftc_path}, running without CFTC.")
            use_cftc = False

    print(f"Data range: {df['date'].min().date()} ~ {df['date'].max().date()}")
    print(f"Total trading days: {len(df)}")

    if use_cftc:
        params = StrategyParams(
            cbot_threshold=0.007,
            spread_zscore_threshold=1.0,
            combined_threshold=0.3,
            ic_window=60,
            ic_mean_window=20,
            ic_threshold=0.03,
            cbot_weight=0.4,
            spread_weight=0.4,
            cftc_weight=0.2,
            cftc_net_pct_threshold=0.10,
            cftc_congestion_cap=0.30,
        )
        print("Strategy config: Dual-Factor + CFTC (weights: CBOT=0.4, Spread=0.4, CFTC=0.2)")
    else:
        params = StrategyParams(
            cbot_threshold=0.007,
            spread_zscore_threshold=1.0,
            combined_threshold=0.3,
            ic_window=60,
            ic_mean_window=20,
            ic_threshold=0.03,
            cbot_weight=0.5,
            spread_weight=0.5,
        )
        print("Strategy config: Dual-Factor (weights: CBOT=0.5, Spread=0.5)")

    if args.mode == "full":
        run_full_backtest(df, params, out_dir, use_cftc=use_cftc)

    elif args.mode == "latest":
        show_latest_signals(df, params, n=10)

    elif args.mode == "sensitivity":
        run_parameter_analysis(df, out_dir)


if __name__ == "__main__":
    main()
