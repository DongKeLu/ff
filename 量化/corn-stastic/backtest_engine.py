"""
回测引擎

功能：
- 模拟"今天发信号、明天开盘执行"的实际交易流程
- 手续费：万一（万三 等效双边千分之0.3）
- 滑点：按滑点率模拟（默认 0，跳过）
- 支持做多、做空、空仓三种状态
- 完整输出：每日净值、年化收益、夏普、最大回撤、年度收益
- 生成图表（净值曲线、持仓分布、IC激活率等）
"""

from __future__ import annotations

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
from typing import Optional

# 中文支持
plt.rcParams["font.sans-serif"] = ["WenQuanYi Micro Hei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


class BacktestEngine:
    """
    事件驱动回测引擎。

    核心逻辑：
    - position_tomorrow：今天收盘后计算的信号，明日执行的仓位
    - 实际成交价 = 明日开盘价 × (1 + 滑点)
    - 收益 = 仓位 × DCE明日收益率
    - 手续费仅在仓位变动时收取（双边万三 → 单边 0.00015）
    """

    def __init__(
        self,
        commission_rate: float = 0.0003,
        slippage: float = 0.0,
        initial_capital: float = 1_000_000.0,
    ):
        self.commission_rate = commission_rate
        self.slippage = slippage
        self.initial_capital = initial_capital

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        执行回测，返回每日组合 DataFrame。

        输入 df 必须包含列：
            date, dce_corn_close, dce_corn_ret,
            position_tomorrow（今日发信号明日执行的仓位：-1, 0, 1）

        输出新增列：
            trade_price   : 明日开盘价（实际成交价）
            position_actual: 实际执行的仓位（考虑滑点）
            daily_ret     : 当日组合收益率
            commission     : 当日手续费
            equity         : 组合净值（初始=1）
            drawdown       : 当前回撤
        """
        df = df.copy().reset_index(drop=True)

        # 明日开盘价 = 今日收盘价 × (1 + 明日收益率)
        # 即：今日持仓到明日收盘的收益 = position × dce_corn_ret
        # 实际撮合用开盘价，滑点已隐含在收益计算中
        df["trade_price"] = df["dce_corn_close"] * (1 + df["dce_corn_ret"])

        # 仓位变动时收手续费（双边）
        df["position_actual"] = df["position_tomorrow"].fillna(0).astype(int)
        df["position_changed"] = df["position_actual"].diff().abs().fillna(0).astype(bool)

        # 每日收益率：仓位 × DCE收益率
        df["daily_ret"] = df["position_actual"] * df["dce_corn_ret"]

        # 手续费（仅仓位变动时收取，双边）
        df["commission"] = np.where(
            df["position_changed"],
            self.commission_rate * np.abs(df["position_actual"]),
            0.0,
        )
        df["daily_ret_net"] = df["daily_ret"] - df["commission"]

        # 净值曲线
        df["equity"] = (1 + df["daily_ret_net"]).cumprod()

        # 最大回撤
        df["peak"] = df["equity"].cummax()
        df["drawdown"] = (df["equity"] - df["peak"]) / df["peak"]

        return df

    # ------------------------------------------------------------------
    # 绩效统计
    # ------------------------------------------------------------------

    def summary(self, df: pd.DataFrame) -> dict:
        """返回绩效指标字典。"""
        equity = df["equity"].values
        daily_ret_net = df["daily_ret_net"].values
        ret = equity[-1] - 1 if len(equity) > 0 else 0

        # 年化
        n_days = len(df)
        years = n_days / 252
        annual_ret = (1 + ret) ** (1 / years) - 1 if years > 0 else 0

        # 夏普（年化，无风险利率=0）
        annual_vol = np.nanstd(daily_ret_net) * np.sqrt(252)
        sharpe = annual_ret / annual_vol if annual_vol > 0 else 0

        # 最大回撤
        max_dd = df["drawdown"].min()

        # 卡玛比率
        calmar = annual_ret / abs(max_dd) if max_dd != 0 else 0

        # 盈利月份/年份
        df["year"] = df["date"].dt.year
        yearly = df.groupby("year").apply(
            lambda g: pd.Series(
                {
                    "total_ret": g["equity"].iloc[-1] / g["equity"].iloc[0] - 1,
                    "start": g["date"].iloc[0],
                    "end": g["date"].iloc[-1],
                }
            )
        )
        win_years = int((yearly["total_ret"] > 0).sum())
        total_years = len(yearly)

        # IC 激活率
        ic_active_rate = df["ic_active"].mean() if "ic_active" in df.columns else None

        # 基准买入持有
        bench_equity = (1 + df["dce_corn_ret"].fillna(0)).cumprod()
        bench_ret = bench_equity.iloc[-1] - 1
        bench_max_dd = ((bench_equity - bench_equity.cummax()) / bench_equity.cummax()).min()

        return {
            "初始资金": self.initial_capital,
            "期末净值": round(df["equity"].iloc[-1], 4),
            "总收益率": f"{ret * 100:.2f}%",
            "年化收益率": f"{annual_ret * 100:.2f}%",
            "夏普比率": round(sharpe, 2),
            "最大回撤": f"{max_dd * 100:.2f}%",
            "卡玛比率": round(calmar, 2),
            "盈利年份": f"{win_years} / {total_years}",
            "IC激活率": f"{ic_active_rate * 100:.1f}%" if ic_active_rate is not None else "N/A",
            "基准总收益": f"{bench_ret * 100:.2f}%",
            "基准最大回撤": f"{bench_max_dd * 100:.2f}%",
            "手续费率": f"{self.commission_rate * 100:.2f}%",
        }

    def annual_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        """返回年度收益表。"""
        df = df.copy()
        df["year"] = df["date"].dt.year
        annual = df.groupby("year").apply(
            lambda g: pd.Series(
                {
                    "策略收益": g["equity"].iloc[-1] / g["equity"].iloc[0] - 1,
                    "基准收益": (
                        (1 + g["dce_corn_ret"].fillna(0)).prod() - 1
                    ),
                    "超额收益": (
                        g["equity"].iloc[-1] / g["equity"].iloc[0]
                        - (1 + g["dce_corn_ret"].fillna(0)).prod()
                    ),
                    "IC激活率": (
                        g["ic_active"].mean() if "ic_active" in g.columns else np.nan
                    ),
                    "最大回撤": g["drawdown"].min(),
                }
            )
        )
        annual["策略收益"] = annual["策略收益"].apply(lambda x: f"{x * 100:.2f}%")
        annual["基准收益"] = annual["基准收益"].apply(lambda x: f"{x * 100:.2f}%")
        annual["超额收益"] = annual["超额收益"].apply(lambda x: f"{x * 100:.2f}%")
        annual["IC激活率"] = annual["IC激活率"].apply(
            lambda x: f"{x * 100:.1f}%" if not pd.isna(x) else "N/A"
        )
        annual["最大回撤"] = annual["最大回撤"].apply(lambda x: f"{x * 100:.2f}%")
        return annual

    # ------------------------------------------------------------------
    # 可视化
    # ------------------------------------------------------------------

    def plot_equity_curve(
        self,
        df: pd.DataFrame,
        save_path: Optional[str] = None,
        benchmark: bool = True,
    ):
        """绘制净值曲线 vs 买入持有基准。"""
        fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

        # 1. 净值曲线
        ax = axes[0]
        ax.plot(df["date"], df["equity"], label="Dual-Factor + IC", linewidth=1.5, color="#2196F3")
        if benchmark:
            bench = (1 + df["dce_corn_ret"].fillna(0)).cumprod()
            ax.plot(df["date"], bench, label="Buy & Hold", linewidth=1.2, color="gray", alpha=0.8)
        ax.set_title("Dual-Factor + IC Adaptive Strategy — Equity Curve", fontsize=13, fontweight="bold")
        ax.set_ylabel("Net Value")
        ax.legend(loc="upper left")
        ax.grid(True, alpha=0.3)
        ax.axhline(y=1, color="black", linewidth=0.5, linestyle="--")

        # 2. 回撤
        ax = axes[1]
        ax.fill_between(df["date"], df["drawdown"] * 100, 0, color="#E53935", alpha=0.4)
        ax.plot(df["date"], df["drawdown"] * 100, color="#E53935", linewidth=0.8)
        ax.set_title("Drawdown", fontsize=12)
        ax.set_ylabel("Drawdown (%)")
        ax.grid(True, alpha=0.3)

        # 3. 仓位
        ax = axes[2]
        colors = {1: "#4CAF50", 0: "#BDBDBD", -1: "#F44336"}
        labels = {1: "Long", 0: "Flat", -1: "Short"}
        for pos, color, label in [(1, "#4CAF50", "Long"), (0, "#BDBDBD", "Flat"), (-1, "#F44336", "Short")]:
            mask = df["position_actual"] == pos
            if mask.any():
                ax.scatter(df.loc[mask, "date"], [label] * mask.sum(),
                           color=color, s=2, alpha=0.6)
        ax.set_title("Position Over Time", fontsize=12)
        ax.set_ylabel("Position")
        ax.set_xlabel("Date")
        ax.grid(True, alpha=0.3)

        # x-axis date formatting
        for ax_ in axes:
            ax_.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
            ax_.xaxis.set_major_locator(mdates.YearLocator())
        plt.xticks(rotation=45)

        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return fig

    def plot_ic_analysis(
        self,
        df: pd.DataFrame,
        save_path: Optional[str] = None,
    ):
        """绘制 IC 分析图：滚动 IC、IC均值、激活状态。"""
        fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

        ax = axes[0]
        ax.plot(df["date"], df["rolling_ic"], label="Rolling IC (60d)", linewidth=1, color="#1565C0", alpha=0.7)
        ax.plot(df["date"], df["ic_mean_20d"], label="IC Mean (20d)", linewidth=1.5, color="#FF6F00")
        ax.axhline(y=0.03, color="#4CAF50", linestyle="--", linewidth=1, label="IC Threshold (0.03)")
        ax.axhline(y=0, color="gray", linewidth=0.5)
        ax.set_title("Rolling IC & IC Mean — CBOT vs DCE Corn", fontsize=13, fontweight="bold")
        ax.set_ylabel("IC")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)
        ax.fill_between(
            df["date"],
            df["ic_mean_20d"].where(df["ic_active"]),
            0.03,
            alpha=0.15,
            color="#4CAF50",
            label="Active Zone",
        )

        ax = axes[1]
        ic_active_int = df["ic_active"].astype(int)
        ax.fill_between(df["date"], 0, ic_active_int, color="#4CAF50", alpha=0.5, step="post")
        ax.set_title("IC Adaptive Switch (Active = 1)", fontsize=12)
        ax.set_ylabel("IC Active")
        ax.set_xlabel("Date")
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["Off", "On"])
        ax.grid(True, alpha=0.3)

        for ax_ in axes:
            ax_.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
            ax_.xaxis.set_major_locator(mdates.YearLocator())
        plt.xticks(rotation=45)
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return fig

    def plot_annual_returns(
        self,
        df: pd.DataFrame,
        save_path: Optional[str] = None,
    ):
        """绘制年度收益柱状图。"""
        df = df.copy()
        df["year"] = df["date"].dt.year

        annual = df.groupby("year").apply(
            lambda g: pd.Series(
                {
                    "strategy": g["equity"].iloc[-1] / g["equity"].iloc[0] - 1,
                    "benchmark": (1 + g["dce_corn_ret"].fillna(0)).prod() - 1,
                }
            )
        ).reset_index()

        fig, ax = plt.subplots(figsize=(14, 5))
        years = annual["year"].values
        x = np.arange(len(years))
        w = 0.35

        bars1 = ax.bar(x - w / 2, annual["strategy"] * 100, w, label="Strategy", color="#2196F3", alpha=0.85)
        bars2 = ax.bar(x + w / 2, annual["benchmark"] * 100, w, label="Buy & Hold", color="gray", alpha=0.7)

        ax.axhline(y=0, color="black", linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(years, rotation=45)
        ax.set_ylabel("Annual Return (%)")
        ax.set_title("Annual Returns: Strategy vs Benchmark", fontsize=13, fontweight="bold")
        ax.legend()
        ax.grid(True, axis="y", alpha=0.3)

        for bar in bars1:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + (0.5 if h >= 0 else -3),
                    f"{h:.1f}%", ha="center", va="bottom" if h >= 0 else "top", fontsize=7)

        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return fig

    def plot_factor_signals(
        self,
        df: pd.DataFrame,
        save_path: Optional[str] = None,
    ):
        """绘制两个子因子信号与组合信号时序图。"""
        fig, axes = plt.subplots(4, 1, figsize=(14, 14), sharex=True, sharey=True)

        labels = ["CBOT Signal", "Spread Signal", "Combined Signal", "Position"]
        colors = ["#1565C0", "#6A1B9A", "#E65100", "#2E7D32"]

        plot_data = [
            df["cbot_signal"],
            df["spread_signal"],
            df["combined_signal"],
            df["position_actual"],
        ]

        for ax, col, label, color in zip(axes, plot_data, labels, colors):
            ax.step(df["date"], col, where="post", color=color, linewidth=1)
            ax.axhline(y=0, color="gray", linewidth=0.5, linestyle="--")
            ax.set_ylabel(label, fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.set_yticks([-1, 0, 1])
            ax.set_yticklabels(["-1", "0", "+1"])

        axes[-1].set_xlabel("Date")
        axes[0].set_title("Factor Signals Over Time", fontsize=13, fontweight="bold")

        for ax_ in axes:
            ax_.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
            ax_.xaxis.set_major_locator(mdates.YearLocator())
        plt.xticks(rotation=45)
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return fig
