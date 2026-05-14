"""
双因子 + IC 自适应择时策略

子因子 1 — CBOT 传导信号
    CBOT 日涨幅 > 0.7%  → 做多 (+1)
    CBOT 日跌幅 < -0.7% → 做空 (-1)
    其他               → 无信号 (0)

子因子 2 — 淀粉-玉米价差信号
    价差 Z-score < -1  → 做多 (+1)
    价差 Z-score > +1  → 做空 (-1)
    其他              → 无信号 (0)

组合信号 = 0.5 × CBOT信号 + 0.5 × 价差信号
    > 0.3  → 多头仓位 (+1)
    < -0.3 → 空头仓位 (-1)
    其余   → 空仓   (0)

IC 自适应开关：
    每天收盘后计算过去 60 天滚动 IC（CBOT收益 与 DCE收益相关性）
    再计算过去 20 天 IC 均值
    IC均值 > 0.03 → 执行组合信号
    IC均值 ≤ 0.03 → 空仓观望

信号执行：今天发信号 → 明天开盘执行
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Optional, Union


@dataclass
class StrategyParams:
    """策略参数集，所有阈值均可配置。"""

    cbot_threshold: float = 0.007
    spread_zscore_threshold: float = 1.0
    combined_threshold: float = 0.3
    ic_window: int = 60
    ic_mean_window: int = 20
    ic_threshold: float = 0.03
    cbot_weight: float = 0.5
    spread_weight: float = 0.5
    cftc_weight: float = 0.0
    cftc_net_pct_threshold: float = 0.10
    cftc_congestion_cap: float = 0.30

    def __post_init__(self):
        total = self.cbot_weight + self.spread_weight + self.cftc_weight
        assert abs(total - 1.0) < 1e-9, \
            f"cbot_weight + spread_weight + cftc_weight must equal 1.0, got {total}"


@dataclass
class DailySignal:
    """单日信号记录。"""

    date: pd.Timestamp
    cbot_ret: float
    spread_zscore: float
    rolling_ic: Optional[float]
    ic_mean_20d: Optional[float]
    ic_active: bool
    cbot_signal: int
    spread_signal: int
    combined_signal: float
    position: int
    cftc_signal: float = 0.0
    fund_net_pct: Optional[float] = None
    fund_concentration: Optional[float] = None


class DualFactorICStrategy:
    """
    双因子 + IC 自适应择时策略。

    使用示例：
        from data_loader import load_all_daily_data, build_merged_dataset, compute_rolling_ic
        from dual_factor_ic_strategy import DualFactorICStrategy, StrategyParams

        data = load_all_daily_data("玉米期货数据.xlsx")
        df   = build_merged_dataset(data)
        df   = compute_rolling_ic(df)
        strat = DualFactorICStrategy(params)
        signals = strat.generate_signals(df)
    """

    def __init__(self, params: Optional[StrategyParams] = None):
        self.params = params or StrategyParams()

    # ------------------------------------------------------------------
    # 子因子
    # ------------------------------------------------------------------

    def compute_cbot_signal(self, cbot_ret: Union[float, np.ndarray]) -> Union[int, np.ndarray]:
        """
        CBOT 传导信号。
        ret >  +threshold → +1（做多）
        ret <  -threshold → -1（做空）
        否则             →  0
        """
        t = self.params.cbot_threshold
        ret = np.asarray(cbot_ret)
        signal = np.where(ret > t, 1, np.where(ret < -t, -1, 0))
        return int(signal[0]) if signal.ndim == 1 and len(signal) == 1 else signal

    def compute_spread_signal(self, spread_zscore: Union[float, np.ndarray]) -> Union[int, np.ndarray]:
        """
        淀粉-玉米价差信号。
        z < -threshold → +1（做多 DCE 玉米，价差偏低）
        z > +threshold → -1（做空 DCE 玉米，价差偏高）
        否则            →  0
        """
        t = self.params.spread_zscore_threshold
        z = np.asarray(spread_zscore)
        signal = np.where(z < -t, 1, np.where(z > t, -1, 0))
        return int(signal[0]) if signal.ndim == 1 and len(signal) == 0 else signal

    def compute_cftc_signal(
        self,
        fund_net_pct: float,
        fund_concentration: float,
    ) -> float:
        """
        CFTC 基金净持仓信号（第三因子）。

        信号逻辑：
        - 净持仓占比 > cftc_net_pct_threshold  → +1（基金净多，看多）
        - 净持仓占比 < -cftc_net_pct_threshold → -1（基金净空，看空）
        - 否则                                  →  0（方向不明确）

        拥挤度衰减：fund_concentration 越高，信号越弱。
        congestion_decay = max(0, 1 - fund_concentration / cftc_congestion_cap)
        最终返回连续信号值 ∈ [-1, 1]。

        缺省值（NaN/0）返回 0。
        """
        if pd.isna(fund_net_pct) or pd.isna(fund_concentration):
            return 0.0

        if fund_net_pct > self.params.cftc_net_pct_threshold:
            direction = 1
        elif fund_net_pct < -self.params.cftc_net_pct_threshold:
            direction = -1
        else:
            direction = 0

        congestion_decay = max(
            0.0,
            1.0 - fund_concentration / self.params.cftc_congestion_cap,
        )
        return direction * congestion_decay

    # ------------------------------------------------------------------
    # 组合信号
    # ------------------------------------------------------------------

    def compute_combined_signal(
        self,
        cbot_signal: int,
        spread_signal: int,
        cftc_signal: float = 0.0,
    ) -> float:
        """组合信号 = 权重1×CBOT + 权重2×价差 + 权重3×CFTC，返回连续值。"""
        return (
            self.params.cbot_weight * cbot_signal
            + self.params.spread_weight * spread_signal
            + self.params.cftc_weight * cftc_signal
        )

    def position_from_combined(self, combined: float) -> int:
        """组合信号 → 仓位。"""
        t = self.params.combined_threshold
        if combined > t:
            return 1
        elif combined < -t:
            return -1
        else:
            return 0

    # ------------------------------------------------------------------
    # 主入口
    # ------------------------------------------------------------------

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        生成每日信号表。

        输入 df 须包含列：
            cbot_ret, spread_zscore, rolling_ic, ic_mean_20d
        可选列（CFTC）：
            fund_net_pct, fund_concentration（不存在时 cftc_weight 必须为 0）

        输出新增列：
            cbot_signal, spread_signal, cftc_signal, combined_signal, position
        """
        df = df.copy()

        df["cbot_signal"] = df["cbot_ret"].apply(self.compute_cbot_signal)
        df["spread_signal"] = df["spread_zscore"].apply(self.compute_spread_signal)

        has_cftc = "fund_net_pct" in df.columns and "fund_concentration" in df.columns
        if has_cftc:
            df["cftc_signal"] = df.apply(
                lambda row: self.compute_cftc_signal(
                    row.get("fund_net_pct", np.nan),
                    row.get("fund_concentration", np.nan),
                ),
                axis=1,
            )
        else:
            df["cftc_signal"] = 0.0

        df["combined_signal"] = (
            df["cbot_signal"] * self.params.cbot_weight
            + df["spread_signal"] * self.params.spread_weight
            + df["cftc_signal"] * self.params.cftc_weight
        )

        df["ic_active"] = df["ic_mean_20d"] > self.params.ic_threshold
        if has_cftc and "fund_net_pct" in df.columns:
            df["ic_active"] = (
                df["ic_active"]
                & (df["fund_net_pct"].fillna(0) < 0.15)
            )

        df["position"] = df.apply(
            lambda row: (
                self.position_from_combined(row["combined_signal"])
                if row["ic_active"]
                else 0
            ),
            axis=1,
        )

        df["position_today"] = df["position"]
        df["position_tomorrow"] = df["position"].shift(1).fillna(0).astype(int)

        return df

    # ------------------------------------------------------------------
    # 每日报告
    # ------------------------------------------------------------------

    def daily_report(self, df: pd.DataFrame) -> pd.DataFrame:
        """生成最近N日信号报告。"""
        cols = [
            "date",
            "cbot_ret",
            "spread_zscore",
            "rolling_ic",
            "ic_mean_20d",
            "ic_active",
            "cbot_signal",
            "spread_signal",
            "cftc_signal",
            "combined_signal",
            "position",
        ]
        present = [c for c in cols if c in df.columns]
        return df[present].tail(10).copy()
