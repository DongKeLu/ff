"""
CFTC 持仓数据获取模块

数据来源: CFTC 官方历史压缩文件
    - Disaggregated Futures Only: fut_disagg_txt_{year}.zip (2017+)
    - Historical combined: fut_disagg_txt_2006_2016.zip (2006-2016)
品种    : CBOT Corn (CFTC Contract Market Code: 002602)

数据说明:
    - Disaggregated 格式将持仓分为:
      Producer/Merchant (实体企业), Swap Dealers (掉期商),
      Managed Money (管理资金/基金), Other Reportable (其他可报告)
    - 每周二数据，周五发布
    - 数据从 2006 年 6 月开始

关键衍生指标:
    fund_net_pct       = (fund_long - fund_short) / open_interest
    fund_concentration = (fund_long + fund_short) / open_interest

用法:
    from cftc_loader import fetch_and_save_cftc_corn, load_and_resample_cftc

    # 一次性获取并存入 CSV
    df = fetch_and_save_cftc_corn(2015, 2025, "data/cftc_corn_disaggregated.csv")

    # 增量更新（只拉取本地最新日期之后的数据）
    df = fetch_and_save_cftc_corn(2025, 2026, "data/cftc_corn_disaggregated.csv", incremental=True)

    # 加载并前向填充为日频
    daily_cftc = load_and_resample_cftc("data/cftc_corn_disaggregated.csv", price_dates)
"""

from __future__ import annotations

import requests
import zipfile
import io
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Optional, List
import warnings

warnings.filterwarnings("ignore", message="Unverified HTTPS request")


# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

# CFTC Disaggregated Futures Only 历史压缩文件 URL 模板
_CFTC_BASE = "https://cftc.gov/files/dea/history"
_HIST_FILE = "fut_disagg_txt_hist_2006_2016.zip"   # 2006-2016 合并文件
_YEAR_FILE = "fut_disagg_txt_{year}.zip"            # 2017+ 年度文件

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; corn-quant-strategy/1.0)",
    "Accept": "application/zip, */*",
}


# ---------------------------------------------------------------------------
# 数据获取
# ---------------------------------------------------------------------------

def _download_zip(url: str, timeout: int = 60) -> bytes:
    """下载并返回 zip 文件内容。"""
    resp = requests.get(url, headers=HEADERS, timeout=timeout, verify=False)
    resp.raise_for_status()
    return resp.content


def _read_corn_from_zip(zip_bytes: bytes, market_filter: str = "CORN - CHICAGO BOARD OF TRADE") -> pd.DataFrame:
    """
    从 zip 内容中解析并过滤出 CBOT Corn 数据。
    自动适配两种日期列名格式（横线 vs 下划线）。
    """
    z = zipfile.ZipFile(io.BytesIO(zip_bytes))
    txt_files = [n for n in z.namelist() if n.lower().endswith(".txt")]
    if not txt_files:
        raise ValueError(f"No .txt file found in zip: {z.namelist()}")

    frames = []
    for fname in txt_files:
        with z.open(fname) as f:
            try:
                df = pd.read_csv(f, low_memory=False, header=0)
                # 标准化列名
                rename_map = {
                    "Report_Date_as_YYYY_MM_DD": "report_date",
                    "Report_Date_as_YYYY-MM-DD": "report_date",
                }
                df.rename(columns=rename_map, inplace=True)
                # 过滤 CBOT Corn
                corn = df[
                    df["Market_and_Exchange_Names"].astype(str).str.contains(
                        market_filter, na=False, regex=False
                    )
                ]
                frames.append(corn)
            except Exception:
                pass

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def fetch_cftc_corn_cot(
    start_year: int,
    end_year: int,
) -> pd.DataFrame:
    """
    从 CFTC 官方历史压缩文件批量下载 Disaggregated Corn COT 数据。

    Parameters
    ----------
    start_year : 起始年份（含）
    end_year   : 结束年份（含）

    Returns
    -------
    DataFrame with raw CFTC fields
    """
    all_corn = []

    for year in range(start_year, end_year + 1):
        print(f"  Fetching {year} ...", end=" ", flush=True)
        try:
            if year <= 2016:
                url = f"{_CFTC_BASE}/{_HIST_FILE}"
            else:
                url = f"{_CFTC_BASE}/{_YEAR_FILE.format(year=year)}"

            zip_bytes = _download_zip(url)
            corn_df = _read_corn_from_zip(zip_bytes)

            if not corn_df.empty:
                # 只取当年数据（某些合并文件包含多年）
                corn_df = corn_df[
                    corn_df["report_date"].astype(str).str.startswith(str(year))
                ]

            print(f"got {len(corn_df)} rows")
            all_corn.append(corn_df)
        except Exception as e:
            print(f"FAILED: {e}")

    if not all_corn:
        return pd.DataFrame()

    result = pd.concat(all_corn, ignore_index=True)
    # 去重（合并文件首尾可能有重叠）
    result.drop_duplicates(
        subset=["report_date", "CFTC_Contract_Market_Code"],
        keep="last",
        inplace=True,
    )
    result.sort_values("report_date", inplace=True)
    result.reset_index(drop=True, inplace=True)
    return result


def fetch_and_save_cftc_corn(
    start_year: int,
    end_year: int,
    csv_path: str,
    incremental: bool = False,
) -> pd.DataFrame:
    """
    获取数据并存入 CSV。

    Parameters
    ----------
    start_year  : 起始年份（含）
    end_year    : 结束年份（含）
    csv_path    : CSV 文件路径
    incremental : True=增量模式（只拉取本地 CSV 最新日期之后的数据）
                  False=全量模式（覆盖写入）

    Returns
    -------
    DataFrame with raw CFTC data
    """
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    if incremental and csv_path.exists():
        existing = pd.read_csv(csv_path, low_memory=False)
        if (
            "Report_Date_as_YYYY_MM_DD" in existing.columns
            and len(existing) > 0
        ):
            latest = pd.to_datetime(
                existing["Report_Date_as_YYYY_MM_DD"], errors="coerce"
            ).max()
            start_year = max(start_year, latest.year + 1)
            print(f"增量更新: 本地最新日期={latest.date()}，从 {start_year} 年开始拉取")
            if start_year > end_year:
                print("已是最新，无需更新。")
                return existing
        else:
            existing = pd.DataFrame()

    print(f"正在从 CFTC 官方文件拉取 {start_year}-{end_year} 年 Corn Disaggregated COT 数据 ...")
    df_raw = fetch_cftc_corn_cot(start_year, end_year)

    if df_raw.empty:
        print("API 返回空数据。")
        return pd.DataFrame()

    if incremental and csv_path.exists():
        combined = pd.concat([existing, df_raw], ignore_index=True)
        combined.drop_duplicates(
            subset=["report_date", "CFTC_Contract_Market_Code"],
            keep="last",
            inplace=True,
        )
        combined.sort_values("report_date", inplace=True)
        combined.reset_index(drop=True, inplace=True)
        combined.to_csv(csv_path, index=False)
        print(f"增量合并完成，共 {len(combined)} 行，已保存到 {csv_path}")
        return combined
    else:
        df_raw.to_csv(csv_path, index=False)
        print(f"全量写入完成，共 {len(df_raw)} 行，已保存到 {csv_path}")
        return df_raw


# ---------------------------------------------------------------------------
# 数据清洗
# ---------------------------------------------------------------------------

def clean_cftc_raw(df: pd.DataFrame) -> pd.DataFrame:
    """
    清洗原始 CFTC 数据，提取关键字段，转换为数值类型。
    """
    # 保留并统一列名
    rename_map = {
        # 两种日期列名格式
        "Report_Date_as_YYYY_MM_DD": "report_date",
        "Report_Date_as_YYYY-MM-DD": "report_date",
        "Open_Interest_All": "open_interest",
        "M_Money_Positions_Long_All": "fund_long",
        "M_Money_Positions_Short_All": "fund_short",
        "M_Money_Positions_Spread_All": "fund_spread",
        "Prod_Merc_Positions_Long_All": "producer_long",
        "Prod_Merc_Positions_Short_All": "producer_short",
        "Swap_Positions_Long_All": "swap_long",
        "Swap__Positions_Short_All": "swap_short",
        "Other_Rept_Positions_Long_All": "other_long",
        "Other_Rept_Positions_Short_All": "other_short",
        "Traders_M_Money_Long_All": "fund_traders_long",
        "Traders_M_Money_Short_All": "fund_traders_short",
    }

    keep_cols = ["report_date", "open_interest",
                 "fund_long", "fund_short", "fund_spread",
                 "producer_long", "producer_short",
                 "swap_long", "swap_short",
                 "other_long", "other_short",
                 "fund_traders_long", "fund_traders_short"]

    df = df.rename(columns=rename_map)
    present = [c for c in keep_cols if c in df.columns]
    df = df[present].copy()

    num_cols = [c for c in df.columns if c != "report_date"]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    # 标准化日期列（适配下划线和非下划线格式）
    for col in df.columns:
        if "report_date" in col.lower():
            df.rename(columns={col: "report_date"}, inplace=True)
            break
    df["report_date"] = pd.to_datetime(
        df["report_date"], errors="coerce"
    ).dt.tz_localize(None)
    df = df.dropna(subset=["report_date"])
    df.sort_values("report_date", inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df


def compute_cftc_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    基于清洗后的数据计算衍生特征。

    新增列:
        fund_net            = fund_long - fund_short  (绝对手数)
        fund_net_pct        = fund_net / open_interest  (净持仓占比，核心信号)
        fund_concentration  = (fund_long + fund_short) / open_interest
        producer_net_pct    = (producer_long - producer_short) / open_interest
        swap_net_pct        = (swap_long - swap_short) / open_interest
        fund_traders_count  = fund_traders_long + fund_traders_short
    """
    oi = df["open_interest"].replace(0, np.nan)
    df["fund_net"] = df["fund_long"] - df["fund_short"]
    df["fund_net_pct"] = df["fund_net"] / oi
    df["fund_concentration"] = (df["fund_long"] + df["fund_short"]) / oi
    df["producer_net_pct"] = (df["producer_long"] - df["producer_short"]) / oi
    df["swap_net_pct"] = (df["swap_long"] - df["swap_short"]) / oi
    df["fund_traders_count"] = (
        df["fund_traders_long"].fillna(0) + df["fund_traders_short"].fillna(0)
    )
    return df


# ---------------------------------------------------------------------------
# 加载 & 周→日 填充
# ---------------------------------------------------------------------------

def load_and_resample_cftc(
    csv_path: str,
    price_dates: pd.Series,
) -> pd.DataFrame:
    """
    将周频 CFTC 数据前向填充到每个交易日。

    CFTC 报告每周二收盘数据，周五发布。
    对于每个 DCE 交易日，使用最近一次（≤当天）的 CFTC 报告。

    Parameters
    ----------
    csv_path    : 本地 CSV 路径
    price_dates : DCE 交易日 Series（pd.Timestamp 或 date-like str）

    Returns
    -------
    DataFrame，index 为 DCE 交易日，columns 包含所有 CFTC 特征
    """
    raw = pd.read_csv(csv_path, low_memory=False)
    clean = clean_cftc_raw(raw)
    feat = compute_cftc_features(clean)

    trading = pd.DataFrame({"date": price_dates})
    trading["date"] = pd.to_datetime(trading["date"])

    # 笛卡尔积：每个交易日 × 每条 CFTC 记录
    trading["_key"] = 1
    cftc_sub = feat[["report_date"]].copy()
    cftc_sub["_key"] = 1

    merged = trading.merge(cftc_sub, on="_key")
    # 保留 CFTC 报告日期 <= 交易日 的记录
    merged = merged[merged["report_date"] <= merged["date"]].copy()

    # 每个交易日取最近一次报告
    idx = merged.groupby("date")["report_date"].idxmax()
    daily = merged.loc[idx, ["date", "report_date"]].merge(
        feat, on="report_date", how="left"
    )
    daily.set_index("date", inplace=True)
    daily.drop(columns=["_key", "report_date"], errors="ignore", inplace=True)
    daily.sort_index(inplace=True)

    return daily


# ---------------------------------------------------------------------------
# 便捷入口
# ---------------------------------------------------------------------------

def ensure_cftc_data(
    csv_path: str,
    start_year: int = 2006,
    end_year: int = None,
) -> Path:
    """
    确保本地 CSV 存在，不存在则自动拉取全量数据。
    """
    if end_year is None:
        end_year = datetime.now().year
    csv_path = Path(csv_path)
    if not csv_path.exists():
        print(f"本地 CFTC 数据不存在，正在拉取 {start_year}-{end_year} 年数据 ...")
        fetch_and_save_cftc_corn(start_year, end_year, str(csv_path), incremental=False)
    else:
        print(f"本地 CFTC 数据已存在: {csv_path}")
    return csv_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="CFTC Corn COT Data Fetcher")
    parser.add_argument("--start", type=int, default=2006, help="起始年份")
    parser.add_argument("--end", type=int, default=None, help="结束年份（默认今年）")
    parser.add_argument("--out", type=str, default="data/cftc_corn_disaggregated.csv", help="输出路径")
    parser.add_argument("--incremental", action="store_true", help="增量更新模式")
    args = parser.parse_args()

    end_year = args.end or datetime.now().year
    df = fetch_and_save_cftc_corn(
        args.start, end_year, args.out, incremental=args.incremental
    )
    if not df.empty:
        cleaned = clean_cftc_raw(df)
        feat = compute_cftc_features(cleaned)
        print(f"\n数据概览（最近5条）:\n{feat.tail(5).to_string()}")
