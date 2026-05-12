"""
Data loaders for corn futures price data and fundamental data.
Supports regime-aware loading: fundamental data only available from 2020+.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional


# Regime boundaries
REGIME_2020 = pd.Timestamp("2020-01-01")
REGIME_2014 = pd.Timestamp("2014-01-01")


def load_price_data(
    price_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Load corn futures price data (DCE, 2004-2026).

    Args:
        price_path: Path to corn_en_no_oi.txt. If None, uses default path.

    Returns:
        DataFrame with columns: date, open, high, low, close, settle, volume
        Sorted by date ascending.
    """
    if price_path is None:
        base = Path(__file__).parent.parent
        price_path = base / "1-量价" / "corn_en_no_oi.txt"

    df = pd.read_csv(
        price_path,
        sep=",",
        header=0,  # has header row
        dtype={"open": float, "high": float, "low": float,
               "close": float, "settle": float, "volume": float},
    )

    # Parse date (may have time suffix like "15:00" in recent rows)
    df["date"] = pd.to_datetime(df["date"].astype(str).str[:10])
    df = df.sort_values("date").reset_index(drop=True)

    # Basic sanity checks (use .ge with skipna-safe comparison)
    assert (df["high"] >= df["low"]).all(), "high < low found"
    assert (df["high"] >= df["open"]).all(), "open outside high"
    assert (df["high"] >= df["close"]).all(), "close outside high"
    assert (df["low"] <= df["open"]).all(), "open outside low"
    assert (df["low"] <= df["close"]).all(), "close outside low"

    df["return_1d"] = df["close"].pct_change()

    return df


def load_fundamental_data(
    fundamental_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Load corn fundamental data (2014-2026).

    The Excel has a two-row header: row 1 = category, row 2 = column names.
    This loader flattens both rows into single column names.

    Args:
        fundamental_path: Path to 国内玉米基本面日度数据.xlsx. If None, uses default.

    Returns:
        DataFrame with date index and all fundamental columns.
        Sorted by date ascending.
    """
    if fundamental_path is None:
        base = Path(__file__).parent.parent
        fundamental_path = base / "2-基本面" / "国内玉米基本面日度数据.xlsx"

    # Read with header on row 1 (0-indexed)
    df = pd.read_excel(fundamental_path, sheet_name=0, header=[0, 1])
    df.columns = [f"{a}|{b}" if pd.notna(b) else a for a, b in df.columns]

    # Flatten the first column (date)
    date_col = df.columns[0]
    df = df.rename(columns={date_col: "date"})
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # Convert all non-date columns to numeric
    for col in df.columns:
        if col != "date":
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def merge_price_fundamental(
    price_df: pd.DataFrame,
    fundamental_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge price and fundamental data on date.

    Args:
        price_df: Price data from load_price_data
        fundamental_df: Fundamental data from load_fundamental_data

    Returns:
        Merged DataFrame with all price columns and all fundamental columns.
        Left join so price data is never truncated.
    """
    merged = price_df.merge(fundamental_df, on="date", how="left")
    return merged


def get_data_summary(df: pd.DataFrame, fundamental_df: Optional[pd.DataFrame] = None) -> dict:
    """
    Generate a data quality summary.

    Args:
        df: Merged or price DataFrame
        fundamental_df: Optional fundamental DataFrame for coverage stats

    Returns:
        Summary dict with date ranges, coverage stats, etc.
    """
    summary = {
        "price_date_range": (df["date"].min(), df["date"].max()),
        "price_total_rows": len(df),
        "price_latest": df.iloc[-1][["date", "close", "volume"]].to_dict(),
        "fundamental_available": fundamental_df is not None,
    }

    if fundamental_df is not None:
        fund_cols = [c for c in fundamental_df.columns if c != "date"]
        summary["fundamental_total_cols"] = len(fund_cols)
        summary["fundamental_date_range"] = (
            fundamental_df["date"].min(),
            fundamental_df["date"].max(),
        )

        # Coverage by period
        pre_2020 = fundamental_df[fundamental_df["date"] < REGIME_2020]
        post_2020 = fundamental_df[fundamental_df["date"] >= REGIME_2020]

        coverage = {}
        for col in fund_cols:
            pre = pre_2020[col].notna().mean()
            post = post_2020[col].notna().mean()
            coverage[col] = {"pre_2020": pre, "post_2020": post}

        summary["fundamental_coverage"] = coverage

    return summary


def load_all(
    price_path: Optional[str] = None,
    fundamental_path: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Convenience function to load and return both price and fundamental data.

    Returns:
        (price_df, fundamental_df)
    """
    price_df = load_price_data(price_path)
    fundamental_df = load_fundamental_data(fundamental_path)
    return price_df, fundamental_df


if __name__ == "__main__":
    price, fund = load_all()
    print(f"Price data: {len(price)} rows, {price['date'].min()} to {price['date'].max()}")
    print(f"Fundamental data: {len(fund)} rows, {fund['date'].min()} to {fund['date'].max()}")
    summary = get_data_summary(price, fund)
    print(f"\nPrice latest: {summary['price_latest']}")
    print(f"\nTop fundamental columns by coverage:")
    cov = summary["fundamental_coverage"]
    sorted_cov = sorted(cov.items(), key=lambda x: x[1]["post_2020"], reverse=True)
    for col, stats in sorted_cov[:10]:
        print(f"  {col}: pre_2020={stats['pre_2020']:.1%}, post_2020={stats['post_2020']:.1%}")
