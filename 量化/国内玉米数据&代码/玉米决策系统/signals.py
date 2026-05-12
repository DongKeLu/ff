"""
Financial logic rules interface.
This module is intentionally left empty — to be filled in by the user
with their own trading rules and experience-based heuristics.

The system will call these functions and merge their output with model predictions.
"""

import pandas as pd
from typing import Dict, List, Any, Optional
from dataclasses import dataclass


@dataclass
class LogicSignal:
    """Output from a financial logic rule."""
    name: str
    direction: str          # "看多" / "中性" / "看空"
    strength: float         # 0-1, how strong the signal is
    description: str       # Human-readable explanation
    triggered: bool = True  # Whether the condition was met


# ──────────────────────────────────────────────────────────────────────────────
# Rule interface — implement your own rules below
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_all_rules(df: pd.DataFrame, horizon: int) -> List[LogicSignal]:
    """
    Evaluate all financial logic rules on the given DataFrame.

    Args:
        df: Feature DataFrame with latest row at position -1
        horizon: Prediction horizon (1, 5, or 20)

    Returns:
        List of LogicSignal objects (one per rule that triggered)
    """
    signals = []

    # ── Add your rules here ──────────────────────────────────────────────────
    # Example format:
    #     if df["ma60_ratio"].iloc[-1] > 1.05:
    #         signals.append(LogicSignal(
    #             name="价格位于MA60上方",
    #             direction="看多",
    #             strength=0.6,
    #             description="当前价格高于MA60 5%以上，确认中长期上升趋势",
    #         ))
    #
    # ────────────────────────────────────────────────────────────────────────

    return signals


def get_rule_summary(signals: List[LogicSignal]) -> Dict[str, Any]:
    """
    Summarize a list of LogicSignals into a combined view.

    Returns:
        Dict with:
            - overall_direction: combined direction
            - overall_strength: 0-1 strength
            - triggered_rules: list of rule names
            - descriptions: list of human-readable descriptions
    """
    if not signals:
        return {
            "overall_direction": "中性",
            "overall_strength": 0.0,
            "triggered_rules": [],
            "descriptions": [],
        }

    direction_map = {"看多": 1, "中性": 0, "看空": -1}
    dir_scores = [direction_map.get(s.direction, 0) * s.strength for s in signals]
    total_strength = sum(s.strength for s in signals)
    weighted_dir = sum(dir_scores) / (total_strength + 1e-9)

    if weighted_dir > 0.3:
        overall_dir = "看多"
    elif weighted_dir < -0.3:
        overall_dir = "看空"
    else:
        overall_dir = "中性"

    return {
        "overall_direction": overall_dir,
        "overall_strength": round(min(total_strength / max(len(signals), 1), 1.0), 3),
        "triggered_rules": [s.name for s in signals],
        "descriptions": [s.description for s in signals],
    }


def detect_divergence(
    ensemble_direction: str,
    logic_direction: str,
) -> Optional[str]:
    """
    Check if model predictions diverge from financial logic rules.

    Returns:
        None if aligned, or a warning string describing the divergence.
    """
    if not logic_direction or logic_direction == "中性":
        return None

    if ensemble_direction == logic_direction:
        return None  # aligned

    return (
        f"背离预警：模型集成信号为「{ensemble_direction}」，"
        f"金融逻辑信号为「{logic_direction}」，方向不一致。"
        f"建议人工复核。"
    )


# ──────────────────────────────────────────────────────────────────────────────
# Backtest helper — for validating your rules on historical data
# ──────────────────────────────────────────────────────────────────────────────

def backtest_rules(
    df: pd.DataFrame,
    horizons: List[int] = [1, 5, 20],
    rules_fn=evaluate_all_rules,
) -> pd.DataFrame:
    """
    Backtest financial logic rules on historical data.

    Args:
        df: Feature DataFrame
        horizons: List of horizons to evaluate
        rules_fn: Function that returns LogicSignals

    Returns:
        DataFrame with columns: date, horizon, direction, strength, actual_return
    """
    records = []
    for i in range(120, len(df) - max(horizons)):
        window = df.iloc[:i+1]
        for h in horizons:
            if i + h >= len(df):
                continue
            actual_return = (df["close"].iloc[i+h] - df["close"].iloc[i]) / df["close"].iloc[i]
            signals = rules_fn(window, h)
            summary = get_rule_summary(signals)
            records.append({
                "date": df["date"].iloc[i],
                "horizon": h,
                "logic_direction": summary["overall_direction"],
                "logic_strength": summary["overall_strength"],
                "actual_return": actual_return,
            })
    return pd.DataFrame(records)
