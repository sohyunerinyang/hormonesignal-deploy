"""
whh_bench/label_confidence.py

The phase label in hormones_and_selfreport.csv is Mira's proprietary
algorithm output, not a directly confirmed ovulation event. This module
re-derives the "Fertility" call independently from raw LH values using a
transparent, published-style rule (LH surge = LH > 1.4x a rolling local
baseline), then flags agreement/disagreement with Mira's label.

This does NOT replace Mira's label. It adds a `label_confidence` column
so downstream users can choose to exclude or down-weight low-confidence
days, following the same evidence-strength-weighting principle used
elsewhere in this benchmark's schema.
"""
import numpy as np
import pandas as pd

LH_SURGE_MULTIPLIER = 1.4
BASELINE_WINDOW_DAYS = 5


def add_label_confidence(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["lh_baseline"] = df.groupby("id")["lh"].transform(
        lambda x: x.rolling(BASELINE_WINDOW_DAYS, min_periods=3, center=True).median()
    )
    df["our_fertility_call"] = (df["lh"] > LH_SURGE_MULTIPLIER * df["lh_baseline"]).astype(int)
    mira_fertility = (df["phase"] == "Fertility").astype(int)

    has_data = df["lh_baseline"].notna() & df["lh"].notna()
    agree = df["our_fertility_call"] == mira_fertility

    df["label_confidence"] = np.select(
        [~has_data, agree], ["unknown", "high"], default="low"
    )
    return df


def confidence_report(df: pd.DataFrame) -> dict:
    valid = df["label_confidence"] != "unknown"
    return {
        "n_total": len(df),
        "n_comparable": int(valid.sum()),
        "agreement_rate": float((df.loc[valid, "label_confidence"] == "high").mean()),
        "counts": df["label_confidence"].value_counts().to_dict(),
    }
