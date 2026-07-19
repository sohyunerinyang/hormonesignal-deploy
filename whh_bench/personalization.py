"""
whh_bench/personalization.py

Per-participant normalization, computed causally (only using days strictly
before the day being featurized). This replaces an earlier version that
z-scored each day against that participant's FULL history, including the
very day being predicted — a look-ahead leak. A model can't know a
person's future average at prediction time in a real deployment, so this
module can't either.

Cost of the fix, stated plainly: each participant's first WINDOW_DAYS days
have no valid rolling baseline yet and are dropped. This is a real, honest
trade-off, not a free upgrade.
"""
import pandas as pd

WINDOW_DAYS = 14
MIN_PERIODS = 5


def add_causal_personalized_features(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """For each feature, adds a `<feature>_causal_z` column: the day's raw
    value, standardized against that participant's own trailing rolling
    mean/std computed from PRIOR days only (via .shift(1))."""
    df = df.copy()
    for f in feature_cols:
        rolling_mean = df.groupby("id")[f].transform(
            lambda x: x.shift(1).rolling(WINDOW_DAYS, min_periods=MIN_PERIODS).mean()
        )
        rolling_std = df.groupby("id")[f].transform(
            lambda x: x.shift(1).rolling(WINDOW_DAYS, min_periods=MIN_PERIODS).std()
        )
        df[f + "_causal_z"] = (df[f] - rolling_mean) / (rolling_std + 1e-6)
    return df


def personalized_feature_names(feature_cols: list[str]) -> list[str]:
    return [f + "_causal_z" for f in feature_cols]
