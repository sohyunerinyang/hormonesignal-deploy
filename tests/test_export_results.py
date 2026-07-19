import numpy as np
import pandas as pd

from whh_bench.export_results import run_single_signal_cv


def _synthetic_df(n_groups, rows_per_group):
    groups = np.repeat(np.arange(n_groups), rows_per_group)
    rng = np.random.default_rng(0)
    n = n_groups * rows_per_group
    return pd.DataFrame(
        {
            "id": groups,
            "feature_causal_z": rng.normal(size=n),
            "phase": np.tile(["Fertility", "Other"], n // 2),
        }
    )


def test_run_single_signal_cv_returns_none_below_group_threshold():
    df = _synthetic_df(n_groups=4, rows_per_group=60)  # < 5 participants
    assert run_single_signal_cv(df, "feature_causal_z") is None


def test_run_single_signal_cv_returns_none_below_row_threshold():
    df = _synthetic_df(n_groups=10, rows_per_group=2)  # < 200 rows
    assert run_single_signal_cv(df, "feature_causal_z") is None


def test_run_single_signal_cv_returns_summary_for_enough_data():
    df = _synthetic_df(n_groups=10, rows_per_group=40)  # 5+ groups, 200+ rows
    result = run_single_signal_cv(df, "feature_causal_z")

    assert result is not None
    assert set(result.keys()) == {"mean", "sd", "n", "majority_baseline"}
    assert 0.0 <= result["mean"] <= 1.0
    assert result["n"] == len(df)


def test_run_single_signal_cv_drops_rows_missing_the_feature():
    df = _synthetic_df(n_groups=10, rows_per_group=40)
    df.loc[df.index[:50], "feature_causal_z"] = np.nan
    result = run_single_signal_cv(df, "feature_causal_z")
    assert result["n"] == len(df) - 50
