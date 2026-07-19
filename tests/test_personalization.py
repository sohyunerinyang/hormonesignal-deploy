import numpy as np
import pandas as pd

from whh_bench.personalization import (
    WINDOW_DAYS,
    MIN_PERIODS,
    add_causal_personalized_features,
    personalized_feature_names,
)


def test_personalized_feature_names_appends_suffix():
    assert personalized_feature_names(["resting_hr", "rmssd"]) == [
        "resting_hr_causal_z",
        "rmssd_causal_z",
    ]


def test_add_causal_personalized_features_adds_one_column_per_feature():
    df = pd.DataFrame(
        {
            "id": ["p1"] * (MIN_PERIODS + 2),
            "day_in_study": range(MIN_PERIODS + 2),
            "resting_hr": np.arange(MIN_PERIODS + 2, dtype=float),
        }
    )
    out = add_causal_personalized_features(df, ["resting_hr"])
    assert "resting_hr_causal_z" in out.columns
    assert len(out) == len(df)


def test_early_rows_are_nan_until_min_periods_reached():
    df = pd.DataFrame(
        {
            "id": ["p1"] * (MIN_PERIODS + 2),
            "day_in_study": range(MIN_PERIODS + 2),
            "resting_hr": np.arange(MIN_PERIODS + 2, dtype=float),
        }
    )
    out = add_causal_personalized_features(df, ["resting_hr"])
    # shift(1) + min_periods means the first MIN_PERIODS rows have no valid baseline
    assert out["resting_hr_causal_z"].iloc[:MIN_PERIODS].isna().all()
    assert out["resting_hr_causal_z"].iloc[MIN_PERIODS:].notna().all()


def test_causal_z_does_not_use_the_current_or_future_day():
    # If personalization looked ahead, changing a later day's value would
    # change an earlier day's z-score. It must not.
    n = MIN_PERIODS + 5
    base = pd.DataFrame(
        {
            "id": ["p1"] * n,
            "day_in_study": range(n),
            "resting_hr": np.linspace(60, 70, n),
        }
    )
    mutated = base.copy()
    mutated.loc[n - 1, "resting_hr"] = 999.0  # blow up the last day only

    out_base = add_causal_personalized_features(base, ["resting_hr"])
    out_mutated = add_causal_personalized_features(mutated, ["resting_hr"])

    pd.testing.assert_series_equal(
        out_base["resting_hr_causal_z"].iloc[:-1],
        out_mutated["resting_hr_causal_z"].iloc[:-1],
    )


def test_personalization_is_per_participant():
    n_per_id = MIN_PERIODS + 2
    df = pd.DataFrame(
        {
            "id": ["p1"] * n_per_id + ["p2"] * n_per_id,
            "day_in_study": list(range(n_per_id)) * 2,
            "resting_hr": list(np.arange(n_per_id, dtype=float))
            + list(np.arange(n_per_id, dtype=float) + 100),
        }
    )
    out = add_causal_personalized_features(df, ["resting_hr"])
    p1_z = out[out["id"] == "p1"]["resting_hr_causal_z"].iloc[MIN_PERIODS:]
    p2_z = out[out["id"] == "p2"]["resting_hr_causal_z"].iloc[MIN_PERIODS:]
    # Same relative trend per participant -> same z-scores despite the offset
    np.testing.assert_allclose(p1_z.values, p2_z.values, rtol=1e-6)
