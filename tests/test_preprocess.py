import pandas as pd

from whh_bench.preprocess import (
    RAW_FEATURES,
    build_joined_dataset,
    load_activity,
    load_hormones,
    load_resting_hr,
)


def _write_csv(path, df):
    df.to_csv(path, index=False)


def test_load_resting_hr_drops_failed_zero_readings(tmp_path):
    _write_csv(
        tmp_path / "resting_heart_rate.csv",
        pd.DataFrame(
            {
                "id": ["p1", "p1", "p1"],
                "day_in_study": [1, 1, 2],
                "value": [60, 0, 65],  # the 0 is a failed device reading
            }
        ),
    )
    out = load_resting_hr(str(tmp_path))
    day1 = out[(out["id"] == "p1") & (out["day_in_study"] == 1)]
    assert len(day1) == 1
    assert day1["resting_hr"].iloc[0] == 60  # the 0-reading must not pull the mean down


def test_load_hormones_drops_rows_without_a_phase_label(tmp_path):
    _write_csv(
        tmp_path / "hormones_and_selfreport.csv",
        pd.DataFrame(
            {
                "id": ["p1", "p1"],
                "study_interval": [1, 1],
                "day_in_study": [1, 2],
                "phase": ["Fertility", None],
                "lh": [1.0, 1.0],
                "estrogen": [1.0, 1.0],
                "pdg": [1.0, 1.0],
            }
        ),
    )
    out = load_hormones(str(tmp_path))
    assert len(out) == 1
    assert out["phase"].iloc[0] == "Fertility"


def test_load_activity_sums_active_minute_categories(tmp_path):
    _write_csv(
        tmp_path / "active_minutes.csv",
        pd.DataFrame(
            {
                "id": ["p1"],
                "day_in_study": [1],
                "lightly": [10],
                "moderately": [5],
                "very": [2],
            }
        ),
    )
    out = load_activity(str(tmp_path))
    assert out["active_total"].iloc[0] == 17


def _write_full_fixture(tmp_path):
    _write_csv(
        tmp_path / "hormones_and_selfreport.csv",
        pd.DataFrame(
            {
                "id": ["p1", "p1"],
                "study_interval": [1, 1],
                "day_in_study": [1, 2],
                "phase": ["Fertility", "Other"],
                "lh": [1.0, 1.0],
                "estrogen": [1.0, 1.0],
                "pdg": [1.0, 1.0],
            }
        ),
    )
    _write_csv(
        tmp_path / "resting_heart_rate.csv",
        pd.DataFrame({"id": ["p1", "p1"], "day_in_study": [1, 2], "value": [60, 62]}),
    )
    _write_csv(
        tmp_path / "computed_temperature.csv",
        pd.DataFrame(
            {
                "id": ["p1", "p1"],
                "sleep_start_day_in_study": [1, 2],
                "nightly_temperature": [97.0, 97.2],
            }
        ),
    )
    _write_csv(
        tmp_path / "heart_rate_variability_details.csv",
        pd.DataFrame({"id": ["p1", "p1"], "day_in_study": [1, 2], "rmssd": [40.0, 42.0]}),
    )
    _write_csv(
        tmp_path / "respiratory_rate_summary.csv",
        pd.DataFrame(
            {
                "id": ["p1", "p1"],
                "day_in_study": [1, 2],
                "full_sleep_breathing_rate": [15.0, 15.5],
            }
        ),
    )
    _write_csv(
        tmp_path / "active_minutes.csv",
        pd.DataFrame(
            {
                "id": ["p1", "p1"],
                "day_in_study": [1, 2],
                "lightly": [10, 12],
                "moderately": [5, 4],
                "very": [2, 3],
            }
        ),
    )
    _write_csv(
        tmp_path / "sleep_score.csv",
        pd.DataFrame({"id": ["p1", "p1"], "day_in_study": [1, 2], "overall_score": [80, 82]}),
    )


def test_build_joined_dataset_produces_one_row_per_id_and_day(tmp_path):
    _write_full_fixture(tmp_path)
    df = build_joined_dataset(str(tmp_path))

    assert len(df) == 2
    assert list(df["day_in_study"]) == [1, 2]
    for feature in RAW_FEATURES:
        assert feature in df.columns


def test_build_joined_dataset_inner_joins_required_signals(tmp_path):
    _write_full_fixture(tmp_path)
    # Drop day 2's resting-hr reading entirely -> day 2 should disappear,
    # since resting_hr/temperature/hrv are inner-joined (required signals).
    _write_csv(
        tmp_path / "resting_heart_rate.csv",
        pd.DataFrame({"id": ["p1"], "day_in_study": [1], "value": [60]}),
    )
    df = build_joined_dataset(str(tmp_path))
    assert len(df) == 1
    assert df["day_in_study"].iloc[0] == 1
