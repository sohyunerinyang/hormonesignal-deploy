import numpy as np
import pandas as pd

from whh_bench.label_confidence import add_label_confidence, confidence_report


def _make_df(lh_values, phases):
    n = len(lh_values)
    return pd.DataFrame(
        {
            "id": ["p1"] * n,
            "day_in_study": range(n),
            "lh": lh_values,
            "phase": phases,
        }
    )


def test_add_label_confidence_flags_agreement_on_clear_surge():
    # Flat baseline of 1.0, then a clear surge (> 1.4x baseline) matching
    # Mira's "Fertility" label -> should be "high" confidence.
    lh = [1.0, 1.0, 1.0, 1.0, 1.0, 3.0, 1.0, 1.0, 1.0]
    phases = ["Other"] * 5 + ["Fertility"] + ["Other"] * 3
    df = add_label_confidence(_make_df(lh, phases))

    surge_idx = 5
    assert df.loc[surge_idx, "our_fertility_call"] == 1
    assert df.loc[surge_idx, "label_confidence"] == "high"


def test_add_label_confidence_flags_disagreement():
    # Same clear surge, but Mira's label says "Other" -> disagreement -> "low".
    lh = [1.0, 1.0, 1.0, 1.0, 1.0, 3.0, 1.0, 1.0, 1.0]
    phases = ["Other"] * 9
    df = add_label_confidence(_make_df(lh, phases))

    surge_idx = 5
    assert df.loc[surge_idx, "our_fertility_call"] == 1
    assert df.loc[surge_idx, "label_confidence"] == "low"


def test_add_label_confidence_marks_unknown_when_no_baseline():
    # min_periods=3 for the rolling baseline -> the very first rows can't
    # have a baseline yet and must be "unknown", not "low"/"high".
    lh = [1.0, np.nan]
    phases = ["Other", "Other"]
    df = add_label_confidence(_make_df(lh, phases))
    assert df.loc[0, "label_confidence"] == "unknown"


def test_confidence_report_excludes_unknown_from_agreement_rate():
    lh = [1.0, np.nan, 1.0, 1.0, 1.0, 3.0, 1.0]
    phases = ["Other"] * 5 + ["Fertility", "Other"]
    df = add_label_confidence(_make_df(lh, phases))
    report = confidence_report(df)

    assert report["n_total"] == len(df)
    assert report["n_comparable"] == int((df["label_confidence"] != "unknown").sum())
    assert 0.0 <= report["agreement_rate"] <= 1.0
    assert set(report["counts"].keys()) <= {"high", "low", "unknown"}
