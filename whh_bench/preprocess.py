"""
whh_bench/preprocess.py

Loads the raw mcPHASES CSVs, joins them on (id, day_in_study), and applies
data-quality fixes discovered during benchmark development.

Known data-quality fix applied here:
    resting_heart_rate.csv contains failed-device-reading rows encoded as
    value == 0 (21% of 2022-round rows, 6% of 2024-round rows). These are
    not physiological signal — they are missing data — and are dropped
    before aggregation. Skipping this step silently deflates the mean and
    inflates the variance of resting_hr, which was mistaken for genuine
    cross-round measurement drift before this was found.
"""
import pandas as pd

MIN_PLAUSIBLE_RESTING_HR = 30  # bpm; values below this are failed readings, not real data


def load_hormones(data_dir: str) -> pd.DataFrame:
    """Ground-truth phase label + raw hormone assay values (LH, estrogen, PdG)."""
    df = pd.read_csv(f"{data_dir}/hormones_and_selfreport.csv", low_memory=False)
    cols = ["id", "study_interval", "day_in_study", "phase", "lh", "estrogen", "pdg"]
    return df[cols].dropna(subset=["phase"])


def load_resting_hr(data_dir: str) -> pd.DataFrame:
    raw = pd.read_csv(f"{data_dir}/resting_heart_rate.csv", low_memory=False)
    raw = raw[raw["value"] >= MIN_PLAUSIBLE_RESTING_HR]  # drop failed-reading zeros
    return (
        raw.groupby(["id", "day_in_study"], as_index=False)["value"]
        .mean()
        .rename(columns={"value": "resting_hr"})
    )


def load_temperature(data_dir: str) -> pd.DataFrame:
    df = pd.read_csv(f"{data_dir}/computed_temperature.csv", low_memory=False)
    df = df.rename(columns={"sleep_start_day_in_study": "day_in_study"})
    return df.groupby(["id", "day_in_study"], as_index=False)["nightly_temperature"].mean()


def load_hrv(data_dir: str) -> pd.DataFrame:
    """Daily mean RMSSD plus intra-day variability (rmssd_std) — the latter
    is computed from the real 5-minute-interval readings, not discarded
    the way a daily-mean-only pipeline would."""
    raw = pd.read_csv(f"{data_dir}/heart_rate_variability_details.csv", low_memory=False)
    return raw.groupby(["id", "day_in_study"], as_index=False).agg(
        rmssd=("rmssd", "mean"), rmssd_std=("rmssd", "std")
    )


def load_respiratory(data_dir: str) -> pd.DataFrame:
    df = pd.read_csv(f"{data_dir}/respiratory_rate_summary.csv", low_memory=False)
    df = df[["id", "day_in_study", "full_sleep_breathing_rate"]]
    return df.groupby(["id", "day_in_study"], as_index=False)["full_sleep_breathing_rate"].mean()


def load_activity(data_dir: str) -> pd.DataFrame:
    df = pd.read_csv(f"{data_dir}/active_minutes.csv", low_memory=False)
    df["active_total"] = df["lightly"] + df["moderately"] + df["very"]
    return df.groupby(["id", "day_in_study"], as_index=False)["active_total"].sum()


def load_sleep_score(data_dir: str) -> pd.DataFrame:
    df = pd.read_csv(f"{data_dir}/sleep_score.csv", low_memory=False)
    df = df[["id", "day_in_study", "overall_score"]]
    return (
        df.groupby(["id", "day_in_study"], as_index=False)["overall_score"]
        .mean()
        .rename(columns={"overall_score": "sleep_score"})
    )


def build_joined_dataset(data_dir: str) -> pd.DataFrame:
    """The single entry point: returns one row per (id, day_in_study) with
    the phase label, raw hormones, and all 6 wearable feature columns."""
    m = load_hormones(data_dir)
    for loader in (load_resting_hr, load_temperature, load_hrv):
        m = m.merge(loader(data_dir), on=["id", "day_in_study"], how="inner")
    for loader in (load_respiratory, load_activity, load_sleep_score):
        m = m.merge(loader(data_dir), on=["id", "day_in_study"], how="left")
    return m.sort_values(["id", "study_interval", "day_in_study"]).reset_index(drop=True)


RAW_FEATURES = [
    "resting_hr",
    "nightly_temperature",
    "rmssd",
    "full_sleep_breathing_rate",
    "active_total",
    "sleep_score",
]
