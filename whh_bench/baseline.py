"""
whh_bench/baseline.py

Run with:  python -m whh_bench.baseline --data-dir /path/to/mcphases/csvs

End-to-end: load real mcPHASES tables -> apply data-quality fixes ->
causal personalization -> independent label-confidence check ->
5-fold GroupKFold evaluation with mean/SD/CI, not a single split.
"""
import argparse
from sklearn.ensemble import RandomForestClassifier

from whh_bench.preprocess import build_joined_dataset, RAW_FEATURES
from whh_bench.personalization import add_causal_personalized_features, personalized_feature_names
from whh_bench.label_confidence import add_label_confidence, confidence_report
from whh_bench.splits import cross_validate
from whh_bench.metrics import summarize_cv, print_report


def make_model(seed: int):
    return RandomForestClassifier(n_estimators=300, random_state=seed, min_samples_leaf=5)


def main(data_dir: str):
    print("Loading and joining mcPHASES tables (resting-HR zero-reading fix applied)...")
    df = build_joined_dataset(data_dir)

    print("\nChecking phase-label confidence against an independent LH-surge rule...")
    df = add_label_confidence(df)
    report = confidence_report(df)
    print(f"  Agreement with Mira's label: {report['agreement_rate']*100:.1f}% "
          f"({report['n_comparable']} comparable days)")
    print(f"  Confidence breakdown: {report['counts']}")

    print("\nApplying causal (look-ahead-safe) personalization...")
    df = add_causal_personalized_features(df, RAW_FEATURES)
    feature_cols = personalized_feature_names(RAW_FEATURES) + ["rmssd_std"]

    df = df.dropna(subset=feature_cols + ["phase"])
    print(f"  Usable rows after rolling-window warmup: {len(df)} "
          f"({df['id'].nunique()} participants)")

    X = df[feature_cols].values
    y = df["phase"].values
    groups = df["id"].values
    baseline = df["phase"].value_counts(normalize=True).max()

    print("\nRunning 5-fold GroupKFold cross-validation (participant-held-out)...")
    fold_accs = cross_validate(X, y, groups, make_model, n_splits=5)
    summary = summarize_cv(fold_accs, baseline)
    print()
    print_report(summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True, help="Directory containing the raw mcPHASES CSVs")
    args = parser.parse_args()
    main(args.data_dir)
