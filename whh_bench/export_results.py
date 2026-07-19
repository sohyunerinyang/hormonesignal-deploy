"""
whh_bench/export_results.py

Runs the full pipeline once against real, locally-held mcPHASES data and
writes a single results.json — this is the file the frontend actually
fetches. Raw data never leaves this machine; only these derived numbers do.

Run with: python -m whh_bench.export_results --data-dir /path/to/csvs --out results/latest_run.json
"""
import argparse
import json
from datetime import datetime, timezone

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

from whh_bench.preprocess import build_joined_dataset, RAW_FEATURES
from whh_bench.personalization import add_causal_personalized_features, personalized_feature_names
from whh_bench.label_confidence import add_label_confidence, confidence_report
from whh_bench.splits import cross_validate, make_group_kfold
from whh_bench.metrics import summarize_cv


def make_model(seed: int):
    return RandomForestClassifier(n_estimators=300, random_state=seed, min_samples_leaf=5)


def run_single_signal_cv(df, feature_col, label_col="phase", group_col="id"):
    sub = df.dropna(subset=[feature_col, label_col])
    if sub[group_col].nunique() < 5 or len(sub) < 200:
        return None
    X = sub[[feature_col]].values
    y = sub[label_col].values
    groups = sub[group_col].values
    accs = cross_validate(X, y, groups, make_model, n_splits=5)
    return {
        "mean": float(accs.mean()),
        "sd": float(accs.std()),
        "n": int(len(sub)),
        "majority_baseline": float(sub[label_col].value_counts(normalize=True).max()),
    }


def main(data_dir: str, out_path: str):
    df = build_joined_dataset(data_dir)
    df = add_label_confidence(df)
    label_conf = confidence_report(df)

    df = add_causal_personalized_features(df, RAW_FEATURES)
    combined_feats = personalized_feature_names(RAW_FEATURES) + ["rmssd_std"]
    df_full = df.dropna(subset=combined_feats + ["phase"])

    X = df_full[combined_feats].values
    y = df_full["phase"].values
    groups = df_full["id"].values
    baseline = df_full["phase"].value_counts(normalize=True).max()

    fold_accs = cross_validate(X, y, groups, make_model, n_splits=5)
    summary = summarize_cv(fold_accs, baseline)

    labels = sorted(np.unique(y))
    agg_cm = np.zeros((len(labels), len(labels)), dtype=int)
    gkf = make_group_kfold(5)
    for tr, te in gkf.split(X, y, groups):
        m_ = make_model(42)
        m_.fit(X[tr], y[tr])
        pred = m_.predict(X[te])
        agg_cm += confusion_matrix(y[te], pred, labels=labels)

    single_signal = {}
    for f in RAW_FEATURES:
        result = run_single_signal_cv(df, f + "_causal_z")
        if result:
            single_signal[f] = result

    results = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "data_snapshot_version": "2026-07-hackathon",
        "phase_definition_version": "Mira-algorithm+PACTS-2025",
        "n_participants": int(df["id"].nunique()),
        "n_rows_total": int(len(df)),
        "label_confidence": label_conf,
        "model": {
            "type": "RandomForestClassifier",
            "n_estimators": 300,
            "cv": "GroupKFold(5), grouped by participant id",
            "features": combined_feats,
        },
        "cv_summary": summary,
        "confusion_matrix": {"labels": labels, "matrix": agg_cm.tolist()},
        "single_signal_cv": single_signal,
    }

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Wrote {out_path}")
    print(json.dumps(results["cv_summary"], indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out", default="results/latest_run.json")
    args = parser.parse_args()
    main(args.data_dir, args.out)
