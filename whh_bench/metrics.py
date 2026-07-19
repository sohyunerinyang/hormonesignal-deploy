"""
whh_bench/metrics.py

Fixed evaluation protocol. Reports mean +/- SD and a 95% CI across folds
instead of a single accuracy number, per the statistical-rigor fix.
"""
import numpy as np


def summarize_cv(fold_accuracies: np.ndarray, baseline: float) -> dict:
    mean = fold_accuracies.mean()
    sd = fold_accuracies.std()
    n = len(fold_accuracies)
    ci_half_width = 1.96 * sd / np.sqrt(n)
    return {
        "mean_accuracy": float(mean),
        "sd_accuracy": float(sd),
        "n_folds": n,
        "ci_95_lower": float(mean - ci_half_width),
        "ci_95_upper": float(mean + ci_half_width),
        "majority_baseline": float(baseline),
        "beats_baseline_significantly": bool(mean - ci_half_width > baseline),
    }


def print_report(summary: dict):
    print(
        f"{summary['n_folds']}-fold accuracy: "
        f"{summary['mean_accuracy']*100:.1f}% \u00b1 {summary['sd_accuracy']*100:.2f}%"
    )
    print(f"95% CI: [{summary['ci_95_lower']*100:.1f}%, {summary['ci_95_upper']*100:.1f}%]")
    print(f"Majority baseline: {summary['majority_baseline']*100:.1f}%")
    print(f"Beats baseline significantly: {summary['beats_baseline_significantly']}")
