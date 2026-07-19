import numpy as np

from whh_bench.metrics import summarize_cv


def test_summarize_cv_basic_stats():
    fold_accs = np.array([0.6, 0.7, 0.8, 0.7, 0.6])
    summary = summarize_cv(fold_accs, baseline=0.5)

    assert summary["n_folds"] == 5
    assert summary["mean_accuracy"] == fold_accs.mean()
    assert summary["sd_accuracy"] == fold_accs.std()
    assert summary["majority_baseline"] == 0.5

    half_width = 1.96 * fold_accs.std() / np.sqrt(5)
    assert summary["ci_95_lower"] == fold_accs.mean() - half_width
    assert summary["ci_95_upper"] == fold_accs.mean() + half_width


def test_summarize_cv_beats_baseline_when_ci_lower_exceeds_it():
    fold_accs = np.array([0.9, 0.9, 0.9, 0.9, 0.9])
    summary = summarize_cv(fold_accs, baseline=0.5)
    assert summary["beats_baseline_significantly"] is True


def test_summarize_cv_does_not_beat_baseline_when_noisy():
    fold_accs = np.array([0.9, 0.1, 0.9, 0.1, 0.5])
    summary = summarize_cv(fold_accs, baseline=0.5)
    assert summary["beats_baseline_significantly"] is False


def test_summarize_cv_returns_plain_python_types():
    fold_accs = np.array([0.6, 0.7])
    summary = summarize_cv(fold_accs, baseline=0.5)
    assert isinstance(summary["mean_accuracy"], float)
    assert isinstance(summary["n_folds"], int)
    assert isinstance(summary["beats_baseline_significantly"], bool)
