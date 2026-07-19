"""
whh_bench/splits.py

Person-level grouped k-fold split — no individual's days ever appear in
both the train and test portion of any fold. This replaced an earlier
single GroupShuffleSplit train/test split, which reported one accuracy
number with no notion of its own variance.
"""
import numpy as np
from sklearn.model_selection import GroupKFold


def make_group_kfold(n_splits: int = 5):
    return GroupKFold(n_splits=n_splits)


def assert_no_leakage(train_groups, test_groups):
    """Hard guarantee, not just a claim — called by tests/test_no_leakage.py."""
    overlap = set(train_groups) & set(test_groups)
    assert not overlap, f"Leakage: {len(overlap)} participant(s) in both train and test"


def cross_validate(X, y, groups, model_factory, n_splits: int = 5, seed: int = 42):
    """Runs GroupKFold, asserts no leakage on every fold, and returns the
    per-fold accuracy array (not just a single mean) so callers can report
    mean +/- SD and a confidence interval instead of one point estimate."""
    from sklearn.metrics import accuracy_score

    gkf = make_group_kfold(n_splits)
    fold_accuracies = []
    for train_idx, test_idx in gkf.split(X, y, groups):
        assert_no_leakage(groups[train_idx], groups[test_idx])
        model = model_factory(seed)
        model.fit(X[train_idx], y[train_idx])
        pred = model.predict(X[test_idx])
        fold_accuracies.append(accuracy_score(y[test_idx], pred))
    return np.array(fold_accuracies)
