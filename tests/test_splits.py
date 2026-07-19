import numpy as np
import pytest
from sklearn.model_selection import GroupKFold

from whh_bench.splits import assert_no_leakage, cross_validate, make_group_kfold


def test_make_group_kfold_returns_group_kfold_with_requested_splits():
    kfold = make_group_kfold(n_splits=3)
    assert isinstance(kfold, GroupKFold)
    assert kfold.get_n_splits() == 3


def test_assert_no_leakage_passes_when_groups_disjoint():
    assert_no_leakage(train_groups=[1, 2, 3], test_groups=[4, 5])


def test_assert_no_leakage_raises_when_groups_overlap():
    with pytest.raises(AssertionError, match="Leakage"):
        assert_no_leakage(train_groups=[1, 2, 3], test_groups=[3, 4])


def _make_constant_model(seed):
    class ConstantModel:
        def fit(self, X, y):
            self.majority_ = np.bincount(y).argmax()
            return self

        def predict(self, X):
            return np.full(len(X), self.majority_)

    return ConstantModel()


def test_cross_validate_returns_one_accuracy_per_fold():
    rng = np.random.default_rng(0)
    n_groups = 10
    rows_per_group = 5
    groups = np.repeat(np.arange(n_groups), rows_per_group)
    X = rng.normal(size=(n_groups * rows_per_group, 2))
    y = np.tile([0, 1], n_groups * rows_per_group // 2)

    fold_accs = cross_validate(X, y, groups, _make_constant_model, n_splits=5)

    assert isinstance(fold_accs, np.ndarray)
    assert len(fold_accs) == 5
    assert ((fold_accs >= 0) & (fold_accs <= 1)).all()


def test_cross_validate_never_lets_a_group_span_train_and_test():
    n_groups = 10
    rows_per_group = 4
    groups = np.repeat(np.arange(n_groups), rows_per_group)
    X = np.arange(n_groups * rows_per_group).reshape(-1, 1)
    y = np.tile([0, 1], n_groups * rows_per_group // 2)

    gkf = make_group_kfold(n_splits=5)
    for train_idx, test_idx in gkf.split(X, y, groups):
        assert_no_leakage(groups[train_idx], groups[test_idx])
