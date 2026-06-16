import numpy as np

from ecg.training.ds004504_rbp_paper import make_stratified_epoch_split


def test_make_stratified_epoch_split_preserves_class_counts() -> None:
    labels = np.asarray([0] * 10 + [1] * 10, dtype=np.int64)

    split = make_stratified_epoch_split(
        labels=labels,
        train_fraction=0.6,
        val_fraction=0.2,
        test_fraction=0.2,
        seed=7,
    )

    assert split.train.shape == (12,)
    assert split.val.shape == (4,)
    assert split.test.shape == (4,)
    assert set(split.train).isdisjoint(set(split.val))
    assert set(split.train).isdisjoint(set(split.test))
    assert set(split.val).isdisjoint(set(split.test))
    assert np.bincount(labels[split.train], minlength=2).tolist() == [6, 6]
    assert np.bincount(labels[split.val], minlength=2).tolist() == [2, 2]
    assert np.bincount(labels[split.test], minlength=2).tolist() == [2, 2]


def test_make_stratified_epoch_split_allows_no_test_partition() -> None:
    labels = np.asarray([0] * 10 + [1] * 10, dtype=np.int64)

    split = make_stratified_epoch_split(
        labels=labels,
        train_fraction=0.8,
        val_fraction=0.2,
        test_fraction=0.0,
        seed=7,
    )

    assert split.train.shape == (16,)
    assert split.val.shape == (4,)
    assert split.test.shape == (0,)
    assert set(split.train).isdisjoint(set(split.val))
    assert np.bincount(labels[split.train], minlength=2).tolist() == [8, 8]
    assert np.bincount(labels[split.val], minlength=2).tolist() == [2, 2]
