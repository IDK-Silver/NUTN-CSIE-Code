import numpy as np
import pytest

from ecg.training.ds004504_rbp_paper import apply_task_to_labels, get_paper_classification_task


def test_apply_binary_paper_task_to_labels() -> None:
    labels = np.asarray([0, 1, 2, 0, 2], dtype=np.int64)
    task = get_paper_classification_task("ad_vs_healthy")

    selected_indices, remapped_labels = apply_task_to_labels(labels, task)

    np.testing.assert_array_equal(selected_indices, np.asarray([0, 2, 3, 4], dtype=np.int64))
    np.testing.assert_array_equal(remapped_labels, np.asarray([0, 1, 0, 1], dtype=np.int64))


def test_get_paper_classification_task_rejects_unknown_task() -> None:
    with pytest.raises(ValueError, match="Unsupported ds004504_rbp_paper task"):
        get_paper_classification_task("unknown")
