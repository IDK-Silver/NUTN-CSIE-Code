import pytest

from ecg.training import compute_classification_metrics


def test_compute_classification_metrics() -> None:
    metrics = compute_classification_metrics(
        y_true=(0, 0, 1, 1),
        y_pred=(0, 1, 1, 1),
        class_names=("negative", "positive"),
    )

    assert metrics["accuracy"] == 0.75
    assert metrics["confusion_matrix"] == [[1, 1], [0, 2]]
    assert metrics["support"] == 4
    assert metrics["per_class"] == [
        {
            "class_index": 0,
            "class_name": "negative",
            "precision": 1.0,
            "recall": 0.5,
            "f1": pytest.approx(2.0 / 3.0),
            "sensitivity": 0.5,
            "specificity": 1.0,
            "support": 2,
        },
        {
            "class_index": 1,
            "class_name": "positive",
            "precision": pytest.approx(2.0 / 3.0),
            "recall": 1.0,
            "f1": 0.8,
            "sensitivity": 1.0,
            "specificity": 0.5,
            "support": 2,
        },
    ]


def test_compute_classification_metrics_rejects_bad_label() -> None:
    with pytest.raises(ValueError, match="outside class_names range"):
        compute_classification_metrics(y_true=(0, 2), y_pred=(0, 1), class_names=("a", "b"))
