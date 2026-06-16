"""Generic classification metrics."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def compute_classification_metrics(
    *,
    y_true: Sequence[int],
    y_pred: Sequence[int],
    class_names: Sequence[str],
) -> dict[str, object]:
    if len(y_true) != len(y_pred):
        raise ValueError(f"y_true and y_pred lengths differ: {len(y_true)} != {len(y_pred)}.")
    if not y_true:
        raise ValueError("Cannot compute metrics for zero samples.")
    if not class_names:
        raise ValueError("class_names must contain at least one class.")

    num_classes = len(class_names)
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    for true_label, pred_label in zip(y_true, y_pred, strict=True):
        if true_label < 0 or true_label >= num_classes:
            raise ValueError(f"y_true contains label outside class_names range: {true_label}.")
        if pred_label < 0 or pred_label >= num_classes:
            raise ValueError(f"y_pred contains label outside class_names range: {pred_label}.")
        confusion[true_label, pred_label] += 1

    total = int(confusion.sum())
    accuracy = float(np.trace(confusion) / total)
    per_class: list[dict[str, object]] = []
    precisions: list[float] = []
    recalls: list[float] = []
    f1_scores: list[float] = []

    for class_index, class_name in enumerate(class_names):
        tp = int(confusion[class_index, class_index])
        fp = int(confusion[:, class_index].sum() - tp)
        fn = int(confusion[class_index, :].sum() - tp)
        tn = int(total - tp - fp - fn)
        precision = tp / (tp + fp) if tp + fp > 0 else 0.0
        recall = tp / (tp + fn) if tp + fn > 0 else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall > 0.0 else 0.0
        specificity = tn / (tn + fp) if tn + fp > 0 else 0.0
        support = int(confusion[class_index, :].sum())

        precisions.append(float(precision))
        recalls.append(float(recall))
        f1_scores.append(float(f1))
        per_class.append(
            {
                "class_index": class_index,
                "class_name": class_name,
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "sensitivity": float(recall),
                "specificity": float(specificity),
                "support": support,
            }
        )

    return {
        "accuracy": accuracy,
        "macro_precision": float(np.mean(precisions)),
        "macro_recall": float(np.mean(recalls)),
        "macro_f1": float(np.mean(f1_scores)),
        "confusion_matrix": confusion.tolist(),
        "per_class": per_class,
        "support": total,
    }
