"""Classification tasks used by the ds004504_rbp_paper reproduction."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


DS004504_LABEL_A = 0
DS004504_LABEL_F = 1
DS004504_LABEL_C = 2


@dataclass(frozen=True)
class PaperClassificationTask:
    task_id: str
    class_names: tuple[str, ...]
    source_label_to_target_label: dict[int, int]


PAPER_CLASSIFICATION_TASKS: tuple[PaperClassificationTask, ...] = (
    PaperClassificationTask(
        task_id="multiclass",
        class_names=("alzheimer", "frontotemporal_dementia", "healthy_control"),
        source_label_to_target_label={
            DS004504_LABEL_A: 0,
            DS004504_LABEL_F: 1,
            DS004504_LABEL_C: 2,
        },
    ),
    PaperClassificationTask(
        task_id="ad_ftd_vs_healthy",
        class_names=("alzheimer_or_frontotemporal_dementia", "healthy_control"),
        source_label_to_target_label={
            DS004504_LABEL_A: 0,
            DS004504_LABEL_F: 0,
            DS004504_LABEL_C: 1,
        },
    ),
    PaperClassificationTask(
        task_id="ad_vs_healthy",
        class_names=("alzheimer", "healthy_control"),
        source_label_to_target_label={
            DS004504_LABEL_A: 0,
            DS004504_LABEL_C: 1,
        },
    ),
    PaperClassificationTask(
        task_id="ftd_vs_healthy",
        class_names=("frontotemporal_dementia", "healthy_control"),
        source_label_to_target_label={
            DS004504_LABEL_F: 0,
            DS004504_LABEL_C: 1,
        },
    ),
)


def get_paper_classification_task(task_id: str) -> PaperClassificationTask:
    for task in PAPER_CLASSIFICATION_TASKS:
        if task.task_id == task_id:
            return task
    supported = ", ".join(task.task_id for task in PAPER_CLASSIFICATION_TASKS)
    raise ValueError(f"Unsupported ds004504_rbp_paper task: {task_id}. Supported tasks: {supported}.")


def apply_task_to_labels(
    labels: npt.NDArray[np.integer],
    task: PaperClassificationTask,
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    if labels.ndim != 1:
        raise ValueError(f"labels must be 1D, got shape {labels.shape}.")

    selected_indices: list[int] = []
    remapped_labels: list[int] = []
    for index, source_label in enumerate(labels.astype(np.int64)):
        source_label_int = int(source_label)
        if source_label_int in task.source_label_to_target_label:
            selected_indices.append(index)
            remapped_labels.append(task.source_label_to_target_label[source_label_int])

    if not selected_indices:
        raise ValueError(f"Task {task.task_id!r} selected zero samples.")

    target_labels = np.asarray(remapped_labels, dtype=np.int64)
    expected_labels = set(range(len(task.class_names)))
    observed_labels = set(int(value) for value in target_labels)
    if observed_labels != expected_labels:
        raise ValueError(
            f"Task {task.task_id!r} does not contain every target class. "
            f"Expected {sorted(expected_labels)}, got {sorted(observed_labels)}."
        )

    return np.asarray(selected_indices, dtype=np.int64), target_labels
