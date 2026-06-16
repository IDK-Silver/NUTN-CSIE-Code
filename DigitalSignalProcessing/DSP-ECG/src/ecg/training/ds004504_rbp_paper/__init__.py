"""ds004504_rbp_paper reproduction helpers."""

from ecg.training.ds004504_rbp_paper.factory import (
    PAPER_BATCH_SIZE,
    PAPER_DENSE_DROPOUT,
    PAPER_DENSE_HIDDEN_DIMS,
    PAPER_LEARNING_RATE,
    PAPER_LSTM_HIDDEN_DIM,
    PAPER_TCN_CHANNELS,
    PAPER_TCN_DILATIONS,
    PAPER_TCN_DROPOUT,
    PAPER_TCN_KERNEL_SIZE,
    PaperExperiment,
    PaperHyperparameters,
    build_paper_experiment,
)
from ecg.training.ds004504_rbp_paper.normalization import MinMaxParams, fit_paper_min_max, transform_min_max
from ecg.training.ds004504_rbp_paper.splits import PaperSplitIndices, make_stratified_epoch_split
from ecg.training.ds004504_rbp_paper.tasks import (
    PaperClassificationTask,
    apply_task_to_labels,
    get_paper_classification_task,
)

__all__ = [
    "PAPER_BATCH_SIZE",
    "PAPER_DENSE_DROPOUT",
    "PAPER_DENSE_HIDDEN_DIMS",
    "PAPER_LEARNING_RATE",
    "PAPER_LSTM_HIDDEN_DIM",
    "PAPER_TCN_CHANNELS",
    "PAPER_TCN_DILATIONS",
    "PAPER_TCN_DROPOUT",
    "PAPER_TCN_KERNEL_SIZE",
    "PaperClassificationTask",
    "PaperExperiment",
    "PaperHyperparameters",
    "PaperSplitIndices",
    "MinMaxParams",
    "apply_task_to_labels",
    "build_paper_experiment",
    "fit_paper_min_max",
    "get_paper_classification_task",
    "make_stratified_epoch_split",
    "transform_min_max",
]
