from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from ecg.training.ds004504_rbp_paper.factory import build_paper_hyperparameters, build_paper_model


JsonObject = dict[str, Any]
CsvRow = dict[str, object]


TASKS = ("multiclass", "ad_ftd_vs_healthy", "ad_vs_healthy", "ftd_vs_healthy")
STANDARD_RBP_TASKS = ("multiclass", "ad_vs_healthy")
SMOTE_TASKS = ("multiclass", "ad_vs_healthy")
LABEL_SWAP_TASKS = ("multiclass", "ftd_vs_healthy")

TASK_TO_PAPER_TABLE = {
    "multiclass": "Table 3",
    "ad_ftd_vs_healthy": "Table 4",
    "ad_vs_healthy": "Table 5",
    "ftd_vs_healthy": "Table 6",
}

STANDARD_RBP_TASK_TO_PAPER_TABLE = {
    "multiclass": "Table 12",
    "ad_vs_healthy": "Table 13",
}

SMOTE_TASK_TO_PAPER_TABLE = {
    "multiclass": "Table 8",
    "ad_vs_healthy": "Table 9",
}

KFOLD_TASK_TO_PAPER_TABLE = {
    "multiclass": "Table 10",
    "ad_vs_healthy": "Table 11",
}

LABEL_SWAP_TASK_TO_PAPER_TABLE = {
    "multiclass": "Table 3",
    "ftd_vs_healthy": "Table 6",
}


PAPER_MODEL_ARCHITECTURE_ROWS: tuple[CsvRow, ...] = (
    {
        "paper_table": "Table 1",
        "layer_order": 1,
        "layer_type": "Input layer",
        "output_shape": "(None, 6, 1)",
        "parameter_count": 0,
        "connected_to": "-",
    },
    {
        "paper_table": "Table 1",
        "layer_order": 2,
        "layer_type": "Conv 1D",
        "output_shape": "(None, 6, 32)",
        "parameter_count": 256,
        "connected_to": "Input layer",
    },
    {
        "paper_table": "Table 1",
        "layer_order": 3,
        "layer_type": "Batch normalization",
        "output_shape": "(None, 6, 32)",
        "parameter_count": 128,
        "connected_to": "Conv1D",
    },
    {
        "paper_table": "Table 1",
        "layer_order": 4,
        "layer_type": "Activation",
        "output_shape": "(None, 6, 32)",
        "parameter_count": 0,
        "connected_to": "Batch normalization",
    },
    {
        "paper_table": "Table 1",
        "layer_order": 5,
        "layer_type": "Spatial dropout 1D",
        "output_shape": "(None, 6, 32)",
        "parameter_count": 0,
        "connected_to": "Activation",
    },
    {
        "paper_table": "Table 1",
        "layer_order": 6,
        "layer_type": "Conv1D",
        "output_shape": "(None, 6, 32)",
        "parameter_count": 7200,
        "connected_to": "Spatial dropout 1D",
    },
    {
        "paper_table": "Table 1",
        "layer_order": 7,
        "layer_type": "Batch normalization",
        "output_shape": "(None, 6, 32)",
        "parameter_count": 128,
        "connected_to": "Conv1D",
    },
    {
        "paper_table": "Table 1",
        "layer_order": 8,
        "layer_type": "Activation",
        "output_shape": "(None, 6, 32)",
        "parameter_count": 0,
        "connected_to": "Batch normalization",
    },
    {
        "paper_table": "Table 1",
        "layer_order": 9,
        "layer_type": "Conv 1D residual",
        "output_shape": "(None, 6, 32)",
        "parameter_count": 64,
        "connected_to": "Input layer",
    },
    {
        "paper_table": "Table 1",
        "layer_order": 10,
        "layer_type": "Spatial dropout 1D",
        "output_shape": "(None, 6, 32)",
        "parameter_count": 0,
        "connected_to": "Activation",
    },
    {
        "paper_table": "Table 1",
        "layer_order": 11,
        "layer_type": "Add",
        "output_shape": "(None, 6, 32)",
        "parameter_count": 0,
        "connected_to": "Conv1D + Spatial dropout",
    },
    {
        "paper_table": "Table 1",
        "layer_order": 12,
        "layer_type": "Conv 1D",
        "output_shape": "(None, 6, 32)",
        "parameter_count": 7200,
        "connected_to": "Add",
    },
    {
        "paper_table": "Table 1",
        "layer_order": 13,
        "layer_type": "Batch normalization",
        "output_shape": "(None, 6, 32)",
        "parameter_count": 128,
        "connected_to": "Conv1D",
    },
    {
        "paper_table": "Table 1",
        "layer_order": 14,
        "layer_type": "Activation",
        "output_shape": "(None, 6, 32)",
        "parameter_count": 0,
        "connected_to": "Batch normalization",
    },
    {
        "paper_table": "Table 1",
        "layer_order": 15,
        "layer_type": "Spatial dropout 1D",
        "output_shape": "(None, 6, 32)",
        "parameter_count": 0,
        "connected_to": "Activation",
    },
    {
        "paper_table": "Table 1",
        "layer_order": 16,
        "layer_type": "Conv 1D",
        "output_shape": "(None, 6, 32)",
        "parameter_count": 7200,
        "connected_to": "Spatial dropout 1D",
    },
    {
        "paper_table": "Table 1",
        "layer_order": 17,
        "layer_type": "Batch normalization",
        "output_shape": "(None, 6, 32)",
        "parameter_count": 128,
        "connected_to": "Conv 1D",
    },
    {
        "paper_table": "Table 1",
        "layer_order": 18,
        "layer_type": "Activation",
        "output_shape": "(None, 6, 32)",
        "parameter_count": 0,
        "connected_to": "Batch normalization",
    },
    {
        "paper_table": "Table 1",
        "layer_order": 19,
        "layer_type": "Conv 1D residual",
        "output_shape": "(None, 6, 32)",
        "parameter_count": 1056,
        "connected_to": "Add",
    },
    {
        "paper_table": "Table 1",
        "layer_order": 20,
        "layer_type": "Spatial dropout 1D",
        "output_shape": "(None, 6, 32)",
        "parameter_count": 0,
        "connected_to": "Activation",
    },
    {
        "paper_table": "Table 1",
        "layer_order": 21,
        "layer_type": "Add",
        "output_shape": "(None, 6, 32)",
        "parameter_count": 0,
        "connected_to": "Conv1D + Spatial dropout",
    },
    {
        "paper_table": "Table 1",
        "layer_order": 22,
        "layer_type": "LSTM",
        "output_shape": "(None, 64)",
        "parameter_count": 24832,
        "connected_to": "Add",
    },
    {
        "paper_table": "Table 1",
        "layer_order": 23,
        "layer_type": "Dense",
        "output_shape": "(None, 128)",
        "parameter_count": 8320,
        "connected_to": "LSTM",
    },
    {
        "paper_table": "Table 1",
        "layer_order": 24,
        "layer_type": "Dropout",
        "output_shape": "(None, 128)",
        "parameter_count": 0,
        "connected_to": "Dense",
    },
    {
        "paper_table": "Table 1",
        "layer_order": 25,
        "layer_type": "Dense",
        "output_shape": "(None, 192)",
        "parameter_count": 24768,
        "connected_to": "Dropout",
    },
    {
        "paper_table": "Table 1",
        "layer_order": 26,
        "layer_type": "Dropout",
        "output_shape": "(None, 192)",
        "parameter_count": 0,
        "connected_to": "Dense",
    },
    {
        "paper_table": "Table 1",
        "layer_order": 27,
        "layer_type": "Dense",
        "output_shape": "(None, 256)",
        "parameter_count": 49408,
        "connected_to": "Dropout",
    },
    {
        "paper_table": "Table 1",
        "layer_order": 28,
        "layer_type": "Dropout",
        "output_shape": "(None, 256)",
        "parameter_count": 0,
        "connected_to": "Dense",
    },
    {
        "paper_table": "Table 1",
        "layer_order": 29,
        "layer_type": "Dense output",
        "output_shape": "(None, 3)",
        "parameter_count": 771,
        "connected_to": "Dropout",
    },
)

PAPER_MODEL_PARAMETER_ROWS: tuple[CsvRow, ...] = (
    {
        "source": "paper_reported",
        "paper_table": "Table 2",
        "scenario": "",
        "task_id": "",
        "parameter": "total_parameters",
        "value": 131587,
        "size": "514.01 KB",
        "notes": "",
    },
    {
        "source": "paper_reported",
        "paper_table": "Table 2",
        "scenario": "",
        "task_id": "",
        "parameter": "trainable_parameters",
        "value": 131331,
        "size": "513.01 KB",
        "notes": "",
    },
    {
        "source": "paper_reported",
        "paper_table": "Table 2",
        "scenario": "",
        "task_id": "",
        "parameter": "non_trainable_parameters",
        "value": 256,
        "size": "1 KB",
        "notes": "",
    },
)

PAPER_CLASSIFICATION_ROWS: tuple[CsvRow, ...] = (
    {
        "source": "paper_reported",
        "paper_table": "Table 3",
        "experiment_kind": "modified_rbp",
        "task_id": "multiclass",
        "class_name": "alzheimer",
        "precision": 0.70,
        "recall": 0.90,
        "f1": 0.79,
        "sensitivity": 0.90,
        "specificity": 0.74,
        "support": 1876,
        "value_scale": "decimal",
        "notes": "",
    },
    {
        "source": "paper_reported",
        "paper_table": "Table 3",
        "experiment_kind": "modified_rbp",
        "task_id": "multiclass",
        "class_name": "frontotemporal_dementia",
        "precision": 1.00,
        "recall": 1.00,
        "f1": 1.00,
        "sensitivity": 1.00,
        "specificity": 1.00,
        "support": 1597,
        "value_scale": "decimal",
        "notes": "Support is inconsistent with ds004504 FTD duration and matches healthy duration.",
    },
    {
        "source": "paper_reported",
        "paper_table": "Table 3",
        "experiment_kind": "modified_rbp",
        "task_id": "multiclass",
        "class_name": "healthy_control",
        "precision": 0.68,
        "recall": 0.35,
        "f1": 0.47,
        "sensitivity": 0.35,
        "specificity": 0.95,
        "support": 1106,
        "value_scale": "decimal",
        "notes": "Support is inconsistent with ds004504 healthy duration and matches FTD duration.",
    },
    {
        "source": "paper_reported",
        "paper_table": "Table 4",
        "experiment_kind": "modified_rbp",
        "task_id": "ad_ftd_vs_healthy",
        "class_name": "alzheimer_or_frontotemporal_dementia",
        "precision": 0.9977,
        "recall": 0.9993,
        "f1": 0.9985,
        "sensitivity": 1.00,
        "specificity": 1.00,
        "support": 2983,
        "value_scale": "decimal",
        "notes": "Binary sensitivity/specificity are carried from the paper's single reported row.",
    },
    {
        "source": "paper_reported",
        "paper_table": "Table 4",
        "experiment_kind": "modified_rbp",
        "task_id": "ad_ftd_vs_healthy",
        "class_name": "healthy_control",
        "precision": 0.9987,
        "recall": 0.9956,
        "f1": 0.9972,
        "sensitivity": 1.00,
        "specificity": 1.00,
        "support": 1596,
        "value_scale": "decimal",
        "notes": "Binary sensitivity/specificity are carried from the paper's single reported row.",
    },
    {
        "source": "paper_reported",
        "paper_table": "Table 5",
        "experiment_kind": "modified_rbp",
        "task_id": "ad_vs_healthy",
        "class_name": "alzheimer",
        "precision": 0.9963,
        "recall": 0.9989,
        "f1": 0.9976,
        "sensitivity": 1.00,
        "specificity": 1.00,
        "support": 1876,
        "value_scale": "decimal",
        "notes": "Binary sensitivity/specificity are carried from the paper's single reported row.",
    },
    {
        "source": "paper_reported",
        "paper_table": "Table 5",
        "experiment_kind": "modified_rbp",
        "task_id": "ad_vs_healthy",
        "class_name": "healthy_control",
        "precision": 0.9987,
        "recall": 0.9956,
        "f1": 0.9972,
        "sensitivity": 1.00,
        "specificity": 1.00,
        "support": 1596,
        "value_scale": "decimal",
        "notes": "Paper table support is 1596; Figure 4c row sum is 1597.",
    },
    {
        "source": "paper_reported",
        "paper_table": "Table 6",
        "experiment_kind": "modified_rbp",
        "task_id": "ftd_vs_healthy",
        "class_name": "frontotemporal_dementia",
        "precision": 0.9994,
        "recall": 0.9956,
        "f1": 0.9975,
        "sensitivity": 1.00,
        "specificity": 1.00,
        "support": 1597,
        "value_scale": "decimal",
        "notes": "Binary sensitivity/specificity are carried from the paper's single reported row.",
    },
    {
        "source": "paper_reported",
        "paper_table": "Table 6",
        "experiment_kind": "modified_rbp",
        "task_id": "ftd_vs_healthy",
        "class_name": "healthy_control",
        "precision": 0.9937,
        "recall": 0.9991,
        "f1": 0.9964,
        "sensitivity": 1.00,
        "specificity": 1.00,
        "support": 1596,
        "value_scale": "decimal",
        "notes": "Paper table support is 1596; Figure 4d row sum and text are 1106.",
    },
    {
        "source": "paper_reported",
        "paper_table": "Table 8",
        "experiment_kind": "smote_modified_rbp",
        "task_id": "multiclass",
        "class_name": "alzheimer",
        "precision": 0.63,
        "recall": 0.71,
        "f1": 0.67,
        "sensitivity": 0.71,
        "specificity": 0.79,
        "support": 1876,
        "value_scale": "decimal",
        "notes": "",
    },
    {
        "source": "paper_reported",
        "paper_table": "Table 8",
        "experiment_kind": "smote_modified_rbp",
        "task_id": "multiclass",
        "class_name": "frontotemporal_dementia",
        "precision": 1.00,
        "recall": 1.00,
        "f1": 1.00,
        "sensitivity": 1.00,
        "specificity": 1.00,
        "support": 1876,
        "value_scale": "decimal",
        "notes": "Table 8 supports are balanced to 1876 for every class.",
    },
    {
        "source": "paper_reported",
        "paper_table": "Table 8",
        "experiment_kind": "smote_modified_rbp",
        "task_id": "multiclass",
        "class_name": "healthy_control",
        "precision": 0.67,
        "recall": 0.58,
        "f1": 0.62,
        "sensitivity": 0.58,
        "specificity": 0.86,
        "support": 1876,
        "value_scale": "decimal",
        "notes": "The paper text reports 77.45% accuracy, but these equal supports and recalls imply 76.33%.",
    },
    {
        "source": "paper_reported",
        "paper_table": "Table 9",
        "experiment_kind": "smote_modified_rbp",
        "task_id": "ad_vs_healthy",
        "class_name": "alzheimer",
        "precision": 99.73,
        "recall": 99.70,
        "f1": 99.71,
        "sensitivity": "",
        "specificity": "",
        "support": 1876,
        "value_scale": "percent",
        "notes": "Paper table reports precision/recall/F1 as percentages, unlike Tables 3-8.",
    },
    {
        "source": "paper_reported",
        "paper_table": "Table 9",
        "experiment_kind": "smote_modified_rbp",
        "task_id": "ad_vs_healthy",
        "class_name": "healthy_control",
        "precision": 99.70,
        "recall": 99.71,
        "f1": 99.73,
        "sensitivity": "",
        "specificity": "",
        "support": 1876,
        "value_scale": "percent",
        "notes": "Paper table reports precision/recall/F1 as percentages, unlike Tables 3-8.",
    },
    {
        "source": "paper_reported",
        "paper_table": "Table 12",
        "experiment_kind": "standard_rbp",
        "task_id": "multiclass",
        "class_name": "alzheimer",
        "precision": 0.60,
        "recall": 0.77,
        "f1": 0.67,
        "sensitivity": "",
        "specificity": "",
        "support": 1876,
        "value_scale": "decimal",
        "notes": "",
    },
    {
        "source": "paper_reported",
        "paper_table": "Table 12",
        "experiment_kind": "standard_rbp",
        "task_id": "multiclass",
        "class_name": "frontotemporal_dementia",
        "precision": 0.68,
        "recall": 0.68,
        "f1": 0.68,
        "sensitivity": "",
        "specificity": "",
        "support": 1597,
        "value_scale": "decimal",
        "notes": "",
    },
    {
        "source": "paper_reported",
        "paper_table": "Table 12",
        "experiment_kind": "standard_rbp",
        "task_id": "multiclass",
        "class_name": "healthy_control",
        "precision": 0.60,
        "recall": 0.33,
        "f1": 0.43,
        "sensitivity": "",
        "specificity": "",
        "support": 1106,
        "value_scale": "decimal",
        "notes": "",
    },
    {
        "source": "paper_reported",
        "paper_table": "Table 13",
        "experiment_kind": "standard_rbp",
        "task_id": "ad_vs_healthy",
        "class_name": "alzheimer",
        "precision": 0.76,
        "recall": 0.81,
        "f1": 0.79,
        "sensitivity": "",
        "specificity": "",
        "support": 1876,
        "value_scale": "decimal",
        "notes": "",
    },
    {
        "source": "paper_reported",
        "paper_table": "Table 13",
        "experiment_kind": "standard_rbp",
        "task_id": "ad_vs_healthy",
        "class_name": "healthy_control",
        "precision": 0.76,
        "recall": 0.71,
        "f1": 0.73,
        "sensitivity": "",
        "specificity": "",
        "support": 1597,
        "value_scale": "decimal",
        "notes": "",
    },
)

PAPER_ACCURACY_ROWS: tuple[CsvRow, ...] = (
    {
        "source": "paper_reported",
        "paper_table": "Table 7",
        "experiment_kind": "modified_rbp",
        "task_id": "ftd_vs_healthy",
        "metric": "accuracy",
        "value": 0.9970,
        "value_scale": "decimal",
        "notes": "",
    },
    {
        "source": "paper_reported",
        "paper_table": "Table 7",
        "experiment_kind": "modified_rbp",
        "task_id": "ad_vs_healthy",
        "metric": "accuracy",
        "value": 0.9974,
        "value_scale": "decimal",
        "notes": "",
    },
    {
        "source": "paper_reported",
        "paper_table": "Table 7",
        "experiment_kind": "modified_rbp",
        "task_id": "ad_ftd_vs_healthy",
        "metric": "accuracy",
        "value": 0.9980,
        "value_scale": "decimal",
        "notes": "",
    },
    {
        "source": "paper_reported",
        "paper_table": "Table 7",
        "experiment_kind": "modified_rbp",
        "task_id": "multiclass",
        "metric": "accuracy",
        "value": 0.8034,
        "value_scale": "decimal",
        "notes": "",
    },
    {
        "source": "paper_text",
        "paper_table": "Table 8",
        "experiment_kind": "smote_modified_rbp",
        "task_id": "multiclass",
        "metric": "accuracy",
        "value": 77.45,
        "value_scale": "percent",
        "notes": "Paper text reports this value; Table 8 row metrics imply a different accuracy.",
    },
    {
        "source": "paper_text",
        "paper_table": "Table 12",
        "experiment_kind": "standard_rbp",
        "task_id": "multiclass",
        "metric": "accuracy",
        "value": 63.03,
        "value_scale": "percent",
        "notes": "Reported in Section 4.5 text.",
    },
    {
        "source": "paper_text",
        "paper_table": "Table 13",
        "experiment_kind": "standard_rbp",
        "task_id": "ad_vs_healthy",
        "metric": "accuracy",
        "value": 76.36,
        "value_scale": "percent",
        "notes": "Reported in Section 4.5 text.",
    },
)

PAPER_KFOLD_ROWS: tuple[CsvRow, ...] = (
    {
        "source": "paper_reported",
        "paper_table": "Table 10",
        "experiment_kind": "kfold_modified_rbp",
        "task_id": "multiclass",
        "fold": 1,
        "train_accuracy_percent": 79.89,
        "test_accuracy_percent": 80.15,
        "notes": "",
    },
    {
        "source": "paper_reported",
        "paper_table": "Table 10",
        "experiment_kind": "kfold_modified_rbp",
        "task_id": "multiclass",
        "fold": 2,
        "train_accuracy_percent": 80.00,
        "test_accuracy_percent": 80.00,
        "notes": "",
    },
    {
        "source": "paper_reported",
        "paper_table": "Table 10",
        "experiment_kind": "kfold_modified_rbp",
        "task_id": "multiclass",
        "fold": 3,
        "train_accuracy_percent": 79.58,
        "test_accuracy_percent": 80.06,
        "notes": "",
    },
    {
        "source": "paper_reported",
        "paper_table": "Table 10",
        "experiment_kind": "kfold_modified_rbp",
        "task_id": "multiclass",
        "fold": 4,
        "train_accuracy_percent": 79.43,
        "test_accuracy_percent": 80.02,
        "notes": "",
    },
    {
        "source": "paper_reported",
        "paper_table": "Table 10",
        "experiment_kind": "kfold_modified_rbp",
        "task_id": "multiclass",
        "fold": 5,
        "train_accuracy_percent": 81.27,
        "test_accuracy_percent": 80.13,
        "notes": "",
    },
    {
        "source": "paper_reported",
        "paper_table": "Table 11",
        "experiment_kind": "kfold_modified_rbp",
        "task_id": "ad_vs_healthy",
        "fold": 1,
        "train_accuracy_percent": 99.82,
        "test_accuracy_percent": 99.86,
        "notes": "",
    },
    {
        "source": "paper_reported",
        "paper_table": "Table 11",
        "experiment_kind": "kfold_modified_rbp",
        "task_id": "ad_vs_healthy",
        "fold": 2,
        "train_accuracy_percent": 99.80,
        "test_accuracy_percent": 99.82,
        "notes": "",
    },
    {
        "source": "paper_reported",
        "paper_table": "Table 11",
        "experiment_kind": "kfold_modified_rbp",
        "task_id": "ad_vs_healthy",
        "fold": 3,
        "train_accuracy_percent": 99.73,
        "test_accuracy_percent": 99.92,
        "notes": "",
    },
    {
        "source": "paper_reported",
        "paper_table": "Table 11",
        "experiment_kind": "kfold_modified_rbp",
        "task_id": "ad_vs_healthy",
        "fold": 4,
        "train_accuracy_percent": 99.61,
        "test_accuracy_percent": 99.86,
        "notes": "",
    },
    {
        "source": "paper_reported",
        "paper_table": "Table 11",
        "experiment_kind": "kfold_modified_rbp",
        "task_id": "ad_vs_healthy",
        "fold": 5,
        "train_accuracy_percent": 99.78,
        "test_accuracy_percent": 99.82,
        "notes": "",
    },
)

PAPER_LITERATURE_ROWS: tuple[CsvRow, ...] = (
    {
        "paper_table": "Table 14",
        "paper": "Ma et al.",
        "model": "Support vector machine",
        "accuracy": "91.5%",
        "feature_engineering": "PHI",
        "xai": "no",
    },
    {
        "paper_table": "Table 14",
        "paper": "Miltiadous et al.",
        "model": "Dual-Input Convolution Encoder Network",
        "accuracy": "83.28%",
        "feature_engineering": "Band power and coherence",
        "xai": "no",
    },
    {
        "paper_table": "Table 14",
        "paper": "Kachare et al.",
        "model": "STEADYNet",
        "accuracy": "97.59%",
        "feature_engineering": "not listed",
        "xai": "no",
    },
    {
        "paper_table": "Table 14",
        "paper": "Chen et al.",
        "model": "Vision transformer + CNN",
        "accuracy": "80.23%",
        "feature_engineering": "frequency channels",
        "xai": "no",
    },
    {
        "paper_table": "Table 14",
        "paper": "This work",
        "model": "Proposed TCN-LSTM",
        "accuracy": "80.34%, 99.7%",
        "feature_engineering": "Modified RBP",
        "xai": "yes",
    },
)

PAPER_CONFUSION_MATRICES: dict[str, tuple[str, str, tuple[str, ...], tuple[tuple[int, ...], ...]]] = {
    "multiclass": (
        "Figure 4a",
        "Table 3",
        ("alzheimer", "frontotemporal_dementia", "healthy_control"),
        ((1693, 4, 179), (1, 1594, 2), (712, 2, 392)),
    ),
    "ad_ftd_vs_healthy": (
        "Figure 4b",
        "Table 4",
        ("alzheimer_or_frontotemporal_dementia", "healthy_control"),
        ((2981, 2), (7, 1589)),
    ),
    "ad_vs_healthy": (
        "Figure 4c",
        "Table 5",
        ("alzheimer", "healthy_control"),
        ((1874, 2), (7, 1590)),
    ),
    "ftd_vs_healthy": (
        "Figure 4d",
        "Table 6",
        ("frontotemporal_dementia", "healthy_control"),
        ((1590, 7), (1, 1105)),
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build CSV tables for ds004504_rbp_paper paper-vs-ours reporting."
    )
    parser.add_argument(
        "--scenario",
        default="paper_literal_80_10_10",
        help="Scenario label written to CSV outputs.",
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=None,
        help="Directory containing one subdirectory per task. Defaults from --scenario.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory where table CSV files will be written. Defaults from --scenario.",
    )
    return parser.parse_args()


def default_runs_dir(scenario: str) -> Path:
    base = Path("data/runs/ds004504_rbp_paper")
    if scenario == "paper_literal_80_10_10":
        return base
    return base / scenario


def default_output_dir(scenario: str) -> Path:
    return Path("data/reports/ds004504_rbp_paper") / scenario / "tables"


def read_json(path: Path) -> JsonObject:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"JSON file must contain an object: {path}")
    return dict(payload)


def require_mapping(value: object, *, path: str) -> JsonObject:
    if not isinstance(value, dict):
        raise ValueError(f"Expected mapping at {path}.")
    return dict(value)


def optional_mapping(value: object) -> JsonObject:
    if isinstance(value, dict):
        return dict(value)
    return {}


def optional_sequence(value: object) -> list[object]:
    if isinstance(value, list):
        return list(value)
    return []


def write_csv(path: Path, fieldnames: tuple[str, ...], rows: list[CsvRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def get_task_run_dir(runs_dir: Path, task_id: str) -> Path:
    return runs_dir / task_id


def get_standard_rbp_task_run_dir(runs_dir: Path, task_id: str) -> Path:
    candidate = runs_dir / "standard_rbp" / task_id
    if candidate.exists():
        return candidate
    return Path("data/runs/ds004504_rbp_paper") / "standard_rbp" / task_id


def get_smote_task_run_dir(runs_dir: Path, task_id: str) -> Path:
    return runs_dir / "smote" / task_id


def get_kfold_task_run_dir(runs_dir: Path, task_id: str) -> Path:
    candidate = runs_dir / "kfold" / task_id
    if candidate.exists():
        return candidate
    return Path("data/runs/ds004504_rbp_paper") / "kfold" / task_id


def get_label_swap_task_run_dir(_runs_dir: Path, task_id: str, *, scenario: str) -> Path:
    if scenario == "paper_literal_80_10_10":
        return Path("data/runs/ds004504_rbp_paper") / "label_swap" / task_id
    candidate = _runs_dir / "label_swap_80_20" / task_id
    if candidate.exists():
        return candidate
    return Path("data/runs/ds004504_rbp_paper") / "label_swap_80_20" / task_id


def get_metrics_payload(run_dir: Path) -> JsonObject:
    metrics_path = run_dir / "test_metrics.json"
    if not metrics_path.exists():
        return {}
    return read_json(metrics_path)


def get_run_payload(run_dir: Path) -> JsonObject:
    run_path = run_dir / "run.json"
    if not run_path.exists():
        return {}
    return read_json(run_path)


def get_history_payload(run_dir: Path) -> list[object]:
    history_path = run_dir / "history.json"
    if not history_path.exists():
        return []
    with history_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, list):
        raise ValueError(f"History JSON must contain a list: {history_path}")
    return payload


def read_dict_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


def metric_value(metrics: JsonObject, key: str) -> object:
    value = metrics.get(key)
    if isinstance(value, int | float):
        return value
    return ""


def merged_config_value(run_payload: JsonObject, section: str, key: str) -> object:
    merged_config = optional_mapping(run_payload.get("merged_config"))
    section_payload = optional_mapping(merged_config.get(section))
    return section_payload.get(key, "")


def build_run_summary_rows(*, scenario: str, runs_dir: Path) -> list[CsvRow]:
    rows: list[CsvRow] = []
    for task_id in TASKS:
        run_dir = get_task_run_dir(runs_dir, task_id)
        run_payload = get_run_payload(run_dir)
        metrics_payload = get_metrics_payload(run_dir)
        metrics = optional_mapping(metrics_payload.get("metrics"))
        status = "available" if run_payload else "missing"
        rows.append(
            {
                "scenario": scenario,
                "task_id": task_id,
                "status": status,
                "run_dir": str(run_dir),
                "experiment_id": run_payload.get("experiment_id", ""),
                "config_path": run_payload.get("config_path", ""),
                "seed": run_payload.get("seed", run_payload.get("resolved_seed", "")),
                "epochs": run_payload.get("epochs", ""),
                "device": run_payload.get("device", merged_config_value(run_payload, "runtime", "device")),
                "test_source": metrics_payload.get(
                    "source", optional_mapping(run_payload.get("evaluation")).get("test_source", "")
                ),
                "train_fraction": optional_mapping(run_payload.get("split_fractions")).get(
                    "train", merged_config_value(run_payload, "split", "train_fraction")
                ),
                "val_fraction": optional_mapping(run_payload.get("split_fractions")).get(
                    "val", merged_config_value(run_payload, "split", "val_fraction")
                ),
                "test_fraction": optional_mapping(run_payload.get("split_fractions")).get(
                    "test", merged_config_value(run_payload, "split", "test_fraction")
                ),
                "accuracy": metric_value(metrics, "accuracy"),
                "macro_precision": metric_value(metrics, "macro_precision"),
                "macro_recall": metric_value(metrics, "macro_recall"),
                "macro_f1": metric_value(metrics, "macro_f1"),
                "support": metric_value(metrics, "support"),
            }
        )
    for task_id in STANDARD_RBP_TASKS:
        run_dir = get_standard_rbp_task_run_dir(runs_dir, task_id)
        run_payload = get_run_payload(run_dir)
        metrics_payload = get_metrics_payload(run_dir)
        metrics = optional_mapping(metrics_payload.get("metrics"))
        status = "available" if run_payload else "missing"
        rows.append(
            {
                "scenario": scenario,
                "task_id": f"standard_rbp/{task_id}",
                "status": status,
                "run_dir": str(run_dir),
                "experiment_id": run_payload.get("experiment_id", ""),
                "config_path": run_payload.get("config_path", ""),
                "seed": run_payload.get("seed", run_payload.get("resolved_seed", "")),
                "epochs": run_payload.get("epochs", ""),
                "device": run_payload.get("device", merged_config_value(run_payload, "runtime", "device")),
                "test_source": metrics_payload.get(
                    "source", optional_mapping(run_payload.get("evaluation")).get("test_source", "")
                ),
                "train_fraction": optional_mapping(run_payload.get("split_fractions")).get(
                    "train", merged_config_value(run_payload, "split", "train_fraction")
                ),
                "val_fraction": optional_mapping(run_payload.get("split_fractions")).get(
                    "val", merged_config_value(run_payload, "split", "val_fraction")
                ),
                "test_fraction": optional_mapping(run_payload.get("split_fractions")).get(
                    "test", merged_config_value(run_payload, "split", "test_fraction")
                ),
                "accuracy": metric_value(metrics, "accuracy"),
                "macro_precision": metric_value(metrics, "macro_precision"),
                "macro_recall": metric_value(metrics, "macro_recall"),
                "macro_f1": metric_value(metrics, "macro_f1"),
                "support": metric_value(metrics, "support"),
            }
        )
    for task_id in SMOTE_TASKS:
        run_dir = get_smote_task_run_dir(runs_dir, task_id)
        run_payload = get_run_payload(run_dir)
        metrics_payload = get_metrics_payload(run_dir)
        metrics = optional_mapping(metrics_payload.get("metrics"))
        status = "available" if run_payload else "missing"
        rows.append(
            {
                "scenario": scenario,
                "task_id": f"smote/{task_id}",
                "status": status,
                "run_dir": str(run_dir),
                "experiment_id": run_payload.get("experiment_id", ""),
                "config_path": run_payload.get("config_path", ""),
                "seed": run_payload.get("seed", run_payload.get("resolved_seed", "")),
                "epochs": run_payload.get("epochs", ""),
                "device": run_payload.get("device", merged_config_value(run_payload, "runtime", "device")),
                "test_source": metrics_payload.get(
                    "source", optional_mapping(run_payload.get("evaluation")).get("test_source", "")
                ),
                "train_fraction": optional_mapping(run_payload.get("split_fractions")).get(
                    "train", merged_config_value(run_payload, "split", "train_fraction")
                ),
                "val_fraction": optional_mapping(run_payload.get("split_fractions")).get(
                    "val", merged_config_value(run_payload, "split", "val_fraction")
                ),
                "test_fraction": optional_mapping(run_payload.get("split_fractions")).get(
                    "test", merged_config_value(run_payload, "split", "test_fraction")
                ),
                "accuracy": metric_value(metrics, "accuracy"),
                "macro_precision": metric_value(metrics, "macro_precision"),
                "macro_recall": metric_value(metrics, "macro_recall"),
                "macro_f1": metric_value(metrics, "macro_f1"),
                "support": metric_value(metrics, "support"),
            }
        )
    for task_id in LABEL_SWAP_TASKS:
        run_dir = get_label_swap_task_run_dir(runs_dir, task_id, scenario=scenario)
        run_payload = get_run_payload(run_dir)
        metrics_payload = get_metrics_payload(run_dir)
        metrics = optional_mapping(metrics_payload.get("metrics"))
        status = "available" if run_payload else "missing"
        rows.append(
            {
                "scenario": scenario,
                "task_id": f"label_swap_80_20/{task_id}",
                "status": status,
                "run_dir": str(run_dir),
                "experiment_id": run_payload.get("experiment_id", ""),
                "config_path": run_payload.get("config_path", ""),
                "seed": run_payload.get("seed", run_payload.get("resolved_seed", "")),
                "epochs": run_payload.get("epochs", ""),
                "device": run_payload.get("device", merged_config_value(run_payload, "runtime", "device")),
                "test_source": metrics_payload.get(
                    "source", optional_mapping(run_payload.get("evaluation")).get("test_source", "")
                ),
                "train_fraction": optional_mapping(run_payload.get("split_fractions")).get(
                    "train", merged_config_value(run_payload, "split", "train_fraction")
                ),
                "val_fraction": optional_mapping(run_payload.get("split_fractions")).get(
                    "val", merged_config_value(run_payload, "split", "val_fraction")
                ),
                "test_fraction": optional_mapping(run_payload.get("split_fractions")).get(
                    "test", merged_config_value(run_payload, "split", "test_fraction")
                ),
                "accuracy": metric_value(metrics, "accuracy"),
                "macro_precision": metric_value(metrics, "macro_precision"),
                "macro_recall": metric_value(metrics, "macro_recall"),
                "macro_f1": metric_value(metrics, "macro_f1"),
                "support": metric_value(metrics, "support"),
            }
        )
    return rows


def build_ours_classification_rows(*, scenario: str, runs_dir: Path) -> list[CsvRow]:
    rows: list[CsvRow] = []
    for task_id in TASKS:
        run_dir = get_task_run_dir(runs_dir, task_id)
        metrics_payload = get_metrics_payload(run_dir)
        metrics = optional_mapping(metrics_payload.get("metrics"))
        per_class = optional_sequence(metrics.get("per_class"))
        for item in per_class:
            item_payload = require_mapping(item, path=f"{run_dir}/test_metrics.json metrics.per_class[]")
            rows.append(
                {
                    "source": "ours",
                    "paper_table": TASK_TO_PAPER_TABLE[task_id],
                    "experiment_kind": "modified_rbp",
                    "scenario": scenario,
                    "task_id": task_id,
                    "class_name": item_payload.get("class_name", ""),
                    "precision": item_payload.get("precision", ""),
                    "recall": item_payload.get("recall", ""),
                    "f1": item_payload.get("f1", ""),
                    "sensitivity": item_payload.get("sensitivity", item_payload.get("recall", "")),
                    "specificity": item_payload.get("specificity", ""),
                    "support": item_payload.get("support", ""),
                    "value_scale": "decimal",
                    "notes": "",
                }
            )
    return rows


def build_ours_standard_rbp_classification_rows(*, scenario: str, runs_dir: Path) -> list[CsvRow]:
    rows: list[CsvRow] = []
    for task_id in STANDARD_RBP_TASKS:
        run_dir = get_standard_rbp_task_run_dir(runs_dir, task_id)
        metrics_payload = get_metrics_payload(run_dir)
        metrics = optional_mapping(metrics_payload.get("metrics"))
        per_class = optional_sequence(metrics.get("per_class"))
        for item in per_class:
            item_payload = require_mapping(item, path=f"{run_dir}/test_metrics.json metrics.per_class[]")
            rows.append(
                {
                    "source": "ours",
                    "paper_table": STANDARD_RBP_TASK_TO_PAPER_TABLE[task_id],
                    "experiment_kind": "standard_rbp",
                    "scenario": scenario,
                    "task_id": task_id,
                    "class_name": item_payload.get("class_name", ""),
                    "precision": item_payload.get("precision", ""),
                    "recall": item_payload.get("recall", ""),
                    "f1": item_payload.get("f1", ""),
                    "sensitivity": item_payload.get("sensitivity", item_payload.get("recall", "")),
                    "specificity": item_payload.get("specificity", ""),
                    "support": item_payload.get("support", ""),
                    "value_scale": "decimal",
                    "notes": "",
                }
            )
    return rows


def build_ours_smote_classification_rows(*, scenario: str, runs_dir: Path) -> list[CsvRow]:
    rows: list[CsvRow] = []
    for task_id in SMOTE_TASKS:
        run_dir = get_smote_task_run_dir(runs_dir, task_id)
        metrics_payload = get_metrics_payload(run_dir)
        metrics = optional_mapping(metrics_payload.get("metrics"))
        per_class = optional_sequence(metrics.get("per_class"))
        for item in per_class:
            item_payload = require_mapping(item, path=f"{run_dir}/test_metrics.json metrics.per_class[]")
            rows.append(
                {
                    "source": "ours",
                    "paper_table": SMOTE_TASK_TO_PAPER_TABLE[task_id],
                    "experiment_kind": "smote_modified_rbp",
                    "scenario": scenario,
                    "task_id": task_id,
                    "class_name": item_payload.get("class_name", ""),
                    "precision": item_payload.get("precision", ""),
                    "recall": item_payload.get("recall", ""),
                    "f1": item_payload.get("f1", ""),
                    "sensitivity": item_payload.get("sensitivity", item_payload.get("recall", "")),
                    "specificity": item_payload.get("specificity", ""),
                    "support": item_payload.get("support", ""),
                    "value_scale": "decimal",
                    "notes": "",
                }
            )
    return rows


def build_ours_label_swap_classification_rows(*, scenario: str, runs_dir: Path) -> list[CsvRow]:
    rows: list[CsvRow] = []
    for task_id in LABEL_SWAP_TASKS:
        run_dir = get_label_swap_task_run_dir(runs_dir, task_id, scenario=scenario)
        metrics_payload = get_metrics_payload(run_dir)
        metrics = optional_mapping(metrics_payload.get("metrics"))
        per_class = optional_sequence(metrics.get("per_class"))
        for item in per_class:
            item_payload = require_mapping(item, path=f"{run_dir}/test_metrics.json metrics.per_class[]")
            rows.append(
                {
                    "source": "ours_label_swap",
                    "paper_table": LABEL_SWAP_TASK_TO_PAPER_TABLE[task_id],
                    "experiment_kind": "modified_rbp_label_swap",
                    "scenario": "label_swap_80_20",
                    "task_id": task_id,
                    "class_name": item_payload.get("class_name", ""),
                    "precision": item_payload.get("precision", ""),
                    "recall": item_payload.get("recall", ""),
                    "f1": item_payload.get("f1", ""),
                    "sensitivity": item_payload.get("sensitivity", item_payload.get("recall", "")),
                    "specificity": item_payload.get("specificity", ""),
                    "support": item_payload.get("support", ""),
                    "value_scale": "decimal",
                    "notes": "Audit run with FTD/Healthy target labels intentionally swapped.",
                }
            )
    return rows


def build_classification_rows(*, scenario: str, runs_dir: Path) -> list[CsvRow]:
    rows: list[CsvRow] = []
    for row in PAPER_CLASSIFICATION_ROWS:
        next_row = dict(row)
        next_row.setdefault("scenario", "paper")
        rows.append(next_row)
    rows.extend(build_ours_classification_rows(scenario=scenario, runs_dir=runs_dir))
    rows.extend(build_ours_standard_rbp_classification_rows(scenario=scenario, runs_dir=runs_dir))
    rows.extend(build_ours_smote_classification_rows(scenario=scenario, runs_dir=runs_dir))
    if scenario in {"paper_literal_80_10_10", "val_as_test_80_20", "fixture_smoke"}:
        rows.extend(build_ours_label_swap_classification_rows(scenario=scenario, runs_dir=runs_dir))
    return rows


def build_paper_confusion_rows() -> list[CsvRow]:
    rows: list[CsvRow] = []
    for task_id, (figure, table, class_names, matrix) in PAPER_CONFUSION_MATRICES.items():
        for true_index, true_class in enumerate(class_names):
            for pred_index, predicted_class in enumerate(class_names):
                rows.append(
                    {
                        "source": "paper_figure",
                        "scenario": "paper",
                        "paper_figure": figure,
                        "paper_table": table,
                        "task_id": task_id,
                        "true_class": true_class,
                        "predicted_class": predicted_class,
                        "count": matrix[true_index][pred_index],
                        "notes": "",
                    }
                )
    return rows


def build_ours_confusion_rows(*, scenario: str, runs_dir: Path) -> list[CsvRow]:
    rows: list[CsvRow] = []
    for task_id in TASKS:
        run_dir = get_task_run_dir(runs_dir, task_id)
        run_payload = get_run_payload(run_dir)
        class_names = tuple(str(value) for value in optional_sequence(run_payload.get("class_names")))
        metrics_payload = get_metrics_payload(run_dir)
        metrics = optional_mapping(metrics_payload.get("metrics"))
        confusion = optional_sequence(metrics.get("confusion_matrix"))
        if not class_names or not confusion:
            continue
        for true_index, true_class in enumerate(class_names):
            true_row = optional_sequence(confusion[true_index]) if true_index < len(confusion) else []
            for pred_index, predicted_class in enumerate(class_names):
                count = true_row[pred_index] if pred_index < len(true_row) else ""
                rows.append(
                    {
                        "source": "ours",
                        "scenario": scenario,
                        "paper_figure": "",
                        "paper_table": TASK_TO_PAPER_TABLE[task_id],
                        "task_id": task_id,
                        "true_class": true_class,
                        "predicted_class": predicted_class,
                        "count": count,
                        "notes": "",
                    }
                )
    return rows


def build_confusion_rows(*, scenario: str, runs_dir: Path) -> list[CsvRow]:
    rows = build_paper_confusion_rows()
    rows.extend(build_ours_confusion_rows(scenario=scenario, runs_dir=runs_dir))
    return rows


def build_support_rows(*, scenario: str, runs_dir: Path) -> list[CsvRow]:
    rows: list[CsvRow] = []
    for row in PAPER_CLASSIFICATION_ROWS:
        rows.append(
            {
                "scenario": "paper",
                "source": row["source"],
                "paper_table": row["paper_table"],
                "task_id": row["task_id"],
                "class_name": row["class_name"],
                "support": row["support"],
                "notes": row["notes"],
            }
        )

    for task_id, (figure, table, class_names, matrix) in PAPER_CONFUSION_MATRICES.items():
        for class_index, class_name in enumerate(class_names):
            support = sum(matrix[class_index])
            rows.append(
                {
                    "scenario": "paper",
                    "source": "paper_figure",
                    "paper_table": table,
                    "task_id": task_id,
                    "class_name": class_name,
                    "support": support,
                    "notes": f"Row sum from {figure}.",
                }
            )

    for row in build_ours_classification_rows(scenario=scenario, runs_dir=runs_dir):
        rows.append(
            {
                "scenario": scenario,
                "source": "ours",
                "paper_table": row["paper_table"],
                "task_id": row["task_id"],
                "class_name": row["class_name"],
                "support": row["support"],
                "notes": "",
            }
        )
    for row in build_ours_standard_rbp_classification_rows(scenario=scenario, runs_dir=runs_dir):
        rows.append(
            {
                "scenario": scenario,
                "source": "ours",
                "paper_table": row["paper_table"],
                "task_id": row["task_id"],
                "class_name": row["class_name"],
                "support": row["support"],
                "notes": "",
            }
        )
    for row in build_ours_smote_classification_rows(scenario=scenario, runs_dir=runs_dir):
        rows.append(
            {
                "scenario": scenario,
                "source": "ours",
                "paper_table": row["paper_table"],
                "task_id": row["task_id"],
                "class_name": row["class_name"],
                "support": row["support"],
                "notes": "",
            }
        )
    if scenario in {"paper_literal_80_10_10", "val_as_test_80_20", "fixture_smoke"}:
        for row in build_ours_label_swap_classification_rows(scenario=scenario, runs_dir=runs_dir):
            rows.append(
                {
                    "scenario": "label_swap_80_20",
                    "source": "ours_label_swap",
                    "paper_table": row["paper_table"],
                    "task_id": row["task_id"],
                    "class_name": row["class_name"],
                    "support": row["support"],
                    "notes": row["notes"],
                }
            )
    return rows


def build_history_rows(*, scenario: str, runs_dir: Path) -> list[CsvRow]:
    rows: list[CsvRow] = []
    for task_id in TASKS:
        run_dir = get_task_run_dir(runs_dir, task_id)
        history_payload = get_history_payload(run_dir)
        for item in history_payload:
            item_payload = require_mapping(item, path=f"{run_dir}/history.json[]")
            train_metrics = optional_mapping(item_payload.get("train_metrics"))
            val_metrics = optional_mapping(item_payload.get("val_metrics"))
            rows.append(
                {
                    "scenario": scenario,
                    "task_id": task_id,
                    "epoch": item_payload.get("epoch", ""),
                    "train_loss": item_payload.get("train_loss", ""),
                    "val_loss": item_payload.get("val_loss", ""),
                    "train_accuracy": train_metrics.get("accuracy", ""),
                    "val_accuracy": val_metrics.get("accuracy", ""),
                    "train_macro_f1": train_metrics.get("macro_f1", ""),
                    "val_macro_f1": val_metrics.get("macro_f1", ""),
                }
            )
    for task_id in STANDARD_RBP_TASKS:
        run_dir = get_standard_rbp_task_run_dir(runs_dir, task_id)
        history_payload = get_history_payload(run_dir)
        for item in history_payload:
            item_payload = require_mapping(item, path=f"{run_dir}/history.json[]")
            train_metrics = optional_mapping(item_payload.get("train_metrics"))
            val_metrics = optional_mapping(item_payload.get("val_metrics"))
            rows.append(
                {
                    "scenario": scenario,
                    "task_id": f"standard_rbp/{task_id}",
                    "epoch": item_payload.get("epoch", ""),
                    "train_loss": item_payload.get("train_loss", ""),
                    "val_loss": item_payload.get("val_loss", ""),
                    "train_accuracy": train_metrics.get("accuracy", ""),
                    "val_accuracy": val_metrics.get("accuracy", ""),
                    "train_macro_f1": train_metrics.get("macro_f1", ""),
                    "val_macro_f1": val_metrics.get("macro_f1", ""),
                }
            )
    for task_id in SMOTE_TASKS:
        run_dir = get_smote_task_run_dir(runs_dir, task_id)
        history_payload = get_history_payload(run_dir)
        for item in history_payload:
            item_payload = require_mapping(item, path=f"{run_dir}/history.json[]")
            train_metrics = optional_mapping(item_payload.get("train_metrics"))
            val_metrics = optional_mapping(item_payload.get("val_metrics"))
            rows.append(
                {
                    "scenario": scenario,
                    "task_id": f"smote/{task_id}",
                    "epoch": item_payload.get("epoch", ""),
                    "train_loss": item_payload.get("train_loss", ""),
                    "val_loss": item_payload.get("val_loss", ""),
                    "train_accuracy": train_metrics.get("accuracy", ""),
                    "val_accuracy": val_metrics.get("accuracy", ""),
                    "train_macro_f1": train_metrics.get("macro_f1", ""),
                    "val_macro_f1": val_metrics.get("macro_f1", ""),
                }
            )
    if scenario in {"paper_literal_80_10_10", "val_as_test_80_20", "fixture_smoke"}:
        for task_id in LABEL_SWAP_TASKS:
            run_dir = get_label_swap_task_run_dir(runs_dir, task_id, scenario=scenario)
            history_payload = get_history_payload(run_dir)
            for item in history_payload:
                item_payload = require_mapping(item, path=f"{run_dir}/history.json[]")
                train_metrics = optional_mapping(item_payload.get("train_metrics"))
                val_metrics = optional_mapping(item_payload.get("val_metrics"))
                rows.append(
                    {
                        "scenario": "label_swap_80_20",
                        "task_id": f"label_swap_80_20/{task_id}",
                        "epoch": item_payload.get("epoch", ""),
                        "train_loss": item_payload.get("train_loss", ""),
                        "val_loss": item_payload.get("val_loss", ""),
                        "train_accuracy": train_metrics.get("accuracy", ""),
                        "val_accuracy": val_metrics.get("accuracy", ""),
                        "train_macro_f1": train_metrics.get("macro_f1", ""),
                        "val_macro_f1": val_metrics.get("macro_f1", ""),
                    }
                )
    return rows


def build_model_parameter_rows(*, scenario: str, runs_dir: Path) -> list[CsvRow]:
    rows: list[CsvRow] = [dict(row) for row in PAPER_MODEL_PARAMETER_ROWS]
    for num_classes in (3, 2):
        hyperparameters = build_paper_hyperparameters(num_classes=num_classes)
        model = build_paper_model(hyperparameters=hyperparameters)
        total_parameters = sum(parameter.numel() for parameter in model.parameters())
        trainable_parameters = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        non_trainable_parameters = total_parameters - trainable_parameters
        for parameter_name, value in (
            ("total_parameters", total_parameters),
            ("trainable_parameters", trainable_parameters),
            ("non_trainable_parameters", non_trainable_parameters),
        ):
            rows.append(
                {
                    "source": "ours_model",
                    "paper_table": "Table 2",
                    "scenario": scenario,
                    "task_id": f"num_classes_{num_classes}",
                    "parameter": parameter_name,
                    "value": value,
                    "size": "",
                    "notes": "Parameter count computed from the project TCN-LSTM implementation.",
                }
            )
    for task_id in TASKS:
        run_dir = get_task_run_dir(runs_dir, task_id)
        run_payload = get_run_payload(run_dir)
        hyperparameters = optional_mapping(run_payload.get("hyperparameters"))
        for key, value in hyperparameters.items():
            rows.append(
                {
                    "source": "ours",
                    "paper_table": "",
                    "scenario": scenario,
                    "task_id": task_id,
                    "parameter": key,
                    "value": json.dumps(value, ensure_ascii=False),
                    "size": "",
                    "notes": "Hyperparameter from run.json, not a parameter-count summary.",
                }
            )
    for task_id in STANDARD_RBP_TASKS:
        run_dir = get_standard_rbp_task_run_dir(runs_dir, task_id)
        run_payload = get_run_payload(run_dir)
        hyperparameters = optional_mapping(run_payload.get("hyperparameters"))
        for key, value in hyperparameters.items():
            rows.append(
                {
                    "source": "ours",
                    "paper_table": "",
                    "scenario": scenario,
                    "task_id": f"standard_rbp/{task_id}",
                    "parameter": key,
                    "value": json.dumps(value, ensure_ascii=False),
                    "size": "",
                    "notes": "Hyperparameter from run.json, not a parameter-count summary.",
                }
            )
    for task_id in SMOTE_TASKS:
        run_dir = get_smote_task_run_dir(runs_dir, task_id)
        run_payload = get_run_payload(run_dir)
        hyperparameters = optional_mapping(run_payload.get("hyperparameters"))
        for key, value in hyperparameters.items():
            rows.append(
                {
                    "source": "ours",
                    "paper_table": "",
                    "scenario": scenario,
                    "task_id": f"smote/{task_id}",
                    "parameter": key,
                    "value": json.dumps(value, ensure_ascii=False),
                    "size": "",
                    "notes": "Hyperparameter from run.json, not a parameter-count summary.",
                }
            )
        smote_config = optional_mapping(run_payload.get("smote"))
        for key, value in smote_config.items():
            rows.append(
                {
                    "source": "ours",
                    "paper_table": "",
                    "scenario": scenario,
                    "task_id": f"smote/{task_id}",
                    "parameter": f"smote.{key}",
                    "value": json.dumps(value, ensure_ascii=False),
                    "size": "",
                    "notes": "SMOTE setting from run.json.",
                }
            )
    return rows


def build_accuracy_rows(*, scenario: str, runs_dir: Path) -> list[CsvRow]:
    rows: list[CsvRow] = [dict(row) for row in PAPER_ACCURACY_ROWS]
    for task_id in TASKS:
        metrics_payload = get_metrics_payload(get_task_run_dir(runs_dir, task_id))
        metrics = optional_mapping(metrics_payload.get("metrics"))
        accuracy = metrics.get("accuracy")
        if isinstance(accuracy, int | float):
            rows.append(
                {
                    "source": "ours",
                    "paper_table": "Table 7",
                    "experiment_kind": "modified_rbp",
                    "scenario": scenario,
                    "task_id": task_id,
                    "metric": "accuracy",
                    "value": accuracy,
                    "value_scale": "decimal",
                    "notes": "",
                }
            )
    for task_id in STANDARD_RBP_TASKS:
        metrics_payload = get_metrics_payload(get_standard_rbp_task_run_dir(runs_dir, task_id))
        metrics = optional_mapping(metrics_payload.get("metrics"))
        accuracy = metrics.get("accuracy")
        if isinstance(accuracy, int | float):
            rows.append(
                {
                    "source": "ours",
                    "paper_table": STANDARD_RBP_TASK_TO_PAPER_TABLE[task_id],
                    "experiment_kind": "standard_rbp",
                    "scenario": scenario,
                    "task_id": task_id,
                    "metric": "accuracy",
                    "value": accuracy,
                    "value_scale": "decimal",
                    "notes": "",
                }
            )
    for task_id in SMOTE_TASKS:
        metrics_payload = get_metrics_payload(get_smote_task_run_dir(runs_dir, task_id))
        metrics = optional_mapping(metrics_payload.get("metrics"))
        accuracy = metrics.get("accuracy")
        if isinstance(accuracy, int | float):
            rows.append(
                {
                    "source": "ours",
                    "paper_table": SMOTE_TASK_TO_PAPER_TABLE[task_id],
                    "experiment_kind": "smote_modified_rbp",
                    "scenario": scenario,
                    "task_id": task_id,
                    "metric": "accuracy",
                    "value": accuracy,
                    "value_scale": "decimal",
                    "notes": "",
                }
            )
    return rows


def comparable_difference(
    *,
    paper_value: object,
    paper_scale: object,
    ours_value: object,
    ours_scale: object,
) -> float | str:
    if not isinstance(paper_value, int | float) or not isinstance(ours_value, int | float):
        return ""
    if paper_scale == ours_scale:
        return float(ours_value) - float(paper_value)
    if paper_scale == "percent" and ours_scale == "decimal":
        return float(ours_value) * 100.0 - float(paper_value)
    if paper_scale == "decimal" and ours_scale == "percent":
        return float(ours_value) / 100.0 - float(paper_value)
    return ""


def parse_float_cell(row: dict[str, str], key: str) -> float | None:
    value = row.get(key, "")
    if value == "":
        return None
    return float(value)


def build_ours_kfold_rows(*, scenario: str, runs_dir: Path) -> list[CsvRow]:
    rows: list[CsvRow] = []
    for task_id, paper_table in KFOLD_TASK_TO_PAPER_TABLE.items():
        fold_summary_path = get_kfold_task_run_dir(runs_dir, task_id) / "fold_summary.csv"
        for row in read_dict_csv(fold_summary_path):
            train_accuracy = parse_float_cell(row, "train_accuracy")
            test_accuracy = parse_float_cell(row, "test_accuracy")
            rows.append(
                {
                    "source": "ours",
                    "scenario": scenario,
                    "paper_table": paper_table,
                    "experiment_kind": "kfold_modified_rbp",
                    "task_id": task_id,
                    "fold": row.get("fold", ""),
                    "train_accuracy_percent": "" if train_accuracy is None else train_accuracy * 100.0,
                    "test_accuracy_percent": "" if test_accuracy is None else test_accuracy * 100.0,
                    "notes": "",
                }
            )
    return rows


def build_kfold_accuracy_rows(*, scenario: str, runs_dir: Path) -> list[CsvRow]:
    rows: list[CsvRow] = []
    for row in PAPER_KFOLD_ROWS:
        next_row = dict(row)
        next_row.setdefault("scenario", "paper")
        rows.append(next_row)
    rows.extend(build_ours_kfold_rows(scenario=scenario, runs_dir=runs_dir))
    return rows


def build_ours_kfold_lookup(*, scenario: str, runs_dir: Path) -> dict[tuple[str, int, str], object]:
    lookup: dict[tuple[str, int, str], object] = {}
    for row in build_ours_kfold_rows(scenario=scenario, runs_dir=runs_dir):
        fold_value = row.get("fold", "")
        if fold_value == "":
            continue
        fold = int(fold_value)
        for metric in ("train_accuracy_percent", "test_accuracy_percent"):
            lookup[(str(row["task_id"]), fold, metric)] = row.get(metric, "")
    return lookup


def build_paper_vs_ours_rows(*, scenario: str, runs_dir: Path) -> list[CsvRow]:
    rows: list[CsvRow] = []
    ours_rows = build_ours_classification_rows(scenario=scenario, runs_dir=runs_dir)
    ours_rows.extend(build_ours_standard_rbp_classification_rows(scenario=scenario, runs_dir=runs_dir))
    ours_rows.extend(build_ours_smote_classification_rows(scenario=scenario, runs_dir=runs_dir))
    ours_metrics: dict[tuple[str, str, str, str], object] = {}
    ours_scales: dict[tuple[str, str, str, str], object] = {}
    for row in ours_rows:
        for metric in ("precision", "recall", "f1", "sensitivity", "specificity", "support"):
            ours_metrics[
                (str(row["experiment_kind"]), str(row["task_id"]), str(row["class_name"]), metric)
            ] = row.get(metric, "")
            ours_scales[
                (str(row["experiment_kind"]), str(row["task_id"]), str(row["class_name"]), metric)
            ] = row.get("value_scale", "")

    for paper_row in PAPER_CLASSIFICATION_ROWS:
        for metric in ("precision", "recall", "f1", "sensitivity", "specificity", "support"):
            paper_value = paper_row.get(metric, "")
            if paper_value == "":
                continue
            key = (str(paper_row["experiment_kind"]), str(paper_row["task_id"]), str(paper_row["class_name"]), metric)
            ours_value = ours_metrics.get(key, "")
            ours_scale = ours_scales.get(key, "")
            ours_available = ours_value != ""
            difference = comparable_difference(
                paper_value=paper_value,
                paper_scale=paper_row["value_scale"],
                ours_value=ours_value,
                ours_scale=ours_scale,
            )
            rows.append(
                {
                    "scenario": scenario,
                    "paper_table": paper_row["paper_table"],
                    "experiment_kind": paper_row["experiment_kind"],
                    "task_id": paper_row["task_id"],
                    "class_name": paper_row["class_name"],
                    "metric": metric,
                    "paper_value": paper_value,
                    "paper_value_scale": paper_row["value_scale"],
                    "ours_value": ours_value if ours_available else "",
                    "ours_value_scale": ours_scale if ours_available else "",
                    "difference": difference,
                    "ours_available": str(ours_available).lower(),
                    "notes": paper_row["notes"],
                }
            )

    ours_accuracy: dict[tuple[str, str], object] = {}
    for task_id in TASKS:
        metrics_payload = get_metrics_payload(get_task_run_dir(runs_dir, task_id))
        metrics = optional_mapping(metrics_payload.get("metrics"))
        if isinstance(metrics.get("accuracy"), int | float):
            ours_accuracy[("modified_rbp", task_id)] = metrics["accuracy"]
    for task_id in STANDARD_RBP_TASKS:
        metrics_payload = get_metrics_payload(get_standard_rbp_task_run_dir(runs_dir, task_id))
        metrics = optional_mapping(metrics_payload.get("metrics"))
        if isinstance(metrics.get("accuracy"), int | float):
            ours_accuracy[("standard_rbp", task_id)] = metrics["accuracy"]
    for task_id in SMOTE_TASKS:
        metrics_payload = get_metrics_payload(get_smote_task_run_dir(runs_dir, task_id))
        metrics = optional_mapping(metrics_payload.get("metrics"))
        if isinstance(metrics.get("accuracy"), int | float):
            ours_accuracy[("smote_modified_rbp", task_id)] = metrics["accuracy"]

    for paper_row in PAPER_ACCURACY_ROWS:
        task_id = str(paper_row["task_id"])
        ours_value = ours_accuracy.get((str(paper_row["experiment_kind"]), task_id), "")
        ours_available = ours_value != ""
        difference = comparable_difference(
            paper_value=paper_row["value"],
            paper_scale=paper_row["value_scale"],
            ours_value=ours_value,
            ours_scale="decimal" if ours_available else "",
        )
        rows.append(
            {
                "scenario": scenario,
                "paper_table": paper_row["paper_table"],
                "experiment_kind": paper_row["experiment_kind"],
                "task_id": task_id,
                "class_name": "",
                "metric": paper_row["metric"],
                "paper_value": paper_row["value"],
                "paper_value_scale": paper_row["value_scale"],
                "ours_value": ours_value if ours_available else "",
                "ours_value_scale": "decimal" if ours_available else "",
                "difference": difference,
                "ours_available": str(ours_available).lower(),
                "notes": paper_row["notes"],
            }
        )

    ours_kfold = build_ours_kfold_lookup(scenario=scenario, runs_dir=runs_dir)
    for row in PAPER_KFOLD_ROWS:
        for metric in ("train_accuracy_percent", "test_accuracy_percent"):
            ours_value = ours_kfold.get((str(row["task_id"]), int(row["fold"]), metric), "")
            ours_available = ours_value != ""
            difference = ""
            if ours_available and isinstance(ours_value, int | float):
                difference = float(ours_value) - float(row[metric])
            rows.append(
                {
                    "scenario": scenario,
                    "paper_table": row["paper_table"],
                    "experiment_kind": row["experiment_kind"],
                    "task_id": row["task_id"],
                    "class_name": "",
                    "metric": f"fold_{row['fold']}_{metric}",
                    "paper_value": row[metric],
                    "paper_value_scale": "percent",
                    "ours_value": ours_value,
                    "ours_value_scale": "percent" if ours_available else "",
                    "difference": difference,
                    "ours_available": str(ours_available).lower(),
                    "notes": "" if ours_available else "K-fold run output is missing.",
                }
            )
    return rows


def build_protocol_manifest_rows(*, scenario: str, runs_dir: Path) -> list[CsvRow]:
    paper_hyperparameters = build_paper_hyperparameters(num_classes=3)
    if scenario == "paper_literal_80_10_10":
        split_protocol = "80/10/10 epoch-level stratified split"
        evaluation_protocol = "reported metrics use independent split.test partition"
        smote_protocol = "simple SMOTE applied to train, val, and test partitions for Table 8-9 shape"
        label_swap_evidence = "cfgs/ds004504_rbp_paper/label_swap/*.yaml"
    elif scenario == "val_as_test_80_20":
        split_protocol = "80/20 epoch-level stratified split"
        evaluation_protocol = "reported metrics use validation partition as test output, matching paper support evidence"
        smote_protocol = "simple SMOTE applied to train and reported val/test partitions for Table 8-9 shape"
        label_swap_evidence = "cfgs/ds004504_rbp_paper/label_swap_80_20/*.yaml"
    elif scenario == "fixture_smoke":
        split_protocol = "fixture-only synthetic run artifacts"
        evaluation_protocol = "fixture metrics; not a paper result"
        smote_protocol = "fixture-only SMOTE-shaped artifacts"
        label_swap_evidence = "scripts/ds004504_rbp_paper/make_fixture_runs.py"
    else:
        split_protocol = "unknown"
        evaluation_protocol = "unknown"
        smote_protocol = "unknown"
        label_swap_evidence = "unknown"

    return [
        {
            "scenario": scenario,
            "component": "raw_dataset",
            "value": "OpenNeuro ds004504 v1.0.5",
            "evidence": "cfgs/ds004504_rbp_paper/base.yaml processing.download_raw_dataset.tag",
            "notes": "Paper data availability statement points to openneuro.ds004504.v1.0.5.",
        },
        {
            "scenario": scenario,
            "component": "preprocessing_source",
            "value": "derivatives/sub-*/eeg/*_task-eyesclosed_eeg.set",
            "evidence": "src/ecg/data/ds004504_rbp_paper.py EEG_GLOB",
            "notes": "Uses preprocessed derivative EEGLAB .set files.",
        },
        {
            "scenario": scenario,
            "component": "epoching",
            "value": "6 seconds with 50% overlap",
            "evidence": "src/ecg/data/ds004504_rbp_paper.py EPOCH_SEC and OVERLAP",
            "notes": "",
        },
        {
            "scenario": scenario,
            "component": "modified_rbp_bands",
            "value": "delta 0.5-4, theta 4-8, alpha 8-16, zaeta 16-24, beta 24-30, gamma 30-45",
            "evidence": "src/ecg/data/ds004504_rbp_paper.py MODIFIED_RBP_BANDS",
            "notes": "Used for Tables 3-11.",
        },
        {
            "scenario": scenario,
            "component": "standard_rbp_bands",
            "value": "delta 0.5-4, theta 4-8, alpha 8-13, beta 13-25, gamma 25-45",
            "evidence": "src/ecg/data/ds004504_rbp_paper.py STANDARD_RBP_BANDS",
            "notes": "Used for Tables 12-13.",
        },
        {
            "scenario": scenario,
            "component": "normalization",
            "value": "paper-style full-task min-max normalization before split",
            "evidence": "src/ecg/training/ds004504_rbp_paper/factory.py build_paper_experiment",
            "notes": "This follows the paper wording but leaks evaluation distribution statistics.",
        },
        {
            "scenario": scenario,
            "component": "split",
            "value": split_protocol,
            "evidence": "cfgs/ds004504_rbp_paper/* and run.json split_fractions",
            "notes": "",
        },
        {
            "scenario": scenario,
            "component": "evaluation",
            "value": evaluation_protocol,
            "evidence": "run.json evaluation.test_source and test_metrics.json source",
            "notes": "",
        },
        {
            "scenario": scenario,
            "component": "model",
            "value": "TCN-LSTM, two TCN blocks, 32 channels, kernel 7, LSTM 64, Dense 128/192/256",
            "evidence": "src/ecg/training/ds004504_rbp_paper/factory.py paper hyperparameters",
            "notes": "",
        },
        {
            "scenario": scenario,
            "component": "model_hyperparameters",
            "value": (
                f"input_dim={paper_hyperparameters.input_dim}, "
                "num_classes=task-dependent, "
                f"tcn_channels={paper_hyperparameters.tcn_channels}, "
                f"tcn_kernel_size={paper_hyperparameters.tcn_kernel_size}, "
                f"tcn_dilations={list(paper_hyperparameters.tcn_dilations)}, "
                f"tcn_dropout={paper_hyperparameters.tcn_dropout}, "
                f"lstm_hidden_dim={paper_hyperparameters.lstm_hidden_dim}, "
                f"dense_hidden_dims={list(paper_hyperparameters.dense_hidden_dims)}, "
                f"dense_dropout={paper_hyperparameters.dense_dropout}"
            ),
            "evidence": "src/ecg/training/ds004504_rbp_paper/factory.py build_paper_hyperparameters",
            "notes": "num_classes is 3 for multiclass tasks and 2 for binary tasks.",
        },
        {
            "scenario": scenario,
            "component": "optimizer",
            "value": "Adam learning_rate=0.0001 batch_size=32",
            "evidence": "src/ecg/training/ds004504_rbp_paper/factory.py PAPER_LEARNING_RATE and PAPER_BATCH_SIZE",
            "notes": "",
        },
        {
            "scenario": scenario,
            "component": "training_runtime",
            "value": "epochs=100, seed=randomly resolved at runtime unless configured, num_workers=4, device=cuda",
            "evidence": "cfgs/ds004504_rbp_paper/base.yaml training/runtime and run.json resolved_seed",
            "notes": "run.json records the actual resolved seed for each run.",
        },
        {
            "scenario": scenario,
            "component": "smote",
            "value": smote_protocol,
            "evidence": "scripts/ds004504_rbp_paper/train_smote.py and cfgs/*/smote/*.yaml",
            "notes": "Paper does not provide source code or exact SMOTE placement.",
        },
        {
            "scenario": scenario,
            "component": "kfold",
            "value": "5-fold epoch-level stratified k-fold",
            "evidence": "scripts/ds004504_rbp_paper/train_kfold.py",
            "notes": "Paper does not specify subject-wise folds.",
        },
        {
            "scenario": scenario,
            "component": "label_swap_audit",
            "value": "FTD/Healthy target-label swap audit for Table 3 and Table 6",
            "evidence": label_swap_evidence,
            "notes": "Audit-only protocol, not the corrected dataset protocol.",
        },
        {
            "scenario": scenario,
            "component": "runs_dir",
            "value": str(runs_dir),
            "evidence": "make_report_csv.py --runs-dir/default_runs_dir",
            "notes": "",
        },
    ]


def build_issue_summary_rows(*, scenario: str) -> list[CsvRow]:
    return [
        {
            "scenario": scenario,
            "issue_id": "split_support_mismatch",
            "severity": "high",
            "affected_tables": "Table 3, Table 4, Table 5, Table 6, Table 7",
            "paper_observation": "Paper text claims 80/10/10 train/validation/test split.",
            "reproduction_implication": "Reported supports are close to a 20% holdout, not a 10% test split.",
            "project_handling": "Provide both paper_literal_80_10_10 and val_as_test_80_20 scenarios; default report uses paper_literal_80_10_10 to follow the paper text, while val_as_test_80_20 is retained for paper-inferred support comparison.",
        },
        {
            "scenario": scenario,
            "issue_id": "ftd_healthy_support_swap",
            "severity": "high",
            "affected_tables": "Table 3, Table 4, Table 6",
            "paper_observation": "FTD and Healthy supports align much better if FTD/Healthy are swapped.",
            "reproduction_implication": "Per-class interpretation of model behavior may be reversed for FTD and Healthy.",
            "project_handling": "Provide label_swap and label_swap_80_20 audit runs and include ours_label_swap rows in Table 3 and Table 6 reports.",
        },
        {
            "scenario": scenario,
            "issue_id": "table6_healthy_support_typo",
            "severity": "medium",
            "affected_tables": "Table 6",
            "paper_observation": "Table 6 reports Healthy support 1596, but Figure 4d and recall imply 1106.",
            "reproduction_implication": "Table support column cannot be used blindly; Figure 4 row sums are needed for consistency checks.",
            "project_handling": "support_comparison.csv includes both paper table support and paper figure row-sum support.",
        },
        {
            "scenario": scenario,
            "issue_id": "smote_accuracy_inconsistency",
            "severity": "medium",
            "affected_tables": "Table 8",
            "paper_observation": "Paper text reports SMOTE accuracy 77.45%, but Table 8 equal supports and recalls imply a different value.",
            "reproduction_implication": "SMOTE headline accuracy should be compared separately from row-level metrics.",
            "project_handling": "accuracy.csv and paper_vs_ours.csv keep Table 8 text accuracy separate from row metrics.",
        },
        {
            "scenario": scenario,
            "issue_id": "unspecified_smote_placement",
            "severity": "medium",
            "affected_tables": "Table 8, Table 9",
            "paper_observation": "Paper does not specify whether SMOTE was applied before split, after split, or only to training data.",
            "reproduction_implication": "Exact SMOTE reproduction is not uniquely determined from the paper.",
            "project_handling": "Use explicit simple SMOTE with documented partitions in protocol_manifest.csv.",
        },
        {
            "scenario": scenario,
            "issue_id": "epoch_level_leakage_risk",
            "severity": "high",
            "affected_tables": "Table 3 through Table 13",
            "paper_observation": "Paper describes epoch-level split and full min-max normalization before split.",
            "reproduction_implication": "Results may be optimistic because adjacent epochs and normalization statistics can leak across partitions.",
            "project_handling": "Reproduce paper-style protocol explicitly and record leakage risk in protocol_manifest.csv; corrected subject-wise protocol is outside Table 1-14 paper reproduction.",
        },
    ]


def build_table_summary_rows(*, scenario: str, runs_dir: Path) -> list[CsvRow]:
    table_purpose = {
        "Table 1": "model architecture",
        "Table 2": "model parameter summary",
        "Table 3": "modified RBP multiclass metrics",
        "Table 4": "modified RBP AD+FTD vs Healthy metrics",
        "Table 5": "modified RBP AD vs Healthy metrics",
        "Table 6": "modified RBP FTD vs Healthy metrics",
        "Table 7": "modified RBP task accuracy",
        "Table 8": "SMOTE multiclass metrics",
        "Table 9": "SMOTE AD vs Healthy metrics",
        "Table 10": "5-fold multiclass accuracy",
        "Table 11": "5-fold AD vs Healthy accuracy",
        "Table 12": "standard RBP multiclass metrics",
        "Table 13": "standard RBP AD vs Healthy metrics",
        "Table 14": "literature comparison",
    }
    table_notes = {
        "Table 1": "Architecture table; no accuracy/support metrics.",
        "Table 2": "Parameter-count table; see model_parameters.csv for paper and ours counts.",
        "Table 14": "Literature comparison table; no ours-side training output.",
    }
    rows: list[CsvRow] = []
    paper_vs_ours = build_paper_vs_ours_rows(scenario=scenario, runs_dir=runs_dir)
    for table in [f"Table {index}" for index in range(1, 15)]:
        table_rows = [row for row in paper_vs_ours if row.get("paper_table") == table]
        paper_metrics = [row for row in table_rows if row.get("metric") == "accuracy"]
        support_rows = [row for row in table_rows if row.get("metric") == "support"]
        paper_accuracy = "; ".join(
            f"{row.get('task_id', '')}={row.get('paper_value', '')}{'%' if row.get('paper_value_scale') == 'percent' else ''}"
            for row in paper_metrics
        )
        ours_accuracy = "; ".join(
            f"{row.get('task_id', '')}={row.get('ours_value', '')}{'%' if row.get('ours_value_scale') == 'percent' else ''}"
            for row in paper_metrics
            if row.get("ours_available") == "true"
        )
        paper_support = "; ".join(
            f"{row.get('task_id', '')}/{row.get('class_name', '')}={row.get('paper_value', '')}"
            for row in support_rows
        )
        ours_support = "; ".join(
            f"{row.get('task_id', '')}/{row.get('class_name', '')}={row.get('ours_value', '')}"
            for row in support_rows
            if row.get("ours_available") == "true"
        )
        rows.append(
            {
                "scenario": scenario,
                "paper_table": table,
                "purpose": table_purpose[table],
                "paper_accuracy": paper_accuracy,
                "ours_accuracy": ours_accuracy,
                "paper_support": paper_support,
                "ours_support": ours_support,
                "notes": table_notes.get(table, "See paper_vs_ours.csv for per-metric differences."),
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    runs_dir = args.runs_dir if args.runs_dir is not None else default_runs_dir(args.scenario)
    output_dir = args.output_dir if args.output_dir is not None else default_output_dir(args.scenario)

    write_csv(
        output_dir / "paper_model_architecture.csv",
        ("paper_table", "layer_order", "layer_type", "output_shape", "parameter_count", "connected_to"),
        [dict(row) for row in PAPER_MODEL_ARCHITECTURE_ROWS],
    )
    write_csv(
        output_dir / "model_parameters.csv",
        ("source", "paper_table", "scenario", "task_id", "parameter", "value", "size", "notes"),
        build_model_parameter_rows(scenario=args.scenario, runs_dir=runs_dir),
    )
    write_csv(
        output_dir / "run_summary.csv",
        (
            "scenario",
            "task_id",
            "status",
            "run_dir",
            "experiment_id",
            "config_path",
            "seed",
            "epochs",
            "device",
            "test_source",
            "train_fraction",
            "val_fraction",
            "test_fraction",
            "accuracy",
            "macro_precision",
            "macro_recall",
            "macro_f1",
            "support",
        ),
        build_run_summary_rows(scenario=args.scenario, runs_dir=runs_dir),
    )
    write_csv(
        output_dir / "classification_metrics.csv",
        (
            "source",
            "paper_table",
            "experiment_kind",
            "scenario",
            "task_id",
            "class_name",
            "precision",
            "recall",
            "f1",
            "sensitivity",
            "specificity",
            "support",
            "value_scale",
            "notes",
        ),
        build_classification_rows(scenario=args.scenario, runs_dir=runs_dir),
    )
    write_csv(
        output_dir / "confusion_matrices.csv",
        (
            "source",
            "scenario",
            "paper_figure",
            "paper_table",
            "task_id",
            "true_class",
            "predicted_class",
            "count",
            "notes",
        ),
        build_confusion_rows(scenario=args.scenario, runs_dir=runs_dir),
    )
    write_csv(
        output_dir / "support_comparison.csv",
        ("scenario", "source", "paper_table", "task_id", "class_name", "support", "notes"),
        build_support_rows(scenario=args.scenario, runs_dir=runs_dir),
    )
    write_csv(
        output_dir / "history.csv",
        (
            "scenario",
            "task_id",
            "epoch",
            "train_loss",
            "val_loss",
            "train_accuracy",
            "val_accuracy",
            "train_macro_f1",
            "val_macro_f1",
        ),
        build_history_rows(scenario=args.scenario, runs_dir=runs_dir),
    )
    write_csv(
        output_dir / "accuracy.csv",
        ("source", "paper_table", "experiment_kind", "scenario", "task_id", "metric", "value", "value_scale", "notes"),
        build_accuracy_rows(scenario=args.scenario, runs_dir=runs_dir),
    )
    write_csv(
        output_dir / "kfold_accuracy.csv",
        (
            "source",
            "scenario",
            "paper_table",
            "experiment_kind",
            "task_id",
            "fold",
            "train_accuracy_percent",
            "test_accuracy_percent",
            "notes",
        ),
        build_kfold_accuracy_rows(scenario=args.scenario, runs_dir=runs_dir),
    )
    write_csv(
        output_dir / "paper_vs_ours.csv",
        (
            "scenario",
            "paper_table",
            "experiment_kind",
            "task_id",
            "class_name",
            "metric",
            "paper_value",
            "paper_value_scale",
            "ours_value",
            "ours_value_scale",
            "difference",
            "ours_available",
            "notes",
        ),
        build_paper_vs_ours_rows(scenario=args.scenario, runs_dir=runs_dir),
    )
    write_csv(
        output_dir / "table_summary.csv",
        (
            "scenario",
            "paper_table",
            "purpose",
            "paper_accuracy",
            "ours_accuracy",
            "paper_support",
            "ours_support",
            "notes",
        ),
        build_table_summary_rows(scenario=args.scenario, runs_dir=runs_dir),
    )
    write_csv(
        output_dir / "protocol_manifest.csv",
        ("scenario", "component", "value", "evidence", "notes"),
        build_protocol_manifest_rows(scenario=args.scenario, runs_dir=runs_dir),
    )
    write_csv(
        output_dir / "issue_summary.csv",
        (
            "scenario",
            "issue_id",
            "severity",
            "affected_tables",
            "paper_observation",
            "reproduction_implication",
            "project_handling",
        ),
        build_issue_summary_rows(scenario=args.scenario),
    )
    write_csv(
        output_dir / "literature_comparison.csv",
        ("paper_table", "paper", "model", "accuracy", "feature_engineering", "xai"),
        [dict(row) for row in PAPER_LITERATURE_ROWS],
    )
    print(f"Wrote report CSV files to {output_dir}")


if __name__ == "__main__":
    main()
