from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path


TASK_CLASSES = {
    "multiclass": ("alzheimer", "frontotemporal_dementia", "healthy_control"),
    "ad_ftd_vs_healthy": ("alzheimer_or_frontotemporal_dementia", "healthy_control"),
    "ad_vs_healthy": ("alzheimer", "healthy_control"),
    "ftd_vs_healthy": ("frontotemporal_dementia", "healthy_control"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create tiny fixture run artifacts for report-chain smoke testing.")
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=Path("data/runs/ds004504_rbp_paper/fixture_smoke"),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")


def class_metrics(class_names: tuple[str, ...], confusion: list[list[int]]) -> dict[str, object]:
    total = sum(sum(row) for row in confusion)
    correct = sum(confusion[index][index] for index in range(len(class_names)))
    per_class: list[dict[str, object]] = []
    for index, class_name in enumerate(class_names):
        tp = confusion[index][index]
        fp = sum(row[index] for row in confusion) - tp
        fn = sum(confusion[index]) - tp
        tn = total - tp - fp - fn
        precision = tp / (tp + fp) if tp + fp > 0 else 0.0
        recall = tp / (tp + fn) if tp + fn > 0 else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall > 0.0 else 0.0
        specificity = tn / (tn + fp) if tn + fp > 0 else 0.0
        per_class.append(
            {
                "class_index": index,
                "class_name": class_name,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "sensitivity": recall,
                "specificity": specificity,
                "support": sum(confusion[index]),
            }
        )
    return {
        "accuracy": correct / total,
        "macro_precision": sum(float(row["precision"]) for row in per_class) / len(per_class),
        "macro_recall": sum(float(row["recall"]) for row in per_class) / len(per_class),
        "macro_f1": sum(float(row["f1"]) for row in per_class) / len(per_class),
        "confusion_matrix": confusion,
        "per_class": per_class,
        "support": total,
    }


def fixture_confusion(class_names: tuple[str, ...]) -> list[list[int]]:
    if len(class_names) == 3:
        return [[8, 1, 1], [1, 7, 2], [2, 1, 7]]
    return [[9, 1], [2, 8]]


def write_predictions(path: Path, class_names: tuple[str, ...], confusion: list[list[int]]) -> None:
    fieldnames = ["index", "source_index", "subject_id", "epoch_start_sec", "y_true", "y_pred"]
    fieldnames.extend(f"logit_{class_name}" for class_name in class_names)
    fieldnames.extend(f"prob_{class_name}" for class_name in class_names)
    path.parent.mkdir(parents=True, exist_ok=True)
    index = 0
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for true_label, row in enumerate(confusion):
            for pred_label, count in enumerate(row):
                for _ in range(count):
                    logits = [0.1 for _ in class_names]
                    logits[pred_label] = 2.0
                    probabilities = [0.1 for _ in class_names]
                    probabilities[pred_label] = 0.8
                    record: dict[str, object] = {
                        "index": index,
                        "source_index": index,
                        "subject_id": f"fixture-{index % 3}",
                        "epoch_start_sec": float(index * 3),
                        "y_true": true_label,
                        "y_pred": pred_label,
                    }
                    for class_name, value in zip(class_names, logits, strict=True):
                        record[f"logit_{class_name}"] = value
                    for class_name, value in zip(class_names, probabilities, strict=True):
                        record[f"prob_{class_name}"] = value
                    writer.writerow(record)
                    index += 1


def write_single_run(run_dir: Path, *, task_id: str, class_names: tuple[str, ...]) -> None:
    confusion = fixture_confusion(class_names)
    metrics = class_metrics(class_names, confusion)
    write_json(
        run_dir / "run.json",
        {
            "experiment_id": f"fixture_{task_id}",
            "config_path": "fixture",
            "task_id": task_id,
            "epochs": 2,
            "seed": 0,
            "device": "cpu",
            "num_workers": 0,
            "protocol": "fixture smoke run; not a paper result",
            "class_names": list(class_names),
            "hyperparameters": {
                "input_dim": 1,
                "num_classes": len(class_names),
                "tcn_channels": 32,
                "tcn_kernel_size": 7,
                "tcn_dilations": [1, 1],
                "tcn_dropout": 0.3,
                "lstm_hidden_dim": 64,
                "dense_hidden_dims": [128, 192, 256],
                "dense_dropout": 0.2,
                "batch_size": 32,
                "learning_rate": 0.0001,
            },
            "split_fractions": {"train": 0.8, "val": 0.2, "test": 0.0},
            "evaluation": {"test_source": "val"},
        },
    )
    write_json(
        run_dir / "history.json",
        [
            {"epoch": 1, "train_loss": 1.0, "val_loss": 1.1, "train_metrics": metrics, "val_metrics": metrics},
            {"epoch": 2, "train_loss": 0.9, "val_loss": 1.0, "train_metrics": metrics, "val_metrics": metrics},
        ],
    )
    write_json(run_dir / "test_metrics.json", {"source": "fixture", "loss": 1.0, "metrics": metrics})
    write_predictions(run_dir / "test_predictions.csv", class_names, confusion)
    (run_dir / "model.pt").write_bytes(b"fixture model placeholder\n")


def write_kfold_run(run_dir: Path, *, task_id: str, class_names: tuple[str, ...]) -> None:
    write_json(
        run_dir / "run.json",
        {
            "experiment_id": f"fixture_kfold_{task_id}",
            "task_id": task_id,
            "protocol": "fixture k-fold smoke run; not a paper result",
            "class_names": list(class_names),
        },
    )
    with (run_dir / "fold_summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=(
                "fold",
                "train_loss",
                "test_loss",
                "train_accuracy",
                "test_accuracy",
                "train_macro_f1",
                "test_macro_f1",
                "train_support",
                "test_support",
            ),
        )
        writer.writeheader()
        for fold in range(1, 6):
            writer.writerow(
                {
                    "fold": fold,
                    "train_loss": 0.9,
                    "test_loss": 1.0,
                    "train_accuracy": 0.8,
                    "test_accuracy": 0.79 + fold * 0.001,
                    "train_macro_f1": 0.8,
                    "test_macro_f1": 0.79,
                    "train_support": 80,
                    "test_support": 20,
                }
            )
    for fold in range(1, 6):
        fold_dir = run_dir / f"fold_{fold}"
        write_single_run(fold_dir, task_id=task_id, class_names=class_names)


def main() -> None:
    args = parse_args()
    if args.runs_dir.exists() and any(args.runs_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Fixture runs dir is not empty: {args.runs_dir}")
    if args.runs_dir.exists() and args.overwrite:
        shutil.rmtree(args.runs_dir)
    for task_id, class_names in TASK_CLASSES.items():
        write_single_run(args.runs_dir / task_id, task_id=task_id, class_names=class_names)
    for task_id in ("multiclass", "ad_vs_healthy"):
        write_single_run(args.runs_dir / "smote" / task_id, task_id=task_id, class_names=TASK_CLASSES[task_id])
        write_single_run(args.runs_dir / "standard_rbp" / task_id, task_id=task_id, class_names=TASK_CLASSES[task_id])
        write_kfold_run(args.runs_dir / "kfold" / task_id, task_id=task_id, class_names=TASK_CLASSES[task_id])
    for task_id in ("multiclass", "ftd_vs_healthy"):
        write_single_run(args.runs_dir / "label_swap_80_20" / task_id, task_id=task_id, class_names=TASK_CLASSES[task_id])
    print(f"Wrote fixture runs to {args.runs_dir}")


if __name__ == "__main__":
    main()
