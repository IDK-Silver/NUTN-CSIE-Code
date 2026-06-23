from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from xml.sax.saxutils import escape


CsvRow = dict[str, str]
Bar = tuple[str, float, str]
Point = tuple[float, float]

COLORS = (
    "#2563eb",
    "#dc2626",
    "#059669",
    "#d97706",
    "#7c3aed",
    "#0891b2",
    "#be123c",
    "#4b5563",
)

TASK_LABELS = {
    "multiclass": "AD vs FTD vs Healthy",
    "ad_ftd_vs_healthy": "AD+FTD vs Healthy",
    "ad_vs_healthy": "AD vs Healthy",
    "ftd_vs_healthy": "FTD vs Healthy",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render SVG plots from ds004504 paper report CSV files.")
    parser.add_argument("--scenario", default="paper_literal_80_10_10")
    parser.add_argument("--tables-dir", type=Path, default=None)
    parser.add_argument("--runs-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def default_tables_dir(scenario: str) -> Path:
    return Path("data/reports/ds004504_rbp_paper") / scenario / "tables"


def default_runs_dir(scenario: str) -> Path:
    base = Path("data/runs/ds004504_rbp_paper")
    if scenario == "paper_literal_80_10_10":
        return base
    return base / scenario


def default_output_dir(scenario: str) -> Path:
    return Path("data/reports/ds004504_rbp_paper") / scenario / "plots"


def read_csv_optional(path: Path) -> list[CsvRow]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def to_float(value: str) -> float | None:
    if value == "":
        return None
    try:
        parsed = float(value)
    except ValueError:
        return None
    if not math.isfinite(parsed):
        return None
    return parsed


def value_as_percent(value: str, scale: str) -> float | None:
    parsed = to_float(value)
    if parsed is None:
        return None
    if scale == "decimal":
        return parsed * 100.0
    return parsed


def svg_document(width: int, height: int, body: str) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">\n'
        '<rect width="100%" height="100%" fill="#ffffff"/>\n'
        f"{body}\n"
        "</svg>\n"
    )


def text(x: float, y: float, value: object, *, size: int = 12, anchor: str = "start", weight: str = "400") -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-family="monospace" font-size="{size}" '
        f'font-weight="{weight}" text-anchor="{anchor}" fill="#111827">{escape(str(value))}</text>'
    )


def write_horizontal_bars(
    path: Path,
    *,
    title: str,
    bars: list[Bar],
    value_suffix: str,
    max_value: float | None = None,
) -> None:
    if not bars:
        write_text(path, svg_document(900, 120, text(30, 60, f"{title}: no data", size=16, weight="700")))
        return
    label_width = 380
    chart_width = 620
    row_height = 28
    top = 70
    width = label_width + chart_width + 170
    height = top + row_height * len(bars) + 40
    computed_max = max(value for _, value, _ in bars)
    scale_max = max_value if max_value is not None else computed_max
    if scale_max <= 0:
        scale_max = 1.0

    parts = [text(24, 34, title, size=18, weight="700")]
    for index, (label, value, color) in enumerate(bars):
        y = top + index * row_height
        bar_width = max(0.0, min(chart_width, chart_width * value / scale_max))
        parts.append(text(24, y + 15, label, size=11))
        parts.append(
            f'<rect x="{label_width}" y="{y:.1f}" width="{bar_width:.1f}" height="18" rx="3" fill="{color}"/>'
        )
        parts.append(text(label_width + bar_width + 8, y + 14, f"{value:.3g}{value_suffix}", size=11))
    write_text(path, svg_document(width, height, "\n".join(parts)))


def write_line_chart(path: Path, *, title: str, series: list[tuple[str, list[Point], str]], y_suffix: str) -> None:
    series = [(name, points, color) for name, points, color in series if points]
    if not series:
        write_text(path, svg_document(900, 120, text(30, 60, f"{title}: no data", size=16, weight="700")))
        return
    width = 980
    height = 520
    left = 80
    top = 60
    chart_width = 760
    chart_height = 360
    all_x = [x for _, points, _ in series for x, _ in points]
    all_y = [y for _, points, _ in series for _, y in points]
    min_x = min(all_x)
    max_x = max(all_x)
    min_y = min(0.0, min(all_y))
    max_y = max(1.0, max(all_y))
    if max_x == min_x:
        max_x = min_x + 1.0
    if max_y == min_y:
        max_y = min_y + 1.0

    def sx(x: float) -> float:
        return left + (x - min_x) / (max_x - min_x) * chart_width

    def sy(y: float) -> float:
        return top + chart_height - (y - min_y) / (max_y - min_y) * chart_height

    parts = [text(24, 34, title, size=18, weight="700")]
    parts.append(f'<rect x="{left}" y="{top}" width="{chart_width}" height="{chart_height}" fill="#f9fafb" stroke="#d1d5db"/>')
    for tick in range(6):
        value = min_y + (max_y - min_y) * tick / 5
        y = sy(value)
        parts.append(f'<line x1="{left}" x2="{left + chart_width}" y1="{y:.1f}" y2="{y:.1f}" stroke="#e5e7eb"/>')
        parts.append(text(left - 10, y + 4, f"{value:.2g}{y_suffix}", size=10, anchor="end"))
    for index, (name, points, color) in enumerate(series):
        path_data = " ".join(("M" if point_index == 0 else "L") + f"{sx(x):.1f},{sy(y):.1f}" for point_index, (x, y) in enumerate(points))
        parts.append(f'<path d="{path_data}" fill="none" stroke="{color}" stroke-width="2.5"/>')
        legend_y = top + index * 22
        parts.append(f'<rect x="{left + chart_width + 28}" y="{legend_y - 11}" width="14" height="14" fill="{color}"/>')
        parts.append(text(left + chart_width + 48, legend_y, name, size=11))
    write_text(path, svg_document(width, height, "\n".join(parts)))


def write_confusion_matrix(path: Path, *, title: str, classes: list[str], matrix: list[list[float]]) -> None:
    if not classes or not matrix:
        write_text(path, svg_document(900, 120, text(30, 60, f"{title}: no data", size=16, weight="700")))
        return
    cell = 84
    left = 210
    top = 90
    width = left + cell * len(classes) + 80
    height = top + cell * len(classes) + 110
    max_count = max(max(row) for row in matrix) if matrix else 1.0
    if max_count <= 0:
        max_count = 1.0
    parts = [text(24, 34, title, size=18, weight="700")]
    for pred_index, class_name in enumerate(classes):
        parts.append(text(left + pred_index * cell + cell / 2, top - 16, class_name, size=10, anchor="middle"))
    for true_index, class_name in enumerate(classes):
        parts.append(text(left - 12, top + true_index * cell + cell / 2 + 4, class_name, size=10, anchor="end"))
        for pred_index in range(len(classes)):
            count = matrix[true_index][pred_index]
            intensity = int(245 - 190 * (count / max_count))
            fill = f"rgb({intensity},{intensity + 8},{255})"
            x = left + pred_index * cell
            y = top + true_index * cell
            parts.append(f'<rect x="{x}" y="{y}" width="{cell}" height="{cell}" fill="{fill}" stroke="#ffffff"/>')
            parts.append(text(x + cell / 2, y + cell / 2 + 4, int(count), size=12, anchor="middle", weight="700"))
    parts.append(text(left + cell * len(classes) / 2, height - 30, "Predicted label", size=12, anchor="middle"))
    parts.append(text(24, top + cell * len(classes) / 2, "True label", size=12))
    write_text(path, svg_document(width, height, "\n".join(parts)))


def source_label(row: CsvRow) -> str:
    source = row.get("source", "")
    scenario = row.get("scenario", "")
    if source == "ours" and scenario:
        return f"ours:{scenario}"
    if source.startswith("paper"):
        return source.replace("_", " ")
    return source


def plot_accuracy(tables_dir: Path, output_dir: Path) -> None:
    rows = read_csv_optional(tables_dir / "accuracy.csv")
    bars: list[Bar] = []
    for row in rows:
        percent = value_as_percent(row.get("value", ""), row.get("value_scale", ""))
        if percent is None:
            continue
        task = TASK_LABELS.get(row.get("task_id", ""), row.get("task_id", ""))
        label = f"{source_label(row)} | {row.get('experiment_kind', '')} | {task}"
        color = COLORS[len(bars) % len(COLORS)]
        bars.append((label, percent, color))
    write_horizontal_bars(output_dir / "accuracy_comparison.svg", title="Accuracy comparison", bars=bars, value_suffix="%", max_value=100.0)


def plot_support(tables_dir: Path, output_dir: Path) -> None:
    rows = read_csv_optional(tables_dir / "support_comparison.csv")
    bars: list[Bar] = []
    for row in rows:
        support = to_float(row.get("support", ""))
        if support is None:
            continue
        label = f"{source_label(row)} | {row.get('paper_table', '')} | {row.get('task_id', '')} | {row.get('class_name', '')}"
        color = COLORS[len(bars) % len(COLORS)]
        bars.append((label, support, color))
    write_horizontal_bars(output_dir / "support_comparison.svg", title="Support comparison", bars=bars, value_suffix="")


def plot_per_class_metrics(tables_dir: Path, output_dir: Path) -> None:
    rows = read_csv_optional(tables_dir / "classification_metrics.csv")
    bars: list[Bar] = []
    for row in rows:
        for metric in ("precision", "recall", "f1"):
            value = value_as_percent(row.get(metric, ""), row.get("value_scale", ""))
            if value is None:
                continue
            label = f"{source_label(row)} | {row.get('paper_table', '')} | {row.get('class_name', '')} | {metric}"
            color = COLORS[len(bars) % len(COLORS)]
            bars.append((label, value, color))
    write_horizontal_bars(
        output_dir / "per_class_precision_recall_f1.svg",
        title="Per-class precision / recall / F1",
        bars=bars,
        value_suffix="%",
        max_value=100.0,
    )


def plot_confusion_matrices(tables_dir: Path, output_dir: Path) -> None:
    rows = read_csv_optional(tables_dir / "confusion_matrices.csv")
    grouped: dict[tuple[str, str, str], list[CsvRow]] = {}
    for row in rows:
        key = (source_label(row), row.get("scenario", ""), row.get("task_id", ""))
        grouped.setdefault(key, []).append(row)
    confusion_dir = output_dir / "confusion_matrices"
    for (source, scenario, task_id), group_rows in grouped.items():
        classes = sorted({row.get("true_class", "") for row in group_rows if row.get("true_class", "")})
        pred_classes = sorted({row.get("predicted_class", "") for row in group_rows if row.get("predicted_class", "")})
        if classes != pred_classes:
            classes = list(dict.fromkeys([row.get("true_class", "") for row in group_rows if row.get("true_class", "")]))
        index = {class_name: class_index for class_index, class_name in enumerate(classes)}
        matrix = [[0.0 for _ in classes] for _ in classes]
        for row in group_rows:
            true_class = row.get("true_class", "")
            predicted_class = row.get("predicted_class", "")
            count = to_float(row.get("count", ""))
            if true_class in index and predicted_class in index and count is not None:
                matrix[index[true_class]][index[predicted_class]] = count
        safe_source = source.replace(":", "_").replace(" ", "_").replace("/", "_")
        safe_scenario = scenario.replace("/", "_") if scenario else "paper"
        path = confusion_dir / f"{safe_source}_{safe_scenario}_{task_id}.svg"
        write_confusion_matrix(path, title=f"{source} {task_id}", classes=classes, matrix=matrix)


def plot_history(tables_dir: Path, output_dir: Path) -> None:
    rows = read_csv_optional(tables_dir / "history.csv")
    grouped: dict[str, list[CsvRow]] = {}
    for row in rows:
        grouped.setdefault(row.get("task_id", ""), []).append(row)
    history_dir = output_dir / "history"
    for task_id, task_rows in grouped.items():
        train_points: list[Point] = []
        val_points: list[Point] = []
        for row in task_rows:
            epoch = to_float(row.get("epoch", ""))
            train_acc = to_float(row.get("train_accuracy", ""))
            val_acc = to_float(row.get("val_accuracy", ""))
            if epoch is not None and train_acc is not None:
                train_points.append((epoch, train_acc))
            if epoch is not None and val_acc is not None:
                val_points.append((epoch, val_acc))
        safe_task = task_id.replace("/", "_")
        write_line_chart(
            history_dir / f"{safe_task}_accuracy.svg",
            title=f"Training history: {task_id}",
            series=[("train accuracy", train_points, COLORS[0]), ("validation accuracy", val_points, COLORS[1])],
            y_suffix="",
        )


def discover_prediction_paths(runs_dir: Path, *, scenario: str) -> list[Path]:
    root_runs = Path("data/runs/ds004504_rbp_paper")
    candidates = [
        runs_dir / task / "test_predictions.csv"
        for task in ("multiclass", "ad_ftd_vs_healthy", "ad_vs_healthy", "ftd_vs_healthy")
    ]
    if scenario == "paper_literal_80_10_10":
        label_swap_dir = root_runs / "label_swap"
    elif scenario == "fixture_smoke":
        label_swap_dir = runs_dir / "label_swap_80_20"
    else:
        label_swap_dir = root_runs / "label_swap_80_20"
    candidates.extend(
        [
            label_swap_dir / "multiclass" / "test_predictions.csv",
            label_swap_dir / "ftd_vs_healthy" / "test_predictions.csv",
            runs_dir / "standard_rbp" / "multiclass" / "test_predictions.csv",
            runs_dir / "standard_rbp" / "ad_vs_healthy" / "test_predictions.csv",
        ]
    )
    candidates.extend(
        [
            runs_dir / "smote" / "multiclass" / "test_predictions.csv",
            runs_dir / "smote" / "ad_vs_healthy" / "test_predictions.csv",
            root_runs / "standard_rbp" / "multiclass" / "test_predictions.csv",
            root_runs / "standard_rbp" / "ad_vs_healthy" / "test_predictions.csv",
        ]
    )
    existing_paths: list[Path] = []
    seen_paths: set[Path] = set()
    for path in candidates:
        if path.exists() and path not in seen_paths:
            existing_paths.append(path)
            seen_paths.add(path)
    return existing_paths


def roc_points(y_true: list[int], scores: list[float], positive_label: int) -> tuple[list[Point], float] | None:
    labels = [1 if label == positive_label else 0 for label in y_true]
    positives = sum(labels)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return None
    order = sorted(range(len(scores)), key=lambda index: scores[index], reverse=True)
    tp = 0
    fp = 0
    points: list[Point] = [(0.0, 0.0)]
    previous_score: float | None = None
    for index in order:
        score = scores[index]
        if previous_score is not None and score != previous_score:
            points.append((fp / negatives, tp / positives))
        if labels[index] == 1:
            tp += 1
        else:
            fp += 1
        previous_score = score
    points.append((fp / negatives, tp / positives))
    points.append((1.0, 1.0))
    auc = 0.0
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        auc += (x1 - x0) * (y0 + y1) / 2.0
    return points, auc


def plot_roc_curves(runs_dir: Path, output_dir: Path, *, scenario: str) -> None:
    prediction_paths = discover_prediction_paths(runs_dir, scenario=scenario)
    roc_dir = output_dir / "roc"
    auc_rows: list[dict[str, object]] = []
    for prediction_path in prediction_paths:
        rows = read_csv_optional(prediction_path)
        if not rows:
            continue
        prob_columns = [column for column in rows[0] if column.startswith("prob_")]
        if not prob_columns:
            continue
        y_true = [int(row["y_true"]) for row in rows if row.get("y_true", "") != ""]
        series: list[tuple[str, list[Point], str]] = []
        for class_index, column in enumerate(prob_columns):
            scores = [float(row[column]) for row in rows]
            result = roc_points(y_true, scores, class_index)
            if result is None:
                continue
            points, auc = result
            class_name = column.removeprefix("prob_")
            series.append((f"{class_name} AUC={auc:.3f}", points, COLORS[class_index % len(COLORS)]))
            auc_rows.append(
                {
                    "prediction_csv": str(prediction_path),
                    "class_index": class_index,
                    "class_name": class_name,
                    "auc": auc,
                }
            )
        if series:
            safe_name = "_".join(prediction_path.parent.parts[-3:]).replace("/", "_")
            write_line_chart(roc_dir / f"{safe_name}_roc.svg", title=f"ROC: {prediction_path.parent}", series=series, y_suffix="")
    if auc_rows:
        auc_path = roc_dir / "roc_auc.csv"
        auc_path.parent.mkdir(parents=True, exist_ok=True)
        with auc_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=("prediction_csv", "class_index", "class_name", "auc"))
            writer.writeheader()
            writer.writerows(auc_rows)


def main() -> None:
    args = parse_args()
    tables_dir = args.tables_dir if args.tables_dir is not None else default_tables_dir(args.scenario)
    runs_dir = args.runs_dir if args.runs_dir is not None else default_runs_dir(args.scenario)
    output_dir = args.output_dir if args.output_dir is not None else default_output_dir(args.scenario)

    plot_accuracy(tables_dir, output_dir)
    plot_support(tables_dir, output_dir)
    plot_per_class_metrics(tables_dir, output_dir)
    plot_confusion_matrices(tables_dir, output_dir)
    plot_history(tables_dir, output_dir)
    plot_roc_curves(runs_dir, output_dir, scenario=args.scenario)
    print(f"Wrote SVG plots to {output_dir}")


if __name__ == "__main__":
    main()
