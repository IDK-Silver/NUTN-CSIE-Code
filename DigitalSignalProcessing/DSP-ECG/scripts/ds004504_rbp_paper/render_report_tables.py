from __future__ import annotations

import argparse
import csv
from pathlib import Path


CsvRow = dict[str, str]

CLASSIFICATION_TABLE_TITLES = {
    "Table 3": "Table 3. Classification metrics for Alzheimer, frontotemporal, and healthy classes.",
    "Table 4": "Table 4. Classification metrics for Alzheimer + frontotemporal disease and healthy classes.",
    "Table 5": "Table 5. Classification metrics for Alzheimer's disease and healthy classes.",
    "Table 6": "Table 6. Classification metrics for frontotemporal disease and healthy classes.",
    "Table 8": "Table 8. Classification metrics with SMOTE data balancing.",
    "Table 9": "Table 9. Classification metrics for Alzheimer's disease and healthy classes with SMOTE balancing.",
    "Table 12": "Table 12. Classification metrics for standard RBP multiclass classification.",
    "Table 13": "Table 13. Classification metrics for standard RBP Alzheimer's disease and healthy classification.",
}

CLASSIFICATION_TABLE_FILES = {
    "Table 3": "table_03_multiclass.md",
    "Table 4": "table_04_ad_ftd_vs_healthy.md",
    "Table 5": "table_05_ad_vs_healthy.md",
    "Table 6": "table_06_ftd_vs_healthy.md",
    "Table 8": "table_08_smote_multiclass.md",
    "Table 9": "table_09_smote_ad_vs_healthy.md",
    "Table 12": "table_12_standard_rbp_multiclass.md",
    "Table 13": "table_13_standard_rbp_ad_vs_healthy.md",
}

ACCURACY_TABLE_FILES = {"Table 7": "table_07_accuracy.md"}
KFOLD_TABLE_FILES = {
    "Table 10": "table_10_kfold_multiclass.md",
    "Table 11": "table_11_kfold_ad_vs_healthy.md",
}

TASK_LABELS = {
    "multiclass": "Alzheimer vs. frontotemporal vs. healthy",
    "ad_ftd_vs_healthy": "Alzheimer + frontotemporal vs. healthy",
    "ad_vs_healthy": "Alzheimer vs. healthy",
    "ftd_vs_healthy": "Frontotemporal vs. healthy",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render paper-shaped Markdown tables from report CSV files.")
    parser.add_argument("--scenario", default="paper_literal_80_10_10")
    parser.add_argument("--tables-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def default_tables_dir(scenario: str) -> Path:
    return Path("data/reports/ds004504_rbp_paper") / scenario / "tables"


def default_output_dir(scenario: str) -> Path:
    return Path("data/reports/ds004504_rbp_paper") / scenario / "paper_tables"


def read_csv_required(path: Path) -> list[CsvRow]:
    if not path.exists():
        raise FileNotFoundError(f"Required report CSV does not exist: {path}")
    with path.open("r", encoding="utf-8", newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def escape_markdown_cell(value: object) -> str:
    text = "" if value is None else str(value)
    return text.replace("|", "\\|").replace("\n", " ")


def markdown_table(headers: tuple[str, ...], rows: list[tuple[object, ...]]) -> str:
    lines = [
        "| " + " | ".join(escape_markdown_cell(header) for header in headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(escape_markdown_cell(cell) for cell in row) + " |")
    return "\n".join(lines)


def format_source(row: CsvRow) -> str:
    source = row.get("source", "")
    scenario = row.get("scenario", "")
    if source == "ours" and scenario:
        return f"ours:{scenario}"
    if source == "paper_reported":
        return "paper"
    if source == "paper_text":
        return "paper text"
    if source == "paper_figure":
        return "paper figure"
    return source


def format_metric(value: str, scale: str) -> str:
    if value == "":
        return ""
    if scale == "percent":
        return f"{value}%"
    return value


def rows_for_table(rows: list[CsvRow], paper_table: str) -> list[CsvRow]:
    return [row for row in rows if row.get("paper_table") == paper_table]


def render_model_architecture(rows: list[CsvRow]) -> str:
    table_rows = [
        (
            row.get("layer_order", ""),
            row.get("layer_type", ""),
            row.get("output_shape", ""),
            row.get("parameter_count", ""),
            row.get("connected_to", ""),
        )
        for row in rows
    ]
    body = markdown_table(("Order", "Layer type", "Output shape", "Parameters", "Connected to"), table_rows)
    return "# Table 1. Model architecture summary\n\n" + body + "\n"


def render_model_parameters(rows: list[CsvRow]) -> str:
    table_rows = [
        (
            format_source(row),
            row.get("task_id", ""),
            row.get("parameter", ""),
            row.get("value", ""),
            row.get("size", ""),
            row.get("notes", ""),
        )
        for row in rows
    ]
    body = markdown_table(("Source", "Task", "Parameter", "Value", "Size", "Notes"), table_rows)
    return "# Table 2. Model parameter summary and run hyperparameters\n\n" + body + "\n"


def render_classification_table(rows: list[CsvRow], paper_table: str) -> str:
    table_rows = []
    for row in rows_for_table(rows, paper_table):
        scale = row.get("value_scale", "")
        table_rows.append(
            (
                format_source(row),
                row.get("class_name", ""),
                format_metric(row.get("precision", ""), scale),
                format_metric(row.get("recall", ""), scale),
                format_metric(row.get("f1", ""), scale),
                format_metric(row.get("sensitivity", ""), scale),
                format_metric(row.get("specificity", ""), scale),
                row.get("support", ""),
                row.get("notes", ""),
            )
        )
    body = markdown_table(
        ("Source", "Class", "Precision", "Recall", "F1 score", "Sensitivity", "Specificity", "Support", "Notes"),
        table_rows,
    )
    return f"# {CLASSIFICATION_TABLE_TITLES[paper_table]}\n\n{body}\n"


def render_accuracy_table(rows: list[CsvRow]) -> str:
    table_rows = []
    for row in rows_for_table(rows, "Table 7"):
        task_id = row.get("task_id", "")
        scale = row.get("value_scale", "")
        table_rows.append(
            (
                format_source(row),
                TASK_LABELS.get(task_id, task_id),
                row.get("metric", ""),
                format_metric(row.get("value", ""), scale),
                row.get("notes", ""),
            )
        )
    body = markdown_table(("Source", "Classification task", "Metric", "Value", "Notes"), table_rows)
    return "# Table 7. Classification accuracy for different dementia classification tasks\n\n" + body + "\n"


def render_kfold_table(rows: list[CsvRow], paper_table: str) -> str:
    title = {
        "Table 10": "Table 10. K-fold validation accuracy for Alzheimer, frontotemporal, and healthy classes.",
        "Table 11": "Table 11. K-fold validation accuracy for Alzheimer and healthy classes.",
    }[paper_table]
    table_rows = [
        (
            format_source(row),
            row.get("fold", ""),
            row.get("train_accuracy_percent", ""),
            row.get("test_accuracy_percent", ""),
            row.get("notes", ""),
        )
        for row in rows_for_table(rows, paper_table)
    ]
    body = markdown_table(("Source", "K-value", "Training accuracy (%)", "Test accuracy (%)", "Notes"), table_rows)
    return f"# {title}\n\n{body}\n"


def render_literature_comparison(rows: list[CsvRow]) -> str:
    table_rows = [
        (
            row.get("paper", ""),
            row.get("model", ""),
            row.get("accuracy", ""),
            row.get("feature_engineering", ""),
            row.get("xai", ""),
        )
        for row in rows
    ]
    body = markdown_table(("Paper", "Model", "Accuracy", "Feature engineering", "XAI"), table_rows)
    return "# Table 14. Model accuracy comparison with existing papers using dataset\n\n" + body + "\n"


def render_protocol_manifest(rows: list[CsvRow]) -> str:
    table_rows = [
        (
            row.get("scenario", ""),
            row.get("component", ""),
            row.get("value", ""),
            row.get("evidence", ""),
            row.get("notes", ""),
        )
        for row in rows
    ]
    body = markdown_table(("Scenario", "Component", "Value", "Evidence", "Notes"), table_rows)
    return "# Protocol manifest\n\n" + body + "\n"


def render_issue_summary(rows: list[CsvRow]) -> str:
    table_rows = [
        (
            row.get("severity", ""),
            row.get("issue_id", ""),
            row.get("affected_tables", ""),
            row.get("paper_observation", ""),
            row.get("reproduction_implication", ""),
            row.get("project_handling", ""),
        )
        for row in rows
    ]
    body = markdown_table(
        ("Severity", "Issue", "Affected tables", "Paper observation", "Reproduction implication", "Project handling"),
        table_rows,
    )
    return "# Reproduction issue summary\n\n" + body + "\n"


def render_table_summary(rows: list[CsvRow]) -> str:
    table_rows = [
        (
            row.get("paper_table", ""),
            row.get("purpose", ""),
            row.get("paper_accuracy", ""),
            row.get("ours_accuracy", ""),
            row.get("paper_support", ""),
            row.get("ours_support", ""),
            row.get("notes", ""),
        )
        for row in rows
    ]
    body = markdown_table(
        ("Table", "Purpose", "Paper accuracy", "Ours accuracy", "Paper support", "Ours support", "Notes"),
        table_rows,
    )
    return "# Table-level comparison summary\n\n" + body + "\n"


def render_confusion_matrices(rows: list[CsvRow]) -> str:
    table_rows = [
        (
            row.get("source", ""),
            row.get("scenario", ""),
            row.get("paper_figure", ""),
            row.get("paper_table", ""),
            row.get("task_id", ""),
            row.get("true_class", ""),
            row.get("predicted_class", ""),
            row.get("count", ""),
            row.get("notes", ""),
        )
        for row in rows
    ]
    body = markdown_table(
        ("Source", "Scenario", "Figure", "Table", "Task", "True class", "Predicted class", "Count", "Notes"),
        table_rows,
    )
    return "# Confusion matrices\n\n" + body + "\n"


def render_index(*, scenario: str, output_files: list[Path]) -> str:
    rows = [(path.stem, path.name) for path in output_files]
    body = markdown_table(("Table", "File"), rows)
    return f"# Rendered paper tables for `{scenario}`\n\n{body}\n"


def scenario_role(scenario: str) -> str:
    if scenario == "paper_literal_80_10_10":
        return "Formal reproduction path following the paper text: 80% train, 10% validation, 10% test."
    if scenario == "val_as_test_80_20":
        return "Audit/comparison path inferred from reported support counts: 80% train, 20% validation-as-test."
    if scenario == "fixture_smoke":
        return "Fixture-only report-chain smoke test; not a paper reproduction result."
    return "Custom scenario supplied on the command line."


def render_combined_report(*, scenario: str, sections: list[tuple[str, str]]) -> str:
    parts = [
        f"# ds004504 paper table reproduction report: `{scenario}`",
        "",
        "This report is generated from normalized CSV outputs. Paper-side values and ours-side outputs are kept together for comparison.",
        "",
        "## Scenario role",
        "",
        scenario_role(scenario),
        "",
        "A formal reproduction is complete only after `audit_reproduction_artifacts.py --fail-on-missing` succeeds for this scenario.",
        "",
    ]
    for filename, content in sections:
        parts.append(f"<!-- source: {filename} -->")
        parts.append(content.strip())
        parts.append("")
    return "\n".join(parts) + "\n"


def main() -> None:
    args = parse_args()
    tables_dir = args.tables_dir if args.tables_dir is not None else default_tables_dir(args.scenario)
    output_dir = args.output_dir if args.output_dir is not None else default_output_dir(args.scenario)

    model_architecture = read_csv_required(tables_dir / "paper_model_architecture.csv")
    model_parameters = read_csv_required(tables_dir / "model_parameters.csv")
    classification_metrics = read_csv_required(tables_dir / "classification_metrics.csv")
    accuracy = read_csv_required(tables_dir / "accuracy.csv")
    kfold_accuracy = read_csv_required(tables_dir / "kfold_accuracy.csv")
    literature_comparison = read_csv_required(tables_dir / "literature_comparison.csv")
    protocol_manifest = read_csv_required(tables_dir / "protocol_manifest.csv")
    issue_summary = read_csv_required(tables_dir / "issue_summary.csv")
    table_summary = read_csv_required(tables_dir / "table_summary.csv")
    confusion_matrices = read_csv_required(tables_dir / "confusion_matrices.csv")

    output_files: list[Path] = []
    report_sections: list[tuple[str, str]] = []
    table_01_path = output_dir / "table_01_model_architecture.md"
    table_01_content = render_model_architecture(model_architecture)
    write_text(table_01_path, table_01_content)
    output_files.append(table_01_path)
    report_sections.append((table_01_path.name, table_01_content))

    table_02_path = output_dir / "table_02_model_parameters.md"
    table_02_content = render_model_parameters(model_parameters)
    write_text(table_02_path, table_02_content)
    output_files.append(table_02_path)
    report_sections.append((table_02_path.name, table_02_content))

    for paper_table, filename in CLASSIFICATION_TABLE_FILES.items():
        path = output_dir / filename
        content = render_classification_table(classification_metrics, paper_table)
        write_text(path, content)
        output_files.append(path)
        report_sections.append((path.name, content))

    table_07_path = output_dir / ACCURACY_TABLE_FILES["Table 7"]
    table_07_content = render_accuracy_table(accuracy)
    write_text(table_07_path, table_07_content)
    output_files.append(table_07_path)
    report_sections.append((table_07_path.name, table_07_content))

    for paper_table, filename in KFOLD_TABLE_FILES.items():
        path = output_dir / filename
        content = render_kfold_table(kfold_accuracy, paper_table)
        write_text(path, content)
        output_files.append(path)
        report_sections.append((path.name, content))

    table_14_path = output_dir / "table_14_literature_comparison.md"
    table_14_content = render_literature_comparison(literature_comparison)
    write_text(table_14_path, table_14_content)
    output_files.append(table_14_path)
    report_sections.append((table_14_path.name, table_14_content))

    protocol_path = output_dir / "protocol_manifest.md"
    protocol_content = render_protocol_manifest(protocol_manifest)
    write_text(protocol_path, protocol_content)
    output_files.append(protocol_path)
    report_sections.insert(0, (protocol_path.name, protocol_content))

    issue_summary_path = output_dir / "issue_summary.md"
    issue_summary_content = render_issue_summary(issue_summary)
    write_text(issue_summary_path, issue_summary_content)
    output_files.append(issue_summary_path)
    report_sections.insert(1, (issue_summary_path.name, issue_summary_content))

    table_summary_path = output_dir / "table_summary.md"
    table_summary_content = render_table_summary(table_summary)
    write_text(table_summary_path, table_summary_content)
    output_files.append(table_summary_path)
    report_sections.insert(2, (table_summary_path.name, table_summary_content))

    confusion_matrices_path = output_dir / "confusion_matrices.md"
    confusion_matrices_content = render_confusion_matrices(confusion_matrices)
    write_text(confusion_matrices_path, confusion_matrices_content)
    output_files.append(confusion_matrices_path)
    report_sections.insert(3, (confusion_matrices_path.name, confusion_matrices_content))

    report_content_by_file = {filename: content for filename, content in report_sections}
    ordered_report_filenames = (
        "protocol_manifest.md",
        "issue_summary.md",
        "table_summary.md",
        "confusion_matrices.md",
        "table_01_model_architecture.md",
        "table_02_model_parameters.md",
        "table_03_multiclass.md",
        "table_04_ad_ftd_vs_healthy.md",
        "table_05_ad_vs_healthy.md",
        "table_06_ftd_vs_healthy.md",
        "table_07_accuracy.md",
        "table_08_smote_multiclass.md",
        "table_09_smote_ad_vs_healthy.md",
        "table_10_kfold_multiclass.md",
        "table_11_kfold_ad_vs_healthy.md",
        "table_12_standard_rbp_multiclass.md",
        "table_13_standard_rbp_ad_vs_healthy.md",
        "table_14_literature_comparison.md",
    )
    report_sections = [
        (filename, report_content_by_file[filename])
        for filename in ordered_report_filenames
        if filename in report_content_by_file
    ]

    report_path = output_dir / "report.md"
    write_text(report_path, render_combined_report(scenario=args.scenario, sections=report_sections))
    output_files.insert(0, report_path)

    write_text(output_dir / "index.md", render_index(scenario=args.scenario, output_files=output_files))
    print(f"Wrote rendered Markdown tables to {output_dir}")


if __name__ == "__main__":
    main()
