from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from report_coverage import (
    EXPECTED_ISSUE_IDS,
    EXPECTED_PAPER_VS_OURS_TABLE_NUMBERS,
    EXPECTED_PROTOCOL_COMPONENTS,
    EXPECTED_TABLE_NUMBERS,
    paper_table_number,
)


DEFAULT_SCENARIO = "paper_literal_80_10_10"

REPORT_CSV_FILES = (
    "paper_model_architecture.csv",
    "model_parameters.csv",
    "run_summary.csv",
    "classification_metrics.csv",
    "confusion_matrices.csv",
    "support_comparison.csv",
    "history.csv",
    "accuracy.csv",
    "kfold_accuracy.csv",
    "paper_vs_ours.csv",
    "table_summary.csv",
    "protocol_manifest.csv",
    "issue_summary.csv",
    "literature_comparison.csv",
)

RENDERED_TABLE_FILES = (
    "index.md",
    "report.md",
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
    "protocol_manifest.md",
    "issue_summary.md",
    "table_summary.md",
    "confusion_matrices.md",
)

ROOT_TASKS = ("multiclass", "ad_ftd_vs_healthy", "ad_vs_healthy", "ftd_vs_healthy")
STANDARD_RBP_TASKS = ("multiclass", "ad_vs_healthy")
KFOLD_TASKS = ("multiclass", "ad_vs_healthy")
SMOTE_TASKS = ("multiclass", "ad_vs_healthy")
LABEL_SWAP_TASKS = ("multiclass", "ftd_vs_healthy")


@dataclass(frozen=True)
class AuditItem:
    category: str
    name: str
    path: str
    status: str
    detail: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit ds004504 paper-table reproduction artifacts.")
    parser.add_argument("--scenario", default=DEFAULT_SCENARIO)
    parser.add_argument("--runs-dir", type=Path, default=None)
    parser.add_argument("--report-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--fail-on-missing", action="store_true")
    return parser.parse_args()


def default_runs_dir(scenario: str) -> Path:
    base = Path("data/runs/ds004504_rbp_paper")
    if scenario == "paper_literal_80_10_10":
        return base
    return base / scenario


def default_report_dir(scenario: str) -> Path:
    return Path("data/reports/ds004504_rbp_paper") / scenario


def default_output_dir(scenario: str) -> Path:
    return Path("data/reports/ds004504_rbp_paper") / scenario / "audit"


def item_for_path(category: str, name: str, path: Path, *, expected: str) -> AuditItem:
    if path.exists():
        if path.is_file():
            detail = f"exists; size={path.stat().st_size}"
        else:
            detail = "exists"
        return AuditItem(category=category, name=name, path=str(path), status="ok", detail=detail)
    return AuditItem(category=category, name=name, path=str(path), status="missing", detail=expected)


def item_for_svg_directory(category: str, name: str, path: Path, *, expected: str) -> AuditItem:
    if not path.exists() or not path.is_dir():
        return AuditItem(category=category, name=name, path=str(path), status="missing", detail=expected)
    svg_paths = sorted(path.glob("*.svg"))
    if not svg_paths:
        return AuditItem(category=category, name=name, path=str(path), status="missing", detail=expected)
    return AuditItem(
        category=category,
        name=name,
        path=str(path),
        status="ok",
        detail=f"contains {len(svg_paths)} SVG file(s)",
    )


def not_applicable_item(category: str, name: str, path: Path, *, reason: str) -> AuditItem:
    return AuditItem(category=category, name=name, path=str(path), status="not_applicable", detail=reason)


def csv_header(path: Path) -> tuple[str, ...]:
    if not path.exists() or not path.is_file():
        return ()
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        try:
            return tuple(next(reader))
        except StopIteration:
            return ()


def csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists() or not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


def audit_expected_table_coverage(
    *,
    category: str,
    name: str,
    path: Path,
    expected_table_numbers: tuple[int, ...],
) -> AuditItem:
    rows = csv_rows(path)
    observed_numbers = {
        table_number
        for row in rows
        if (table_number := paper_table_number(row.get("paper_table", ""))) is not None
    }
    missing_numbers = [number for number in expected_table_numbers if number not in observed_numbers]
    if not rows:
        return AuditItem(
            category=category,
            name=name,
            path=str(path),
            status="missing",
            detail="CSV has no readable rows for table coverage audit",
        )
    if missing_numbers:
        missing_text = ", ".join(f"Table {number}" for number in missing_numbers)
        return AuditItem(
            category=category,
            name=name,
            path=str(path),
            status="stale",
            detail=f"missing expected table coverage: {missing_text}",
        )
    return AuditItem(
        category=category,
        name=name,
        path=str(path),
        status="ok",
        detail=f"covers expected tables: {', '.join(f'Table {number}' for number in expected_table_numbers)}",
    )


def audit_expected_issue_coverage(path: Path) -> AuditItem:
    rows = csv_rows(path)
    observed_ids = {row.get("issue_id", "") for row in rows}
    missing_ids = [issue_id for issue_id in EXPECTED_ISSUE_IDS if issue_id not in observed_ids]
    if not rows:
        return AuditItem(
            category="report_content",
            name="issue_summary:issue_id_coverage",
            path=str(path),
            status="missing",
            detail="CSV has no readable rows for issue coverage audit",
        )
    if missing_ids:
        return AuditItem(
            category="report_content",
            name="issue_summary:issue_id_coverage",
            path=str(path),
            status="stale",
            detail=f"missing expected issue ids: {', '.join(missing_ids)}",
        )
    return AuditItem(
        category="report_content",
        name="issue_summary:issue_id_coverage",
        path=str(path),
        status="ok",
        detail=f"covers expected issue ids: {', '.join(EXPECTED_ISSUE_IDS)}",
    )


def audit_expected_protocol_coverage(path: Path) -> AuditItem:
    rows = csv_rows(path)
    observed_components = {row.get("component", "") for row in rows}
    missing_components = [component for component in EXPECTED_PROTOCOL_COMPONENTS if component not in observed_components]
    if not rows:
        return AuditItem(
            category="report_content",
            name="protocol_manifest:component_coverage",
            path=str(path),
            status="missing",
            detail="CSV has no readable rows for protocol component coverage audit",
        )
    if missing_components:
        return AuditItem(
            category="report_content",
            name="protocol_manifest:component_coverage",
            path=str(path),
            status="stale",
            detail=f"missing expected protocol components: {', '.join(missing_components)}",
        )
    return AuditItem(
        category="report_content",
        name="protocol_manifest:component_coverage",
        path=str(path),
        status="ok",
        detail=f"covers expected protocol components: {', '.join(EXPECTED_PROTOCOL_COMPONENTS)}",
    )


def audit_prediction_csv(category: str, name: str, path: Path) -> list[AuditItem]:
    items = [item_for_path(category, name, path, expected="prediction CSV should be generated by training script")]
    if not path.exists():
        return items
    header = csv_header(path)
    if any(column.startswith("prob_") for column in header):
        items.append(
            AuditItem(
                category=category,
                name=f"{name}:probability_columns",
                path=str(path),
                status="ok",
                detail="contains prob_* columns for ROC/AUC plotting",
            )
        )
    else:
        items.append(
            AuditItem(
                category=category,
                name=f"{name}:probability_columns",
                path=str(path),
                status="stale",
                detail="missing prob_* columns; rerun training to enable ROC/AUC plots",
            )
        )
    return items


def root_run_dir_for_scenario(runs_dir: Path, scenario: str, task_id: str) -> Path:
    if scenario == "paper_literal_80_10_10":
        return runs_dir / task_id
    return runs_dir / task_id


def audit_single_run(category: str, name: str, run_dir: Path) -> list[AuditItem]:
    items = [
        item_for_path(category, f"{name}:run_json", run_dir / "run.json", expected="run metadata should exist"),
        item_for_path(category, f"{name}:history", run_dir / "history.json", expected="training history should exist"),
        item_for_path(category, f"{name}:metrics", run_dir / "test_metrics.json", expected="reported metrics should exist"),
        item_for_path(category, f"{name}:model", run_dir / "model.pt", expected="trained model checkpoint should exist"),
    ]
    items.extend(audit_prediction_csv(category, f"{name}:predictions", run_dir / "test_predictions.csv"))
    return items


def audit_kfold_run(category: str, name: str, run_dir: Path) -> list[AuditItem]:
    items = [
        item_for_path(category, f"{name}:run_json", run_dir / "run.json", expected="k-fold run metadata should exist"),
        item_for_path(category, f"{name}:fold_summary", run_dir / "fold_summary.csv", expected="k-fold summary should exist"),
    ]
    for fold_index in range(1, 6):
        fold_dir = run_dir / f"fold_{fold_index}"
        items.extend(
            [
                item_for_path(
                    category,
                    f"{name}:fold_{fold_index}:history",
                    fold_dir / "history.json",
                    expected="fold training history should exist",
                ),
                item_for_path(
                    category,
                    f"{name}:fold_{fold_index}:metrics",
                    fold_dir / "test_metrics.json",
                    expected="fold test metrics should exist",
                ),
            ]
        )
        items.extend(audit_prediction_csv(category, f"{name}:fold_{fold_index}:predictions", fold_dir / "test_predictions.csv"))
    return items


def smote_run_dir_for_scenario(root_runs: Path, scenario: str, task_id: str) -> Path:
    if scenario == "paper_literal_80_10_10":
        return root_runs / "smote" / task_id
    if scenario == "fixture_smoke":
        return root_runs / "fixture_smoke" / "smote" / task_id
    return root_runs / "val_as_test_80_20" / "smote" / task_id


def build_audit_items(*, scenario: str, runs_dir: Path, report_dir: Path) -> list[AuditItem]:
    root_runs = Path("data/runs/ds004504_rbp_paper")
    processed_dataset_paths = (
        (
            "modified_rbp_h5",
            Path("data/processed_raw_dataset/ds004504_rbp_paper.h5"),
            "modified RBP H5 is required for Tables 3-11",
        ),
        (
            "modified_rbp_manifest",
            Path("data/processed_raw_dataset/ds004504_rbp_paper_manifest.json"),
            "modified RBP manifest is required for Tables 3-11",
        ),
        (
            "standard_rbp_h5",
            Path("data/processed_raw_dataset/ds004504_standard_rbp_paper.h5"),
            "standard RBP H5 is required for Tables 12-13",
        ),
        (
            "standard_rbp_manifest",
            Path("data/processed_raw_dataset/ds004504_standard_rbp_paper_manifest.json"),
            "standard RBP manifest is required for Tables 12-13",
        ),
    )
    items: list[AuditItem] = []
    for name, path, expected in processed_dataset_paths:
        if scenario == "fixture_smoke":
            items.append(
                not_applicable_item(
                    "processed_dataset",
                    name,
                    path,
                    reason="fixture_smoke does not process EEG; this audit only checks report-chain plumbing",
                )
            )
        else:
            items.append(item_for_path("processed_dataset", name, path, expected=expected))

    for task_id in ROOT_TASKS:
        run_dir = root_run_dir_for_scenario(runs_dir, scenario, task_id)
        items.extend(audit_single_run("training_run", f"{scenario}:{task_id}", run_dir))

    for task_id in SMOTE_TASKS:
        items.extend(
            audit_single_run(
                "smote_run",
                f"smote:{task_id}",
                smote_run_dir_for_scenario(root_runs, scenario, task_id),
            )
        )

    if scenario in {"paper_literal_80_10_10", "val_as_test_80_20", "fixture_smoke"}:
        for task_id in LABEL_SWAP_TASKS:
            label_swap_dir = root_runs / "label_swap_80_20" / task_id
            if scenario == "paper_literal_80_10_10":
                label_swap_dir = root_runs / "label_swap" / task_id
            elif scenario == "fixture_smoke":
                label_swap_dir = root_runs / "fixture_smoke" / "label_swap_80_20" / task_id
            items.extend(
                audit_single_run(
                    "label_swap_run",
                    f"label_swap:{task_id}",
                    label_swap_dir,
                )
            )

    for task_id in STANDARD_RBP_TASKS:
        standard_dir = root_runs / "standard_rbp" / task_id
        if scenario == "fixture_smoke":
            standard_dir = root_runs / "fixture_smoke" / "standard_rbp" / task_id
        items.extend(audit_single_run("standard_rbp_run", f"standard_rbp:{task_id}", standard_dir))

    for task_id in KFOLD_TASKS:
        kfold_dir = root_runs / "kfold" / task_id
        if scenario == "fixture_smoke":
            kfold_dir = root_runs / "fixture_smoke" / "kfold" / task_id
        items.extend(audit_kfold_run("kfold_run", f"kfold:{task_id}", kfold_dir))

    tables_dir = report_dir / "tables"
    for filename in REPORT_CSV_FILES:
        items.append(
            item_for_path(
                "report_csv",
                filename,
                tables_dir / filename,
                expected="report CSV should be generated by make_report_csv.py",
            )
        )
    items.append(
        audit_expected_table_coverage(
            category="report_content",
            name="table_summary:table_1_to_14_coverage",
            path=tables_dir / "table_summary.csv",
            expected_table_numbers=EXPECTED_TABLE_NUMBERS,
        )
    )
    items.append(
        audit_expected_table_coverage(
            category="report_content",
            name="paper_vs_ours:table_3_to_13_coverage",
            path=tables_dir / "paper_vs_ours.csv",
            expected_table_numbers=EXPECTED_PAPER_VS_OURS_TABLE_NUMBERS,
        )
    )
    items.append(audit_expected_issue_coverage(tables_dir / "issue_summary.csv"))
    items.append(audit_expected_protocol_coverage(tables_dir / "protocol_manifest.csv"))

    paper_tables_dir = report_dir / "paper_tables"
    for filename in RENDERED_TABLE_FILES:
        items.append(
            item_for_path(
                "rendered_table",
                filename,
                paper_tables_dir / filename,
                expected="paper-shaped Markdown table should be generated by render_report_tables.py",
            )
        )

    plots_dir = report_dir / "plots"
    items.append(
        item_for_path(
            "pipeline",
            "execution_manifest",
            report_dir / "pipeline_execution_manifest.json",
            expected="pipeline execution manifest should be generated by run_reproduction_pipeline.py",
        )
    )
    for filename in ("accuracy_comparison.svg", "support_comparison.svg", "per_class_precision_recall_f1.svg"):
        items.append(
            item_for_path(
                "plot",
                filename,
                plots_dir / filename,
                expected="SVG plot should be generated by plot_report.py",
            )
        )
    items.append(
        item_for_svg_directory(
            "plot",
            "confusion_matrix_svgs",
            plots_dir / "confusion_matrices",
            expected="confusion matrix SVG heatmaps should be generated by plot_report.py",
        )
    )
    items.append(
        item_for_svg_directory(
            "plot",
            "training_history_svgs",
            plots_dir / "history",
            expected="training/validation history SVG plots should be generated by plot_report.py",
        )
    )
    items.append(
        item_for_path(
            "plot",
            "roc_auc_csv",
            plots_dir / "roc" / "roc_auc.csv",
            expected="ROC/AUC CSV is generated only when prediction CSV files contain prob_* columns",
        )
    )
    return items


def summarize(items: list[AuditItem]) -> dict[str, int]:
    summary: dict[str, int] = {}
    for item in items:
        summary[item.status] = summary.get(item.status, 0) + 1
    return dict(sorted(summary.items()))


def render_markdown(items: list[AuditItem], *, scenario: str, summary: dict[str, int]) -> str:
    lines = [f"# Reproduction artifact audit for `{scenario}`", ""]
    lines.append("## Summary")
    lines.append("")
    lines.append("| Status | Count |")
    lines.append("| --- | --- |")
    for status, count in summary.items():
        lines.append(f"| {status} | {count} |")
    lines.append("")
    lines.append("## Items")
    lines.append("")
    lines.append("| Status | Category | Name | Path | Detail |")
    lines.append("| --- | --- | --- | --- | --- |")
    for item in items:
        lines.append(f"| {item.status} | {item.category} | {item.name} | `{item.path}` | {item.detail} |")
    lines.append("")
    return "\n".join(lines)


def write_audit(output_dir: Path, *, scenario: str, items: list[AuditItem]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = summarize(items)
    with (output_dir / "audit.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "scenario": scenario,
                "summary": summary,
                "items": [asdict(item) for item in items],
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
        f.write("\n")
    (output_dir / "audit.md").write_text(render_markdown(items, scenario=scenario, summary=summary), encoding="utf-8")


def main() -> None:
    args = parse_args()
    runs_dir = args.runs_dir if args.runs_dir is not None else default_runs_dir(args.scenario)
    report_dir = args.report_dir if args.report_dir is not None else default_report_dir(args.scenario)
    output_dir = args.output_dir if args.output_dir is not None else default_output_dir(args.scenario)
    items = build_audit_items(scenario=args.scenario, runs_dir=runs_dir, report_dir=report_dir)
    write_audit(output_dir, scenario=args.scenario, items=items)
    summary = summarize(items)
    print(f"Wrote audit to {output_dir}")
    print(summary)
    failing_statuses = {"missing", "stale"}
    if args.scenario != "fixture_smoke":
        failing_statuses.add("not_applicable")
    if args.fail_on_missing and any(item.status in failing_statuses for item in items):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
