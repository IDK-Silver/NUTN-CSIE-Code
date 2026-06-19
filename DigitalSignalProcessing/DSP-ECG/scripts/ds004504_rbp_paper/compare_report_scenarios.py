from __future__ import annotations

import argparse
import csv
from pathlib import Path

from report_coverage import (
    EXPECTED_FULL_COMPARISON_SCENARIOS,
    EXPECTED_ISSUE_IDS,
    EXPECTED_PAPER_VS_OURS_TABLE_NUMBERS,
    EXPECTED_PROTOCOL_COMPONENTS,
    EXPECTED_TABLE_NUMBERS,
    paper_table_number,
)


CsvRow = dict[str, str]

DEFAULT_SCENARIOS = EXPECTED_FULL_COMPARISON_SCENARIOS
SCENARIO_ROLES = {
    "paper_literal_80_10_10": "Formal reproduction path that follows the paper text: 80% train, 10% validation, 10% test.",
    "val_as_test_80_20": "Audit/comparison path inferred from reported support counts: 80% train, 20% validation-as-test.",
    "fixture_smoke": "Report-chain fixture only; not a paper reproduction result.",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare generated ds004504 report summaries across scenarios.")
    parser.add_argument("--scenario", action="append", default=None)
    parser.add_argument(
        "--reports-root",
        type=Path,
        default=Path("data/reports/ds004504_rbp_paper"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/reports/ds004504_rbp_paper/scenario_comparison"),
    )
    parser.add_argument("--fail-on-missing", action="store_true")
    return parser.parse_args()


def read_csv_optional(path: Path) -> list[CsvRow]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


def write_csv(path: Path, fieldnames: tuple[str, ...], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(headers: tuple[str, ...], rows: list[tuple[object, ...]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(value).replace("|", "\\|") for value in row) + " |")
    return "\n".join(lines)


def collect_table_summary_rows(*, scenarios: tuple[str, ...], reports_root: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for scenario in scenarios:
        path = reports_root / scenario / "tables" / "table_summary.csv"
        scenario_rows = read_csv_optional(path)
        if not scenario_rows:
            rows.append(
                {
                    "scenario": scenario,
                    "status": "missing",
                    "paper_table": "",
                    "purpose": "",
                    "paper_accuracy": "",
                    "ours_accuracy": "",
                    "paper_support": "",
                    "ours_support": "",
                    "notes": f"Missing {path}",
                }
            )
            continue
        for row in scenario_rows:
            next_row: dict[str, object] = {"scenario": scenario, "status": "available"}
            next_row.update(row)
            rows.append(next_row)
    return rows


def coverage_row(
    *,
    scenario: str,
    artifact: str,
    expected_values: tuple[object, ...],
    observed_values: set[object],
    source_available: bool,
) -> dict[str, object]:
    missing_values = [value for value in expected_values if value not in observed_values]
    if not source_available:
        status = "missing"
        notes = "source CSV is missing or has no available rows"
    elif missing_values:
        status = "stale"
        notes = "coverage is incomplete"
    else:
        status = "ok"
        notes = "coverage is complete"
    return {
        "scenario": scenario,
        "status": status,
        "artifact": artifact,
        "expected": ", ".join(str(value) for value in expected_values),
        "observed": ", ".join(str(value) for value in sorted(observed_values, key=str)),
        "missing": ", ".join(str(value) for value in missing_values),
        "notes": notes,
    }


def collect_coverage_rows(
    *,
    scenarios: tuple[str, ...],
    table_rows: list[dict[str, object]],
    protocol_rows: list[dict[str, object]],
    paper_vs_ours_rows: list[dict[str, object]],
    issue_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for scenario in scenarios:
        scenario_table_rows = [
            row for row in table_rows if row.get("scenario") == scenario and row.get("status") == "available"
        ]
        rows.append(
            coverage_row(
                scenario=scenario,
                artifact="table_summary.csv:paper_table",
                expected_values=tuple(f"Table {number}" for number in EXPECTED_TABLE_NUMBERS),
                observed_values={
                    f"Table {table_number}"
                    for row in scenario_table_rows
                    if (table_number := paper_table_number(row.get("paper_table", ""))) is not None
                },
                source_available=bool(scenario_table_rows),
            )
        )

        scenario_paper_vs_ours_rows = [
            row for row in paper_vs_ours_rows if row.get("scenario") == scenario and row.get("status") == "available"
        ]
        rows.append(
            coverage_row(
                scenario=scenario,
                artifact="paper_vs_ours.csv:paper_table",
                expected_values=tuple(f"Table {number}" for number in EXPECTED_PAPER_VS_OURS_TABLE_NUMBERS),
                observed_values={
                    f"Table {table_number}"
                    for row in scenario_paper_vs_ours_rows
                    if (table_number := paper_table_number(row.get("paper_table", ""))) is not None
                },
                source_available=bool(scenario_paper_vs_ours_rows),
            )
        )

        scenario_protocol_rows = [
            row for row in protocol_rows if row.get("scenario") == scenario and row.get("status") == "available"
        ]
        rows.append(
            coverage_row(
                scenario=scenario,
                artifact="protocol_manifest.csv:component",
                expected_values=EXPECTED_PROTOCOL_COMPONENTS,
                observed_values={str(row.get("component", "")) for row in scenario_protocol_rows},
                source_available=bool(scenario_protocol_rows),
            )
        )

        scenario_issue_rows = [
            row for row in issue_rows if row.get("scenario") == scenario and row.get("status") == "available"
        ]
        rows.append(
            coverage_row(
                scenario=scenario,
                artifact="issue_summary.csv:issue_id",
                expected_values=EXPECTED_ISSUE_IDS,
                observed_values={str(row.get("issue_id", "")) for row in scenario_issue_rows},
                source_available=bool(scenario_issue_rows),
            )
        )
    return rows


def collect_protocol_rows(*, scenarios: tuple[str, ...], reports_root: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for scenario in scenarios:
        path = reports_root / scenario / "tables" / "protocol_manifest.csv"
        scenario_rows = read_csv_optional(path)
        if not scenario_rows:
            rows.append(
                {
                    "scenario": scenario,
                    "status": "missing",
                    "component": "",
                    "value": "",
                    "evidence": "",
                    "notes": f"Missing {path}",
                }
            )
            continue
        for row in scenario_rows:
            rows.append(
                {
                    "scenario": scenario,
                    "status": "available",
                    "component": row.get("component", ""),
                    "value": row.get("value", ""),
                    "evidence": row.get("evidence", ""),
                    "notes": row.get("notes", ""),
                }
            )
    return rows


def collect_paper_vs_ours_rows(*, scenarios: tuple[str, ...], reports_root: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for scenario in scenarios:
        path = reports_root / scenario / "tables" / "paper_vs_ours.csv"
        scenario_rows = read_csv_optional(path)
        if not scenario_rows:
            rows.append(
                {
                    "scenario": scenario,
                    "status": "missing",
                    "paper_table": "",
                    "experiment_kind": "",
                    "task_id": "",
                    "class_name": "",
                    "metric": "",
                    "paper_value": "",
                    "paper_value_scale": "",
                    "ours_value": "",
                    "ours_value_scale": "",
                    "difference": "",
                    "ours_available": "",
                    "notes": f"Missing {path}",
                }
            )
            continue
        for row in scenario_rows:
            next_row: dict[str, object] = {"scenario": scenario, "status": "available"}
            next_row.update(row)
            rows.append(next_row)
    return rows


def collect_issue_summary_rows(*, scenarios: tuple[str, ...], reports_root: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for scenario in scenarios:
        path = reports_root / scenario / "tables" / "issue_summary.csv"
        scenario_rows = read_csv_optional(path)
        if not scenario_rows:
            rows.append(
                {
                    "scenario": scenario,
                    "status": "missing",
                    "issue_id": "",
                    "severity": "",
                    "affected_tables": "",
                    "paper_observation": "",
                    "reproduction_implication": "",
                    "project_handling": "",
                    "notes": f"Missing {path}",
                }
            )
            continue
        for row in scenario_rows:
            next_row: dict[str, object] = {"scenario": scenario, "status": "available", "notes": ""}
            next_row.update(row)
            rows.append(next_row)
    return rows


def render_markdown(
    *,
    scenarios: tuple[str, ...],
    table_rows: list[dict[str, object]],
    protocol_rows: list[dict[str, object]],
    paper_vs_ours_rows: list[dict[str, object]],
    issue_rows: list[dict[str, object]],
    coverage_rows: list[dict[str, object]],
) -> str:
    summary_rows = [
        (
            row.get("scenario", ""),
            row.get("status", ""),
            row.get("paper_table", ""),
            row.get("purpose", ""),
            row.get("paper_accuracy", ""),
            row.get("ours_accuracy", ""),
            row.get("paper_support", ""),
            row.get("ours_support", ""),
        )
        for row in table_rows
    ]
    protocol_table_rows = [
        (
            row.get("scenario", ""),
            row.get("status", ""),
            row.get("component", ""),
            row.get("value", ""),
            row.get("notes", ""),
        )
        for row in protocol_rows
    ]
    paper_vs_ours_table_rows = [
        (
            row.get("scenario", ""),
            row.get("status", ""),
            row.get("paper_table", ""),
            row.get("experiment_kind", ""),
            row.get("task_id", ""),
            row.get("class_name", ""),
            row.get("metric", ""),
            row.get("paper_value", ""),
            row.get("ours_value", ""),
            row.get("difference", ""),
            row.get("ours_available", ""),
        )
        for row in paper_vs_ours_rows
    ]
    issue_table_rows = [
        (
            row.get("scenario", ""),
            row.get("status", ""),
            row.get("issue_id", ""),
            row.get("severity", ""),
            row.get("affected_tables", ""),
            row.get("paper_observation", ""),
            row.get("reproduction_implication", ""),
            row.get("project_handling", ""),
        )
        for row in issue_rows
    ]
    coverage_table_rows = [
        (
            row.get("scenario", ""),
            row.get("status", ""),
            row.get("artifact", ""),
            row.get("expected", ""),
            row.get("observed", ""),
            row.get("missing", ""),
            row.get("notes", ""),
        )
        for row in coverage_rows
    ]
    scenario_role_rows = [
        (f"`{scenario}`", SCENARIO_ROLES.get(scenario, "Custom scenario supplied on the command line."))
        for scenario in scenarios
    ]
    return (
        "# Scenario comparison\n\n"
        f"Compared scenarios: `{', '.join(scenarios)}`\n\n"
        "## Scenario roles\n\n"
        + markdown_table(("Scenario", "Role"), scenario_role_rows)
        + "\n\n"
        "## Coverage summary\n\n"
        + markdown_table(("Scenario", "Status", "Artifact", "Expected", "Observed", "Missing", "Notes"), coverage_table_rows)
        + "\n\n## Table summary\n\n"
        + markdown_table(
            (
                "Scenario",
                "Status",
                "Table",
                "Purpose",
                "Paper accuracy",
                "Ours accuracy",
                "Paper support",
                "Ours support",
            ),
            summary_rows,
        )
        + "\n\n## Protocol summary\n\n"
        + markdown_table(("Scenario", "Status", "Component", "Value", "Notes"), protocol_table_rows)
        + "\n\n## Paper vs ours details\n\n"
        + markdown_table(
            (
                "Scenario",
                "Status",
                "Table",
                "Experiment",
                "Task",
                "Class",
                "Metric",
                "Paper value",
                "Ours value",
                "Difference",
                "Ours available",
            ),
            paper_vs_ours_table_rows,
        )
        + "\n\n## Issue summary\n\n"
        + markdown_table(
            (
                "Scenario",
                "Status",
                "Issue",
                "Severity",
                "Affected tables",
                "Paper observation",
                "Reproduction implication",
                "Project handling",
            ),
            issue_table_rows,
        )
        + "\n"
    )


def main() -> None:
    args = parse_args()
    scenarios = tuple(args.scenario) if args.scenario is not None else DEFAULT_SCENARIOS
    table_rows = collect_table_summary_rows(scenarios=scenarios, reports_root=args.reports_root)
    protocol_rows = collect_protocol_rows(scenarios=scenarios, reports_root=args.reports_root)
    paper_vs_ours_rows = collect_paper_vs_ours_rows(scenarios=scenarios, reports_root=args.reports_root)
    issue_rows = collect_issue_summary_rows(scenarios=scenarios, reports_root=args.reports_root)
    coverage_rows = collect_coverage_rows(
        scenarios=scenarios,
        table_rows=table_rows,
        protocol_rows=protocol_rows,
        paper_vs_ours_rows=paper_vs_ours_rows,
        issue_rows=issue_rows,
    )

    write_csv(
        args.output_dir / "scenario_coverage.csv",
        ("scenario", "status", "artifact", "expected", "observed", "missing", "notes"),
        coverage_rows,
    )
    write_csv(
        args.output_dir / "scenario_table_summary.csv",
        (
            "scenario",
            "status",
            "paper_table",
            "purpose",
            "paper_accuracy",
            "ours_accuracy",
            "paper_support",
            "ours_support",
            "notes",
        ),
        table_rows,
    )
    write_csv(
        args.output_dir / "scenario_protocol_summary.csv",
        ("scenario", "status", "component", "value", "evidence", "notes"),
        protocol_rows,
    )
    write_csv(
        args.output_dir / "scenario_paper_vs_ours.csv",
        (
            "scenario",
            "status",
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
        paper_vs_ours_rows,
    )
    write_csv(
        args.output_dir / "scenario_issue_summary.csv",
        (
            "scenario",
            "status",
            "issue_id",
            "severity",
            "affected_tables",
            "paper_observation",
            "reproduction_implication",
            "project_handling",
            "notes",
        ),
        issue_rows,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "scenario_comparison.md").write_text(
        render_markdown(
            scenarios=scenarios,
            table_rows=table_rows,
            protocol_rows=protocol_rows,
            paper_vs_ours_rows=paper_vs_ours_rows,
            issue_rows=issue_rows,
            coverage_rows=coverage_rows,
        ),
        encoding="utf-8",
    )
    print(f"Wrote scenario comparison to {args.output_dir}")
    failing_statuses = {"missing", "stale"}
    if args.fail_on_missing and any(
        row.get("status") in failing_statuses
        for row in (*table_rows, *protocol_rows, *paper_vs_ours_rows, *issue_rows, *coverage_rows)
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
