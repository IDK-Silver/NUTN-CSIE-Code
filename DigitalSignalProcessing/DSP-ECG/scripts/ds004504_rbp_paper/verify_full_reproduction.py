from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from report_coverage import (
    EXPECTED_COMPARISON_SCRIPT,
    EXPECTED_FULL_COMPARISON_SCENARIOS,
    EXPECTED_PIPELINE_SCRIPT,
    EXPECTED_SCENARIO_COMPARISON_FILES,
    EXPECTED_SCENARIO_COVERAGE_ARTIFACTS,
    EXPECTED_VERIFIER_SCRIPT,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify full ds004504 paper reproduction completion evidence."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/reports/ds004504_rbp_paper/full_comparison_execution_manifest.json"),
        help="Full-comparison execution manifest written by run_full_comparison.py.",
    )
    parser.add_argument(
        "--coverage",
        type=Path,
        default=Path("data/reports/ds004504_rbp_paper/scenario_comparison/scenario_coverage.csv"),
        help="Scenario coverage CSV written by compare_report_scenarios.py.",
    )
    parser.add_argument(
        "--allow-verifying",
        action="store_true",
        help="Allow overall_status=verifying when called from run_full_comparison.py before the verifier step is recorded.",
    )
    return parser.parse_args()


def read_json_object(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return dict(payload)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


def is_verifier_record(item: dict[str, Any]) -> bool:
    command_text = str(item.get("command_text", ""))
    command = item.get("command")
    if "scripts/ds004504_rbp_paper/verify_full_reproduction.py" in command_text:
        return True
    if isinstance(command, list):
        return any(str(part) == "scripts/ds004504_rbp_paper/verify_full_reproduction.py" for part in command)
    return False


def verify_manifest(path: Path, *, coverage_path: Path, allow_verifying: bool) -> list[str]:
    errors: list[str] = []
    if not path.exists():
        return [f"missing manifest: {path}"]
    payload = read_json_object(path)
    overall_status = payload.get("overall_status")
    workflow_status = payload.get("workflow_status", overall_status)
    if payload.get("execute") is not True:
        errors.append(f"manifest execute must be true, got {payload.get('execute')!r}")
    if payload.get("mode") != "EXECUTE":
        errors.append(f"manifest mode must be 'EXECUTE', got {payload.get('mode')!r}")
    if payload.get("coverage") != str(coverage_path):
        errors.append(f"manifest coverage must be {str(coverage_path)!r}, got {payload.get('coverage')!r}")
    if workflow_status != "completed":
        errors.append(f"manifest workflow_status must be completed, got {workflow_status!r}")
    allowed_overall_statuses = {"completed", "verifying"} if allow_verifying else {"completed"}
    if overall_status not in allowed_overall_statuses:
        allowed_text = " or ".join(sorted(allowed_overall_statuses))
        errors.append(f"manifest overall_status must be {allowed_text}, got {overall_status!r}")
    expected_script_fields = {
        "pipeline_script": EXPECTED_PIPELINE_SCRIPT,
        "comparison_script": EXPECTED_COMPARISON_SCRIPT,
        "verifier_script": EXPECTED_VERIFIER_SCRIPT,
    }
    for field_name, expected_script in expected_script_fields.items():
        observed_script = payload.get(field_name)
        if observed_script != expected_script:
            errors.append(f"manifest {field_name} must be {expected_script!r}, got {observed_script!r}")
    execution_results = payload.get("execution_results")
    scenarios = payload.get("scenarios")
    skip_val_as_test = payload.get("skip_val_as_test")
    if payload.get("allow_partial_outputs"):
        errors.append("manifest allow_partial_outputs must be false for full reproduction completion evidence")
    if skip_val_as_test:
        errors.append("manifest skip_val_as_test must be false for full reproduction comparison evidence")
    if not isinstance(scenarios, list):
        errors.append("manifest scenarios must be a list")
    else:
        required_scenarios = set(EXPECTED_FULL_COMPARISON_SCENARIOS)
        observed_scenarios = {str(scenario) for scenario in scenarios}
        missing_scenarios = sorted(required_scenarios - observed_scenarios)
        if missing_scenarios:
            errors.append(f"manifest scenarios missing required full-comparison scenarios: {', '.join(missing_scenarios)}")
    if not isinstance(execution_results, list) or not execution_results:
        errors.append("manifest execution_results must be a non-empty list")
        return errors
    commands = payload.get("commands")
    expected_workflow_count: int | None = None
    if isinstance(commands, list):
        observed_command_scripts: set[str] = set()
        observed_pipeline_scenarios: set[str] = set()
        expected_command_prefix = ("uv", "run", "python")
        for command_number, item in enumerate(commands, start=1):
            if not isinstance(item, dict):
                errors.append(f"manifest commands[{command_number}] must be an object")
                continue
            command_text = item.get("command_text")
            if not isinstance(command_text, str) or not command_text.strip():
                errors.append(f"manifest commands[{command_number}].command_text must be a non-empty string")
            command = item.get("command")
            if not isinstance(command, list):
                errors.append(f"manifest commands[{command_number}].command must be a list")
                continue
            command_prefix = tuple(str(part) for part in command[:3])
            if command_prefix != expected_command_prefix:
                errors.append(
                    f"manifest commands[{command_number}] must start with uv run python, "
                    f"got {' '.join(command_prefix)!r}"
                )
            if len(command) >= 4:
                observed_command_scripts.add(str(command[3]))
            if len(command) >= 4 and str(command[3]) == EXPECTED_PIPELINE_SCRIPT:
                for command_index, command_part in enumerate(command):
                    if str(command_part) == "--scenario" and command_index + 1 < len(command):
                        observed_pipeline_scenarios.add(str(command[command_index + 1]))
        expected_command_scripts = {
            EXPECTED_PIPELINE_SCRIPT,
            EXPECTED_COMPARISON_SCRIPT,
            EXPECTED_VERIFIER_SCRIPT,
        }
        missing_command_scripts = sorted(expected_command_scripts - observed_command_scripts)
        if missing_command_scripts:
            errors.append(
                "manifest commands missing expected script entrypoints: " + ", ".join(missing_command_scripts)
            )
        missing_pipeline_scenarios = sorted(set(EXPECTED_FULL_COMPARISON_SCENARIOS) - observed_pipeline_scenarios)
        if missing_pipeline_scenarios:
            errors.append(
                "manifest pipeline commands missing expected scenarios: " + ", ".join(missing_pipeline_scenarios)
            )
        expected_workflow_count = sum(
            1 for item in commands if isinstance(item, dict) and not is_verifier_record(item)
        )
        expected_verifier_count = sum(
            1 for item in commands if isinstance(item, dict) and is_verifier_record(item)
        )
    else:
        errors.append("manifest commands must be a list")
        expected_verifier_count = None
    workflow_result_count = 0
    verifier_result_count = 0
    for index, item in enumerate(execution_results, start=1):
        if not isinstance(item, dict):
            errors.append(f"manifest execution_results[{index}] must be an object")
            continue
        status = item.get("status")
        if status == "completed" and item.get("returncode") != 0:
            errors.append(
                f"manifest execution_results[{index}] completed result must have returncode 0, "
                f"got {item.get('returncode')!r}"
            )
        if is_verifier_record(item):
            verifier_result_count += 1
            if status != "completed":
                errors.append(f"manifest verifier result status must be completed, got {status!r}")
            continue
        workflow_result_count += 1
        if status != "completed":
            errors.append(f"manifest execution_results[{index}] status must be completed, got {status!r}")
    if expected_workflow_count is not None and workflow_result_count != expected_workflow_count:
        errors.append(
            "manifest workflow execution result count must match non-verifier command count, "
            f"got {workflow_result_count}, expected {expected_workflow_count}"
        )
    if overall_status == "completed" and expected_verifier_count is not None:
        if verifier_result_count != expected_verifier_count:
            errors.append(
                "manifest verifier execution result count must match verifier command count when overall_status is completed, "
                f"got {verifier_result_count}, expected {expected_verifier_count}"
            )
    return errors


def verify_coverage(path: Path) -> list[str]:
    errors: list[str] = []
    if not path.exists():
        return [f"missing scenario coverage CSV: {path}"]
    rows = read_csv_rows(path)
    if not rows:
        return [f"scenario coverage CSV has no rows: {path}"]
    required_scenarios = set(EXPECTED_FULL_COMPARISON_SCENARIOS)
    observed_scenarios = {row.get("scenario", "") for row in rows}
    missing_scenarios = sorted(required_scenarios - observed_scenarios)
    if missing_scenarios:
        errors.append(f"scenario coverage CSV missing required scenarios: {', '.join(missing_scenarios)}")
    unexpected_scenarios = sorted(observed_scenarios - required_scenarios)
    if unexpected_scenarios:
        errors.append(f"scenario coverage CSV contains unexpected scenarios: {', '.join(unexpected_scenarios)}")
    for scenario in sorted(required_scenarios):
        scenario_rows = [row for row in rows if row.get("scenario", "") == scenario]
        scenario_artifacts = {row.get("artifact", "") for row in scenario_rows}
        artifact_counts: dict[str, int] = {}
        for scenario_row in scenario_rows:
            artifact = scenario_row.get("artifact", "")
            artifact_counts[artifact] = artifact_counts.get(artifact, 0) + 1
        duplicate_artifacts = sorted(artifact for artifact, count in artifact_counts.items() if count > 1)
        if duplicate_artifacts:
            errors.append(
                f"scenario coverage CSV contains duplicate artifacts for {scenario}: "
                + ", ".join(duplicate_artifacts)
            )
        missing_artifacts = sorted(set(EXPECTED_SCENARIO_COVERAGE_ARTIFACTS) - scenario_artifacts)
        if missing_artifacts:
            errors.append(
                f"scenario coverage CSV missing required artifacts for {scenario}: {', '.join(missing_artifacts)}"
            )
        for required_artifact in EXPECTED_SCENARIO_COVERAGE_ARTIFACTS:
            matching_rows = [row for row in scenario_rows if row.get("artifact", "") == required_artifact]
            for matching_row in matching_rows:
                status = matching_row.get("status", "")
                if status != "ok":
                    missing = matching_row.get("missing", "")
                    errors.append(
                        "scenario coverage CSV required artifact must be ok: "
                        f"scenario={scenario}, artifact={required_artifact}, status={status}, missing={missing}"
                    )
    for index, row in enumerate(rows, start=1):
        status = row.get("status", "")
        if status != "ok":
            artifact = row.get("artifact", "")
            scenario = row.get("scenario", "")
            missing = row.get("missing", "")
            errors.append(
                f"coverage row {index} failed: scenario={scenario}, artifact={artifact}, status={status}, missing={missing}"
            )
    return errors


def verify_comparison_artifacts(coverage_path: Path) -> list[str]:
    errors: list[str] = []
    comparison_dir = coverage_path.parent
    for filename in EXPECTED_SCENARIO_COMPARISON_FILES:
        path = comparison_dir / filename
        if not path.exists() or not path.is_file():
            errors.append(f"missing scenario comparison artifact: {path}")
            continue
        if path.stat().st_size == 0:
            errors.append(f"scenario comparison artifact is empty: {path}")
            continue
        if path.suffix == ".csv":
            rows = read_csv_rows(path)
            if not rows:
                errors.append(f"scenario comparison CSV has no data rows: {path}")
                continue
            observed_scenarios = {row.get("scenario", "") for row in rows}
            required_scenarios = set(EXPECTED_FULL_COMPARISON_SCENARIOS)
            missing_scenarios = sorted(required_scenarios - observed_scenarios)
            if missing_scenarios:
                errors.append(
                    f"scenario comparison CSV missing required scenarios in {path}: "
                    + ", ".join(missing_scenarios)
                )
            unexpected_scenarios = sorted(observed_scenarios - required_scenarios)
            if unexpected_scenarios:
                errors.append(
                    f"scenario comparison CSV contains unexpected scenarios in {path}: "
                    + ", ".join(unexpected_scenarios)
                )
    return errors


def manifest_step_summary(path: Path) -> dict[str, object]:
    payload = read_json_object(path)
    execution_results = payload.get("execution_results")
    if not isinstance(execution_results, list):
        return {
            "workflow_status": payload.get("workflow_status", ""),
            "overall_status": payload.get("overall_status", ""),
            "workflow_steps": 0,
            "recorded_verifier_steps": 0,
        }
    workflow_steps = 0
    verifier_steps = 0
    for item in execution_results:
        if not isinstance(item, dict):
            continue
        if is_verifier_record(item):
            verifier_steps += 1
        else:
            workflow_steps += 1
    return {
        "workflow_status": payload.get("workflow_status", ""),
        "overall_status": payload.get("overall_status", ""),
        "workflow_steps": workflow_steps,
        "recorded_verifier_steps": verifier_steps,
    }


def main() -> None:
    args = parse_args()
    errors = [
        *verify_manifest(args.manifest, coverage_path=args.coverage, allow_verifying=args.allow_verifying),
        *verify_coverage(args.coverage),
        *verify_comparison_artifacts(args.coverage),
    ]
    if errors:
        print("Full reproduction evidence verification failed:")
        for error in errors:
            print(f"- {error}")
        raise SystemExit(1)
    step_summary = manifest_step_summary(args.manifest)
    print("Full reproduction evidence verification passed.")
    print(f"- manifest: {args.manifest}")
    print(f"- coverage: {args.coverage}")
    print(f"- workflow_status: {step_summary['workflow_status']}")
    print(f"- overall_status: {step_summary['overall_status']}")
    print(f"- workflow_steps: {step_summary['workflow_steps']}")
    print(f"- recorded_verifier_steps: {step_summary['recorded_verifier_steps']}")


if __name__ == "__main__":
    main()
