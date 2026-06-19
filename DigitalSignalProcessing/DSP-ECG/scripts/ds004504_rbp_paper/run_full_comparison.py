from __future__ import annotations

import argparse
import json
import shlex
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from report_coverage import (
    EXPECTED_COMPARISON_SCRIPT,
    EXPECTED_FORMAL_REPRODUCTION_SCENARIO,
    EXPECTED_PIPELINE_SCRIPT,
    EXPECTED_SUPPORT_AUDIT_SCENARIO,
    EXPECTED_VERIFIER_SCRIPT,
)


PAPER_LITERAL_SCENARIO = EXPECTED_FORMAL_REPRODUCTION_SCENARIO
SUPPORT_AUDIT_SCENARIO = EXPECTED_SUPPORT_AUDIT_SCENARIO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the paper-literal reproduction, optional support-audit scenario, "
            "and final scenario comparison gate."
        )
    )
    parser.add_argument("--execute", action="store_true", help="Run commands instead of printing the plan only.")
    parser.add_argument(
        "--include-download",
        action="store_true",
        help="Pass --include-download to scenario pipelines.",
    )
    parser.add_argument(
        "--no-skip-existing",
        action="store_true",
        help="Pass --no-skip-existing to scenario pipelines.",
    )
    parser.add_argument(
        "--allow-partial-outputs",
        action="store_true",
        help="Pass --allow-partial-outputs to scenario pipelines.",
    )
    parser.add_argument(
        "--skip-val-as-test",
        action="store_true",
        help="Only run the paper-literal scenario and skip the 80/20 support-audit comparison scenario.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/reports/ds004504_rbp_paper/full_comparison_execution_manifest.json"),
        help="Path for writing the full-comparison command/execution manifest.",
    )
    parser.add_argument(
        "--coverage",
        type=Path,
        default=Path("data/reports/ds004504_rbp_paper/scenario_comparison/scenario_coverage.csv"),
        help="Scenario coverage CSV checked by the final verifier.",
    )
    return parser.parse_args()


def pipeline_command(scenario: str, args: argparse.Namespace) -> tuple[str, ...]:
    command = [
        "uv",
        "run",
        "python",
        EXPECTED_PIPELINE_SCRIPT,
        "--scenario",
        scenario,
        "--execute",
    ]
    if args.include_download:
        command.append("--include-download")
    if args.no_skip_existing:
        command.append("--no-skip-existing")
    if args.allow_partial_outputs:
        command.append("--allow-partial-outputs")
    return tuple(command)


def comparison_command() -> tuple[str, ...]:
    return (
        "uv",
        "run",
        "python",
        EXPECTED_COMPARISON_SCRIPT,
        "--fail-on-missing",
    )


def verifier_command(args: argparse.Namespace) -> tuple[str, ...]:
    return (
        "uv",
        "run",
        "python",
        EXPECTED_VERIFIER_SCRIPT,
        "--manifest",
        str(args.manifest),
        "--coverage",
        str(args.coverage),
        "--allow-verifying",
    )


def is_verifier_command(command: tuple[str, ...]) -> bool:
    return len(command) >= 4 and command[3] == EXPECTED_VERIFIER_SCRIPT


def build_commands(args: argparse.Namespace) -> tuple[tuple[str, ...], ...]:
    commands = [pipeline_command(PAPER_LITERAL_SCENARIO, args)]
    if not args.skip_val_as_test:
        commands.append(pipeline_command(SUPPORT_AUDIT_SCENARIO, args))
        commands.append(comparison_command())
        commands.append(verifier_command(args))
    return tuple(commands)


def planned_scenarios(args: argparse.Namespace) -> tuple[str, ...]:
    scenarios = [PAPER_LITERAL_SCENARIO]
    if not args.skip_val_as_test:
        scenarios.append(SUPPORT_AUDIT_SCENARIO)
    return tuple(scenarios)


def write_manifest(
    path: Path,
    *,
    args: argparse.Namespace,
    mode: str,
    scope: str,
    overall_status: str,
    workflow_status: str,
    commands: tuple[tuple[str, ...], ...],
    execution_results: list[dict[str, object]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": datetime.now(tz=UTC).isoformat(),
        "mode": mode,
        "scope": scope,
        "overall_status": overall_status,
        "workflow_status": workflow_status,
        "execute": args.execute,
        "include_download": args.include_download,
        "no_skip_existing": args.no_skip_existing,
        "allow_partial_outputs": args.allow_partial_outputs,
        "skip_val_as_test": args.skip_val_as_test,
        "scenarios": list(planned_scenarios(args)),
        "coverage": str(args.coverage),
        "pipeline_script": EXPECTED_PIPELINE_SCRIPT,
        "comparison_script": EXPECTED_COMPARISON_SCRIPT,
        "verifier_script": EXPECTED_VERIFIER_SCRIPT,
        "verifier_allow_verifying": not args.skip_val_as_test,
        "commands": [
            {
                "index": index,
                "command": list(command),
                "command_text": shlex.join(command),
            }
            for index, command in enumerate(commands, start=1)
        ],
        "execution_results": execution_results,
    }
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")


def main() -> None:
    args = parse_args()
    commands = build_commands(args)
    mode = "EXECUTE" if args.execute else "PLAN ONLY"
    scope = (
        "paper-literal reproduction only; scenario comparison is skipped"
        if args.skip_val_as_test
        else (
            "paper-literal reproduction, 80/20 support-audit scenario, "
            "final scenario comparison gate, and final verifier"
        )
    )
    print(f"Mode: {mode}")
    print(f"Scope: {scope}")
    execution_results: list[dict[str, object]] = []
    for index, command in enumerate(commands, start=1):
        prefix = "running" if args.execute else "would run"
        print(f"[{index}/{len(commands)}] {prefix}: {shlex.join(command)}")
        if args.execute:
            if is_verifier_command(command):
                write_manifest(
                    args.manifest,
                    args=args,
                    mode=mode,
                    scope=scope,
                    overall_status="verifying",
                    workflow_status="completed",
                    commands=commands,
                    execution_results=execution_results,
                )
            try:
                subprocess.run(command, check=True)
            except subprocess.CalledProcessError as exc:
                execution_results.append(
                    {
                        "index": index,
                        "command": list(command),
                        "command_text": shlex.join(command),
                        "status": "failed",
                        "returncode": exc.returncode,
                    }
                )
                write_manifest(
                    args.manifest,
                    args=args,
                    mode=mode,
                    scope=scope,
                    overall_status="failed",
                    workflow_status="completed" if is_verifier_command(command) else "failed",
                    commands=commands,
                    execution_results=execution_results,
                )
                print(f"Wrote full-comparison manifest to {args.manifest}")
                raise
            execution_results.append(
                {
                    "index": index,
                    "command": list(command),
                    "command_text": shlex.join(command),
                    "status": "completed",
                    "returncode": 0,
                }
            )
            print(f"[{index}/{len(commands)}] completed")
        else:
            execution_results.append(
                {
                    "index": index,
                    "command": list(command),
                    "command_text": shlex.join(command),
                    "status": "planned",
                    "returncode": "",
                }
            )
    if not args.execute:
        print("No commands were executed. Re-run with --execute to start the pipelines.")
    write_manifest(
        args.manifest,
        args=args,
        mode=mode,
        scope=scope,
        overall_status="completed" if args.execute else "planned",
        workflow_status="completed" if args.execute else "planned",
        commands=commands,
        execution_results=execution_results,
    )
    print(f"Wrote full-comparison manifest to {args.manifest}")


if __name__ == "__main__":
    main()
