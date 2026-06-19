from __future__ import annotations

import argparse
import json
import shlex
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path


DEFAULT_SCENARIO = "paper_literal_80_10_10"


@dataclass(frozen=True)
class PipelineStep:
    stage: str
    name: str
    command: tuple[str, ...]
    marker: Path | None

    def shell_text(self) -> str:
        return " ".join(shlex.quote(part) for part in self.command)


@dataclass(frozen=True)
class StepExecutionResult:
    stage: str
    name: str
    command: tuple[str, ...]
    marker: Path | None
    status: str
    returncode: int | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Print or execute the full ds004504 paper-table reproduction pipeline. "
            "By default this only prints the command plan; pass --execute to run it."
        )
    )
    parser.add_argument(
        "--scenario",
        default=DEFAULT_SCENARIO,
        choices=("paper_literal_80_10_10", "val_as_test_80_20", "fixture_smoke"),
        help="Report scenario for final CSV/table/plot generation.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually execute the command plan. Without this flag, commands are only printed.",
    )
    parser.add_argument(
        "--include-download",
        action="store_true",
        help="Include the OpenNeuro download step before processing.",
    )
    parser.add_argument(
        "--no-skip-existing",
        action="store_true",
        help="Do not skip steps whose output marker already exists.",
    )
    parser.add_argument(
        "--allow-partial-outputs",
        action="store_true",
        help=(
            "Allow execution even when a training output directory exists but its marker is missing. "
            "By default, execution stops early because train scripts use overwrite=false."
        ),
    )
    parser.add_argument(
        "--stage",
        action="append",
        choices=("download", "process", "train", "report"),
        help="Run only selected stages. May be passed multiple times. Defaults to all stages.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Optional path for writing a JSON execution manifest. Defaults under data/reports for the scenario.",
    )
    return parser.parse_args()


def uv_command(*parts: str) -> tuple[str, ...]:
    return ("uv", "run", *parts)


def python_script(script: str, *args: str) -> tuple[str, ...]:
    return uv_command("python", script, *args)


def modified_rbp_training_steps() -> tuple[PipelineStep, ...]:
    return (
        PipelineStep(
            stage="train",
            name="paper_literal_multiclass",
            command=python_script(
                "scripts/ds004504_rbp_paper/train.py",
                "cfgs/ds004504_rbp_paper/multiclass.yaml",
            ),
            marker=Path("data/runs/ds004504_rbp_paper/multiclass/test_metrics.json"),
        ),
        PipelineStep(
            stage="train",
            name="paper_literal_ad_ftd_vs_healthy",
            command=python_script(
                "scripts/ds004504_rbp_paper/train.py",
                "cfgs/ds004504_rbp_paper/ad_ftd_vs_healthy.yaml",
            ),
            marker=Path("data/runs/ds004504_rbp_paper/ad_ftd_vs_healthy/test_metrics.json"),
        ),
        PipelineStep(
            stage="train",
            name="paper_literal_ad_vs_healthy",
            command=python_script(
                "scripts/ds004504_rbp_paper/train.py",
                "cfgs/ds004504_rbp_paper/ad_vs_healthy.yaml",
            ),
            marker=Path("data/runs/ds004504_rbp_paper/ad_vs_healthy/test_metrics.json"),
        ),
        PipelineStep(
            stage="train",
            name="paper_literal_ftd_vs_healthy",
            command=python_script(
                "scripts/ds004504_rbp_paper/train.py",
                "cfgs/ds004504_rbp_paper/ftd_vs_healthy.yaml",
            ),
            marker=Path("data/runs/ds004504_rbp_paper/ftd_vs_healthy/test_metrics.json"),
        ),
    )


def val_as_test_training_steps() -> tuple[PipelineStep, ...]:
    return (
        PipelineStep(
            stage="train",
            name="val_as_test_multiclass",
            command=python_script(
                "scripts/ds004504_rbp_paper/train.py",
                "cfgs/ds004504_rbp_paper/val_as_test_80_20/multiclass.yaml",
            ),
            marker=Path("data/runs/ds004504_rbp_paper/val_as_test_80_20/multiclass/test_metrics.json"),
        ),
        PipelineStep(
            stage="train",
            name="val_as_test_ad_ftd_vs_healthy",
            command=python_script(
                "scripts/ds004504_rbp_paper/train.py",
                "cfgs/ds004504_rbp_paper/val_as_test_80_20/ad_ftd_vs_healthy.yaml",
            ),
            marker=Path("data/runs/ds004504_rbp_paper/val_as_test_80_20/ad_ftd_vs_healthy/test_metrics.json"),
        ),
        PipelineStep(
            stage="train",
            name="val_as_test_ad_vs_healthy",
            command=python_script(
                "scripts/ds004504_rbp_paper/train.py",
                "cfgs/ds004504_rbp_paper/val_as_test_80_20/ad_vs_healthy.yaml",
            ),
            marker=Path("data/runs/ds004504_rbp_paper/val_as_test_80_20/ad_vs_healthy/test_metrics.json"),
        ),
        PipelineStep(
            stage="train",
            name="val_as_test_ftd_vs_healthy",
            command=python_script(
                "scripts/ds004504_rbp_paper/train.py",
                "cfgs/ds004504_rbp_paper/val_as_test_80_20/ftd_vs_healthy.yaml",
            ),
            marker=Path("data/runs/ds004504_rbp_paper/val_as_test_80_20/ftd_vs_healthy/test_metrics.json"),
        ),
    )


def smote_training_steps(*, scenario: str) -> tuple[PipelineStep, ...]:
    if scenario == "paper_literal_80_10_10":
        return (
            PipelineStep(
                stage="train",
                name="smote_multiclass",
                command=python_script(
                    "scripts/ds004504_rbp_paper/train_smote.py",
                    "cfgs/ds004504_rbp_paper/smote/multiclass.yaml",
                ),
                marker=Path("data/runs/ds004504_rbp_paper/smote/multiclass/test_metrics.json"),
            ),
            PipelineStep(
                stage="train",
                name="smote_ad_vs_healthy",
                command=python_script(
                    "scripts/ds004504_rbp_paper/train_smote.py",
                    "cfgs/ds004504_rbp_paper/smote/ad_vs_healthy.yaml",
                ),
                marker=Path("data/runs/ds004504_rbp_paper/smote/ad_vs_healthy/test_metrics.json"),
            ),
        )
    if scenario == "val_as_test_80_20":
        return (
            PipelineStep(
                stage="train",
                name="smote_multiclass",
                command=python_script(
                    "scripts/ds004504_rbp_paper/train_smote.py",
                    "cfgs/ds004504_rbp_paper/val_as_test_80_20/smote/multiclass.yaml",
                ),
                marker=Path("data/runs/ds004504_rbp_paper/val_as_test_80_20/smote/multiclass/test_metrics.json"),
            ),
            PipelineStep(
                stage="train",
                name="smote_ad_vs_healthy",
                command=python_script(
                    "scripts/ds004504_rbp_paper/train_smote.py",
                    "cfgs/ds004504_rbp_paper/val_as_test_80_20/smote/ad_vs_healthy.yaml",
                ),
                marker=Path("data/runs/ds004504_rbp_paper/val_as_test_80_20/smote/ad_vs_healthy/test_metrics.json"),
            ),
        )
    raise ValueError(f"Unsupported scenario: {scenario}")


def auxiliary_training_steps(*, scenario: str) -> tuple[PipelineStep, ...]:
    steps: list[PipelineStep] = [
        PipelineStep(
            stage="train",
            name="standard_rbp_multiclass",
            command=python_script(
                "scripts/ds004504_rbp_paper/train.py",
                "cfgs/ds004504_rbp_paper/standard_rbp/multiclass.yaml",
            ),
            marker=Path("data/runs/ds004504_rbp_paper/standard_rbp/multiclass/test_metrics.json"),
        ),
        PipelineStep(
            stage="train",
            name="standard_rbp_ad_vs_healthy",
            command=python_script(
                "scripts/ds004504_rbp_paper/train.py",
                "cfgs/ds004504_rbp_paper/standard_rbp/ad_vs_healthy.yaml",
            ),
            marker=Path("data/runs/ds004504_rbp_paper/standard_rbp/ad_vs_healthy/test_metrics.json"),
        ),
        PipelineStep(
            stage="train",
            name="kfold_multiclass",
            command=python_script(
                "scripts/ds004504_rbp_paper/train_kfold.py",
                "cfgs/ds004504_rbp_paper/kfold/multiclass.yaml",
            ),
            marker=Path("data/runs/ds004504_rbp_paper/kfold/multiclass/fold_summary.csv"),
        ),
        PipelineStep(
            stage="train",
            name="kfold_ad_vs_healthy",
            command=python_script(
                "scripts/ds004504_rbp_paper/train_kfold.py",
                "cfgs/ds004504_rbp_paper/kfold/ad_vs_healthy.yaml",
            ),
            marker=Path("data/runs/ds004504_rbp_paper/kfold/ad_vs_healthy/fold_summary.csv"),
        ),
    ]
    if scenario == "paper_literal_80_10_10":
        steps.extend(
            [
                PipelineStep(
                    stage="train",
                    name="label_swap_multiclass",
                    command=python_script(
                        "scripts/ds004504_rbp_paper/train.py",
                        "cfgs/ds004504_rbp_paper/label_swap/multiclass.yaml",
                    ),
                    marker=Path("data/runs/ds004504_rbp_paper/label_swap/multiclass/test_metrics.json"),
                ),
                PipelineStep(
                    stage="train",
                    name="label_swap_ftd_vs_healthy",
                    command=python_script(
                        "scripts/ds004504_rbp_paper/train.py",
                        "cfgs/ds004504_rbp_paper/label_swap/ftd_vs_healthy.yaml",
                    ),
                    marker=Path("data/runs/ds004504_rbp_paper/label_swap/ftd_vs_healthy/test_metrics.json"),
                ),
            ]
        )
    elif scenario == "val_as_test_80_20":
        steps.extend(
            [
                PipelineStep(
                    stage="train",
                    name="label_swap_multiclass",
                    command=python_script(
                        "scripts/ds004504_rbp_paper/train.py",
                        "cfgs/ds004504_rbp_paper/label_swap_80_20/multiclass.yaml",
                    ),
                    marker=Path("data/runs/ds004504_rbp_paper/label_swap_80_20/multiclass/test_metrics.json"),
                ),
                PipelineStep(
                    stage="train",
                    name="label_swap_ftd_vs_healthy",
                    command=python_script(
                        "scripts/ds004504_rbp_paper/train.py",
                        "cfgs/ds004504_rbp_paper/label_swap_80_20/ftd_vs_healthy.yaml",
                    ),
                    marker=Path("data/runs/ds004504_rbp_paper/label_swap_80_20/ftd_vs_healthy/test_metrics.json"),
                ),
            ]
        )
    steps.extend(smote_training_steps(scenario=scenario))
    return tuple(steps)


def build_steps(*, scenario: str, include_download: bool) -> tuple[PipelineStep, ...]:
    steps: list[PipelineStep] = []
    if scenario == "fixture_smoke":
        return (
            PipelineStep(
                stage="train",
                name="make_fixture_runs",
                command=python_script(
                    "scripts/ds004504_rbp_paper/make_fixture_runs.py",
                    "--overwrite",
                ),
                marker=Path("data/runs/ds004504_rbp_paper/fixture_smoke/multiclass/test_metrics.json"),
            ),
            PipelineStep(
                stage="report",
                name="make_report_csv",
                command=python_script(
                    "scripts/ds004504_rbp_paper/make_report_csv.py",
                    "--scenario",
                    scenario,
                ),
                marker=Path("data/reports/ds004504_rbp_paper") / scenario / "tables" / "paper_vs_ours.csv",
            ),
            PipelineStep(
                stage="report",
                name="render_report_tables",
                command=python_script(
                    "scripts/ds004504_rbp_paper/render_report_tables.py",
                    "--scenario",
                    scenario,
                ),
                marker=Path("data/reports/ds004504_rbp_paper") / scenario / "paper_tables" / "index.md",
            ),
            PipelineStep(
                stage="report",
                name="plot_report",
                command=python_script(
                    "scripts/ds004504_rbp_paper/plot_report.py",
                    "--scenario",
                    scenario,
                ),
                marker=Path("data/reports/ds004504_rbp_paper") / scenario / "plots" / "accuracy_comparison.svg",
            ),
            PipelineStep(
                stage="report",
                name="audit_reproduction_artifacts",
                command=python_script(
                    "scripts/ds004504_rbp_paper/audit_reproduction_artifacts.py",
                    "--scenario",
                    scenario,
                    "--fail-on-missing",
                ),
                marker=Path("data/reports/ds004504_rbp_paper") / scenario / "audit" / "audit.json",
            ),
        )
    if include_download:
        steps.append(
            PipelineStep(
                stage="download",
                name="download_ds004504_v1_0_5",
                command=uv_command(
                    "download_raw_dataset",
                    "--dataset",
                    "ds004504",
                    "--tag",
                    "1.0.5",
                    "--target-dir",
                    "data/raw/ds004504",
                ),
                marker=Path("data/raw/ds004504/participants.tsv"),
            )
        )

    steps.extend(
        [
            PipelineStep(
                stage="process",
                name="process_modified_rbp",
                command=uv_command(
                    "process_raw_dataset",
                    "--dataset",
                    "ds004504_rbp_paper",
                    "--raw-dir",
                    "data/raw/ds004504",
                    "--output",
                    "data/processed_raw_dataset/ds004504_rbp_paper.h5",
                    "--manifest",
                    "data/processed_raw_dataset/ds004504_rbp_paper_manifest.json",
                ),
                marker=Path("data/processed_raw_dataset/ds004504_rbp_paper_manifest.json"),
            ),
            PipelineStep(
                stage="process",
                name="process_standard_rbp",
                command=uv_command(
                    "process_raw_dataset",
                    "--dataset",
                    "ds004504_standard_rbp_paper",
                    "--raw-dir",
                    "data/raw/ds004504",
                    "--output",
                    "data/processed_raw_dataset/ds004504_standard_rbp_paper.h5",
                    "--manifest",
                    "data/processed_raw_dataset/ds004504_standard_rbp_paper_manifest.json",
                ),
                marker=Path("data/processed_raw_dataset/ds004504_standard_rbp_paper_manifest.json"),
            ),
        ]
    )

    if scenario == "paper_literal_80_10_10":
        steps.extend(modified_rbp_training_steps())
    elif scenario == "val_as_test_80_20":
        steps.extend(val_as_test_training_steps())
    else:
        raise ValueError(f"Unsupported scenario: {scenario}")

    steps.extend(auxiliary_training_steps(scenario=scenario))
    steps.extend(
        [
            PipelineStep(
                stage="report",
                name="make_report_csv",
                command=python_script(
                    "scripts/ds004504_rbp_paper/make_report_csv.py",
                    "--scenario",
                    scenario,
                ),
                marker=Path("data/reports/ds004504_rbp_paper") / scenario / "tables" / "paper_vs_ours.csv",
            ),
            PipelineStep(
                stage="report",
                name="render_report_tables",
                command=python_script(
                    "scripts/ds004504_rbp_paper/render_report_tables.py",
                    "--scenario",
                    scenario,
                ),
                marker=Path("data/reports/ds004504_rbp_paper") / scenario / "paper_tables" / "index.md",
            ),
            PipelineStep(
                stage="report",
                name="plot_report",
                command=python_script(
                    "scripts/ds004504_rbp_paper/plot_report.py",
                    "--scenario",
                    scenario,
                ),
                marker=Path("data/reports/ds004504_rbp_paper") / scenario / "plots" / "accuracy_comparison.svg",
            ),
            PipelineStep(
                stage="report",
                name="audit_reproduction_artifacts",
                command=python_script(
                    "scripts/ds004504_rbp_paper/audit_reproduction_artifacts.py",
                    "--scenario",
                    scenario,
                ),
                marker=Path("data/reports/ds004504_rbp_paper") / scenario / "audit" / "audit.json",
            ),
        ]
    )
    return tuple(steps)


def selected_steps(steps: tuple[PipelineStep, ...], stages: tuple[str, ...] | None) -> tuple[PipelineStep, ...]:
    if stages is None:
        return steps
    selected = set(stages)
    return tuple(step for step in steps if step.stage in selected)


def print_plan(steps: tuple[PipelineStep, ...], *, skip_existing: bool) -> None:
    for index, step in enumerate(steps, start=1):
        marker_text = str(step.marker) if step.marker is not None else ""
        status = "ready"
        if should_skip_existing_step(step, skip_existing=skip_existing):
            status = "skip-existing"
        elif is_partial_training_output(step):
            status = "partial-output"
        print(f"{index:02d}. [{step.stage}] {step.name} ({status})")
        if marker_text:
            print(f"    marker: {marker_text}")
        print(f"    {step.shell_text()}")


def step_status(step: PipelineStep, *, skip_existing: bool) -> str:
    if should_skip_existing_step(step, skip_existing=skip_existing):
        return "skip-existing"
    if is_partial_training_output(step):
        return "partial-output"
    return "ready"


def should_skip_existing_step(step: PipelineStep, *, skip_existing: bool) -> bool:
    if not skip_existing:
        return False
    if step.stage == "report":
        return False
    if step.name == "make_fixture_runs":
        return False
    return step.marker is not None and step.marker.exists()


def write_execution_manifest(
    path: Path,
    *,
    scenario: str,
    execute: bool,
    skip_existing: bool,
    steps: tuple[PipelineStep, ...],
    execution_results: tuple[StepExecutionResult, ...] = (),
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "scenario": scenario,
        "execute": execute,
        "skip_existing": skip_existing,
        "steps": [
            {
                "stage": step.stage,
                "name": step.name,
                "command": list(step.command),
                "command_text": step.shell_text(),
                "marker": "" if step.marker is None else str(step.marker),
                "status_at_plan_time": step_status(step, skip_existing=skip_existing),
            }
            for step in steps
        ],
        "execution_results": [
            {
                "stage": result.stage,
                "name": result.name,
                "command": list(result.command),
                "command_text": " ".join(shlex.quote(part) for part in result.command),
                "marker": "" if result.marker is None else str(result.marker),
                "status": result.status,
                "returncode": result.returncode,
            }
            for result in execution_results
        ],
    }
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")


def is_partial_training_output(step: PipelineStep) -> bool:
    if step.name == "make_fixture_runs":
        return False
    if step.stage != "train" or step.marker is None or step.marker.exists():
        return False
    output_dir = step.marker.parent
    return output_dir.exists() and any(output_dir.iterdir())


def partial_training_outputs(steps: tuple[PipelineStep, ...]) -> tuple[PipelineStep, ...]:
    return tuple(step for step in steps if is_partial_training_output(step))


def run_steps(steps: tuple[PipelineStep, ...], *, skip_existing: bool) -> tuple[StepExecutionResult, ...]:
    results: list[StepExecutionResult] = []
    for index, step in enumerate(steps, start=1):
        if should_skip_existing_step(step, skip_existing=skip_existing):
            print(f"[{index}/{len(steps)}] skip existing {step.name}: {step.marker}")
            results.append(
                StepExecutionResult(
                    stage=step.stage,
                    name=step.name,
                    command=step.command,
                    marker=step.marker,
                    status="skipped_existing",
                    returncode=None,
                )
            )
            continue
        print(f"[{index}/{len(steps)}] running {step.name}")
        print(step.shell_text())
        completed = subprocess.run(step.command, check=False)
        status = "completed" if completed.returncode == 0 else "failed"
        results.append(
            StepExecutionResult(
                stage=step.stage,
                name=step.name,
                command=step.command,
                marker=step.marker,
                status=status,
                returncode=completed.returncode,
            )
        )
        if completed.returncode != 0:
            break
    return tuple(results)


def first_failed_result(results: tuple[StepExecutionResult, ...]) -> StepExecutionResult | None:
    for result in results:
        if result.status == "failed":
            return result
    return None


def main() -> None:
    args = parse_args()
    stages = tuple(args.stage) if args.stage is not None else None
    steps = selected_steps(
        build_steps(scenario=args.scenario, include_download=args.include_download),
        stages,
    )
    skip_existing = not args.no_skip_existing
    print_plan(steps, skip_existing=skip_existing)
    manifest_path = args.manifest
    if manifest_path is None:
        manifest_path = Path("data/reports/ds004504_rbp_paper") / args.scenario / "pipeline_execution_manifest.json"
    write_execution_manifest(
        manifest_path,
        scenario=args.scenario,
        execute=bool(args.execute),
        skip_existing=skip_existing,
        steps=steps,
    )
    print(f"Wrote execution manifest: {manifest_path}")
    if args.execute:
        partial_steps = partial_training_outputs(steps)
        if partial_steps and not args.allow_partial_outputs:
            names = ", ".join(f"{step.name} -> {step.marker.parent}" for step in partial_steps if step.marker is not None)
            raise SystemExit(
                "Refusing to execute with partial training output directories because training configs use "
                f"overwrite=false: {names}. Remove those directories, complete those runs, or pass "
                "--allow-partial-outputs to let the underlying command fail or handle it."
            )
        execution_results = run_steps(steps, skip_existing=skip_existing)
        write_execution_manifest(
            manifest_path,
            scenario=args.scenario,
            execute=True,
            skip_existing=skip_existing,
            steps=steps,
            execution_results=execution_results,
        )
        print(f"Updated execution manifest with results: {manifest_path}")
        failed_result = first_failed_result(execution_results)
        if failed_result is not None:
            raise subprocess.CalledProcessError(
                failed_result.returncode if failed_result.returncode is not None else 1,
                failed_result.command,
            )


if __name__ == "__main__":
    main()
