from __future__ import annotations

import re


EXPECTED_TABLE_NUMBERS = tuple(range(1, 15))
EXPECTED_PAPER_VS_OURS_TABLE_NUMBERS = tuple(range(3, 14))
EXPECTED_ISSUE_IDS = (
    "split_support_mismatch",
    "ftd_healthy_support_swap",
    "table6_healthy_support_typo",
    "smote_accuracy_inconsistency",
    "unspecified_smote_placement",
    "epoch_level_leakage_risk",
)
EXPECTED_PROTOCOL_COMPONENTS = (
    "raw_dataset",
    "preprocessing_source",
    "epoching",
    "modified_rbp_bands",
    "standard_rbp_bands",
    "normalization",
    "split",
    "evaluation",
    "model",
    "model_hyperparameters",
    "optimizer",
    "training_runtime",
    "smote",
    "kfold",
    "label_swap_audit",
    "runs_dir",
)
EXPECTED_SCENARIO_COVERAGE_ARTIFACTS = (
    "table_summary.csv:paper_table",
    "paper_vs_ours.csv:paper_table",
    "protocol_manifest.csv:component",
    "issue_summary.csv:issue_id",
)
EXPECTED_SCENARIO_COMPARISON_FILES = (
    "scenario_table_summary.csv",
    "scenario_protocol_summary.csv",
    "scenario_paper_vs_ours.csv",
    "scenario_issue_summary.csv",
    "scenario_comparison.md",
)
EXPECTED_FORMAL_REPRODUCTION_SCENARIO = "paper_literal_80_10_10"
EXPECTED_SUPPORT_AUDIT_SCENARIO = "val_as_test_80_20"
EXPECTED_FULL_COMPARISON_SCENARIOS = (
    EXPECTED_FORMAL_REPRODUCTION_SCENARIO,
    EXPECTED_SUPPORT_AUDIT_SCENARIO,
)
EXPECTED_PIPELINE_SCRIPT = "scripts/ds004504_rbp_paper/run_reproduction_pipeline.py"
EXPECTED_COMPARISON_SCRIPT = "scripts/ds004504_rbp_paper/compare_report_scenarios.py"
EXPECTED_VERIFIER_SCRIPT = "scripts/ds004504_rbp_paper/verify_full_reproduction.py"


def paper_table_number(value: object) -> int | None:
    match = re.search(r"\bTable\s+(\d+)\b", str(value))
    if match is None:
        return None
    return int(match.group(1))
