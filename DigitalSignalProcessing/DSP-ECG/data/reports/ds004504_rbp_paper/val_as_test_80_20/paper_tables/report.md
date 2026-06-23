# ds004504 paper table reproduction report: `val_as_test_80_20`

This report is generated from normalized CSV outputs. Paper-side values and ours-side outputs are kept together for comparison.

## Scenario role

Audit/comparison path inferred from reported support counts: 80% train, 20% validation-as-test.

A formal reproduction is complete only after `audit_reproduction_artifacts.py --fail-on-missing` succeeds for this scenario.

<!-- source: protocol_manifest.md -->
# Protocol manifest

| Scenario | Component | Value | Evidence | Notes |
| --- | --- | --- | --- | --- |
| val_as_test_80_20 | raw_dataset | OpenNeuro ds004504 v1.0.5 | cfgs/ds004504_rbp_paper/base.yaml processing.download_raw_dataset.tag | Paper data availability statement points to openneuro.ds004504.v1.0.5. |
| val_as_test_80_20 | preprocessing_source | derivatives/sub-*/eeg/*_task-eyesclosed_eeg.set | src/ecg/data/ds004504_rbp_paper.py EEG_GLOB | Uses preprocessed derivative EEGLAB .set files. |
| val_as_test_80_20 | epoching | 6 seconds with 50% overlap | src/ecg/data/ds004504_rbp_paper.py EPOCH_SEC and OVERLAP |  |
| val_as_test_80_20 | modified_rbp_bands | delta 0.5-4, theta 4-8, alpha 8-16, zaeta 16-24, beta 24-30, gamma 30-45 | src/ecg/data/ds004504_rbp_paper.py MODIFIED_RBP_BANDS | Used for Tables 3-11. |
| val_as_test_80_20 | standard_rbp_bands | delta 0.5-4, theta 4-8, alpha 8-13, beta 13-25, gamma 25-45 | src/ecg/data/ds004504_rbp_paper.py STANDARD_RBP_BANDS | Used for Tables 12-13. |
| val_as_test_80_20 | normalization | paper-style full-task min-max normalization before split | src/ecg/training/ds004504_rbp_paper/factory.py build_paper_experiment | This follows the paper wording but leaks evaluation distribution statistics. |
| val_as_test_80_20 | split | 80/20 epoch-level stratified split | cfgs/ds004504_rbp_paper/* and run.json split_fractions |  |
| val_as_test_80_20 | evaluation | reported metrics use validation partition as test output, matching paper support evidence | run.json evaluation.test_source and test_metrics.json source |  |
| val_as_test_80_20 | model | TCN-LSTM, two TCN blocks, 32 channels, kernel 7, LSTM 64, Dense 128/192/256 | src/ecg/training/ds004504_rbp_paper/factory.py paper hyperparameters |  |
| val_as_test_80_20 | model_hyperparameters | input_dim=1, num_classes=task-dependent, tcn_channels=32, tcn_kernel_size=7, tcn_dilations=[1, 1], tcn_dropout=0.3, lstm_hidden_dim=64, dense_hidden_dims=[128, 192, 256], dense_dropout=0.2 | src/ecg/training/ds004504_rbp_paper/factory.py build_paper_hyperparameters | num_classes is 3 for multiclass tasks and 2 for binary tasks. |
| val_as_test_80_20 | optimizer | Adam learning_rate=0.0001 batch_size=32 | src/ecg/training/ds004504_rbp_paper/factory.py PAPER_LEARNING_RATE and PAPER_BATCH_SIZE |  |
| val_as_test_80_20 | training_runtime | epochs=100, seed=randomly resolved at runtime unless configured, num_workers=4, device=cuda | cfgs/ds004504_rbp_paper/base.yaml training/runtime and run.json resolved_seed | run.json records the actual resolved seed for each run. |
| val_as_test_80_20 | smote | simple SMOTE applied to train and reported val/test partitions for Table 8-9 shape | scripts/ds004504_rbp_paper/train_smote.py and cfgs/*/smote/*.yaml | Paper does not provide source code or exact SMOTE placement. |
| val_as_test_80_20 | kfold | 5-fold epoch-level stratified k-fold | scripts/ds004504_rbp_paper/train_kfold.py | Paper does not specify subject-wise folds. |
| val_as_test_80_20 | label_swap_audit | FTD/Healthy target-label swap audit for Table 3 and Table 6 | cfgs/ds004504_rbp_paper/label_swap_80_20/*.yaml | Audit-only protocol, not the corrected dataset protocol. |
| val_as_test_80_20 | runs_dir | data/runs/ds004504_rbp_paper/val_as_test_80_20 | make_report_csv.py --runs-dir/default_runs_dir |  |

<!-- source: issue_summary.md -->
# Reproduction issue summary

| Severity | Issue | Affected tables | Paper observation | Reproduction implication | Project handling |
| --- | --- | --- | --- | --- | --- |
| high | split_support_mismatch | Table 3, Table 4, Table 5, Table 6, Table 7 | Paper text claims 80/10/10 train/validation/test split. | Reported supports are close to a 20% holdout, not a 10% test split. | Provide both paper_literal_80_10_10 and val_as_test_80_20 scenarios; default report uses paper_literal_80_10_10 to follow the paper text, while val_as_test_80_20 is retained for paper-inferred support comparison. |
| high | ftd_healthy_support_swap | Table 3, Table 4, Table 6 | FTD and Healthy supports align much better if FTD/Healthy are swapped. | Per-class interpretation of model behavior may be reversed for FTD and Healthy. | Provide label_swap and label_swap_80_20 audit runs and include ours_label_swap rows in Table 3 and Table 6 reports. |
| medium | table6_healthy_support_typo | Table 6 | Table 6 reports Healthy support 1596, but Figure 4d and recall imply 1106. | Table support column cannot be used blindly; Figure 4 row sums are needed for consistency checks. | support_comparison.csv includes both paper table support and paper figure row-sum support. |
| medium | smote_accuracy_inconsistency | Table 8 | Paper text reports SMOTE accuracy 77.45%, but Table 8 equal supports and recalls imply a different value. | SMOTE headline accuracy should be compared separately from row-level metrics. | accuracy.csv and paper_vs_ours.csv keep Table 8 text accuracy separate from row metrics. |
| medium | unspecified_smote_placement | Table 8, Table 9 | Paper does not specify whether SMOTE was applied before split, after split, or only to training data. | Exact SMOTE reproduction is not uniquely determined from the paper. | Use explicit simple SMOTE with documented partitions in protocol_manifest.csv. |
| high | epoch_level_leakage_risk | Table 3 through Table 13 | Paper describes epoch-level split and full min-max normalization before split. | Results may be optimistic because adjacent epochs and normalization statistics can leak across partitions. | Reproduce paper-style protocol explicitly and record leakage risk in protocol_manifest.csv; corrected subject-wise protocol is outside Table 1-14 paper reproduction. |

<!-- source: table_summary.md -->
# Table-level comparison summary

| Table | Purpose | Paper accuracy | Ours accuracy | Paper support | Ours support | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| Table 1 | model architecture |  |  |  |  | Architecture table; no accuracy/support metrics. |
| Table 2 | model parameter summary |  |  |  |  | Parameter-count table; see model_parameters.csv for paper and ours counts. |
| Table 3 | modified RBP multiclass metrics |  |  | multiclass/alzheimer=1876; multiclass/frontotemporal_dementia=1597; multiclass/healthy_control=1106 | multiclass/alzheimer=1932; multiclass/frontotemporal_dementia=1100; multiclass/healthy_control=1599 | See paper_vs_ours.csv for per-metric differences. |
| Table 4 | modified RBP AD+FTD vs Healthy metrics |  |  | ad_ftd_vs_healthy/alzheimer_or_frontotemporal_dementia=2983; ad_ftd_vs_healthy/healthy_control=1596 | ad_ftd_vs_healthy/alzheimer_or_frontotemporal_dementia=3032; ad_ftd_vs_healthy/healthy_control=1599 | See paper_vs_ours.csv for per-metric differences. |
| Table 5 | modified RBP AD vs Healthy metrics |  |  | ad_vs_healthy/alzheimer=1876; ad_vs_healthy/healthy_control=1596 | ad_vs_healthy/alzheimer=1932; ad_vs_healthy/healthy_control=1599 | See paper_vs_ours.csv for per-metric differences. |
| Table 6 | modified RBP FTD vs Healthy metrics |  |  | ftd_vs_healthy/frontotemporal_dementia=1597; ftd_vs_healthy/healthy_control=1596 | ftd_vs_healthy/frontotemporal_dementia=1100; ftd_vs_healthy/healthy_control=1599 | See paper_vs_ours.csv for per-metric differences. |
| Table 7 | modified RBP task accuracy | ftd_vs_healthy=0.997; ad_vs_healthy=0.9974; ad_ftd_vs_healthy=0.998; multiclass=0.8034 | ftd_vs_healthy=0.7121155983697666; ad_vs_healthy=0.7187765505522515; ad_ftd_vs_healthy=0.7354782984236666; multiclass=0.5789246383070611 |  |  | See paper_vs_ours.csv for per-metric differences. |
| Table 8 | SMOTE multiclass metrics | multiclass=77.45% | multiclass=0.5488267770876466 | multiclass/alzheimer=1876; multiclass/frontotemporal_dementia=1876; multiclass/healthy_control=1876 | multiclass/alzheimer=1932; multiclass/frontotemporal_dementia=1932; multiclass/healthy_control=1932 | See paper_vs_ours.csv for per-metric differences. |
| Table 9 | SMOTE AD vs Healthy metrics |  |  | ad_vs_healthy/alzheimer=1876; ad_vs_healthy/healthy_control=1876 | ad_vs_healthy/alzheimer=1932; ad_vs_healthy/healthy_control=1932 | See paper_vs_ours.csv for per-metric differences. |
| Table 10 | 5-fold multiclass accuracy |  |  |  |  | See paper_vs_ours.csv for per-metric differences. |
| Table 11 | 5-fold AD vs Healthy accuracy |  |  |  |  | See paper_vs_ours.csv for per-metric differences. |
| Table 12 | standard RBP multiclass metrics | multiclass=63.03% | multiclass=0.5621225194132873 | multiclass/alzheimer=1876; multiclass/frontotemporal_dementia=1597; multiclass/healthy_control=1106 | multiclass/alzheimer=967; multiclass/frontotemporal_dementia=551; multiclass/healthy_control=800 | See paper_vs_ours.csv for per-metric differences. |
| Table 13 | standard RBP AD vs Healthy metrics | ad_vs_healthy=76.36% | ad_vs_healthy=0.7176004527447651 | ad_vs_healthy/alzheimer=1876; ad_vs_healthy/healthy_control=1597 | ad_vs_healthy/alzheimer=967; ad_vs_healthy/healthy_control=800 | See paper_vs_ours.csv for per-metric differences. |
| Table 14 | literature comparison |  |  |  |  | Literature comparison table; no ours-side training output. |

<!-- source: confusion_matrices.md -->
# Confusion matrices

| Source | Scenario | Figure | Table | Task | True class | Predicted class | Count | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| paper_figure | paper | Figure 4a | Table 3 | multiclass | alzheimer | alzheimer | 1693 |  |
| paper_figure | paper | Figure 4a | Table 3 | multiclass | alzheimer | frontotemporal_dementia | 4 |  |
| paper_figure | paper | Figure 4a | Table 3 | multiclass | alzheimer | healthy_control | 179 |  |
| paper_figure | paper | Figure 4a | Table 3 | multiclass | frontotemporal_dementia | alzheimer | 1 |  |
| paper_figure | paper | Figure 4a | Table 3 | multiclass | frontotemporal_dementia | frontotemporal_dementia | 1594 |  |
| paper_figure | paper | Figure 4a | Table 3 | multiclass | frontotemporal_dementia | healthy_control | 2 |  |
| paper_figure | paper | Figure 4a | Table 3 | multiclass | healthy_control | alzheimer | 712 |  |
| paper_figure | paper | Figure 4a | Table 3 | multiclass | healthy_control | frontotemporal_dementia | 2 |  |
| paper_figure | paper | Figure 4a | Table 3 | multiclass | healthy_control | healthy_control | 392 |  |
| paper_figure | paper | Figure 4b | Table 4 | ad_ftd_vs_healthy | alzheimer_or_frontotemporal_dementia | alzheimer_or_frontotemporal_dementia | 2981 |  |
| paper_figure | paper | Figure 4b | Table 4 | ad_ftd_vs_healthy | alzheimer_or_frontotemporal_dementia | healthy_control | 2 |  |
| paper_figure | paper | Figure 4b | Table 4 | ad_ftd_vs_healthy | healthy_control | alzheimer_or_frontotemporal_dementia | 7 |  |
| paper_figure | paper | Figure 4b | Table 4 | ad_ftd_vs_healthy | healthy_control | healthy_control | 1589 |  |
| paper_figure | paper | Figure 4c | Table 5 | ad_vs_healthy | alzheimer | alzheimer | 1874 |  |
| paper_figure | paper | Figure 4c | Table 5 | ad_vs_healthy | alzheimer | healthy_control | 2 |  |
| paper_figure | paper | Figure 4c | Table 5 | ad_vs_healthy | healthy_control | alzheimer | 7 |  |
| paper_figure | paper | Figure 4c | Table 5 | ad_vs_healthy | healthy_control | healthy_control | 1590 |  |
| paper_figure | paper | Figure 4d | Table 6 | ftd_vs_healthy | frontotemporal_dementia | frontotemporal_dementia | 1590 |  |
| paper_figure | paper | Figure 4d | Table 6 | ftd_vs_healthy | frontotemporal_dementia | healthy_control | 7 |  |
| paper_figure | paper | Figure 4d | Table 6 | ftd_vs_healthy | healthy_control | frontotemporal_dementia | 1 |  |
| paper_figure | paper | Figure 4d | Table 6 | ftd_vs_healthy | healthy_control | healthy_control | 1105 |  |
| ours | val_as_test_80_20 |  | Table 3 | multiclass | alzheimer | alzheimer | 1525 |  |
| ours | val_as_test_80_20 |  | Table 3 | multiclass | alzheimer | frontotemporal_dementia | 44 |  |
| ours | val_as_test_80_20 |  | Table 3 | multiclass | alzheimer | healthy_control | 363 |  |
| ours | val_as_test_80_20 |  | Table 3 | multiclass | frontotemporal_dementia | alzheimer | 743 |  |
| ours | val_as_test_80_20 |  | Table 3 | multiclass | frontotemporal_dementia | frontotemporal_dementia | 147 |  |
| ours | val_as_test_80_20 |  | Table 3 | multiclass | frontotemporal_dementia | healthy_control | 210 |  |
| ours | val_as_test_80_20 |  | Table 3 | multiclass | healthy_control | alzheimer | 558 |  |
| ours | val_as_test_80_20 |  | Table 3 | multiclass | healthy_control | frontotemporal_dementia | 32 |  |
| ours | val_as_test_80_20 |  | Table 3 | multiclass | healthy_control | healthy_control | 1009 |  |
| ours | val_as_test_80_20 |  | Table 4 | ad_ftd_vs_healthy | alzheimer_or_frontotemporal_dementia | alzheimer_or_frontotemporal_dementia | 2708 |  |
| ours | val_as_test_80_20 |  | Table 4 | ad_ftd_vs_healthy | alzheimer_or_frontotemporal_dementia | healthy_control | 324 |  |
| ours | val_as_test_80_20 |  | Table 4 | ad_ftd_vs_healthy | healthy_control | alzheimer_or_frontotemporal_dementia | 901 |  |
| ours | val_as_test_80_20 |  | Table 4 | ad_ftd_vs_healthy | healthy_control | healthy_control | 698 |  |
| ours | val_as_test_80_20 |  | Table 5 | ad_vs_healthy | alzheimer | alzheimer | 1470 |  |
| ours | val_as_test_80_20 |  | Table 5 | ad_vs_healthy | alzheimer | healthy_control | 462 |  |
| ours | val_as_test_80_20 |  | Table 5 | ad_vs_healthy | healthy_control | alzheimer | 531 |  |
| ours | val_as_test_80_20 |  | Table 5 | ad_vs_healthy | healthy_control | healthy_control | 1068 |  |
| ours | val_as_test_80_20 |  | Table 6 | ftd_vs_healthy | frontotemporal_dementia | frontotemporal_dementia | 817 |  |
| ours | val_as_test_80_20 |  | Table 6 | ftd_vs_healthy | frontotemporal_dementia | healthy_control | 283 |  |
| ours | val_as_test_80_20 |  | Table 6 | ftd_vs_healthy | healthy_control | frontotemporal_dementia | 494 |  |
| ours | val_as_test_80_20 |  | Table 6 | ftd_vs_healthy | healthy_control | healthy_control | 1105 |  |

<!-- source: table_01_model_architecture.md -->
# Table 1. Model architecture summary

| Order | Layer type | Output shape | Parameters | Connected to |
| --- | --- | --- | --- | --- |
| 1 | Input layer | (None, 6, 1) | 0 | - |
| 2 | Conv 1D | (None, 6, 32) | 256 | Input layer |
| 3 | Batch normalization | (None, 6, 32) | 128 | Conv1D |
| 4 | Activation | (None, 6, 32) | 0 | Batch normalization |
| 5 | Spatial dropout 1D | (None, 6, 32) | 0 | Activation |
| 6 | Conv1D | (None, 6, 32) | 7200 | Spatial dropout 1D |
| 7 | Batch normalization | (None, 6, 32) | 128 | Conv1D |
| 8 | Activation | (None, 6, 32) | 0 | Batch normalization |
| 9 | Conv 1D residual | (None, 6, 32) | 64 | Input layer |
| 10 | Spatial dropout 1D | (None, 6, 32) | 0 | Activation |
| 11 | Add | (None, 6, 32) | 0 | Conv1D + Spatial dropout |
| 12 | Conv 1D | (None, 6, 32) | 7200 | Add |
| 13 | Batch normalization | (None, 6, 32) | 128 | Conv1D |
| 14 | Activation | (None, 6, 32) | 0 | Batch normalization |
| 15 | Spatial dropout 1D | (None, 6, 32) | 0 | Activation |
| 16 | Conv 1D | (None, 6, 32) | 7200 | Spatial dropout 1D |
| 17 | Batch normalization | (None, 6, 32) | 128 | Conv 1D |
| 18 | Activation | (None, 6, 32) | 0 | Batch normalization |
| 19 | Conv 1D residual | (None, 6, 32) | 1056 | Add |
| 20 | Spatial dropout 1D | (None, 6, 32) | 0 | Activation |
| 21 | Add | (None, 6, 32) | 0 | Conv1D + Spatial dropout |
| 22 | LSTM | (None, 64) | 24832 | Add |
| 23 | Dense | (None, 128) | 8320 | LSTM |
| 24 | Dropout | (None, 128) | 0 | Dense |
| 25 | Dense | (None, 192) | 24768 | Dropout |
| 26 | Dropout | (None, 192) | 0 | Dense |
| 27 | Dense | (None, 256) | 49408 | Dropout |
| 28 | Dropout | (None, 256) | 0 | Dense |
| 29 | Dense output | (None, 3) | 771 | Dropout |

<!-- source: table_02_model_parameters.md -->
# Table 2. Model parameter summary and run hyperparameters

| Source | Task | Parameter | Value | Size | Notes |
| --- | --- | --- | --- | --- | --- |
| paper |  | total_parameters | 131587 | 514.01 KB |  |
| paper |  | trainable_parameters | 131331 | 513.01 KB |  |
| paper |  | non_trainable_parameters | 256 | 1 KB |  |
| ours_model | num_classes_3 | total_parameters | 131587 |  | Parameter count computed from the project TCN-LSTM implementation. |
| ours_model | num_classes_3 | trainable_parameters | 131587 |  | Parameter count computed from the project TCN-LSTM implementation. |
| ours_model | num_classes_3 | non_trainable_parameters | 0 |  | Parameter count computed from the project TCN-LSTM implementation. |
| ours_model | num_classes_2 | total_parameters | 131330 |  | Parameter count computed from the project TCN-LSTM implementation. |
| ours_model | num_classes_2 | trainable_parameters | 131330 |  | Parameter count computed from the project TCN-LSTM implementation. |
| ours_model | num_classes_2 | non_trainable_parameters | 0 |  | Parameter count computed from the project TCN-LSTM implementation. |
| ours:val_as_test_80_20 | multiclass | input_dim | 1 |  | Hyperparameter from run.json, not a parameter-count summary. |
| ours:val_as_test_80_20 | multiclass | num_classes | 3 |  | Hyperparameter from run.json, not a parameter-count summary. |
| ours:val_as_test_80_20 | multiclass | tcn_channels | 32 |  | Hyperparameter from run.json, not a parameter-count summary. |
| ours:val_as_test_80_20 | multiclass | tcn_kernel_size | 7 |  | Hyperparameter from run.json, not a parameter-count summary. |
| ours:val_as_test_80_20 | multiclass | tcn_dilations | [1, 1] |  | Hyperparameter from run.json, not a parameter-count summary. |
| ours:val_as_test_80_20 | multiclass | tcn_dropout | 0.3 |  | Hyperparameter from run.json, not a parameter-count summary. |
| ours:val_as_test_80_20 | multiclass | lstm_hidden_dim | 64 |  | Hyperparameter from run.json, not a parameter-count summary. |
| ours:val_as_test_80_20 | multiclass | dense_hidden_dims | [128, 192, 256] |  | Hyperparameter from run.json, not a parameter-count summary. |
| ours:val_as_test_80_20 | multiclass | dense_dropout | 0.2 |  | Hyperparameter from run.json, not a parameter-count summary. |
| ours:val_as_test_80_20 | multiclass | batch_size | 32 |  | Hyperparameter from run.json, not a parameter-count summary. |
| ours:val_as_test_80_20 | multiclass | learning_rate | 0.0001 |  | Hyperparameter from run.json, not a parameter-count summary. |
| ours:val_as_test_80_20 | ad_ftd_vs_healthy | input_dim | 1 |  | Hyperparameter from run.json, not a parameter-count summary. |
| ours:val_as_test_80_20 | ad_ftd_vs_healthy | num_classes | 2 |  | Hyperparameter from run.json, not a parameter-count summary. |
| ours:val_as_test_80_20 | ad_ftd_vs_healthy | tcn_channels | 32 |  | Hyperparameter from run.json, not a parameter-count summary. |
| ours:val_as_test_80_20 | ad_ftd_vs_healthy | tcn_kernel_size | 7 |  | Hyperparameter from run.json, not a parameter-count summary. |
| ours:val_as_test_80_20 | ad_ftd_vs_healthy | tcn_dilations | [1, 1] |  | Hyperparameter from run.json, not a parameter-count summary. |
| ours:val_as_test_80_20 | ad_ftd_vs_healthy | tcn_dropout | 0.3 |  | Hyperparameter from run.json, not a parameter-count summary. |
| ours:val_as_test_80_20 | ad_ftd_vs_healthy | lstm_hidden_dim | 64 |  | Hyperparameter from run.json, not a parameter-count summary. |
| ours:val_as_test_80_20 | ad_ftd_vs_healthy | dense_hidden_dims | [128, 192, 256] |  | Hyperparameter from run.json, not a parameter-count summary. |
| ours:val_as_test_80_20 | ad_ftd_vs_healthy | dense_dropout | 0.2 |  | Hyperparameter from run.json, not a parameter-count summary. |
| ours:val_as_test_80_20 | ad_ftd_vs_healthy | batch_size | 32 |  | Hyperparameter from run.json, not a parameter-count summary. |
| ours:val_as_test_80_20 | ad_ftd_vs_healthy | learning_rate | 0.0001 |  | Hyperparameter from run.json, not a parameter-count summary. |
| ours:val_as_test_80_20 | ad_vs_healthy | input_dim | 1 |  | Hyperparameter from run.json, not a parameter-count summary. |
| ours:val_as_test_80_20 | ad_vs_healthy | num_classes | 2 |  | Hyperparameter from run.json, not a parameter-count summary. |
| ours:val_as_test_80_20 | ad_vs_healthy | tcn_channels | 32 |  | Hyperparameter from run.json, not a parameter-count summary. |
| ours:val_as_test_80_20 | ad_vs_healthy | tcn_kernel_size | 7 |  | Hyperparameter from run.json, not a parameter-count summary. |
| ours:val_as_test_80_20 | ad_vs_healthy | tcn_dilations | [1, 1] |  | Hyperparameter from run.json, not a parameter-count summary. |
| ours:val_as_test_80_20 | ad_vs_healthy | tcn_dropout | 0.3 |  | Hyperparameter from run.json, not a parameter-count summary. |
| ours:val_as_test_80_20 | ad_vs_healthy | lstm_hidden_dim | 64 |  | Hyperparameter from run.json, not a parameter-count summary. |
| ours:val_as_test_80_20 | ad_vs_healthy | dense_hidden_dims | [128, 192, 256] |  | Hyperparameter from run.json, not a parameter-count summary. |
| ours:val_as_test_80_20 | ad_vs_healthy | dense_dropout | 0.2 |  | Hyperparameter from run.json, not a parameter-count summary. |
| ours:val_as_test_80_20 | ad_vs_healthy | batch_size | 32 |  | Hyperparameter from run.json, not a parameter-count summary. |
| ours:val_as_test_80_20 | ad_vs_healthy | learning_rate | 0.0001 |  | Hyperparameter from run.json, not a parameter-count summary. |
| ours:val_as_test_80_20 | ftd_vs_healthy | input_dim | 1 |  | Hyperparameter from run.json, not a parameter-count summary. |
| ours:val_as_test_80_20 | ftd_vs_healthy | num_classes | 2 |  | Hyperparameter from run.json, not a parameter-count summary. |
| ours:val_as_test_80_20 | ftd_vs_healthy | tcn_channels | 32 |  | Hyperparameter from run.json, not a parameter-count summary. |
| ours:val_as_test_80_20 | ftd_vs_healthy | tcn_kernel_size | 7 |  | Hyperparameter from run.json, not a parameter-count summary. |
| ours:val_as_test_80_20 | ftd_vs_healthy | tcn_dilations | [1, 1] |  | Hyperparameter from run.json, not a parameter-count summary. |
| ours:val_as_test_80_20 | ftd_vs_healthy | tcn_dropout | 0.3 |  | Hyperparameter from run.json, not a parameter-count summary. |
| ours:val_as_test_80_20 | ftd_vs_healthy | lstm_hidden_dim | 64 |  | Hyperparameter from run.json, not a parameter-count summary. |
| ours:val_as_test_80_20 | ftd_vs_healthy | dense_hidden_dims | [128, 192, 256] |  | Hyperparameter from run.json, not a parameter-count summary. |
| ours:val_as_test_80_20 | ftd_vs_healthy | dense_dropout | 0.2 |  | Hyperparameter from run.json, not a parameter-count summary. |
| ours:val_as_test_80_20 | ftd_vs_healthy | batch_size | 32 |  | Hyperparameter from run.json, not a parameter-count summary. |
| ours:val_as_test_80_20 | ftd_vs_healthy | learning_rate | 0.0001 |  | Hyperparameter from run.json, not a parameter-count summary. |
| ours:val_as_test_80_20 | standard_rbp/multiclass | input_dim | 1 |  | Hyperparameter from run.json, not a parameter-count summary. |
| ours:val_as_test_80_20 | standard_rbp/multiclass | num_classes | 3 |  | Hyperparameter from run.json, not a parameter-count summary. |
| ours:val_as_test_80_20 | standard_rbp/multiclass | tcn_channels | 32 |  | Hyperparameter from run.json, not a parameter-count summary. |
| ours:val_as_test_80_20 | standard_rbp/multiclass | tcn_kernel_size | 7 |  | Hyperparameter from run.json, not a parameter-count summary. |
| ours:val_as_test_80_20 | standard_rbp/multiclass | tcn_dilations | [1, 1] |  | Hyperparameter from run.json, not a parameter-count summary. |
| ours:val_as_test_80_20 | standard_rbp/multiclass | tcn_dropout | 0.3 |  | Hyperparameter from run.json, not a parameter-count summary. |
| ours:val_as_test_80_20 | standard_rbp/multiclass | lstm_hidden_dim | 64 |  | Hyperparameter from run.json, not a parameter-count summary. |
| ours:val_as_test_80_20 | standard_rbp/multiclass | dense_hidden_dims | [128, 192, 256] |  | Hyperparameter from run.json, not a parameter-count summary. |
| ours:val_as_test_80_20 | standard_rbp/multiclass | dense_dropout | 0.2 |  | Hyperparameter from run.json, not a parameter-count summary. |
| ours:val_as_test_80_20 | standard_rbp/multiclass | batch_size | 32 |  | Hyperparameter from run.json, not a parameter-count summary. |
| ours:val_as_test_80_20 | standard_rbp/multiclass | learning_rate | 0.0001 |  | Hyperparameter from run.json, not a parameter-count summary. |
| ours:val_as_test_80_20 | standard_rbp/ad_vs_healthy | input_dim | 1 |  | Hyperparameter from run.json, not a parameter-count summary. |
| ours:val_as_test_80_20 | standard_rbp/ad_vs_healthy | num_classes | 2 |  | Hyperparameter from run.json, not a parameter-count summary. |
| ours:val_as_test_80_20 | standard_rbp/ad_vs_healthy | tcn_channels | 32 |  | Hyperparameter from run.json, not a parameter-count summary. |
| ours:val_as_test_80_20 | standard_rbp/ad_vs_healthy | tcn_kernel_size | 7 |  | Hyperparameter from run.json, not a parameter-count summary. |
| ours:val_as_test_80_20 | standard_rbp/ad_vs_healthy | tcn_dilations | [1, 1] |  | Hyperparameter from run.json, not a parameter-count summary. |
| ours:val_as_test_80_20 | standard_rbp/ad_vs_healthy | tcn_dropout | 0.3 |  | Hyperparameter from run.json, not a parameter-count summary. |
| ours:val_as_test_80_20 | standard_rbp/ad_vs_healthy | lstm_hidden_dim | 64 |  | Hyperparameter from run.json, not a parameter-count summary. |
| ours:val_as_test_80_20 | standard_rbp/ad_vs_healthy | dense_hidden_dims | [128, 192, 256] |  | Hyperparameter from run.json, not a parameter-count summary. |
| ours:val_as_test_80_20 | standard_rbp/ad_vs_healthy | dense_dropout | 0.2 |  | Hyperparameter from run.json, not a parameter-count summary. |
| ours:val_as_test_80_20 | standard_rbp/ad_vs_healthy | batch_size | 32 |  | Hyperparameter from run.json, not a parameter-count summary. |
| ours:val_as_test_80_20 | standard_rbp/ad_vs_healthy | learning_rate | 0.0001 |  | Hyperparameter from run.json, not a parameter-count summary. |
| ours:val_as_test_80_20 | smote/multiclass | input_dim | 1 |  | Hyperparameter from run.json, not a parameter-count summary. |
| ours:val_as_test_80_20 | smote/multiclass | num_classes | 3 |  | Hyperparameter from run.json, not a parameter-count summary. |
| ours:val_as_test_80_20 | smote/multiclass | tcn_channels | 32 |  | Hyperparameter from run.json, not a parameter-count summary. |
| ours:val_as_test_80_20 | smote/multiclass | tcn_kernel_size | 7 |  | Hyperparameter from run.json, not a parameter-count summary. |
| ours:val_as_test_80_20 | smote/multiclass | tcn_dilations | [1, 1] |  | Hyperparameter from run.json, not a parameter-count summary. |
| ours:val_as_test_80_20 | smote/multiclass | tcn_dropout | 0.3 |  | Hyperparameter from run.json, not a parameter-count summary. |
| ours:val_as_test_80_20 | smote/multiclass | lstm_hidden_dim | 64 |  | Hyperparameter from run.json, not a parameter-count summary. |
| ours:val_as_test_80_20 | smote/multiclass | dense_hidden_dims | [128, 192, 256] |  | Hyperparameter from run.json, not a parameter-count summary. |
| ours:val_as_test_80_20 | smote/multiclass | dense_dropout | 0.2 |  | Hyperparameter from run.json, not a parameter-count summary. |
| ours:val_as_test_80_20 | smote/multiclass | batch_size | 32 |  | Hyperparameter from run.json, not a parameter-count summary. |
| ours:val_as_test_80_20 | smote/multiclass | learning_rate | 0.0001 |  | Hyperparameter from run.json, not a parameter-count summary. |
| ours:val_as_test_80_20 | smote/multiclass | smote.target_strategy | "max_class_count" |  | SMOTE setting from run.json. |
| ours:val_as_test_80_20 | smote/multiclass | smote.k_neighbors | 5 |  | SMOTE setting from run.json. |
| ours:val_as_test_80_20 | smote/multiclass | smote.apply_to_partitions | ["train", "val"] |  | SMOTE setting from run.json. |
| ours:val_as_test_80_20 | smote/multiclass | smote.partition_summary | [{"partition": "train", "input_count": 18520, "output_count": 23181, "synthetic_count": 4661, "class_counts": {"0": 7727, "1": 7727, "2": 7727}}, {"partition": "val", "input_count": 4631, "output_count": 5796, "synthetic_count": 1165, "class_counts": {"0": 1932, "1": 1932, "2": 1932}}] |  | SMOTE setting from run.json. |
| ours:val_as_test_80_20 | smote/ad_vs_healthy | input_dim | 1 |  | Hyperparameter from run.json, not a parameter-count summary. |
| ours:val_as_test_80_20 | smote/ad_vs_healthy | num_classes | 2 |  | Hyperparameter from run.json, not a parameter-count summary. |
| ours:val_as_test_80_20 | smote/ad_vs_healthy | tcn_channels | 32 |  | Hyperparameter from run.json, not a parameter-count summary. |
| ours:val_as_test_80_20 | smote/ad_vs_healthy | tcn_kernel_size | 7 |  | Hyperparameter from run.json, not a parameter-count summary. |
| ours:val_as_test_80_20 | smote/ad_vs_healthy | tcn_dilations | [1, 1] |  | Hyperparameter from run.json, not a parameter-count summary. |
| ours:val_as_test_80_20 | smote/ad_vs_healthy | tcn_dropout | 0.3 |  | Hyperparameter from run.json, not a parameter-count summary. |
| ours:val_as_test_80_20 | smote/ad_vs_healthy | lstm_hidden_dim | 64 |  | Hyperparameter from run.json, not a parameter-count summary. |
| ours:val_as_test_80_20 | smote/ad_vs_healthy | dense_hidden_dims | [128, 192, 256] |  | Hyperparameter from run.json, not a parameter-count summary. |
| ours:val_as_test_80_20 | smote/ad_vs_healthy | dense_dropout | 0.2 |  | Hyperparameter from run.json, not a parameter-count summary. |
| ours:val_as_test_80_20 | smote/ad_vs_healthy | batch_size | 32 |  | Hyperparameter from run.json, not a parameter-count summary. |
| ours:val_as_test_80_20 | smote/ad_vs_healthy | learning_rate | 0.0001 |  | Hyperparameter from run.json, not a parameter-count summary. |
| ours:val_as_test_80_20 | smote/ad_vs_healthy | smote.target_strategy | "max_class_count" |  | SMOTE setting from run.json. |
| ours:val_as_test_80_20 | smote/ad_vs_healthy | smote.k_neighbors | 5 |  | SMOTE setting from run.json. |
| ours:val_as_test_80_20 | smote/ad_vs_healthy | smote.apply_to_partitions | ["train", "val"] |  | SMOTE setting from run.json. |
| ours:val_as_test_80_20 | smote/ad_vs_healthy | smote.partition_summary | [{"partition": "train", "input_count": 14123, "output_count": 15454, "synthetic_count": 1331, "class_counts": {"0": 7727, "1": 7727}}, {"partition": "val", "input_count": 3531, "output_count": 3864, "synthetic_count": 333, "class_counts": {"0": 1932, "1": 1932}}] |  | SMOTE setting from run.json. |

<!-- source: table_03_multiclass.md -->
# Table 3. Classification metrics for Alzheimer, frontotemporal, and healthy classes.

| Source | Class | Precision | Recall | F1 score | Sensitivity | Specificity | Support | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| paper | alzheimer | 0.7 | 0.9 | 0.79 | 0.9 | 0.74 | 1876 |  |
| paper | frontotemporal_dementia | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1597 | Support is inconsistent with ds004504 FTD duration and matches healthy duration. |
| paper | healthy_control | 0.68 | 0.35 | 0.47 | 0.35 | 0.95 | 1106 | Support is inconsistent with ds004504 healthy duration and matches FTD duration. |
| ours:val_as_test_80_20 | alzheimer | 0.5396319886765747 | 0.7893374741200828 | 0.6410256410256411 | 0.7893374741200828 | 0.5179696183771767 | 1932 |  |
| ours:val_as_test_80_20 | frontotemporal_dementia | 0.6591928251121076 | 0.13363636363636364 | 0.22222222222222224 | 0.13363636363636364 | 0.978476352308128 | 1100 |  |
| ours:val_as_test_80_20 | healthy_control | 0.6378002528445006 | 0.631019387116948 | 0.6343917007230431 | 0.631019387116948 | 0.8110158311345647 | 1599 |  |
| ours_label_swap | alzheimer | 0.5451541850220264 | 0.7686335403726708 | 0.6378865979381443 | 0.7686335403726708 | 0.5409410892923305 | 1932 | Audit run with FTD/Healthy target labels intentionally swapped. |
| ours_label_swap | frontotemporal_dementia | 0.6191616766467066 | 0.6466541588492808 | 0.6326093606607526 | 0.6466541588492808 | 0.7902374670184696 | 1599 | Audit run with FTD/Healthy target labels intentionally swapped. |
| ours_label_swap | healthy_control | 0.6835443037974683 | 0.14727272727272728 | 0.24233358264771876 | 0.14727272727272728 | 0.9787595581988106 | 1100 | Audit run with FTD/Healthy target labels intentionally swapped. |

<!-- source: table_04_ad_ftd_vs_healthy.md -->
# Table 4. Classification metrics for Alzheimer + frontotemporal disease and healthy classes.

| Source | Class | Precision | Recall | F1 score | Sensitivity | Specificity | Support | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| paper | alzheimer_or_frontotemporal_dementia | 0.9977 | 0.9993 | 0.9985 | 1.0 | 1.0 | 2983 | Binary sensitivity/specificity are carried from the paper's single reported row. |
| paper | healthy_control | 0.9987 | 0.9956 | 0.9972 | 1.0 | 1.0 | 1596 | Binary sensitivity/specificity are carried from the paper's single reported row. |
| ours:val_as_test_80_20 | alzheimer_or_frontotemporal_dementia | 0.7503463563313937 | 0.8931398416886543 | 0.8155398283391054 | 0.8931398416886543 | 0.4365228267667292 | 3032 |  |
| ours:val_as_test_80_20 | healthy_control | 0.6829745596868885 | 0.4365228267667292 | 0.5326211369706219 | 0.4365228267667292 | 0.8931398416886543 | 1599 |  |

<!-- source: table_05_ad_vs_healthy.md -->
# Table 5. Classification metrics for Alzheimer's disease and healthy classes.

| Source | Class | Precision | Recall | F1 score | Sensitivity | Specificity | Support | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| paper | alzheimer | 0.9963 | 0.9989 | 0.9976 | 1.0 | 1.0 | 1876 | Binary sensitivity/specificity are carried from the paper's single reported row. |
| paper | healthy_control | 0.9987 | 0.9956 | 0.9972 | 1.0 | 1.0 | 1596 | Paper table support is 1596; Figure 4c row sum is 1597. |
| ours:val_as_test_80_20 | alzheimer | 0.7346326836581709 | 0.7608695652173914 | 0.7475209763539283 | 0.7608695652173914 | 0.6679174484052532 | 1932 |  |
| ours:val_as_test_80_20 | healthy_control | 0.6980392156862745 | 0.6679174484052532 | 0.6826462128475551 | 0.6679174484052532 | 0.7608695652173914 | 1599 |  |

<!-- source: table_06_ftd_vs_healthy.md -->
# Table 6. Classification metrics for frontotemporal disease and healthy classes.

| Source | Class | Precision | Recall | F1 score | Sensitivity | Specificity | Support | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| paper | frontotemporal_dementia | 0.9994 | 0.9956 | 0.9975 | 1.0 | 1.0 | 1597 | Binary sensitivity/specificity are carried from the paper's single reported row. |
| paper | healthy_control | 0.9937 | 0.9991 | 0.9964 | 1.0 | 1.0 | 1596 | Paper table support is 1596; Figure 4d row sum and text are 1106. |
| ours:val_as_test_80_20 | frontotemporal_dementia | 0.6231884057971014 | 0.7427272727272727 | 0.6777270841974283 | 0.7427272727272727 | 0.6910569105691057 | 1100 |  |
| ours:val_as_test_80_20 | healthy_control | 0.7961095100864554 | 0.6910569105691057 | 0.7398727820555742 | 0.6910569105691057 | 0.7427272727272727 | 1599 |  |
| ours_label_swap | frontotemporal_dementia | 0.7856115107913669 | 0.6829268292682927 | 0.7306791569086651 | 0.6829268292682927 | 0.7290909090909091 | 1599 | Audit run with FTD/Healthy target labels intentionally swapped. |
| ours_label_swap | healthy_control | 0.612681436210848 | 0.7290909090909091 | 0.6658364466583645 | 0.7290909090909091 | 0.6829268292682927 | 1100 | Audit run with FTD/Healthy target labels intentionally swapped. |

<!-- source: table_07_accuracy.md -->
# Table 7. Classification accuracy for different dementia classification tasks

| Source | Classification task | Metric | Value | Notes |
| --- | --- | --- | --- | --- |
| paper | Frontotemporal vs. healthy | accuracy | 0.997 |  |
| paper | Alzheimer vs. healthy | accuracy | 0.9974 |  |
| paper | Alzheimer + frontotemporal vs. healthy | accuracy | 0.998 |  |
| paper | Alzheimer vs. frontotemporal vs. healthy | accuracy | 0.8034 |  |
| ours:val_as_test_80_20 | Alzheimer vs. frontotemporal vs. healthy | accuracy | 0.5789246383070611 |  |
| ours:val_as_test_80_20 | Alzheimer + frontotemporal vs. healthy | accuracy | 0.7354782984236666 |  |
| ours:val_as_test_80_20 | Alzheimer vs. healthy | accuracy | 0.7187765505522515 |  |
| ours:val_as_test_80_20 | Frontotemporal vs. healthy | accuracy | 0.7121155983697666 |  |

<!-- source: table_08_smote_multiclass.md -->
# Table 8. Classification metrics with SMOTE data balancing.

| Source | Class | Precision | Recall | F1 score | Sensitivity | Specificity | Support | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| paper | alzheimer | 0.63 | 0.71 | 0.67 | 0.71 | 0.79 | 1876 |  |
| paper | frontotemporal_dementia | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1876 | Table 8 supports are balanced to 1876 for every class. |
| paper | healthy_control | 0.67 | 0.58 | 0.62 | 0.58 | 0.86 | 1876 | The paper text reports 77.45% accuracy, but these equal supports and recalls imply 76.33%. |
| ours:val_as_test_80_20 | alzheimer | 0.5125170687300865 | 0.582815734989648 | 0.5454105110196174 | 0.582815734989648 | 0.7228260869565217 | 1932 |  |
| ours:val_as_test_80_20 | frontotemporal_dementia | 0.5197923426346528 | 0.41459627329192544 | 0.4612726749208177 | 0.41459627329192544 | 0.8084886128364389 | 1932 |  |
| ours:val_as_test_80_20 | healthy_control | 0.60932944606414 | 0.6490683229813664 | 0.6285714285714286 | 0.6490683229813664 | 0.7919254658385093 | 1932 |  |

<!-- source: table_09_smote_ad_vs_healthy.md -->
# Table 9. Classification metrics for Alzheimer's disease and healthy classes with SMOTE balancing.

| Source | Class | Precision | Recall | F1 score | Sensitivity | Specificity | Support | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| paper | alzheimer | 99.73% | 99.7% | 99.71% |  |  | 1876 | Paper table reports precision/recall/F1 as percentages, unlike Tables 3-8. |
| paper | healthy_control | 99.7% | 99.71% | 99.73% |  |  | 1876 | Paper table reports precision/recall/F1 as percentages, unlike Tables 3-8. |
| ours:val_as_test_80_20 | alzheimer | 0.7419540229885058 | 0.6682194616977226 | 0.7031590413943356 | 0.6682194616977226 | 0.7675983436853002 | 1932 |  |
| ours:val_as_test_80_20 | healthy_control | 0.698210922787194 | 0.7675983436853002 | 0.7312623274161736 | 0.7675983436853002 | 0.6682194616977226 | 1932 |  |

<!-- source: table_10_kfold_multiclass.md -->
# Table 10. K-fold validation accuracy for Alzheimer, frontotemporal, and healthy classes.

| Source | K-value | Training accuracy (%) | Test accuracy (%) | Notes |
| --- | --- | --- | --- | --- |
| paper | 1 | 79.89 | 80.15 |  |
| paper | 2 | 80.0 | 80.0 |  |
| paper | 3 | 79.58 | 80.06 |  |
| paper | 4 | 79.43 | 80.02 |  |
| paper | 5 | 81.27 | 80.13 |  |
| ours:val_as_test_80_20 | 1 | 56.92764578833693 | 58.30274238825308 |  |
| ours:val_as_test_80_20 | 2 | 57.44060475161987 | 57.24465558194775 |  |
| ours:val_as_test_80_20 | 3 | 57.5670860104746 | 57.170626349892004 |  |
| ours:val_as_test_80_20 | 4 | 57.334917121105775 | 57.66738660907127 |  |
| ours:val_as_test_80_20 | 5 | 56.97548860814167 | 58.2847267228343 |  |

<!-- source: table_11_kfold_ad_vs_healthy.md -->
# Table 11. K-fold validation accuracy for Alzheimer and healthy classes.

| Source | K-value | Training accuracy (%) | Test accuracy (%) | Notes |
| --- | --- | --- | --- | --- |
| paper | 1 | 99.82 | 99.86 |  |
| paper | 2 | 99.8 | 99.82 |  |
| paper | 3 | 99.73 | 99.92 |  |
| paper | 4 | 99.61 | 99.86 |  |
| paper | 5 | 99.78 | 99.82 |  |
| ours:val_as_test_80_20 | 1 | 71.5570346243716 | 71.76437269895214 |  |
| ours:val_as_test_80_20 | 2 | 71.42250230121078 | 72.61399037099972 |  |
| ours:val_as_test_80_20 | 3 | 71.59951851589605 | 72.89719626168224 |  |
| ours:val_as_test_80_20 | 4 | 71.81193797351837 | 69.95185499858397 |  |
| ours:val_as_test_80_20 | 5 | 71.5024072500708 | 72.1813031161473 |  |

<!-- source: table_12_standard_rbp_multiclass.md -->
# Table 12. Classification metrics for standard RBP multiclass classification.

| Source | Class | Precision | Recall | F1 score | Sensitivity | Specificity | Support | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| paper | alzheimer | 0.6 | 0.77 | 0.67 |  |  | 1876 |  |
| paper | frontotemporal_dementia | 0.68 | 0.68 | 0.68 |  |  | 1597 |  |
| paper | healthy_control | 0.6 | 0.33 | 0.43 |  |  | 1106 |  |
| ours:val_as_test_80_20 | alzheimer | 0.5328125 | 0.7052740434332989 | 0.6070315976858034 | 0.7052740434332989 | 0.5573649148778682 | 967 |  |
| ours:val_as_test_80_20 | frontotemporal_dementia | 0.5656565656565656 | 0.20326678765880218 | 0.29906542056074764 | 0.20326678765880218 | 0.9513299377475948 | 551 |  |
| ours:val_as_test_80_20 | healthy_control | 0.6059523809523809 | 0.63625 | 0.6207317073170731 | 0.63625 | 0.7819499341238472 | 800 |  |

<!-- source: table_13_standard_rbp_ad_vs_healthy.md -->
# Table 13. Classification metrics for standard RBP Alzheimer's disease and healthy classification.

| Source | Class | Precision | Recall | F1 score | Sensitivity | Specificity | Support | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| paper | alzheimer | 0.76 | 0.81 | 0.79 |  |  | 1876 |  |
| paper | healthy_control | 0.76 | 0.71 | 0.73 |  |  | 1597 |  |
| ours:val_as_test_80_20 | alzheimer | 0.7174721189591078 | 0.7983453981385729 | 0.7557513460597161 | 0.7983453981385729 | 0.62 | 967 |  |
| ours:val_as_test_80_20 | healthy_control | 0.7178002894356006 | 0.62 | 0.6653252850435949 | 0.62 | 0.7983453981385729 | 800 |  |

<!-- source: table_14_literature_comparison.md -->
# Table 14. Model accuracy comparison with existing papers using dataset

| Paper | Model | Accuracy | Feature engineering | XAI |
| --- | --- | --- | --- | --- |
| Ma et al. | Support vector machine | 91.5% | PHI | no |
| Miltiadous et al. | Dual-Input Convolution Encoder Network | 83.28% | Band power and coherence | no |
| Kachare et al. | STEADYNet | 97.59% | not listed | no |
| Chen et al. | Vision transformer + CNN | 80.23% | frequency channels | no |
| This work | Proposed TCN-LSTM | 80.34%, 99.7% | Modified RBP | yes |

