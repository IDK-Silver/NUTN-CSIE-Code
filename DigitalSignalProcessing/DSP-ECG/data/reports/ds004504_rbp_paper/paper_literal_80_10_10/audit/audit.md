# Reproduction artifact audit for `paper_literal_80_10_10`

## Summary

| Status | Count |
| --- | --- |
| ok | 149 |
| stale | 4 |

## Items

| Status | Category | Name | Path | Detail |
| --- | --- | --- | --- | --- |
| ok | processed_dataset | modified_rbp_h5 | `data/processed_raw_dataset/ds004504_rbp_paper.h5` | exists; size=10104675 |
| ok | processed_dataset | modified_rbp_manifest | `data/processed_raw_dataset/ds004504_rbp_paper_manifest.json` | exists; size=31712 |
| ok | processed_dataset | standard_rbp_h5 | `data/processed_raw_dataset/ds004504_standard_rbp_paper.h5` | exists; size=8559803 |
| ok | processed_dataset | standard_rbp_manifest | `data/processed_raw_dataset/ds004504_standard_rbp_paper_manifest.json` | exists; size=31686 |
| ok | training_run | paper_literal_80_10_10:multiclass:run_json | `data/runs/ds004504_rbp_paper/multiclass/run.json` | exists; size=536007 |
| ok | training_run | paper_literal_80_10_10:multiclass:history | `data/runs/ds004504_rbp_paper/multiclass/history.json` | exists; size=296944 |
| ok | training_run | paper_literal_80_10_10:multiclass:metrics | `data/runs/ds004504_rbp_paper/multiclass/test_metrics.json` | exists; size=1331 |
| ok | training_run | paper_literal_80_10_10:multiclass:model | `data/runs/ds004504_rbp_paper/multiclass/model.pt` | exists; size=540170 |
| ok | training_run | paper_literal_80_10_10:multiclass:predictions | `data/runs/ds004504_rbp_paper/multiclass/test_predictions.csv` | exists; size=22091 |
| stale | training_run | paper_literal_80_10_10:multiclass:predictions:probability_columns | `data/runs/ds004504_rbp_paper/multiclass/test_predictions.csv` | missing prob_* columns; rerun training to enable ROC/AUC plots |
| ok | training_run | paper_literal_80_10_10:ad_ftd_vs_healthy:run_json | `data/runs/ds004504_rbp_paper/ad_ftd_vs_healthy/run.json` | exists; size=536045 |
| ok | training_run | paper_literal_80_10_10:ad_ftd_vs_healthy:history | `data/runs/ds004504_rbp_paper/ad_ftd_vs_healthy/history.json` | exists; size=220554 |
| ok | training_run | paper_literal_80_10_10:ad_ftd_vs_healthy:metrics | `data/runs/ds004504_rbp_paper/ad_ftd_vs_healthy/test_metrics.json` | exists; size=967 |
| ok | training_run | paper_literal_80_10_10:ad_ftd_vs_healthy:model | `data/runs/ds004504_rbp_paper/ad_ftd_vs_healthy/model.pt` | exists; size=539210 |
| ok | training_run | paper_literal_80_10_10:ad_ftd_vs_healthy:predictions | `data/runs/ds004504_rbp_paper/ad_ftd_vs_healthy/test_predictions.csv` | exists; size=22081 |
| stale | training_run | paper_literal_80_10_10:ad_ftd_vs_healthy:predictions:probability_columns | `data/runs/ds004504_rbp_paper/ad_ftd_vs_healthy/test_predictions.csv` | missing prob_* columns; rerun training to enable ROC/AUC plots |
| ok | training_run | paper_literal_80_10_10:ad_vs_healthy:run_json | `data/runs/ds004504_rbp_paper/ad_vs_healthy/run.json` | exists; size=404066 |
| ok | training_run | paper_literal_80_10_10:ad_vs_healthy:history | `data/runs/ds004504_rbp_paper/ad_vs_healthy/history.json` | exists; size=215069 |
| ok | training_run | paper_literal_80_10_10:ad_vs_healthy:metrics | `data/runs/ds004504_rbp_paper/ad_vs_healthy/test_metrics.json` | exists; size=939 |
| ok | training_run | paper_literal_80_10_10:ad_vs_healthy:model | `data/runs/ds004504_rbp_paper/ad_vs_healthy/model.pt` | exists; size=539146 |
| ok | training_run | paper_literal_80_10_10:ad_vs_healthy:predictions | `data/runs/ds004504_rbp_paper/ad_vs_healthy/test_predictions.csv` | exists; size=16581 |
| stale | training_run | paper_literal_80_10_10:ad_vs_healthy:predictions:probability_columns | `data/runs/ds004504_rbp_paper/ad_vs_healthy/test_predictions.csv` | missing prob_* columns; rerun training to enable ROC/AUC plots |
| ok | training_run | paper_literal_80_10_10:ftd_vs_healthy:run_json | `data/runs/ds004504_rbp_paper/ftd_vs_healthy/run.json` | exists; size=314968 |
| ok | training_run | paper_literal_80_10_10:ftd_vs_healthy:history | `data/runs/ds004504_rbp_paper/ftd_vs_healthy/history.json` | exists; size=217007 |
| ok | training_run | paper_literal_80_10_10:ftd_vs_healthy:metrics | `data/runs/ds004504_rbp_paper/ftd_vs_healthy/test_metrics.json` | exists; size=956 |
| ok | training_run | paper_literal_80_10_10:ftd_vs_healthy:model | `data/runs/ds004504_rbp_paper/ftd_vs_healthy/model.pt` | exists; size=539146 |
| ok | training_run | paper_literal_80_10_10:ftd_vs_healthy:predictions | `data/runs/ds004504_rbp_paper/ftd_vs_healthy/test_predictions.csv` | exists; size=12421 |
| stale | training_run | paper_literal_80_10_10:ftd_vs_healthy:predictions:probability_columns | `data/runs/ds004504_rbp_paper/ftd_vs_healthy/test_predictions.csv` | missing prob_* columns; rerun training to enable ROC/AUC plots |
| ok | smote_run | smote:multiclass:run_json | `data/runs/ds004504_rbp_paper/smote/multiclass/run.json` | exists; size=537535 |
| ok | smote_run | smote:multiclass:history | `data/runs/ds004504_rbp_paper/smote/multiclass/history.json` | exists; size=301971 |
| ok | smote_run | smote:multiclass:metrics | `data/runs/ds004504_rbp_paper/smote/multiclass/test_metrics.json` | exists; size=1396 |
| ok | smote_run | smote:multiclass:model | `data/runs/ds004504_rbp_paper/smote/multiclass/model.pt` | exists; size=540170 |
| ok | smote_run | smote:multiclass:predictions | `data/runs/ds004504_rbp_paper/smote/multiclass/test_predictions.csv` | exists; size=427115 |
| ok | smote_run | smote:multiclass:predictions:probability_columns | `data/runs/ds004504_rbp_paper/smote/multiclass/test_predictions.csv` | contains prob_* columns for ROC/AUC plotting |
| ok | smote_run | smote:ad_vs_healthy:run_json | `data/runs/ds004504_rbp_paper/smote/ad_vs_healthy/run.json` | exists; size=405535 |
| ok | smote_run | smote:ad_vs_healthy:history | `data/runs/ds004504_rbp_paper/smote/ad_vs_healthy/history.json` | exists; size=215035 |
| ok | smote_run | smote:ad_vs_healthy:metrics | `data/runs/ds004504_rbp_paper/smote/ad_vs_healthy/test_metrics.json` | exists; size=999 |
| ok | smote_run | smote:ad_vs_healthy:model | `data/runs/ds004504_rbp_paper/smote/ad_vs_healthy/model.pt` | exists; size=539146 |
| ok | smote_run | smote:ad_vs_healthy:predictions | `data/runs/ds004504_rbp_paper/smote/ad_vs_healthy/test_predictions.csv` | exists; size=206569 |
| ok | smote_run | smote:ad_vs_healthy:predictions:probability_columns | `data/runs/ds004504_rbp_paper/smote/ad_vs_healthy/test_predictions.csv` | contains prob_* columns for ROC/AUC plotting |
| ok | label_swap_run | label_swap:multiclass:run_json | `data/runs/ds004504_rbp_paper/label_swap/multiclass/run.json` | exists; size=536427 |
| ok | label_swap_run | label_swap:multiclass:history | `data/runs/ds004504_rbp_paper/label_swap/multiclass/history.json` | exists; size=295652 |
| ok | label_swap_run | label_swap:multiclass:metrics | `data/runs/ds004504_rbp_paper/label_swap/multiclass/test_metrics.json` | exists; size=1366 |
| ok | label_swap_run | label_swap:multiclass:model | `data/runs/ds004504_rbp_paper/label_swap/multiclass/model.pt` | exists; size=540234 |
| ok | label_swap_run | label_swap:multiclass:predictions | `data/runs/ds004504_rbp_paper/label_swap/multiclass/test_predictions.csv` | exists; size=340957 |
| ok | label_swap_run | label_swap:multiclass:predictions:probability_columns | `data/runs/ds004504_rbp_paper/label_swap/multiclass/test_predictions.csv` | contains prob_* columns for ROC/AUC plotting |
| ok | label_swap_run | label_swap:ftd_vs_healthy:run_json | `data/runs/ds004504_rbp_paper/label_swap/ftd_vs_healthy/run.json` | exists; size=315390 |
| ok | label_swap_run | label_swap:ftd_vs_healthy:history | `data/runs/ds004504_rbp_paper/label_swap/ftd_vs_healthy/history.json` | exists; size=217322 |
| ok | label_swap_run | label_swap:ftd_vs_healthy:metrics | `data/runs/ds004504_rbp_paper/label_swap/ftd_vs_healthy/test_metrics.json` | exists; size=974 |
| ok | label_swap_run | label_swap:ftd_vs_healthy:model | `data/runs/ds004504_rbp_paper/label_swap/ftd_vs_healthy/model.pt` | exists; size=539210 |
| ok | label_swap_run | label_swap:ftd_vs_healthy:predictions | `data/runs/ds004504_rbp_paper/label_swap/ftd_vs_healthy/test_predictions.csv` | exists; size=144960 |
| ok | label_swap_run | label_swap:ftd_vs_healthy:predictions:probability_columns | `data/runs/ds004504_rbp_paper/label_swap/ftd_vs_healthy/test_predictions.csv` | contains prob_* columns for ROC/AUC plotting |
| ok | standard_rbp_run | standard_rbp:multiclass:run_json | `data/runs/ds004504_rbp_paper/standard_rbp/multiclass/run.json` | exists; size=536424 |
| ok | standard_rbp_run | standard_rbp:multiclass:history | `data/runs/ds004504_rbp_paper/standard_rbp/multiclass/history.json` | exists; size=298896 |
| ok | standard_rbp_run | standard_rbp:multiclass:metrics | `data/runs/ds004504_rbp_paper/standard_rbp/multiclass/test_metrics.json` | exists; size=1360 |
| ok | standard_rbp_run | standard_rbp:multiclass:model | `data/runs/ds004504_rbp_paper/standard_rbp/multiclass/model.pt` | exists; size=540234 |
| ok | standard_rbp_run | standard_rbp:multiclass:predictions | `data/runs/ds004504_rbp_paper/standard_rbp/multiclass/test_predictions.csv` | exists; size=340609 |
| ok | standard_rbp_run | standard_rbp:multiclass:predictions:probability_columns | `data/runs/ds004504_rbp_paper/standard_rbp/multiclass/test_predictions.csv` | contains prob_* columns for ROC/AUC plotting |
| ok | standard_rbp_run | standard_rbp:ad_vs_healthy:run_json | `data/runs/ds004504_rbp_paper/standard_rbp/ad_vs_healthy/run.json` | exists; size=404480 |
| ok | standard_rbp_run | standard_rbp:ad_vs_healthy:history | `data/runs/ds004504_rbp_paper/standard_rbp/ad_vs_healthy/history.json` | exists; size=215010 |
| ok | standard_rbp_run | standard_rbp:ad_vs_healthy:metrics | `data/runs/ds004504_rbp_paper/standard_rbp/ad_vs_healthy/test_metrics.json` | exists; size=954 |
| ok | standard_rbp_run | standard_rbp:ad_vs_healthy:model | `data/runs/ds004504_rbp_paper/standard_rbp/ad_vs_healthy/model.pt` | exists; size=539146 |
| ok | standard_rbp_run | standard_rbp:ad_vs_healthy:predictions | `data/runs/ds004504_rbp_paper/standard_rbp/ad_vs_healthy/test_predictions.csv` | exists; size=188181 |
| ok | standard_rbp_run | standard_rbp:ad_vs_healthy:predictions:probability_columns | `data/runs/ds004504_rbp_paper/standard_rbp/ad_vs_healthy/test_predictions.csv` | contains prob_* columns for ROC/AUC plotting |
| ok | kfold_run | kfold:multiclass:run_json | `data/runs/ds004504_rbp_paper/kfold/multiclass/run.json` | exists; size=2159228 |
| ok | kfold_run | kfold:multiclass:fold_summary | `data/runs/ds004504_rbp_paper/kfold/multiclass/fold_summary.csv` | exists; size=741 |
| ok | kfold_run | kfold:multiclass:fold_1:history | `data/runs/ds004504_rbp_paper/kfold/multiclass/fold_1/history.json` | exists; size=298348 |
| ok | kfold_run | kfold:multiclass:fold_1:metrics | `data/runs/ds004504_rbp_paper/kfold/multiclass/fold_1/test_metrics.json` | exists; size=1400 |
| ok | kfold_run | kfold:multiclass:fold_1:predictions | `data/runs/ds004504_rbp_paper/kfold/multiclass/fold_1/test_predictions.csv` | exists; size=681981 |
| ok | kfold_run | kfold:multiclass:fold_1:predictions:probability_columns | `data/runs/ds004504_rbp_paper/kfold/multiclass/fold_1/test_predictions.csv` | contains prob_* columns for ROC/AUC plotting |
| ok | kfold_run | kfold:multiclass:fold_2:history | `data/runs/ds004504_rbp_paper/kfold/multiclass/fold_2/history.json` | exists; size=299016 |
| ok | kfold_run | kfold:multiclass:fold_2:metrics | `data/runs/ds004504_rbp_paper/kfold/multiclass/fold_2/test_metrics.json` | exists; size=1399 |
| ok | kfold_run | kfold:multiclass:fold_2:predictions | `data/runs/ds004504_rbp_paper/kfold/multiclass/fold_2/test_predictions.csv` | exists; size=681630 |
| ok | kfold_run | kfold:multiclass:fold_2:predictions:probability_columns | `data/runs/ds004504_rbp_paper/kfold/multiclass/fold_2/test_predictions.csv` | contains prob_* columns for ROC/AUC plotting |
| ok | kfold_run | kfold:multiclass:fold_3:history | `data/runs/ds004504_rbp_paper/kfold/multiclass/fold_3/history.json` | exists; size=299705 |
| ok | kfold_run | kfold:multiclass:fold_3:metrics | `data/runs/ds004504_rbp_paper/kfold/multiclass/fold_3/test_metrics.json` | exists; size=1367 |
| ok | kfold_run | kfold:multiclass:fold_3:predictions | `data/runs/ds004504_rbp_paper/kfold/multiclass/fold_3/test_predictions.csv` | exists; size=681275 |
| ok | kfold_run | kfold:multiclass:fold_3:predictions:probability_columns | `data/runs/ds004504_rbp_paper/kfold/multiclass/fold_3/test_predictions.csv` | contains prob_* columns for ROC/AUC plotting |
| ok | kfold_run | kfold:multiclass:fold_4:history | `data/runs/ds004504_rbp_paper/kfold/multiclass/fold_4/history.json` | exists; size=298792 |
| ok | kfold_run | kfold:multiclass:fold_4:metrics | `data/runs/ds004504_rbp_paper/kfold/multiclass/fold_4/test_metrics.json` | exists; size=1397 |
| ok | kfold_run | kfold:multiclass:fold_4:predictions | `data/runs/ds004504_rbp_paper/kfold/multiclass/fold_4/test_predictions.csv` | exists; size=681194 |
| ok | kfold_run | kfold:multiclass:fold_4:predictions:probability_columns | `data/runs/ds004504_rbp_paper/kfold/multiclass/fold_4/test_predictions.csv` | contains prob_* columns for ROC/AUC plotting |
| ok | kfold_run | kfold:multiclass:fold_5:history | `data/runs/ds004504_rbp_paper/kfold/multiclass/fold_5/history.json` | exists; size=300167 |
| ok | kfold_run | kfold:multiclass:fold_5:metrics | `data/runs/ds004504_rbp_paper/kfold/multiclass/fold_5/test_metrics.json` | exists; size=1398 |
| ok | kfold_run | kfold:multiclass:fold_5:predictions | `data/runs/ds004504_rbp_paper/kfold/multiclass/fold_5/test_predictions.csv` | exists; size=681892 |
| ok | kfold_run | kfold:multiclass:fold_5:predictions:probability_columns | `data/runs/ds004504_rbp_paper/kfold/multiclass/fold_5/test_predictions.csv` | contains prob_* columns for ROC/AUC plotting |
| ok | kfold_run | kfold:ad_vs_healthy:run_json | `data/runs/ds004504_rbp_paper/kfold/ad_vs_healthy/run.json` | exists; size=1631503 |
| ok | kfold_run | kfold:ad_vs_healthy:fold_summary | `data/runs/ds004504_rbp_paper/kfold/ad_vs_healthy/fold_summary.csv` | exists; size=749 |
| ok | kfold_run | kfold:ad_vs_healthy:fold_1:history | `data/runs/ds004504_rbp_paper/kfold/ad_vs_healthy/fold_1/history.json` | exists; size=215536 |
| ok | kfold_run | kfold:ad_vs_healthy:fold_1:metrics | `data/runs/ds004504_rbp_paper/kfold/ad_vs_healthy/fold_1/test_metrics.json` | exists; size=1005 |
| ok | kfold_run | kfold:ad_vs_healthy:fold_1:predictions | `data/runs/ds004504_rbp_paper/kfold/ad_vs_healthy/fold_1/test_predictions.csv` | exists; size=377940 |
| ok | kfold_run | kfold:ad_vs_healthy:fold_1:predictions:probability_columns | `data/runs/ds004504_rbp_paper/kfold/ad_vs_healthy/fold_1/test_predictions.csv` | contains prob_* columns for ROC/AUC plotting |
| ok | kfold_run | kfold:ad_vs_healthy:fold_2:history | `data/runs/ds004504_rbp_paper/kfold/ad_vs_healthy/fold_2/history.json` | exists; size=215673 |
| ok | kfold_run | kfold:ad_vs_healthy:fold_2:metrics | `data/runs/ds004504_rbp_paper/kfold/ad_vs_healthy/fold_2/test_metrics.json` | exists; size=1005 |
| ok | kfold_run | kfold:ad_vs_healthy:fold_2:predictions | `data/runs/ds004504_rbp_paper/kfold/ad_vs_healthy/fold_2/test_predictions.csv` | exists; size=377700 |
| ok | kfold_run | kfold:ad_vs_healthy:fold_2:predictions:probability_columns | `data/runs/ds004504_rbp_paper/kfold/ad_vs_healthy/fold_2/test_predictions.csv` | contains prob_* columns for ROC/AUC plotting |
| ok | kfold_run | kfold:ad_vs_healthy:fold_3:history | `data/runs/ds004504_rbp_paper/kfold/ad_vs_healthy/fold_3/history.json` | exists; size=215452 |
| ok | kfold_run | kfold:ad_vs_healthy:fold_3:metrics | `data/runs/ds004504_rbp_paper/kfold/ad_vs_healthy/fold_3/test_metrics.json` | exists; size=1005 |
| ok | kfold_run | kfold:ad_vs_healthy:fold_3:predictions | `data/runs/ds004504_rbp_paper/kfold/ad_vs_healthy/fold_3/test_predictions.csv` | exists; size=377446 |
| ok | kfold_run | kfold:ad_vs_healthy:fold_3:predictions:probability_columns | `data/runs/ds004504_rbp_paper/kfold/ad_vs_healthy/fold_3/test_predictions.csv` | contains prob_* columns for ROC/AUC plotting |
| ok | kfold_run | kfold:ad_vs_healthy:fold_4:history | `data/runs/ds004504_rbp_paper/kfold/ad_vs_healthy/fold_4/history.json` | exists; size=215594 |
| ok | kfold_run | kfold:ad_vs_healthy:fold_4:metrics | `data/runs/ds004504_rbp_paper/kfold/ad_vs_healthy/fold_4/test_metrics.json` | exists; size=1001 |
| ok | kfold_run | kfold:ad_vs_healthy:fold_4:predictions | `data/runs/ds004504_rbp_paper/kfold/ad_vs_healthy/fold_4/test_predictions.csv` | exists; size=377301 |
| ok | kfold_run | kfold:ad_vs_healthy:fold_4:predictions:probability_columns | `data/runs/ds004504_rbp_paper/kfold/ad_vs_healthy/fold_4/test_predictions.csv` | contains prob_* columns for ROC/AUC plotting |
| ok | kfold_run | kfold:ad_vs_healthy:fold_5:history | `data/runs/ds004504_rbp_paper/kfold/ad_vs_healthy/fold_5/history.json` | exists; size=215646 |
| ok | kfold_run | kfold:ad_vs_healthy:fold_5:metrics | `data/runs/ds004504_rbp_paper/kfold/ad_vs_healthy/fold_5/test_metrics.json` | exists; size=1000 |
| ok | kfold_run | kfold:ad_vs_healthy:fold_5:predictions | `data/runs/ds004504_rbp_paper/kfold/ad_vs_healthy/fold_5/test_predictions.csv` | exists; size=377599 |
| ok | kfold_run | kfold:ad_vs_healthy:fold_5:predictions:probability_columns | `data/runs/ds004504_rbp_paper/kfold/ad_vs_healthy/fold_5/test_predictions.csv` | contains prob_* columns for ROC/AUC plotting |
| ok | report_csv | paper_model_architecture.csv | `data/reports/ds004504_rbp_paper/paper_literal_80_10_10/tables/paper_model_architecture.csv` | exists; size=1630 |
| ok | report_csv | model_parameters.csv | `data/reports/ds004504_rbp_paper/paper_literal_80_10_10/tables/model_parameters.csv` | exists; size=14532 |
| ok | report_csv | run_summary.csv | `data/reports/ds004504_rbp_paper/paper_literal_80_10_10/tables/run_summary.csv` | exists; size=3185 |
| ok | report_csv | classification_metrics.csv | `data/reports/ds004504_rbp_paper/paper_literal_80_10_10/tables/classification_metrics.csv` | exists; size=7826 |
| ok | report_csv | confusion_matrices.csv | `data/reports/ds004504_rbp_paper/paper_literal_80_10_10/tables/confusion_matrices.csv` | exists; size=4007 |
| ok | report_csv | support_comparison.csv | `data/reports/ds004504_rbp_paper/paper_literal_80_10_10/tables/support_comparison.csv` | exists; size=5105 |
| ok | report_csv | history.csv | `data/reports/ds004504_rbp_paper/paper_literal_80_10_10/tables/history.csv` | exists; size=159976 |
| ok | report_csv | accuracy.csv | `data/reports/ds004504_rbp_paper/paper_literal_80_10_10/tables/accuracy.csv` | exists; size=1564 |
| ok | report_csv | kfold_accuracy.csv | `data/reports/ds004504_rbp_paper/paper_literal_80_10_10/tables/kfold_accuracy.csv` | exists; size=1963 |
| ok | report_csv | paper_vs_ours.csv | `data/reports/ds004504_rbp_paper/paper_literal_80_10_10/tables/paper_vs_ours.csv` | exists; size=23955 |
| ok | report_csv | table_summary.csv | `data/reports/ds004504_rbp_paper/paper_literal_80_10_10/tables/table_summary.csv` | exists; size=3429 |
| ok | report_csv | protocol_manifest.csv | `data/reports/ds004504_rbp_paper/paper_literal_80_10_10/tables/protocol_manifest.csv` | exists; size=3123 |
| ok | report_csv | issue_summary.csv | `data/reports/ds004504_rbp_paper/paper_literal_80_10_10/tables/issue_summary.csv` | exists; size=2328 |
| ok | report_csv | literature_comparison.csv | `data/reports/ds004504_rbp_paper/paper_literal_80_10_10/tables/literature_comparison.csv` | exists; size=419 |
| ok | report_content | table_summary:table_1_to_14_coverage | `data/reports/ds004504_rbp_paper/paper_literal_80_10_10/tables/table_summary.csv` | covers expected tables: Table 1, Table 2, Table 3, Table 4, Table 5, Table 6, Table 7, Table 8, Table 9, Table 10, Table 11, Table 12, Table 13, Table 14 |
| ok | report_content | paper_vs_ours:table_3_to_13_coverage | `data/reports/ds004504_rbp_paper/paper_literal_80_10_10/tables/paper_vs_ours.csv` | covers expected tables: Table 3, Table 4, Table 5, Table 6, Table 7, Table 8, Table 9, Table 10, Table 11, Table 12, Table 13 |
| ok | report_content | issue_summary:issue_id_coverage | `data/reports/ds004504_rbp_paper/paper_literal_80_10_10/tables/issue_summary.csv` | covers expected issue ids: split_support_mismatch, ftd_healthy_support_swap, table6_healthy_support_typo, smote_accuracy_inconsistency, unspecified_smote_placement, epoch_level_leakage_risk |
| ok | report_content | protocol_manifest:component_coverage | `data/reports/ds004504_rbp_paper/paper_literal_80_10_10/tables/protocol_manifest.csv` | covers expected protocol components: raw_dataset, preprocessing_source, epoching, modified_rbp_bands, standard_rbp_bands, normalization, split, evaluation, model, model_hyperparameters, optimizer, training_runtime, smote, kfold, label_swap_audit, runs_dir |
| ok | rendered_table | index.md | `data/reports/ds004504_rbp_paper/paper_literal_80_10_10/paper_tables/index.md` | exists; size=1152 |
| ok | rendered_table | report.md | `data/reports/ds004504_rbp_paper/paper_literal_80_10_10/paper_tables/report.md` | exists; size=43215 |
| ok | rendered_table | table_01_model_architecture.md | `data/reports/ds004504_rbp_paper/paper_literal_80_10_10/paper_tables/table_01_model_architecture.md` | exists; size=1718 |
| ok | rendered_table | table_02_model_parameters.md | `data/reports/ds004504_rbp_paper/paper_literal_80_10_10/paper_tables/table_02_model_parameters.md` | exists; size=15310 |
| ok | rendered_table | table_03_multiclass.md | `data/reports/ds004504_rbp_paper/paper_literal_80_10_10/paper_tables/table_03_multiclass.md` | exists; size=1697 |
| ok | rendered_table | table_04_ad_ftd_vs_healthy.md | `data/reports/ds004504_rbp_paper/paper_literal_80_10_10/paper_tables/table_04_ad_ftd_vs_healthy.md` | exists; size=897 |
| ok | rendered_table | table_05_ad_vs_healthy.md | `data/reports/ds004504_rbp_paper/paper_literal_80_10_10/paper_tables/table_05_ad_vs_healthy.md` | exists; size=802 |
| ok | rendered_table | table_06_ftd_vs_healthy.md | `data/reports/ds004504_rbp_paper/paper_literal_80_10_10/paper_tables/table_06_ftd_vs_healthy.md` | exists; size=1248 |
| ok | rendered_table | table_07_accuracy.md | `data/reports/ds004504_rbp_paper/paper_literal_80_10_10/paper_tables/table_07_accuracy.md` | exists; size=842 |
| ok | rendered_table | table_08_smote_multiclass.md | `data/reports/ds004504_rbp_paper/paper_literal_80_10_10/paper_tables/table_08_smote_multiclass.md` | exists; size=1068 |
| ok | rendered_table | table_09_smote_ad_vs_healthy.md | `data/reports/ds004504_rbp_paper/paper_literal_80_10_10/paper_tables/table_09_smote_ad_vs_healthy.md` | exists; size=855 |
| ok | rendered_table | table_10_kfold_multiclass.md | `data/reports/ds004504_rbp_paper/paper_literal_80_10_10/paper_tables/table_10_kfold_multiclass.md` | exists; size=755 |
| ok | rendered_table | table_11_kfold_ad_vs_healthy.md | `data/reports/ds004504_rbp_paper/paper_literal_80_10_10/paper_tables/table_11_kfold_ad_vs_healthy.md` | exists; size=736 |
| ok | rendered_table | table_12_standard_rbp_multiclass.md | `data/reports/ds004504_rbp_paper/paper_literal_80_10_10/paper_tables/table_12_standard_rbp_multiclass.md` | exists; size=894 |
| ok | rendered_table | table_13_standard_rbp_ad_vs_healthy.md | `data/reports/ds004504_rbp_paper/paper_literal_80_10_10/paper_tables/table_13_standard_rbp_ad_vs_healthy.md` | exists; size=658 |
| ok | rendered_table | table_14_literature_comparison.md | `data/reports/ds004504_rbp_paper/paper_literal_80_10_10/paper_tables/table_14_literature_comparison.md` | exists; size=532 |
| ok | rendered_table | protocol_manifest.md | `data/reports/ds004504_rbp_paper/paper_literal_80_10_10/paper_tables/protocol_manifest.md` | exists; size=3349 |
| ok | rendered_table | issue_summary.md | `data/reports/ds004504_rbp_paper/paper_literal_80_10_10/paper_tables/issue_summary.md` | exists; size=2321 |
| ok | rendered_table | table_summary.md | `data/reports/ds004504_rbp_paper/paper_literal_80_10_10/paper_tables/table_summary.md` | exists; size=3395 |
| ok | rendered_table | confusion_matrices.md | `data/reports/ds004504_rbp_paper/paper_literal_80_10_10/paper_tables/confusion_matrices.md` | exists; size=4887 |
| ok | pipeline | execution_manifest | `data/reports/ds004504_rbp_paper/paper_literal_80_10_10/pipeline_execution_manifest.json` | exists; size=10126 |
| ok | plot | accuracy_comparison.svg | `data/reports/ds004504_rbp_paper/paper_literal_80_10_10/plots/accuracy_comparison.svg` | exists; size=6074 |
| ok | plot | support_comparison.svg | `data/reports/ds004504_rbp_paper/paper_literal_80_10_10/plots/support_comparison.svg` | exists; size=20824 |
| ok | plot | per_class_precision_recall_f1.svg | `data/reports/ds004504_rbp_paper/paper_literal_80_10_10/plots/per_class_precision_recall_f1.svg` | exists; size=50562 |
| ok | plot | confusion_matrix_svgs | `data/reports/ds004504_rbp_paper/paper_literal_80_10_10/plots/confusion_matrices` | contains 8 SVG file(s) |
| ok | plot | training_history_svgs | `data/reports/ds004504_rbp_paper/paper_literal_80_10_10/plots/history` | contains 10 SVG file(s) |
| ok | plot | roc_auc_csv | `data/reports/ds004504_rbp_paper/paper_literal_80_10_10/plots/roc/roc_auc.csv` | exists; size=1699 |
