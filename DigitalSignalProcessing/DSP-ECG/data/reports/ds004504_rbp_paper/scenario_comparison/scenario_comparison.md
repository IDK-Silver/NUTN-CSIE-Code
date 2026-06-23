# Scenario comparison

Compared scenarios: `paper_literal_80_10_10, val_as_test_80_20`

## Scenario roles

| Scenario | Role |
| --- | --- |
| `paper_literal_80_10_10` | Formal reproduction path that follows the paper text: 80% train, 10% validation, 10% test. |
| `val_as_test_80_20` | Audit/comparison path inferred from reported support counts: 80% train, 20% validation-as-test. |

## Coverage summary

| Scenario | Status | Artifact | Expected | Observed | Missing | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| paper_literal_80_10_10 | ok | table_summary.csv:paper_table | Table 1, Table 2, Table 3, Table 4, Table 5, Table 6, Table 7, Table 8, Table 9, Table 10, Table 11, Table 12, Table 13, Table 14 | Table 1, Table 10, Table 11, Table 12, Table 13, Table 14, Table 2, Table 3, Table 4, Table 5, Table 6, Table 7, Table 8, Table 9 |  | coverage is complete |
| paper_literal_80_10_10 | ok | paper_vs_ours.csv:paper_table | Table 3, Table 4, Table 5, Table 6, Table 7, Table 8, Table 9, Table 10, Table 11, Table 12, Table 13 | Table 10, Table 11, Table 12, Table 13, Table 3, Table 4, Table 5, Table 6, Table 7, Table 8, Table 9 |  | coverage is complete |
| paper_literal_80_10_10 | ok | protocol_manifest.csv:component | raw_dataset, preprocessing_source, epoching, modified_rbp_bands, standard_rbp_bands, normalization, split, evaluation, model, model_hyperparameters, optimizer, training_runtime, smote, kfold, label_swap_audit, runs_dir | epoching, evaluation, kfold, label_swap_audit, model, model_hyperparameters, modified_rbp_bands, normalization, optimizer, preprocessing_source, raw_dataset, runs_dir, smote, split, standard_rbp_bands, training_runtime |  | coverage is complete |
| paper_literal_80_10_10 | ok | issue_summary.csv:issue_id | split_support_mismatch, ftd_healthy_support_swap, table6_healthy_support_typo, smote_accuracy_inconsistency, unspecified_smote_placement, epoch_level_leakage_risk | epoch_level_leakage_risk, ftd_healthy_support_swap, smote_accuracy_inconsistency, split_support_mismatch, table6_healthy_support_typo, unspecified_smote_placement |  | coverage is complete |
| val_as_test_80_20 | ok | table_summary.csv:paper_table | Table 1, Table 2, Table 3, Table 4, Table 5, Table 6, Table 7, Table 8, Table 9, Table 10, Table 11, Table 12, Table 13, Table 14 | Table 1, Table 10, Table 11, Table 12, Table 13, Table 14, Table 2, Table 3, Table 4, Table 5, Table 6, Table 7, Table 8, Table 9 |  | coverage is complete |
| val_as_test_80_20 | ok | paper_vs_ours.csv:paper_table | Table 3, Table 4, Table 5, Table 6, Table 7, Table 8, Table 9, Table 10, Table 11, Table 12, Table 13 | Table 10, Table 11, Table 12, Table 13, Table 3, Table 4, Table 5, Table 6, Table 7, Table 8, Table 9 |  | coverage is complete |
| val_as_test_80_20 | ok | protocol_manifest.csv:component | raw_dataset, preprocessing_source, epoching, modified_rbp_bands, standard_rbp_bands, normalization, split, evaluation, model, model_hyperparameters, optimizer, training_runtime, smote, kfold, label_swap_audit, runs_dir | epoching, evaluation, kfold, label_swap_audit, model, model_hyperparameters, modified_rbp_bands, normalization, optimizer, preprocessing_source, raw_dataset, runs_dir, smote, split, standard_rbp_bands, training_runtime |  | coverage is complete |
| val_as_test_80_20 | ok | issue_summary.csv:issue_id | split_support_mismatch, ftd_healthy_support_swap, table6_healthy_support_typo, smote_accuracy_inconsistency, unspecified_smote_placement, epoch_level_leakage_risk | epoch_level_leakage_risk, ftd_healthy_support_swap, smote_accuracy_inconsistency, split_support_mismatch, table6_healthy_support_typo, unspecified_smote_placement |  | coverage is complete |

## Table summary

| Scenario | Status | Table | Purpose | Paper accuracy | Ours accuracy | Paper support | Ours support |
| --- | --- | --- | --- | --- | --- | --- | --- |
| paper_literal_80_10_10 | available | Table 1 | model architecture |  |  |  |  |
| paper_literal_80_10_10 | available | Table 2 | model parameter summary |  |  |  |  |
| paper_literal_80_10_10 | available | Table 3 | modified RBP multiclass metrics |  |  | multiclass/alzheimer=1876; multiclass/frontotemporal_dementia=1597; multiclass/healthy_control=1106 | multiclass/alzheimer=967; multiclass/frontotemporal_dementia=551; multiclass/healthy_control=800 |
| paper_literal_80_10_10 | available | Table 4 | modified RBP AD+FTD vs Healthy metrics |  |  | ad_ftd_vs_healthy/alzheimer_or_frontotemporal_dementia=2983; ad_ftd_vs_healthy/healthy_control=1596 | ad_ftd_vs_healthy/alzheimer_or_frontotemporal_dementia=1517; ad_ftd_vs_healthy/healthy_control=800 |
| paper_literal_80_10_10 | available | Table 5 | modified RBP AD vs Healthy metrics |  |  | ad_vs_healthy/alzheimer=1876; ad_vs_healthy/healthy_control=1596 | ad_vs_healthy/alzheimer=967; ad_vs_healthy/healthy_control=800 |
| paper_literal_80_10_10 | available | Table 6 | modified RBP FTD vs Healthy metrics |  |  | ftd_vs_healthy/frontotemporal_dementia=1597; ftd_vs_healthy/healthy_control=1596 | ftd_vs_healthy/frontotemporal_dementia=551; ftd_vs_healthy/healthy_control=800 |
| paper_literal_80_10_10 | available | Table 7 | modified RBP task accuracy | ftd_vs_healthy=0.997; ad_vs_healthy=0.9974; ad_ftd_vs_healthy=0.998; multiclass=0.8034 | ftd_vs_healthy=0.7424130273871207; ad_vs_healthy=0.7368421052631579; ad_ftd_vs_healthy=0.7466551575312904; multiclass=0.5720448662640207 |  |  |
| paper_literal_80_10_10 | available | Table 8 | SMOTE multiclass metrics | multiclass=77.45% | multiclass=0.5415374008962427 | multiclass/alzheimer=1876; multiclass/frontotemporal_dementia=1876; multiclass/healthy_control=1876 | multiclass/alzheimer=967; multiclass/frontotemporal_dementia=967; multiclass/healthy_control=967 |
| paper_literal_80_10_10 | available | Table 9 | SMOTE AD vs Healthy metrics |  |  | ad_vs_healthy/alzheimer=1876; ad_vs_healthy/healthy_control=1876 | ad_vs_healthy/alzheimer=967; ad_vs_healthy/healthy_control=967 |
| paper_literal_80_10_10 | available | Table 10 | 5-fold multiclass accuracy |  |  |  |  |
| paper_literal_80_10_10 | available | Table 11 | 5-fold AD vs Healthy accuracy |  |  |  |  |
| paper_literal_80_10_10 | available | Table 12 | standard RBP multiclass metrics | multiclass=63.03% | multiclass=0.5621225194132873 | multiclass/alzheimer=1876; multiclass/frontotemporal_dementia=1597; multiclass/healthy_control=1106 | multiclass/alzheimer=967; multiclass/frontotemporal_dementia=551; multiclass/healthy_control=800 |
| paper_literal_80_10_10 | available | Table 13 | standard RBP AD vs Healthy metrics | ad_vs_healthy=76.36% | ad_vs_healthy=0.7176004527447651 | ad_vs_healthy/alzheimer=1876; ad_vs_healthy/healthy_control=1597 | ad_vs_healthy/alzheimer=967; ad_vs_healthy/healthy_control=800 |
| paper_literal_80_10_10 | available | Table 14 | literature comparison |  |  |  |  |
| val_as_test_80_20 | available | Table 1 | model architecture |  |  |  |  |
| val_as_test_80_20 | available | Table 2 | model parameter summary |  |  |  |  |
| val_as_test_80_20 | available | Table 3 | modified RBP multiclass metrics |  |  | multiclass/alzheimer=1876; multiclass/frontotemporal_dementia=1597; multiclass/healthy_control=1106 | multiclass/alzheimer=1932; multiclass/frontotemporal_dementia=1100; multiclass/healthy_control=1599 |
| val_as_test_80_20 | available | Table 4 | modified RBP AD+FTD vs Healthy metrics |  |  | ad_ftd_vs_healthy/alzheimer_or_frontotemporal_dementia=2983; ad_ftd_vs_healthy/healthy_control=1596 | ad_ftd_vs_healthy/alzheimer_or_frontotemporal_dementia=3032; ad_ftd_vs_healthy/healthy_control=1599 |
| val_as_test_80_20 | available | Table 5 | modified RBP AD vs Healthy metrics |  |  | ad_vs_healthy/alzheimer=1876; ad_vs_healthy/healthy_control=1596 | ad_vs_healthy/alzheimer=1932; ad_vs_healthy/healthy_control=1599 |
| val_as_test_80_20 | available | Table 6 | modified RBP FTD vs Healthy metrics |  |  | ftd_vs_healthy/frontotemporal_dementia=1597; ftd_vs_healthy/healthy_control=1596 | ftd_vs_healthy/frontotemporal_dementia=1100; ftd_vs_healthy/healthy_control=1599 |
| val_as_test_80_20 | available | Table 7 | modified RBP task accuracy | ftd_vs_healthy=0.997; ad_vs_healthy=0.9974; ad_ftd_vs_healthy=0.998; multiclass=0.8034 | ftd_vs_healthy=0.7121155983697666; ad_vs_healthy=0.7187765505522515; ad_ftd_vs_healthy=0.7354782984236666; multiclass=0.5789246383070611 |  |  |
| val_as_test_80_20 | available | Table 8 | SMOTE multiclass metrics | multiclass=77.45% | multiclass=0.5488267770876466 | multiclass/alzheimer=1876; multiclass/frontotemporal_dementia=1876; multiclass/healthy_control=1876 | multiclass/alzheimer=1932; multiclass/frontotemporal_dementia=1932; multiclass/healthy_control=1932 |
| val_as_test_80_20 | available | Table 9 | SMOTE AD vs Healthy metrics |  |  | ad_vs_healthy/alzheimer=1876; ad_vs_healthy/healthy_control=1876 | ad_vs_healthy/alzheimer=1932; ad_vs_healthy/healthy_control=1932 |
| val_as_test_80_20 | available | Table 10 | 5-fold multiclass accuracy |  |  |  |  |
| val_as_test_80_20 | available | Table 11 | 5-fold AD vs Healthy accuracy |  |  |  |  |
| val_as_test_80_20 | available | Table 12 | standard RBP multiclass metrics | multiclass=63.03% | multiclass=0.5621225194132873 | multiclass/alzheimer=1876; multiclass/frontotemporal_dementia=1597; multiclass/healthy_control=1106 | multiclass/alzheimer=967; multiclass/frontotemporal_dementia=551; multiclass/healthy_control=800 |
| val_as_test_80_20 | available | Table 13 | standard RBP AD vs Healthy metrics | ad_vs_healthy=76.36% | ad_vs_healthy=0.7176004527447651 | ad_vs_healthy/alzheimer=1876; ad_vs_healthy/healthy_control=1597 | ad_vs_healthy/alzheimer=967; ad_vs_healthy/healthy_control=800 |
| val_as_test_80_20 | available | Table 14 | literature comparison |  |  |  |  |

## Protocol summary

| Scenario | Status | Component | Value | Notes |
| --- | --- | --- | --- | --- |
| paper_literal_80_10_10 | available | raw_dataset | OpenNeuro ds004504 v1.0.5 | Paper data availability statement points to openneuro.ds004504.v1.0.5. |
| paper_literal_80_10_10 | available | preprocessing_source | derivatives/sub-*/eeg/*_task-eyesclosed_eeg.set | Uses preprocessed derivative EEGLAB .set files. |
| paper_literal_80_10_10 | available | epoching | 6 seconds with 50% overlap |  |
| paper_literal_80_10_10 | available | modified_rbp_bands | delta 0.5-4, theta 4-8, alpha 8-16, zaeta 16-24, beta 24-30, gamma 30-45 | Used for Tables 3-11. |
| paper_literal_80_10_10 | available | standard_rbp_bands | delta 0.5-4, theta 4-8, alpha 8-13, beta 13-25, gamma 25-45 | Used for Tables 12-13. |
| paper_literal_80_10_10 | available | normalization | paper-style full-task min-max normalization before split | This follows the paper wording but leaks evaluation distribution statistics. |
| paper_literal_80_10_10 | available | split | 80/10/10 epoch-level stratified split |  |
| paper_literal_80_10_10 | available | evaluation | reported metrics use independent split.test partition |  |
| paper_literal_80_10_10 | available | model | TCN-LSTM, two TCN blocks, 32 channels, kernel 7, LSTM 64, Dense 128/192/256 |  |
| paper_literal_80_10_10 | available | model_hyperparameters | input_dim=1, num_classes=task-dependent, tcn_channels=32, tcn_kernel_size=7, tcn_dilations=[1, 1], tcn_dropout=0.3, lstm_hidden_dim=64, dense_hidden_dims=[128, 192, 256], dense_dropout=0.2 | num_classes is 3 for multiclass tasks and 2 for binary tasks. |
| paper_literal_80_10_10 | available | optimizer | Adam learning_rate=0.0001 batch_size=32 |  |
| paper_literal_80_10_10 | available | training_runtime | epochs=100, seed=randomly resolved at runtime unless configured, num_workers=4, device=cuda | run.json records the actual resolved seed for each run. |
| paper_literal_80_10_10 | available | smote | simple SMOTE applied to train, val, and test partitions for Table 8-9 shape | Paper does not provide source code or exact SMOTE placement. |
| paper_literal_80_10_10 | available | kfold | 5-fold epoch-level stratified k-fold | Paper does not specify subject-wise folds. |
| paper_literal_80_10_10 | available | label_swap_audit | FTD/Healthy target-label swap audit for Table 3 and Table 6 | Audit-only protocol, not the corrected dataset protocol. |
| paper_literal_80_10_10 | available | runs_dir | data/runs/ds004504_rbp_paper |  |
| val_as_test_80_20 | available | raw_dataset | OpenNeuro ds004504 v1.0.5 | Paper data availability statement points to openneuro.ds004504.v1.0.5. |
| val_as_test_80_20 | available | preprocessing_source | derivatives/sub-*/eeg/*_task-eyesclosed_eeg.set | Uses preprocessed derivative EEGLAB .set files. |
| val_as_test_80_20 | available | epoching | 6 seconds with 50% overlap |  |
| val_as_test_80_20 | available | modified_rbp_bands | delta 0.5-4, theta 4-8, alpha 8-16, zaeta 16-24, beta 24-30, gamma 30-45 | Used for Tables 3-11. |
| val_as_test_80_20 | available | standard_rbp_bands | delta 0.5-4, theta 4-8, alpha 8-13, beta 13-25, gamma 25-45 | Used for Tables 12-13. |
| val_as_test_80_20 | available | normalization | paper-style full-task min-max normalization before split | This follows the paper wording but leaks evaluation distribution statistics. |
| val_as_test_80_20 | available | split | 80/20 epoch-level stratified split |  |
| val_as_test_80_20 | available | evaluation | reported metrics use validation partition as test output, matching paper support evidence |  |
| val_as_test_80_20 | available | model | TCN-LSTM, two TCN blocks, 32 channels, kernel 7, LSTM 64, Dense 128/192/256 |  |
| val_as_test_80_20 | available | model_hyperparameters | input_dim=1, num_classes=task-dependent, tcn_channels=32, tcn_kernel_size=7, tcn_dilations=[1, 1], tcn_dropout=0.3, lstm_hidden_dim=64, dense_hidden_dims=[128, 192, 256], dense_dropout=0.2 | num_classes is 3 for multiclass tasks and 2 for binary tasks. |
| val_as_test_80_20 | available | optimizer | Adam learning_rate=0.0001 batch_size=32 |  |
| val_as_test_80_20 | available | training_runtime | epochs=100, seed=randomly resolved at runtime unless configured, num_workers=4, device=cuda | run.json records the actual resolved seed for each run. |
| val_as_test_80_20 | available | smote | simple SMOTE applied to train and reported val/test partitions for Table 8-9 shape | Paper does not provide source code or exact SMOTE placement. |
| val_as_test_80_20 | available | kfold | 5-fold epoch-level stratified k-fold | Paper does not specify subject-wise folds. |
| val_as_test_80_20 | available | label_swap_audit | FTD/Healthy target-label swap audit for Table 3 and Table 6 | Audit-only protocol, not the corrected dataset protocol. |
| val_as_test_80_20 | available | runs_dir | data/runs/ds004504_rbp_paper/val_as_test_80_20 |  |

## Paper vs ours details

| Scenario | Status | Table | Experiment | Task | Class | Metric | Paper value | Ours value | Difference | Ours available |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| paper_literal_80_10_10 | available | Table 3 | modified_rbp | multiclass | alzheimer | precision | 0.7 | 0.5398729710656316 | -0.16012702893436837 | true |
| paper_literal_80_10_10 | available | Table 3 | modified_rbp | multiclass | alzheimer | recall | 0.9 | 0.7911065149948294 | -0.1088934850051706 | true |
| paper_literal_80_10_10 | available | Table 3 | modified_rbp | multiclass | alzheimer | f1 | 0.79 | 0.6417785234899329 | -0.14822147651006712 | true |
| paper_literal_80_10_10 | available | Table 3 | modified_rbp | multiclass | alzheimer | sensitivity | 0.9 | 0.7911065149948294 | -0.1088934850051706 | true |
| paper_literal_80_10_10 | available | Table 3 | modified_rbp | multiclass | alzheimer | specificity | 0.74 | 0.5173945225758697 | -0.22260547742413028 | true |
| paper_literal_80_10_10 | available | Table 3 | modified_rbp | multiclass | alzheimer | support | 1876 | 967 | -909.0 | true |
| paper_literal_80_10_10 | available | Table 3 | modified_rbp | multiclass | frontotemporal_dementia | precision | 1.0 | 0.66 | -0.33999999999999997 | true |
| paper_literal_80_10_10 | available | Table 3 | modified_rbp | multiclass | frontotemporal_dementia | recall | 1.0 | 0.17967332123411978 | -0.8203266787658803 | true |
| paper_literal_80_10_10 | available | Table 3 | modified_rbp | multiclass | frontotemporal_dementia | f1 | 1.0 | 0.28245363766048504 | -0.717546362339515 | true |
| paper_literal_80_10_10 | available | Table 3 | modified_rbp | multiclass | frontotemporal_dementia | sensitivity | 1.0 | 0.17967332123411978 | -0.8203266787658803 | true |
| paper_literal_80_10_10 | available | Table 3 | modified_rbp | multiclass | frontotemporal_dementia | specificity | 1.0 | 0.9711375212224108 | -0.028862478777589184 | true |
| paper_literal_80_10_10 | available | Table 3 | modified_rbp | multiclass | frontotemporal_dementia | support | 1597 | 551 | -1046.0 | true |
| paper_literal_80_10_10 | available | Table 3 | modified_rbp | multiclass | healthy_control | precision | 0.68 | 0.6151797603195739 | -0.06482023968042616 | true |
| paper_literal_80_10_10 | available | Table 3 | modified_rbp | multiclass | healthy_control | recall | 0.35 | 0.5775 | 0.22750000000000004 | true |
| paper_literal_80_10_10 | available | Table 3 | modified_rbp | multiclass | healthy_control | f1 | 0.47 | 0.5957446808510637 | 0.12574468085106372 | true |
| paper_literal_80_10_10 | available | Table 3 | modified_rbp | multiclass | healthy_control | sensitivity | 0.35 | 0.5775 | 0.22750000000000004 | true |
| paper_literal_80_10_10 | available | Table 3 | modified_rbp | multiclass | healthy_control | specificity | 0.95 | 0.8096179183135704 | -0.14038208168642952 | true |
| paper_literal_80_10_10 | available | Table 3 | modified_rbp | multiclass | healthy_control | support | 1106 | 800 | -306.0 | true |
| paper_literal_80_10_10 | available | Table 4 | modified_rbp | ad_ftd_vs_healthy | alzheimer_or_frontotemporal_dementia | precision | 0.9977 | 0.7895392278953923 | -0.2081607721046077 | true |
| paper_literal_80_10_10 | available | Table 4 | modified_rbp | ad_ftd_vs_healthy | alzheimer_or_frontotemporal_dementia | recall | 0.9993 | 0.8358602504943968 | -0.16343974950560314 | true |
| paper_literal_80_10_10 | available | Table 4 | modified_rbp | ad_ftd_vs_healthy | alzheimer_or_frontotemporal_dementia | f1 | 0.9985 | 0.8120397054114634 | -0.18646029458853663 | true |
| paper_literal_80_10_10 | available | Table 4 | modified_rbp | ad_ftd_vs_healthy | alzheimer_or_frontotemporal_dementia | sensitivity | 1.0 | 0.8358602504943968 | -0.16413974950560317 | true |
| paper_literal_80_10_10 | available | Table 4 | modified_rbp | ad_ftd_vs_healthy | alzheimer_or_frontotemporal_dementia | specificity | 1.0 | 0.5775 | -0.4225 | true |
| paper_literal_80_10_10 | available | Table 4 | modified_rbp | ad_ftd_vs_healthy | alzheimer_or_frontotemporal_dementia | support | 2983 | 1517 | -1466.0 | true |
| paper_literal_80_10_10 | available | Table 4 | modified_rbp | ad_ftd_vs_healthy | healthy_control | precision | 0.9987 | 0.6497890295358649 | -0.3489109704641351 | true |
| paper_literal_80_10_10 | available | Table 4 | modified_rbp | ad_ftd_vs_healthy | healthy_control | recall | 0.9956 | 0.5775 | -0.4181 | true |
| paper_literal_80_10_10 | available | Table 4 | modified_rbp | ad_ftd_vs_healthy | healthy_control | f1 | 0.9972 | 0.6115155526141628 | -0.3856844473858372 | true |
| paper_literal_80_10_10 | available | Table 4 | modified_rbp | ad_ftd_vs_healthy | healthy_control | sensitivity | 1.0 | 0.5775 | -0.4225 | true |
| paper_literal_80_10_10 | available | Table 4 | modified_rbp | ad_ftd_vs_healthy | healthy_control | specificity | 1.0 | 0.8358602504943968 | -0.16413974950560317 | true |
| paper_literal_80_10_10 | available | Table 4 | modified_rbp | ad_ftd_vs_healthy | healthy_control | support | 1596 | 800 | -796.0 | true |
| paper_literal_80_10_10 | available | Table 5 | modified_rbp | ad_vs_healthy | alzheimer | precision | 0.9963 | 0.7446393762183235 | -0.2516606237816764 | true |
| paper_literal_80_10_10 | available | Table 5 | modified_rbp | ad_vs_healthy | alzheimer | recall | 0.9989 | 0.7900723888314375 | -0.20882761116856252 | true |
| paper_literal_80_10_10 | available | Table 5 | modified_rbp | ad_vs_healthy | alzheimer | f1 | 0.9976 | 0.7666833918715504 | -0.2309166081284496 | true |
| paper_literal_80_10_10 | available | Table 5 | modified_rbp | ad_vs_healthy | alzheimer | sensitivity | 1.0 | 0.7900723888314375 | -0.2099276111685625 | true |
| paper_literal_80_10_10 | available | Table 5 | modified_rbp | ad_vs_healthy | alzheimer | specificity | 1.0 | 0.6725 | -0.3275 | true |
| paper_literal_80_10_10 | available | Table 5 | modified_rbp | ad_vs_healthy | alzheimer | support | 1876 | 967 | -909.0 | true |
| paper_literal_80_10_10 | available | Table 5 | modified_rbp | ad_vs_healthy | healthy_control | precision | 0.9987 | 0.7260458839406208 | -0.2726541160593793 | true |
| paper_literal_80_10_10 | available | Table 5 | modified_rbp | ad_vs_healthy | healthy_control | recall | 0.9956 | 0.6725 | -0.32310000000000005 | true |
| paper_literal_80_10_10 | available | Table 5 | modified_rbp | ad_vs_healthy | healthy_control | f1 | 0.9972 | 0.6982478909798833 | -0.2989521090201167 | true |
| paper_literal_80_10_10 | available | Table 5 | modified_rbp | ad_vs_healthy | healthy_control | sensitivity | 1.0 | 0.6725 | -0.3275 | true |
| paper_literal_80_10_10 | available | Table 5 | modified_rbp | ad_vs_healthy | healthy_control | specificity | 1.0 | 0.7900723888314375 | -0.2099276111685625 | true |
| paper_literal_80_10_10 | available | Table 5 | modified_rbp | ad_vs_healthy | healthy_control | support | 1596 | 800 | -796.0 | true |
| paper_literal_80_10_10 | available | Table 6 | modified_rbp | ftd_vs_healthy | frontotemporal_dementia | precision | 0.9994 | 0.6994106090373281 | -0.29998939096267185 | true |
| paper_literal_80_10_10 | available | Table 6 | modified_rbp | ftd_vs_healthy | frontotemporal_dementia | recall | 0.9956 | 0.6460980036297641 | -0.34950199637023593 | true |
| paper_literal_80_10_10 | available | Table 6 | modified_rbp | ftd_vs_healthy | frontotemporal_dementia | f1 | 0.9975 | 0.6716981132075471 | -0.3258018867924529 | true |
| paper_literal_80_10_10 | available | Table 6 | modified_rbp | ftd_vs_healthy | frontotemporal_dementia | sensitivity | 1.0 | 0.6460980036297641 | -0.3539019963702359 | true |
| paper_literal_80_10_10 | available | Table 6 | modified_rbp | ftd_vs_healthy | frontotemporal_dementia | specificity | 1.0 | 0.80875 | -0.19125000000000003 | true |
| paper_literal_80_10_10 | available | Table 6 | modified_rbp | ftd_vs_healthy | frontotemporal_dementia | support | 1597 | 551 | -1046.0 | true |
| paper_literal_80_10_10 | available | Table 6 | modified_rbp | ftd_vs_healthy | healthy_control | precision | 0.9937 | 0.7684085510688836 | -0.2252914489311164 | true |
| paper_literal_80_10_10 | available | Table 6 | modified_rbp | ftd_vs_healthy | healthy_control | recall | 0.9991 | 0.80875 | -0.19035000000000002 | true |
| paper_literal_80_10_10 | available | Table 6 | modified_rbp | ftd_vs_healthy | healthy_control | f1 | 0.9964 | 0.7880633373934226 | -0.20833666260657735 | true |
| paper_literal_80_10_10 | available | Table 6 | modified_rbp | ftd_vs_healthy | healthy_control | sensitivity | 1.0 | 0.80875 | -0.19125000000000003 | true |
| paper_literal_80_10_10 | available | Table 6 | modified_rbp | ftd_vs_healthy | healthy_control | specificity | 1.0 | 0.6460980036297641 | -0.3539019963702359 | true |
| paper_literal_80_10_10 | available | Table 6 | modified_rbp | ftd_vs_healthy | healthy_control | support | 1596 | 800 | -796.0 | true |
| paper_literal_80_10_10 | available | Table 8 | smote_modified_rbp | multiclass | alzheimer | precision | 0.63 | 0.5439560439560439 | -0.0860439560439561 | true |
| paper_literal_80_10_10 | available | Table 8 | smote_modified_rbp | multiclass | alzheimer | recall | 0.71 | 0.4095139607032058 | -0.3004860392967942 | true |
| paper_literal_80_10_10 | available | Table 8 | smote_modified_rbp | multiclass | alzheimer | f1 | 0.67 | 0.4672566371681416 | -0.20274336283185845 | true |
| paper_literal_80_10_10 | available | Table 8 | smote_modified_rbp | multiclass | alzheimer | sensitivity | 0.71 | 0.4095139607032058 | -0.3004860392967942 | true |
| paper_literal_80_10_10 | available | Table 8 | smote_modified_rbp | multiclass | alzheimer | specificity | 0.79 | 0.828335056876939 | 0.038335056876938944 | true |
| paper_literal_80_10_10 | available | Table 8 | smote_modified_rbp | multiclass | alzheimer | support | 1876 | 967 | -909.0 | true |
| paper_literal_80_10_10 | available | Table 8 | smote_modified_rbp | multiclass | frontotemporal_dementia | precision | 1.0 | 0.4912621359223301 | -0.5087378640776699 | true |
| paper_literal_80_10_10 | available | Table 8 | smote_modified_rbp | multiclass | frontotemporal_dementia | recall | 1.0 | 0.5232678386763185 | -0.47673216132368146 | true |
| paper_literal_80_10_10 | available | Table 8 | smote_modified_rbp | multiclass | frontotemporal_dementia | f1 | 1.0 | 0.5067601402103155 | -0.49323985978968454 | true |
| paper_literal_80_10_10 | available | Table 8 | smote_modified_rbp | multiclass | frontotemporal_dementia | sensitivity | 1.0 | 0.5232678386763185 | -0.47673216132368146 | true |
| paper_literal_80_10_10 | available | Table 8 | smote_modified_rbp | multiclass | frontotemporal_dementia | specificity | 1.0 | 0.7290589451913133 | -0.2709410548086867 | true |
| paper_literal_80_10_10 | available | Table 8 | smote_modified_rbp | multiclass | frontotemporal_dementia | support | 1876 | 967 | -909.0 | true |
| paper_literal_80_10_10 | available | Table 8 | smote_modified_rbp | multiclass | healthy_control | precision | 0.67 | 0.5853018372703412 | -0.0846981627296588 | true |
| paper_literal_80_10_10 | available | Table 8 | smote_modified_rbp | multiclass | healthy_control | recall | 0.58 | 0.6918304033092038 | 0.1118304033092038 | true |
| paper_literal_80_10_10 | available | Table 8 | smote_modified_rbp | multiclass | healthy_control | f1 | 0.62 | 0.6341232227488152 | 0.014123222748815167 | true |
| paper_literal_80_10_10 | available | Table 8 | smote_modified_rbp | multiclass | healthy_control | sensitivity | 0.58 | 0.6918304033092038 | 0.1118304033092038 | true |
| paper_literal_80_10_10 | available | Table 8 | smote_modified_rbp | multiclass | healthy_control | specificity | 0.86 | 0.7549120992761117 | -0.1050879007238883 | true |
| paper_literal_80_10_10 | available | Table 8 | smote_modified_rbp | multiclass | healthy_control | support | 1876 | 967 | -909.0 | true |
| paper_literal_80_10_10 | available | Table 9 | smote_modified_rbp | ad_vs_healthy | alzheimer | precision | 99.73 | 0.6986027944111777 | -29.869720558882236 | true |
| paper_literal_80_10_10 | available | Table 9 | smote_modified_rbp | ad_vs_healthy | alzheimer | recall | 99.7 | 0.7238883143743536 | -27.311168562564646 | true |
| paper_literal_80_10_10 | available | Table 9 | smote_modified_rbp | ad_vs_healthy | alzheimer | f1 | 99.71 | 0.7110208227526664 | -28.607917724733355 | true |
| paper_literal_80_10_10 | available | Table 9 | smote_modified_rbp | ad_vs_healthy | alzheimer | support | 1876 | 967 | 94824.0 | true |
| paper_literal_80_10_10 | available | Table 9 | smote_modified_rbp | ad_vs_healthy | healthy_control | precision | 99.7 | 0.7135193133047211 | -28.34806866952789 | true |
| paper_literal_80_10_10 | available | Table 9 | smote_modified_rbp | ad_vs_healthy | healthy_control | recall | 99.71 | 0.687693898655636 | -30.940610134436398 | true |
| paper_literal_80_10_10 | available | Table 9 | smote_modified_rbp | ad_vs_healthy | healthy_control | f1 | 99.73 | 0.7003686150605583 | -29.693138493944176 | true |
| paper_literal_80_10_10 | available | Table 9 | smote_modified_rbp | ad_vs_healthy | healthy_control | support | 1876 | 967 | 94824.0 | true |
| paper_literal_80_10_10 | available | Table 12 | standard_rbp | multiclass | alzheimer | precision | 0.6 | 0.5328125 | -0.06718749999999996 | true |
| paper_literal_80_10_10 | available | Table 12 | standard_rbp | multiclass | alzheimer | recall | 0.77 | 0.7052740434332989 | -0.0647259565667011 | true |
| paper_literal_80_10_10 | available | Table 12 | standard_rbp | multiclass | alzheimer | f1 | 0.67 | 0.6070315976858034 | -0.06296840231419665 | true |
| paper_literal_80_10_10 | available | Table 12 | standard_rbp | multiclass | alzheimer | support | 1876 | 967 | -909.0 | true |
| paper_literal_80_10_10 | available | Table 12 | standard_rbp | multiclass | frontotemporal_dementia | precision | 0.68 | 0.5656565656565656 | -0.11434343434343441 | true |
| paper_literal_80_10_10 | available | Table 12 | standard_rbp | multiclass | frontotemporal_dementia | recall | 0.68 | 0.20326678765880218 | -0.47673321234119787 | true |
| paper_literal_80_10_10 | available | Table 12 | standard_rbp | multiclass | frontotemporal_dementia | f1 | 0.68 | 0.29906542056074764 | -0.3809345794392524 | true |
| paper_literal_80_10_10 | available | Table 12 | standard_rbp | multiclass | frontotemporal_dementia | support | 1597 | 551 | -1046.0 | true |
| paper_literal_80_10_10 | available | Table 12 | standard_rbp | multiclass | healthy_control | precision | 0.6 | 0.6059523809523809 | 0.005952380952380931 | true |
| paper_literal_80_10_10 | available | Table 12 | standard_rbp | multiclass | healthy_control | recall | 0.33 | 0.63625 | 0.30624999999999997 | true |
| paper_literal_80_10_10 | available | Table 12 | standard_rbp | multiclass | healthy_control | f1 | 0.43 | 0.6207317073170731 | 0.19073170731707306 | true |
| paper_literal_80_10_10 | available | Table 12 | standard_rbp | multiclass | healthy_control | support | 1106 | 800 | -306.0 | true |
| paper_literal_80_10_10 | available | Table 13 | standard_rbp | ad_vs_healthy | alzheimer | precision | 0.76 | 0.7174721189591078 | -0.042527881040892224 | true |
| paper_literal_80_10_10 | available | Table 13 | standard_rbp | ad_vs_healthy | alzheimer | recall | 0.81 | 0.7983453981385729 | -0.011654601861427194 | true |
| paper_literal_80_10_10 | available | Table 13 | standard_rbp | ad_vs_healthy | alzheimer | f1 | 0.79 | 0.7557513460597161 | -0.03424865394028398 | true |
| paper_literal_80_10_10 | available | Table 13 | standard_rbp | ad_vs_healthy | alzheimer | support | 1876 | 967 | -909.0 | true |
| paper_literal_80_10_10 | available | Table 13 | standard_rbp | ad_vs_healthy | healthy_control | precision | 0.76 | 0.7178002894356006 | -0.04219971056439942 | true |
| paper_literal_80_10_10 | available | Table 13 | standard_rbp | ad_vs_healthy | healthy_control | recall | 0.71 | 0.62 | -0.08999999999999997 | true |
| paper_literal_80_10_10 | available | Table 13 | standard_rbp | ad_vs_healthy | healthy_control | f1 | 0.73 | 0.6653252850435949 | -0.06467471495640509 | true |
| paper_literal_80_10_10 | available | Table 13 | standard_rbp | ad_vs_healthy | healthy_control | support | 1597 | 800 | -797.0 | true |
| paper_literal_80_10_10 | available | Table 7 | modified_rbp | ftd_vs_healthy |  | accuracy | 0.997 | 0.7424130273871207 | -0.2545869726128793 | true |
| paper_literal_80_10_10 | available | Table 7 | modified_rbp | ad_vs_healthy |  | accuracy | 0.9974 | 0.7368421052631579 | -0.2605578947368421 | true |
| paper_literal_80_10_10 | available | Table 7 | modified_rbp | ad_ftd_vs_healthy |  | accuracy | 0.998 | 0.7466551575312904 | -0.2513448424687096 | true |
| paper_literal_80_10_10 | available | Table 7 | modified_rbp | multiclass |  | accuracy | 0.8034 | 0.5720448662640207 | -0.23135513373597927 | true |
| paper_literal_80_10_10 | available | Table 8 | smote_modified_rbp | multiclass |  | accuracy | 77.45 | 0.5415374008962427 | -23.29625991037573 | true |
| paper_literal_80_10_10 | available | Table 12 | standard_rbp | multiclass |  | accuracy | 63.03 | 0.5621225194132873 | -6.81774805867127 | true |
| paper_literal_80_10_10 | available | Table 13 | standard_rbp | ad_vs_healthy |  | accuracy | 76.36 | 0.7176004527447651 | -4.599954725523489 | true |
| paper_literal_80_10_10 | available | Table 10 | kfold_modified_rbp | multiclass |  | fold_1_train_accuracy_percent | 79.89 | 56.92764578833693 | -22.96235421166307 | true |
| paper_literal_80_10_10 | available | Table 10 | kfold_modified_rbp | multiclass |  | fold_1_test_accuracy_percent | 80.15 | 58.30274238825308 | -21.847257611746926 | true |
| paper_literal_80_10_10 | available | Table 10 | kfold_modified_rbp | multiclass |  | fold_2_train_accuracy_percent | 80.0 | 57.44060475161987 | -22.559395248380127 | true |
| paper_literal_80_10_10 | available | Table 10 | kfold_modified_rbp | multiclass |  | fold_2_test_accuracy_percent | 80.0 | 57.24465558194775 | -22.755344418052253 | true |
| paper_literal_80_10_10 | available | Table 10 | kfold_modified_rbp | multiclass |  | fold_3_train_accuracy_percent | 79.58 | 57.5670860104746 | -22.0129139895254 | true |
| paper_literal_80_10_10 | available | Table 10 | kfold_modified_rbp | multiclass |  | fold_3_test_accuracy_percent | 80.06 | 57.170626349892004 | -22.889373650108 | true |
| paper_literal_80_10_10 | available | Table 10 | kfold_modified_rbp | multiclass |  | fold_4_train_accuracy_percent | 79.43 | 57.334917121105775 | -22.09508287889423 | true |
| paper_literal_80_10_10 | available | Table 10 | kfold_modified_rbp | multiclass |  | fold_4_test_accuracy_percent | 80.02 | 57.66738660907127 | -22.352613390928724 | true |
| paper_literal_80_10_10 | available | Table 10 | kfold_modified_rbp | multiclass |  | fold_5_train_accuracy_percent | 81.27 | 56.97548860814167 | -24.29451139185833 | true |
| paper_literal_80_10_10 | available | Table 10 | kfold_modified_rbp | multiclass |  | fold_5_test_accuracy_percent | 80.13 | 58.2847267228343 | -21.845273277165695 | true |
| paper_literal_80_10_10 | available | Table 11 | kfold_modified_rbp | ad_vs_healthy |  | fold_1_train_accuracy_percent | 99.82 | 71.5570346243716 | -28.262965375628397 | true |
| paper_literal_80_10_10 | available | Table 11 | kfold_modified_rbp | ad_vs_healthy |  | fold_1_test_accuracy_percent | 99.86 | 71.76437269895214 | -28.095627301047855 | true |
| paper_literal_80_10_10 | available | Table 11 | kfold_modified_rbp | ad_vs_healthy |  | fold_2_train_accuracy_percent | 99.8 | 71.42250230121078 | -28.377497698789213 | true |
| paper_literal_80_10_10 | available | Table 11 | kfold_modified_rbp | ad_vs_healthy |  | fold_2_test_accuracy_percent | 99.82 | 72.61399037099972 | -27.206009629000278 | true |
| paper_literal_80_10_10 | available | Table 11 | kfold_modified_rbp | ad_vs_healthy |  | fold_3_train_accuracy_percent | 99.73 | 71.59951851589605 | -28.130481484103953 | true |
| paper_literal_80_10_10 | available | Table 11 | kfold_modified_rbp | ad_vs_healthy |  | fold_3_test_accuracy_percent | 99.92 | 72.89719626168224 | -27.022803738317762 | true |
| paper_literal_80_10_10 | available | Table 11 | kfold_modified_rbp | ad_vs_healthy |  | fold_4_train_accuracy_percent | 99.61 | 71.81193797351837 | -27.79806202648163 | true |
| paper_literal_80_10_10 | available | Table 11 | kfold_modified_rbp | ad_vs_healthy |  | fold_4_test_accuracy_percent | 99.86 | 69.95185499858397 | -29.90814500141603 | true |
| paper_literal_80_10_10 | available | Table 11 | kfold_modified_rbp | ad_vs_healthy |  | fold_5_train_accuracy_percent | 99.78 | 71.5024072500708 | -28.277592749929198 | true |
| paper_literal_80_10_10 | available | Table 11 | kfold_modified_rbp | ad_vs_healthy |  | fold_5_test_accuracy_percent | 99.82 | 72.1813031161473 | -27.63869688385269 | true |
| val_as_test_80_20 | available | Table 3 | modified_rbp | multiclass | alzheimer | precision | 0.7 | 0.5396319886765747 | -0.16036801132342526 | true |
| val_as_test_80_20 | available | Table 3 | modified_rbp | multiclass | alzheimer | recall | 0.9 | 0.7893374741200828 | -0.1106625258799172 | true |
| val_as_test_80_20 | available | Table 3 | modified_rbp | multiclass | alzheimer | f1 | 0.79 | 0.6410256410256411 | -0.14897435897435896 | true |
| val_as_test_80_20 | available | Table 3 | modified_rbp | multiclass | alzheimer | sensitivity | 0.9 | 0.7893374741200828 | -0.1106625258799172 | true |
| val_as_test_80_20 | available | Table 3 | modified_rbp | multiclass | alzheimer | specificity | 0.74 | 0.5179696183771767 | -0.2220303816228233 | true |
| val_as_test_80_20 | available | Table 3 | modified_rbp | multiclass | alzheimer | support | 1876 | 1932 | 56.0 | true |
| val_as_test_80_20 | available | Table 3 | modified_rbp | multiclass | frontotemporal_dementia | precision | 1.0 | 0.6591928251121076 | -0.34080717488789236 | true |
| val_as_test_80_20 | available | Table 3 | modified_rbp | multiclass | frontotemporal_dementia | recall | 1.0 | 0.13363636363636364 | -0.8663636363636363 | true |
| val_as_test_80_20 | available | Table 3 | modified_rbp | multiclass | frontotemporal_dementia | f1 | 1.0 | 0.22222222222222224 | -0.7777777777777778 | true |
| val_as_test_80_20 | available | Table 3 | modified_rbp | multiclass | frontotemporal_dementia | sensitivity | 1.0 | 0.13363636363636364 | -0.8663636363636363 | true |
| val_as_test_80_20 | available | Table 3 | modified_rbp | multiclass | frontotemporal_dementia | specificity | 1.0 | 0.978476352308128 | -0.021523647691871983 | true |
| val_as_test_80_20 | available | Table 3 | modified_rbp | multiclass | frontotemporal_dementia | support | 1597 | 1100 | -497.0 | true |
| val_as_test_80_20 | available | Table 3 | modified_rbp | multiclass | healthy_control | precision | 0.68 | 0.6378002528445006 | -0.042199747155499456 | true |
| val_as_test_80_20 | available | Table 3 | modified_rbp | multiclass | healthy_control | recall | 0.35 | 0.631019387116948 | 0.28101938711694807 | true |
| val_as_test_80_20 | available | Table 3 | modified_rbp | multiclass | healthy_control | f1 | 0.47 | 0.6343917007230431 | 0.1643917007230431 | true |
| val_as_test_80_20 | available | Table 3 | modified_rbp | multiclass | healthy_control | sensitivity | 0.35 | 0.631019387116948 | 0.28101938711694807 | true |
| val_as_test_80_20 | available | Table 3 | modified_rbp | multiclass | healthy_control | specificity | 0.95 | 0.8110158311345647 | -0.1389841688654353 | true |
| val_as_test_80_20 | available | Table 3 | modified_rbp | multiclass | healthy_control | support | 1106 | 1599 | 493.0 | true |
| val_as_test_80_20 | available | Table 4 | modified_rbp | ad_ftd_vs_healthy | alzheimer_or_frontotemporal_dementia | precision | 0.9977 | 0.7503463563313937 | -0.24735364366860635 | true |
| val_as_test_80_20 | available | Table 4 | modified_rbp | ad_ftd_vs_healthy | alzheimer_or_frontotemporal_dementia | recall | 0.9993 | 0.8931398416886543 | -0.10616015831134562 | true |
| val_as_test_80_20 | available | Table 4 | modified_rbp | ad_ftd_vs_healthy | alzheimer_or_frontotemporal_dementia | f1 | 0.9985 | 0.8155398283391054 | -0.18296017166089462 | true |
| val_as_test_80_20 | available | Table 4 | modified_rbp | ad_ftd_vs_healthy | alzheimer_or_frontotemporal_dementia | sensitivity | 1.0 | 0.8931398416886543 | -0.10686015831134565 | true |
| val_as_test_80_20 | available | Table 4 | modified_rbp | ad_ftd_vs_healthy | alzheimer_or_frontotemporal_dementia | specificity | 1.0 | 0.4365228267667292 | -0.5634771732332708 | true |
| val_as_test_80_20 | available | Table 4 | modified_rbp | ad_ftd_vs_healthy | alzheimer_or_frontotemporal_dementia | support | 2983 | 3032 | 49.0 | true |
| val_as_test_80_20 | available | Table 4 | modified_rbp | ad_ftd_vs_healthy | healthy_control | precision | 0.9987 | 0.6829745596868885 | -0.31572544031311156 | true |
| val_as_test_80_20 | available | Table 4 | modified_rbp | ad_ftd_vs_healthy | healthy_control | recall | 0.9956 | 0.4365228267667292 | -0.5590771732332709 | true |
| val_as_test_80_20 | available | Table 4 | modified_rbp | ad_ftd_vs_healthy | healthy_control | f1 | 0.9972 | 0.5326211369706219 | -0.4645788630293781 | true |
| val_as_test_80_20 | available | Table 4 | modified_rbp | ad_ftd_vs_healthy | healthy_control | sensitivity | 1.0 | 0.4365228267667292 | -0.5634771732332708 | true |
| val_as_test_80_20 | available | Table 4 | modified_rbp | ad_ftd_vs_healthy | healthy_control | specificity | 1.0 | 0.8931398416886543 | -0.10686015831134565 | true |
| val_as_test_80_20 | available | Table 4 | modified_rbp | ad_ftd_vs_healthy | healthy_control | support | 1596 | 1599 | 3.0 | true |
| val_as_test_80_20 | available | Table 5 | modified_rbp | ad_vs_healthy | alzheimer | precision | 0.9963 | 0.7346326836581709 | -0.26166731634182905 | true |
| val_as_test_80_20 | available | Table 5 | modified_rbp | ad_vs_healthy | alzheimer | recall | 0.9989 | 0.7608695652173914 | -0.23803043478260866 | true |
| val_as_test_80_20 | available | Table 5 | modified_rbp | ad_vs_healthy | alzheimer | f1 | 0.9976 | 0.7475209763539283 | -0.2500790236460717 | true |
| val_as_test_80_20 | available | Table 5 | modified_rbp | ad_vs_healthy | alzheimer | sensitivity | 1.0 | 0.7608695652173914 | -0.23913043478260865 | true |
| val_as_test_80_20 | available | Table 5 | modified_rbp | ad_vs_healthy | alzheimer | specificity | 1.0 | 0.6679174484052532 | -0.33208255159474676 | true |
| val_as_test_80_20 | available | Table 5 | modified_rbp | ad_vs_healthy | alzheimer | support | 1876 | 1932 | 56.0 | true |
| val_as_test_80_20 | available | Table 5 | modified_rbp | ad_vs_healthy | healthy_control | precision | 0.9987 | 0.6980392156862745 | -0.30066078431372556 | true |
| val_as_test_80_20 | available | Table 5 | modified_rbp | ad_vs_healthy | healthy_control | recall | 0.9956 | 0.6679174484052532 | -0.3276825515947468 | true |
| val_as_test_80_20 | available | Table 5 | modified_rbp | ad_vs_healthy | healthy_control | f1 | 0.9972 | 0.6826462128475551 | -0.31455378715244486 | true |
| val_as_test_80_20 | available | Table 5 | modified_rbp | ad_vs_healthy | healthy_control | sensitivity | 1.0 | 0.6679174484052532 | -0.33208255159474676 | true |
| val_as_test_80_20 | available | Table 5 | modified_rbp | ad_vs_healthy | healthy_control | specificity | 1.0 | 0.7608695652173914 | -0.23913043478260865 | true |
| val_as_test_80_20 | available | Table 5 | modified_rbp | ad_vs_healthy | healthy_control | support | 1596 | 1599 | 3.0 | true |
| val_as_test_80_20 | available | Table 6 | modified_rbp | ftd_vs_healthy | frontotemporal_dementia | precision | 0.9994 | 0.6231884057971014 | -0.3762115942028985 | true |
| val_as_test_80_20 | available | Table 6 | modified_rbp | ftd_vs_healthy | frontotemporal_dementia | recall | 0.9956 | 0.7427272727272727 | -0.25287272727272736 | true |
| val_as_test_80_20 | available | Table 6 | modified_rbp | ftd_vs_healthy | frontotemporal_dementia | f1 | 0.9975 | 0.6777270841974283 | -0.3197729158025717 | true |
| val_as_test_80_20 | available | Table 6 | modified_rbp | ftd_vs_healthy | frontotemporal_dementia | sensitivity | 1.0 | 0.7427272727272727 | -0.2572727272727273 | true |
| val_as_test_80_20 | available | Table 6 | modified_rbp | ftd_vs_healthy | frontotemporal_dementia | specificity | 1.0 | 0.6910569105691057 | -0.3089430894308943 | true |
| val_as_test_80_20 | available | Table 6 | modified_rbp | ftd_vs_healthy | frontotemporal_dementia | support | 1597 | 1100 | -497.0 | true |
| val_as_test_80_20 | available | Table 6 | modified_rbp | ftd_vs_healthy | healthy_control | precision | 0.9937 | 0.7961095100864554 | -0.19759048991354466 | true |
| val_as_test_80_20 | available | Table 6 | modified_rbp | ftd_vs_healthy | healthy_control | recall | 0.9991 | 0.6910569105691057 | -0.3080430894308943 | true |
| val_as_test_80_20 | available | Table 6 | modified_rbp | ftd_vs_healthy | healthy_control | f1 | 0.9964 | 0.7398727820555742 | -0.2565272179444258 | true |
| val_as_test_80_20 | available | Table 6 | modified_rbp | ftd_vs_healthy | healthy_control | sensitivity | 1.0 | 0.6910569105691057 | -0.3089430894308943 | true |
| val_as_test_80_20 | available | Table 6 | modified_rbp | ftd_vs_healthy | healthy_control | specificity | 1.0 | 0.7427272727272727 | -0.2572727272727273 | true |
| val_as_test_80_20 | available | Table 6 | modified_rbp | ftd_vs_healthy | healthy_control | support | 1596 | 1599 | 3.0 | true |
| val_as_test_80_20 | available | Table 8 | smote_modified_rbp | multiclass | alzheimer | precision | 0.63 | 0.5125170687300865 | -0.1174829312699135 | true |
| val_as_test_80_20 | available | Table 8 | smote_modified_rbp | multiclass | alzheimer | recall | 0.71 | 0.582815734989648 | -0.12718426501035196 | true |
| val_as_test_80_20 | available | Table 8 | smote_modified_rbp | multiclass | alzheimer | f1 | 0.67 | 0.5454105110196174 | -0.12458948898038269 | true |
| val_as_test_80_20 | available | Table 8 | smote_modified_rbp | multiclass | alzheimer | sensitivity | 0.71 | 0.582815734989648 | -0.12718426501035196 | true |
| val_as_test_80_20 | available | Table 8 | smote_modified_rbp | multiclass | alzheimer | specificity | 0.79 | 0.7228260869565217 | -0.0671739130434783 | true |
| val_as_test_80_20 | available | Table 8 | smote_modified_rbp | multiclass | alzheimer | support | 1876 | 1932 | 56.0 | true |
| val_as_test_80_20 | available | Table 8 | smote_modified_rbp | multiclass | frontotemporal_dementia | precision | 1.0 | 0.5197923426346528 | -0.4802076573653472 | true |
| val_as_test_80_20 | available | Table 8 | smote_modified_rbp | multiclass | frontotemporal_dementia | recall | 1.0 | 0.41459627329192544 | -0.5854037267080745 | true |
| val_as_test_80_20 | available | Table 8 | smote_modified_rbp | multiclass | frontotemporal_dementia | f1 | 1.0 | 0.4612726749208177 | -0.5387273250791823 | true |
| val_as_test_80_20 | available | Table 8 | smote_modified_rbp | multiclass | frontotemporal_dementia | sensitivity | 1.0 | 0.41459627329192544 | -0.5854037267080745 | true |
| val_as_test_80_20 | available | Table 8 | smote_modified_rbp | multiclass | frontotemporal_dementia | specificity | 1.0 | 0.8084886128364389 | -0.1915113871635611 | true |
| val_as_test_80_20 | available | Table 8 | smote_modified_rbp | multiclass | frontotemporal_dementia | support | 1876 | 1932 | 56.0 | true |
| val_as_test_80_20 | available | Table 8 | smote_modified_rbp | multiclass | healthy_control | precision | 0.67 | 0.60932944606414 | -0.06067055393586007 | true |
| val_as_test_80_20 | available | Table 8 | smote_modified_rbp | multiclass | healthy_control | recall | 0.58 | 0.6490683229813664 | 0.06906832298136645 | true |
| val_as_test_80_20 | available | Table 8 | smote_modified_rbp | multiclass | healthy_control | f1 | 0.62 | 0.6285714285714286 | 0.008571428571428563 | true |
| val_as_test_80_20 | available | Table 8 | smote_modified_rbp | multiclass | healthy_control | sensitivity | 0.58 | 0.6490683229813664 | 0.06906832298136645 | true |
| val_as_test_80_20 | available | Table 8 | smote_modified_rbp | multiclass | healthy_control | specificity | 0.86 | 0.7919254658385093 | -0.06807453416149067 | true |
| val_as_test_80_20 | available | Table 8 | smote_modified_rbp | multiclass | healthy_control | support | 1876 | 1932 | 56.0 | true |
| val_as_test_80_20 | available | Table 9 | smote_modified_rbp | ad_vs_healthy | alzheimer | precision | 99.73 | 0.7419540229885058 | -25.534597701149423 | true |
| val_as_test_80_20 | available | Table 9 | smote_modified_rbp | ad_vs_healthy | alzheimer | recall | 99.7 | 0.6682194616977226 | -32.87805383022774 | true |
| val_as_test_80_20 | available | Table 9 | smote_modified_rbp | ad_vs_healthy | alzheimer | f1 | 99.71 | 0.7031590413943356 | -29.394095860566438 | true |
| val_as_test_80_20 | available | Table 9 | smote_modified_rbp | ad_vs_healthy | alzheimer | support | 1876 | 1932 | 191324.0 | true |
| val_as_test_80_20 | available | Table 9 | smote_modified_rbp | ad_vs_healthy | healthy_control | precision | 99.7 | 0.698210922787194 | -29.878907721280598 | true |
| val_as_test_80_20 | available | Table 9 | smote_modified_rbp | ad_vs_healthy | healthy_control | recall | 99.71 | 0.7675983436853002 | -22.95016563146997 | true |
| val_as_test_80_20 | available | Table 9 | smote_modified_rbp | ad_vs_healthy | healthy_control | f1 | 99.73 | 0.7312623274161736 | -26.603767258382646 | true |
| val_as_test_80_20 | available | Table 9 | smote_modified_rbp | ad_vs_healthy | healthy_control | support | 1876 | 1932 | 191324.0 | true |
| val_as_test_80_20 | available | Table 12 | standard_rbp | multiclass | alzheimer | precision | 0.6 | 0.5328125 | -0.06718749999999996 | true |
| val_as_test_80_20 | available | Table 12 | standard_rbp | multiclass | alzheimer | recall | 0.77 | 0.7052740434332989 | -0.0647259565667011 | true |
| val_as_test_80_20 | available | Table 12 | standard_rbp | multiclass | alzheimer | f1 | 0.67 | 0.6070315976858034 | -0.06296840231419665 | true |
| val_as_test_80_20 | available | Table 12 | standard_rbp | multiclass | alzheimer | support | 1876 | 967 | -909.0 | true |
| val_as_test_80_20 | available | Table 12 | standard_rbp | multiclass | frontotemporal_dementia | precision | 0.68 | 0.5656565656565656 | -0.11434343434343441 | true |
| val_as_test_80_20 | available | Table 12 | standard_rbp | multiclass | frontotemporal_dementia | recall | 0.68 | 0.20326678765880218 | -0.47673321234119787 | true |
| val_as_test_80_20 | available | Table 12 | standard_rbp | multiclass | frontotemporal_dementia | f1 | 0.68 | 0.29906542056074764 | -0.3809345794392524 | true |
| val_as_test_80_20 | available | Table 12 | standard_rbp | multiclass | frontotemporal_dementia | support | 1597 | 551 | -1046.0 | true |
| val_as_test_80_20 | available | Table 12 | standard_rbp | multiclass | healthy_control | precision | 0.6 | 0.6059523809523809 | 0.005952380952380931 | true |
| val_as_test_80_20 | available | Table 12 | standard_rbp | multiclass | healthy_control | recall | 0.33 | 0.63625 | 0.30624999999999997 | true |
| val_as_test_80_20 | available | Table 12 | standard_rbp | multiclass | healthy_control | f1 | 0.43 | 0.6207317073170731 | 0.19073170731707306 | true |
| val_as_test_80_20 | available | Table 12 | standard_rbp | multiclass | healthy_control | support | 1106 | 800 | -306.0 | true |
| val_as_test_80_20 | available | Table 13 | standard_rbp | ad_vs_healthy | alzheimer | precision | 0.76 | 0.7174721189591078 | -0.042527881040892224 | true |
| val_as_test_80_20 | available | Table 13 | standard_rbp | ad_vs_healthy | alzheimer | recall | 0.81 | 0.7983453981385729 | -0.011654601861427194 | true |
| val_as_test_80_20 | available | Table 13 | standard_rbp | ad_vs_healthy | alzheimer | f1 | 0.79 | 0.7557513460597161 | -0.03424865394028398 | true |
| val_as_test_80_20 | available | Table 13 | standard_rbp | ad_vs_healthy | alzheimer | support | 1876 | 967 | -909.0 | true |
| val_as_test_80_20 | available | Table 13 | standard_rbp | ad_vs_healthy | healthy_control | precision | 0.76 | 0.7178002894356006 | -0.04219971056439942 | true |
| val_as_test_80_20 | available | Table 13 | standard_rbp | ad_vs_healthy | healthy_control | recall | 0.71 | 0.62 | -0.08999999999999997 | true |
| val_as_test_80_20 | available | Table 13 | standard_rbp | ad_vs_healthy | healthy_control | f1 | 0.73 | 0.6653252850435949 | -0.06467471495640509 | true |
| val_as_test_80_20 | available | Table 13 | standard_rbp | ad_vs_healthy | healthy_control | support | 1597 | 800 | -797.0 | true |
| val_as_test_80_20 | available | Table 7 | modified_rbp | ftd_vs_healthy |  | accuracy | 0.997 | 0.7121155983697666 | -0.28488440163023343 | true |
| val_as_test_80_20 | available | Table 7 | modified_rbp | ad_vs_healthy |  | accuracy | 0.9974 | 0.7187765505522515 | -0.2786234494477484 | true |
| val_as_test_80_20 | available | Table 7 | modified_rbp | ad_ftd_vs_healthy |  | accuracy | 0.998 | 0.7354782984236666 | -0.26252170157633337 | true |
| val_as_test_80_20 | available | Table 7 | modified_rbp | multiclass |  | accuracy | 0.8034 | 0.5789246383070611 | -0.2244753616929389 | true |
| val_as_test_80_20 | available | Table 8 | smote_modified_rbp | multiclass |  | accuracy | 77.45 | 0.5488267770876466 | -22.567322291235342 | true |
| val_as_test_80_20 | available | Table 12 | standard_rbp | multiclass |  | accuracy | 63.03 | 0.5621225194132873 | -6.81774805867127 | true |
| val_as_test_80_20 | available | Table 13 | standard_rbp | ad_vs_healthy |  | accuracy | 76.36 | 0.7176004527447651 | -4.599954725523489 | true |
| val_as_test_80_20 | available | Table 10 | kfold_modified_rbp | multiclass |  | fold_1_train_accuracy_percent | 79.89 | 56.92764578833693 | -22.96235421166307 | true |
| val_as_test_80_20 | available | Table 10 | kfold_modified_rbp | multiclass |  | fold_1_test_accuracy_percent | 80.15 | 58.30274238825308 | -21.847257611746926 | true |
| val_as_test_80_20 | available | Table 10 | kfold_modified_rbp | multiclass |  | fold_2_train_accuracy_percent | 80.0 | 57.44060475161987 | -22.559395248380127 | true |
| val_as_test_80_20 | available | Table 10 | kfold_modified_rbp | multiclass |  | fold_2_test_accuracy_percent | 80.0 | 57.24465558194775 | -22.755344418052253 | true |
| val_as_test_80_20 | available | Table 10 | kfold_modified_rbp | multiclass |  | fold_3_train_accuracy_percent | 79.58 | 57.5670860104746 | -22.0129139895254 | true |
| val_as_test_80_20 | available | Table 10 | kfold_modified_rbp | multiclass |  | fold_3_test_accuracy_percent | 80.06 | 57.170626349892004 | -22.889373650108 | true |
| val_as_test_80_20 | available | Table 10 | kfold_modified_rbp | multiclass |  | fold_4_train_accuracy_percent | 79.43 | 57.334917121105775 | -22.09508287889423 | true |
| val_as_test_80_20 | available | Table 10 | kfold_modified_rbp | multiclass |  | fold_4_test_accuracy_percent | 80.02 | 57.66738660907127 | -22.352613390928724 | true |
| val_as_test_80_20 | available | Table 10 | kfold_modified_rbp | multiclass |  | fold_5_train_accuracy_percent | 81.27 | 56.97548860814167 | -24.29451139185833 | true |
| val_as_test_80_20 | available | Table 10 | kfold_modified_rbp | multiclass |  | fold_5_test_accuracy_percent | 80.13 | 58.2847267228343 | -21.845273277165695 | true |
| val_as_test_80_20 | available | Table 11 | kfold_modified_rbp | ad_vs_healthy |  | fold_1_train_accuracy_percent | 99.82 | 71.5570346243716 | -28.262965375628397 | true |
| val_as_test_80_20 | available | Table 11 | kfold_modified_rbp | ad_vs_healthy |  | fold_1_test_accuracy_percent | 99.86 | 71.76437269895214 | -28.095627301047855 | true |
| val_as_test_80_20 | available | Table 11 | kfold_modified_rbp | ad_vs_healthy |  | fold_2_train_accuracy_percent | 99.8 | 71.42250230121078 | -28.377497698789213 | true |
| val_as_test_80_20 | available | Table 11 | kfold_modified_rbp | ad_vs_healthy |  | fold_2_test_accuracy_percent | 99.82 | 72.61399037099972 | -27.206009629000278 | true |
| val_as_test_80_20 | available | Table 11 | kfold_modified_rbp | ad_vs_healthy |  | fold_3_train_accuracy_percent | 99.73 | 71.59951851589605 | -28.130481484103953 | true |
| val_as_test_80_20 | available | Table 11 | kfold_modified_rbp | ad_vs_healthy |  | fold_3_test_accuracy_percent | 99.92 | 72.89719626168224 | -27.022803738317762 | true |
| val_as_test_80_20 | available | Table 11 | kfold_modified_rbp | ad_vs_healthy |  | fold_4_train_accuracy_percent | 99.61 | 71.81193797351837 | -27.79806202648163 | true |
| val_as_test_80_20 | available | Table 11 | kfold_modified_rbp | ad_vs_healthy |  | fold_4_test_accuracy_percent | 99.86 | 69.95185499858397 | -29.90814500141603 | true |
| val_as_test_80_20 | available | Table 11 | kfold_modified_rbp | ad_vs_healthy |  | fold_5_train_accuracy_percent | 99.78 | 71.5024072500708 | -28.277592749929198 | true |
| val_as_test_80_20 | available | Table 11 | kfold_modified_rbp | ad_vs_healthy |  | fold_5_test_accuracy_percent | 99.82 | 72.1813031161473 | -27.63869688385269 | true |

## Issue summary

| Scenario | Status | Issue | Severity | Affected tables | Paper observation | Reproduction implication | Project handling |
| --- | --- | --- | --- | --- | --- | --- | --- |
| paper_literal_80_10_10 | available | split_support_mismatch | high | Table 3, Table 4, Table 5, Table 6, Table 7 | Paper text claims 80/10/10 train/validation/test split. | Reported supports are close to a 20% holdout, not a 10% test split. | Provide both paper_literal_80_10_10 and val_as_test_80_20 scenarios; default report uses paper_literal_80_10_10 to follow the paper text, while val_as_test_80_20 is retained for paper-inferred support comparison. |
| paper_literal_80_10_10 | available | ftd_healthy_support_swap | high | Table 3, Table 4, Table 6 | FTD and Healthy supports align much better if FTD/Healthy are swapped. | Per-class interpretation of model behavior may be reversed for FTD and Healthy. | Provide label_swap and label_swap_80_20 audit runs and include ours_label_swap rows in Table 3 and Table 6 reports. |
| paper_literal_80_10_10 | available | table6_healthy_support_typo | medium | Table 6 | Table 6 reports Healthy support 1596, but Figure 4d and recall imply 1106. | Table support column cannot be used blindly; Figure 4 row sums are needed for consistency checks. | support_comparison.csv includes both paper table support and paper figure row-sum support. |
| paper_literal_80_10_10 | available | smote_accuracy_inconsistency | medium | Table 8 | Paper text reports SMOTE accuracy 77.45%, but Table 8 equal supports and recalls imply a different value. | SMOTE headline accuracy should be compared separately from row-level metrics. | accuracy.csv and paper_vs_ours.csv keep Table 8 text accuracy separate from row metrics. |
| paper_literal_80_10_10 | available | unspecified_smote_placement | medium | Table 8, Table 9 | Paper does not specify whether SMOTE was applied before split, after split, or only to training data. | Exact SMOTE reproduction is not uniquely determined from the paper. | Use explicit simple SMOTE with documented partitions in protocol_manifest.csv. |
| paper_literal_80_10_10 | available | epoch_level_leakage_risk | high | Table 3 through Table 13 | Paper describes epoch-level split and full min-max normalization before split. | Results may be optimistic because adjacent epochs and normalization statistics can leak across partitions. | Reproduce paper-style protocol explicitly and record leakage risk in protocol_manifest.csv; corrected subject-wise protocol is outside Table 1-14 paper reproduction. |
| val_as_test_80_20 | available | split_support_mismatch | high | Table 3, Table 4, Table 5, Table 6, Table 7 | Paper text claims 80/10/10 train/validation/test split. | Reported supports are close to a 20% holdout, not a 10% test split. | Provide both paper_literal_80_10_10 and val_as_test_80_20 scenarios; default report uses paper_literal_80_10_10 to follow the paper text, while val_as_test_80_20 is retained for paper-inferred support comparison. |
| val_as_test_80_20 | available | ftd_healthy_support_swap | high | Table 3, Table 4, Table 6 | FTD and Healthy supports align much better if FTD/Healthy are swapped. | Per-class interpretation of model behavior may be reversed for FTD and Healthy. | Provide label_swap and label_swap_80_20 audit runs and include ours_label_swap rows in Table 3 and Table 6 reports. |
| val_as_test_80_20 | available | table6_healthy_support_typo | medium | Table 6 | Table 6 reports Healthy support 1596, but Figure 4d and recall imply 1106. | Table support column cannot be used blindly; Figure 4 row sums are needed for consistency checks. | support_comparison.csv includes both paper table support and paper figure row-sum support. |
| val_as_test_80_20 | available | smote_accuracy_inconsistency | medium | Table 8 | Paper text reports SMOTE accuracy 77.45%, but Table 8 equal supports and recalls imply a different value. | SMOTE headline accuracy should be compared separately from row-level metrics. | accuracy.csv and paper_vs_ours.csv keep Table 8 text accuracy separate from row metrics. |
| val_as_test_80_20 | available | unspecified_smote_placement | medium | Table 8, Table 9 | Paper does not specify whether SMOTE was applied before split, after split, or only to training data. | Exact SMOTE reproduction is not uniquely determined from the paper. | Use explicit simple SMOTE with documented partitions in protocol_manifest.csv. |
| val_as_test_80_20 | available | epoch_level_leakage_risk | high | Table 3 through Table 13 | Paper describes epoch-level split and full min-max normalization before split. | Results may be optimistic because adjacent epochs and normalization statistics can leak across partitions. | Reproduce paper-style protocol explicitly and record leakage risk in protocol_manifest.csv; corrected subject-wise protocol is outside Table 1-14 paper reproduction. |
