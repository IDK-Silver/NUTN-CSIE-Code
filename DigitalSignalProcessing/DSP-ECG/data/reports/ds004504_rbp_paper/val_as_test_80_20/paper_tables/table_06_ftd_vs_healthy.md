# Table 6. Classification metrics for frontotemporal disease and healthy classes.

| Source | Class | Precision | Recall | F1 score | Sensitivity | Specificity | Support | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| paper | frontotemporal_dementia | 0.9994 | 0.9956 | 0.9975 | 1.0 | 1.0 | 1597 | Binary sensitivity/specificity are carried from the paper's single reported row. |
| paper | healthy_control | 0.9937 | 0.9991 | 0.9964 | 1.0 | 1.0 | 1596 | Paper table support is 1596; Figure 4d row sum and text are 1106. |
| ours:val_as_test_80_20 | frontotemporal_dementia | 0.6231884057971014 | 0.7427272727272727 | 0.6777270841974283 | 0.7427272727272727 | 0.6910569105691057 | 1100 |  |
| ours:val_as_test_80_20 | healthy_control | 0.7961095100864554 | 0.6910569105691057 | 0.7398727820555742 | 0.6910569105691057 | 0.7427272727272727 | 1599 |  |
| ours_label_swap | frontotemporal_dementia | 0.7856115107913669 | 0.6829268292682927 | 0.7306791569086651 | 0.6829268292682927 | 0.7290909090909091 | 1599 | Audit run with FTD/Healthy target labels intentionally swapped. |
| ours_label_swap | healthy_control | 0.612681436210848 | 0.7290909090909091 | 0.6658364466583645 | 0.7290909090909091 | 0.6829268292682927 | 1100 | Audit run with FTD/Healthy target labels intentionally swapped. |
