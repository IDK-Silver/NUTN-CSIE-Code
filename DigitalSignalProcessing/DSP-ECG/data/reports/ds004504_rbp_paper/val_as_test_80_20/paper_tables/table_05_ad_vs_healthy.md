# Table 5. Classification metrics for Alzheimer's disease and healthy classes.

| Source | Class | Precision | Recall | F1 score | Sensitivity | Specificity | Support | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| paper | alzheimer | 0.9963 | 0.9989 | 0.9976 | 1.0 | 1.0 | 1876 | Binary sensitivity/specificity are carried from the paper's single reported row. |
| paper | healthy_control | 0.9987 | 0.9956 | 0.9972 | 1.0 | 1.0 | 1596 | Paper table support is 1596; Figure 4c row sum is 1597. |
| ours:val_as_test_80_20 | alzheimer | 0.7346326836581709 | 0.7608695652173914 | 0.7475209763539283 | 0.7608695652173914 | 0.6679174484052532 | 1932 |  |
| ours:val_as_test_80_20 | healthy_control | 0.6980392156862745 | 0.6679174484052532 | 0.6826462128475551 | 0.6679174484052532 | 0.7608695652173914 | 1599 |  |
