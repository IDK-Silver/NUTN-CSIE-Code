# Table 6. Classification metrics for frontotemporal disease and healthy classes.

| Source | Class | Precision | Recall | F1 score | Sensitivity | Specificity | Support | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| paper | frontotemporal_dementia | 0.9994 | 0.9956 | 0.9975 | 1.0 | 1.0 | 1597 | Binary sensitivity/specificity are carried from the paper's single reported row. |
| paper | healthy_control | 0.9937 | 0.9991 | 0.9964 | 1.0 | 1.0 | 1596 | Paper table support is 1596; Figure 4d row sum and text are 1106. |
| ours:paper_literal_80_10_10 | frontotemporal_dementia | 0.6994106090373281 | 0.6460980036297641 | 0.6716981132075471 | 0.6460980036297641 | 0.80875 | 551 |  |
| ours:paper_literal_80_10_10 | healthy_control | 0.7684085510688836 | 0.80875 | 0.7880633373934226 | 0.80875 | 0.6460980036297641 | 800 |  |
| ours_label_swap | frontotemporal_dementia | 0.7903682719546742 | 0.6975 | 0.7410358565737052 | 0.6975 | 0.7313974591651543 | 800 | Audit run with FTD/Healthy target labels intentionally swapped. |
| ours_label_swap | healthy_control | 0.6248062015503876 | 0.7313974591651543 | 0.6739130434782609 | 0.7313974591651543 | 0.6975 | 551 | Audit run with FTD/Healthy target labels intentionally swapped. |
