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
