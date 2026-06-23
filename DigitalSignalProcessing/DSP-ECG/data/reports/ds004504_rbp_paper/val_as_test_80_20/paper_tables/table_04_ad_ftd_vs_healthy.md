# Table 4. Classification metrics for Alzheimer + frontotemporal disease and healthy classes.

| Source | Class | Precision | Recall | F1 score | Sensitivity | Specificity | Support | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| paper | alzheimer_or_frontotemporal_dementia | 0.9977 | 0.9993 | 0.9985 | 1.0 | 1.0 | 2983 | Binary sensitivity/specificity are carried from the paper's single reported row. |
| paper | healthy_control | 0.9987 | 0.9956 | 0.9972 | 1.0 | 1.0 | 1596 | Binary sensitivity/specificity are carried from the paper's single reported row. |
| ours:val_as_test_80_20 | alzheimer_or_frontotemporal_dementia | 0.7503463563313937 | 0.8931398416886543 | 0.8155398283391054 | 0.8931398416886543 | 0.4365228267667292 | 3032 |  |
| ours:val_as_test_80_20 | healthy_control | 0.6829745596868885 | 0.4365228267667292 | 0.5326211369706219 | 0.4365228267667292 | 0.8931398416886543 | 1599 |  |
