# Table 3. Classification metrics for Alzheimer, frontotemporal, and healthy classes.

| Source | Class | Precision | Recall | F1 score | Sensitivity | Specificity | Support | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| paper | alzheimer | 0.7 | 0.9 | 0.79 | 0.9 | 0.74 | 1876 |  |
| paper | frontotemporal_dementia | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1597 | Support is inconsistent with ds004504 FTD duration and matches healthy duration. |
| paper | healthy_control | 0.68 | 0.35 | 0.47 | 0.35 | 0.95 | 1106 | Support is inconsistent with ds004504 healthy duration and matches FTD duration. |
| ours:paper_literal_80_10_10 | alzheimer | 0.5398729710656316 | 0.7911065149948294 | 0.6417785234899329 | 0.7911065149948294 | 0.5173945225758697 | 967 |  |
| ours:paper_literal_80_10_10 | frontotemporal_dementia | 0.66 | 0.17967332123411978 | 0.28245363766048504 | 0.17967332123411978 | 0.9711375212224108 | 551 |  |
| ours:paper_literal_80_10_10 | healthy_control | 0.6151797603195739 | 0.5775 | 0.5957446808510637 | 0.5775 | 0.8096179183135704 | 800 |  |
| ours_label_swap | alzheimer | 0.5253595760787282 | 0.717683557394002 | 0.6066433566433566 | 0.717683557394002 | 0.535899333826795 | 967 | Audit run with FTD/Healthy target labels intentionally swapped. |
| ours_label_swap | frontotemporal_dementia | 0.5770528683914511 | 0.64125 | 0.6074600355239786 | 0.64125 | 0.7523056653491436 | 800 | Audit run with FTD/Healthy target labels intentionally swapped. |
| ours_label_swap | healthy_control | 0.6018518518518519 | 0.11796733212341198 | 0.19726858877086492 | 0.11796733212341198 | 0.9756649688737974 | 551 | Audit run with FTD/Healthy target labels intentionally swapped. |
