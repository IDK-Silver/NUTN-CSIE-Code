# Table 8. Classification metrics with SMOTE data balancing.

| Source | Class | Precision | Recall | F1 score | Sensitivity | Specificity | Support | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| paper | alzheimer | 0.63 | 0.71 | 0.67 | 0.71 | 0.79 | 1876 |  |
| paper | frontotemporal_dementia | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1876 | Table 8 supports are balanced to 1876 for every class. |
| paper | healthy_control | 0.67 | 0.58 | 0.62 | 0.58 | 0.86 | 1876 | The paper text reports 77.45% accuracy, but these equal supports and recalls imply 76.33%. |
| ours:paper_literal_80_10_10 | alzheimer | 0.5439560439560439 | 0.4095139607032058 | 0.4672566371681416 | 0.4095139607032058 | 0.828335056876939 | 967 |  |
| ours:paper_literal_80_10_10 | frontotemporal_dementia | 0.4912621359223301 | 0.5232678386763185 | 0.5067601402103155 | 0.5232678386763185 | 0.7290589451913133 | 967 |  |
| ours:paper_literal_80_10_10 | healthy_control | 0.5853018372703412 | 0.6918304033092038 | 0.6341232227488152 | 0.6918304033092038 | 0.7549120992761117 | 967 |  |
