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
