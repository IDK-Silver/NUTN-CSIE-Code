# Table 8. Classification metrics with SMOTE data balancing.

| Source | Class | Precision | Recall | F1 score | Sensitivity | Specificity | Support | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| paper | alzheimer | 0.63 | 0.71 | 0.67 | 0.71 | 0.79 | 1876 |  |
| paper | frontotemporal_dementia | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1876 | Table 8 supports are balanced to 1876 for every class. |
| paper | healthy_control | 0.67 | 0.58 | 0.62 | 0.58 | 0.86 | 1876 | The paper text reports 77.45% accuracy, but these equal supports and recalls imply 76.33%. |
| ours:val_as_test_80_20 | alzheimer | 0.5125170687300865 | 0.582815734989648 | 0.5454105110196174 | 0.582815734989648 | 0.7228260869565217 | 1932 |  |
| ours:val_as_test_80_20 | frontotemporal_dementia | 0.5197923426346528 | 0.41459627329192544 | 0.4612726749208177 | 0.41459627329192544 | 0.8084886128364389 | 1932 |  |
| ours:val_as_test_80_20 | healthy_control | 0.60932944606414 | 0.6490683229813664 | 0.6285714285714286 | 0.6490683229813664 | 0.7919254658385093 | 1932 |  |
