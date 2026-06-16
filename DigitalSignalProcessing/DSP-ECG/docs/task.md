# Task

## 目標

建立一套可以重現 paper 圖表的資料與繪圖流程。

## 目前已有資料

- `data/processed_raw_dataset/ds004504_rbp_paper.h5`
- `data/processed_raw_dataset/ds004504_rbp_paper_manifest.json`
- `data/runs/ds004504_rbp_paper/*/run.json`
- `data/runs/ds004504_rbp_paper/*/history.json`
- `data/runs/ds004504_rbp_paper/*/test_metrics.json`
- `data/runs/ds004504_rbp_paper/*/test_predictions.csv`
- `data/runs/ds004504_rbp_paper/val_as_test_80_20/*`

## 缺口

### 1. Prediction Score Export

狀態：已實作於 training script；舊的 runs 需要重跑才會產生新欄位。

新 `test_predictions.csv` 會輸出：

- `source_index`
- `subject_id`
- `epoch_start_sec`
- `y_true`
- `y_pred`
- `logit_*`
- `prob_*`

### 2. Report CSV

需要新增 script，把 training outputs 整理成 CSV。

目標 script：

- `scripts/ds004504_rbp_paper/make_report_csv.py`

目標輸出：

- `data/reports/ds004504_rbp_paper/<scenario>/tables/run_summary.csv`
- `data/reports/ds004504_rbp_paper/<scenario>/tables/paper_vs_ours.csv`
- `data/reports/ds004504_rbp_paper/<scenario>/tables/classification_metrics.csv`
- `data/reports/ds004504_rbp_paper/<scenario>/tables/confusion_matrices.csv`
- `data/reports/ds004504_rbp_paper/<scenario>/tables/support_comparison.csv`
- `data/reports/ds004504_rbp_paper/<scenario>/tables/history.csv`
- `data/reports/ds004504_rbp_paper/<scenario>/tables/model_parameters.csv`

### 3. Plot Script

需要新增 script，只讀 CSV，不直接讀 training JSON。

目標 script：

- `scripts/ds004504_rbp_paper/plot_report.py`

目標輸出：

- accuracy comparison
- support comparison
- confusion matrices
- train/validation curves
- per-class precision/recall/F1
- ROC/AUC

### 4. K-fold

論文有 k-fold tables，目前缺 k-fold runner。

需要補：

- fold split indices
- fold train/test metrics
- fold history
- fold predictions
- fold summary CSV

### 5. SMOTE

論文有 SMOTE 相關表格，目前缺 SMOTE training flow。

需要補：

- SMOTE 後 class counts
- balanced training set 記錄
- SMOTE metrics
- SMOTE predictions

### 6. Standard RBP vs Modified RBP

目前只有 `ds004504_rbp_paper` modified RBP。

需要補：

- standard RBP processed dataset
- standard RBP training configs
- standard RBP metrics
- modified vs standard comparison CSV

### 7. SHAP

論文 Figure 6-14 需要 SHAP artifacts，目前完全缺。

需要補：

- background samples
- explained samples
- feature values
- per-class SHAP values
- global mean absolute SHAP values
- SHAP summary plot data
- SHAP heatmap data

### 8. ICA Components

論文 Figure 2 需要 ICA component artifacts。

目前 H5 只有 RBP features，沒有 ICA components、ICLabel、component maps。

需要補：

- ICA components
- ICLabel output
- removed component metadata
- component visualization data

## 優先順序

1. 補 prediction score export。
2. 做 `make_report_csv.py`。
3. 做 `plot_report.py`。
4. 補 k-fold runner。
5. 補 SMOTE flow。
6. 補 standard RBP dataset。
7. 補 SHAP pipeline。
8. 評估是否要補 ICA component reproduction。

## 套件需求

可能需要：

- `matplotlib`
- `pandas`
- `scikit-learn`
- `imbalanced-learn`
- `shap`
