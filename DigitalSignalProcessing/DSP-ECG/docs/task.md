# Task

## 目標

建立一套可以重現 paper 圖表的資料與繪圖流程。

## 完成狀態

目前狀態是「pipeline、configs、reporting、plotting、audit 已建立」，不是「Table 1-14 復現 artifacts 已完成」。

正式完成條件以 [復現完成條件](復現完成條件.md) 為準。必須完成正式 pipeline 執行，並讓以下 command 成功退出：

```bash
uv run python scripts/ds004504_rbp_paper/audit_reproduction_artifacts.py \
  --scenario paper_literal_80_10_10 \
  --fail-on-missing
```

`fixture_smoke` 只能驗證報表鏈，不能當作論文復現數值。

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

狀態：已新增 `scripts/ds004504_rbp_paper/make_report_csv.py`。

目前此 script 先把論文 Table 1-14 的 paper-side 數字固定成 CSV，並讀取既有 training outputs 產生 ours-side 的 run summary、classification metrics、confusion matrices、history 與 hyperparameters。SMOTE、K-fold、standard RBP 的 ours-side pipeline 已接進正式 reproduction pipeline；正式數值仍需實際執行 pipeline 後才會產生。

Table 2 會另外從專案的 TCN-LSTM 實作直接計算 ours-side total/trainable/non-trainable parameter counts，用來和論文參數量比較。

使用方式：

```bash
uv run python scripts/ds004504_rbp_paper/make_report_csv.py \
  --scenario paper_literal_80_10_10
```

80/20 validation-as-test scenario：

```bash
uv run python scripts/ds004504_rbp_paper/make_report_csv.py \
  --scenario val_as_test_80_20
```

已新增 script：

- `scripts/ds004504_rbp_paper/make_report_csv.py`

輸出：

- `data/reports/ds004504_rbp_paper/<scenario>/tables/run_summary.csv`
- `data/reports/ds004504_rbp_paper/<scenario>/tables/paper_vs_ours.csv`
- `data/reports/ds004504_rbp_paper/<scenario>/tables/classification_metrics.csv`
- `data/reports/ds004504_rbp_paper/<scenario>/tables/confusion_matrices.csv`
- `data/reports/ds004504_rbp_paper/<scenario>/tables/support_comparison.csv`
- `data/reports/ds004504_rbp_paper/<scenario>/tables/history.csv`
- `data/reports/ds004504_rbp_paper/<scenario>/tables/model_parameters.csv`
- `data/reports/ds004504_rbp_paper/<scenario>/tables/paper_model_architecture.csv`
- `data/reports/ds004504_rbp_paper/<scenario>/tables/accuracy.csv`
- `data/reports/ds004504_rbp_paper/<scenario>/tables/kfold_accuracy.csv`
- `data/reports/ds004504_rbp_paper/<scenario>/tables/protocol_manifest.csv`
- `data/reports/ds004504_rbp_paper/<scenario>/tables/literature_comparison.csv`

### 3. Table Renderer 與 Plot Script

狀態：已新增 Markdown table renderer 與 stdlib SVG plot script。

已新增 script：

- `scripts/ds004504_rbp_paper/audit_reproduction_artifacts.py`
- `scripts/ds004504_rbp_paper/run_full_comparison.py`
- `scripts/ds004504_rbp_paper/run_reproduction_pipeline.py`
- `scripts/ds004504_rbp_paper/render_report_tables.py`
- `scripts/ds004504_rbp_paper/plot_report.py`
- `scripts/ds004504_rbp_paper/verify_full_reproduction.py`

`run_full_comparison.py` 是完整入口；預設只列出 plan，加入 `--execute` 才會依序執行 paper-literal pipeline、80/20 support-audit pipeline 與 scenario comparison gate。

完整 Table 1-14 command plan：

```bash
uv run python scripts/ds004504_rbp_paper/run_reproduction_pipeline.py \
  --scenario paper_literal_80_10_10
```

`paper_literal_80_10_10` 是正式照論文文字方法的主線。`val_as_test_80_20` 只用於檢查論文 reported support 疑點。

快速 smoke test：

```bash
uv run python scripts/ds004504_rbp_paper/run_reproduction_pipeline.py \
  --scenario fixture_smoke \
  --execute
```

執行完整 pipeline：

```bash
uv run python scripts/ds004504_rbp_paper/run_reproduction_pipeline.py \
  --scenario paper_literal_80_10_10 \
  --execute
```

完整 pipeline 的 report stage 會依序產生 CSV、Markdown tables、SVG plots 與 artifact audit。

稽核 Table 1-14 所需 artifacts：

```bash
uv run python scripts/ds004504_rbp_paper/audit_reproduction_artifacts.py \
  --scenario paper_literal_80_10_10 \
  --fail-on-missing
```

這個 script 只讀 `make_report_csv.py` 產生的 CSV，不直接讀 training JSON，會把 Table 1-14 render 成接近論文形狀的 Markdown tables。

使用方式：

```bash
uv run python scripts/ds004504_rbp_paper/render_report_tables.py \
  --scenario paper_literal_80_10_10
```

輸出：

- `data/reports/ds004504_rbp_paper/<scenario>/paper_tables/index.md`
- `data/reports/ds004504_rbp_paper/<scenario>/paper_tables/report.md`
- `data/reports/ds004504_rbp_paper/<scenario>/paper_tables/table_01_model_architecture.md`
- `data/reports/ds004504_rbp_paper/<scenario>/paper_tables/table_02_model_parameters.md`
- `data/reports/ds004504_rbp_paper/<scenario>/paper_tables/table_03_multiclass.md`
- `data/reports/ds004504_rbp_paper/<scenario>/paper_tables/table_04_ad_ftd_vs_healthy.md`
- `data/reports/ds004504_rbp_paper/<scenario>/paper_tables/table_05_ad_vs_healthy.md`
- `data/reports/ds004504_rbp_paper/<scenario>/paper_tables/table_06_ftd_vs_healthy.md`
- `data/reports/ds004504_rbp_paper/<scenario>/paper_tables/table_07_accuracy.md`
- `data/reports/ds004504_rbp_paper/<scenario>/paper_tables/table_08_smote_multiclass.md`
- `data/reports/ds004504_rbp_paper/<scenario>/paper_tables/table_09_smote_ad_vs_healthy.md`
- `data/reports/ds004504_rbp_paper/<scenario>/paper_tables/table_10_kfold_multiclass.md`
- `data/reports/ds004504_rbp_paper/<scenario>/paper_tables/table_11_kfold_ad_vs_healthy.md`
- `data/reports/ds004504_rbp_paper/<scenario>/paper_tables/table_12_standard_rbp_multiclass.md`
- `data/reports/ds004504_rbp_paper/<scenario>/paper_tables/table_13_standard_rbp_ad_vs_healthy.md`
- `data/reports/ds004504_rbp_paper/<scenario>/paper_tables/table_14_literature_comparison.md`
- `data/reports/ds004504_rbp_paper/<scenario>/paper_tables/protocol_manifest.md`
- `data/reports/ds004504_rbp_paper/<scenario>/paper_tables/confusion_matrices.md`

已新增圖像 script：

- `scripts/ds004504_rbp_paper/plot_report.py`

使用方式：

```bash
uv run python scripts/ds004504_rbp_paper/plot_report.py \
  --scenario paper_literal_80_10_10
```

目標輸出：

- accuracy comparison
- support comparison
- confusion matrices
- train/validation curves
- per-class precision/recall/F1
- ROC/AUC

### 4. K-fold

狀態：已新增 k-fold runner 與 configs，結果 artifact 尚未實際產生。

新增 script：

- `scripts/ds004504_rbp_paper/train_kfold.py`

Table 10-11 commands：

```bash
uv run python scripts/ds004504_rbp_paper/train_kfold.py \
  cfgs/ds004504_rbp_paper/kfold/multiclass.yaml

uv run python scripts/ds004504_rbp_paper/train_kfold.py \
  cfgs/ds004504_rbp_paper/kfold/ad_vs_healthy.yaml
```

`make_report_csv.py` 已會讀取：

- `data/runs/ds004504_rbp_paper/kfold/multiclass/fold_summary.csv`
- `data/runs/ds004504_rbp_paper/kfold/ad_vs_healthy/fold_summary.csv`

已補：

- fold split indices
- fold train/test metrics
- fold history
- fold predictions
- fold summary CSV

### 5. SMOTE

狀態：已新增 paper-literal `80/10/10` 與 `val_as_test_80_20` 的 SMOTE training flow，結果 artifact 尚未實際產生。

新增 script：

- `scripts/ds004504_rbp_paper/train_smote.py`

Table 8-9 paper-literal `80/10/10` commands：

```bash
uv run python scripts/ds004504_rbp_paper/train_smote.py \
  cfgs/ds004504_rbp_paper/smote/multiclass.yaml

uv run python scripts/ds004504_rbp_paper/train_smote.py \
  cfgs/ds004504_rbp_paper/smote/ad_vs_healthy.yaml
```

這是正式照論文文字 `80/10/10` split 產生 Table 8-9 的 SMOTE tables。由於論文未說明 SMOTE placement，此專案在 protocol manifest 中明確記錄 simple SMOTE 的 partition placement。

若要檢查 paper-inferred `80/20 validation-as-test` 疑點，使用：

```bash
uv run python scripts/ds004504_rbp_paper/train_smote.py \
  cfgs/ds004504_rbp_paper/val_as_test_80_20/smote/multiclass.yaml

uv run python scripts/ds004504_rbp_paper/train_smote.py \
  cfgs/ds004504_rbp_paper/val_as_test_80_20/smote/ad_vs_healthy.yaml
```

已補：

- SMOTE 後 class counts
- balanced training set 記錄
- SMOTE metrics
- SMOTE predictions

### 5.1 Label-swap audit

狀態：已新增 paper-literal `80/10/10` 與 `val_as_test_80_20` 的 FTD/Healthy label-swap audit configs。

這不是 corrected protocol，而是用來檢查論文 Table 3 與 Table 6 的 FTD/Healthy support 疑點。

Paper-literal `80/10/10` commands：

```bash
uv run python scripts/ds004504_rbp_paper/train.py \
  cfgs/ds004504_rbp_paper/label_swap/multiclass.yaml

uv run python scripts/ds004504_rbp_paper/train.py \
  cfgs/ds004504_rbp_paper/label_swap/ftd_vs_healthy.yaml
```

Paper-inferred `80/20 validation-as-test` commands：

```bash
uv run python scripts/ds004504_rbp_paper/train.py \
  cfgs/ds004504_rbp_paper/label_swap_80_20/multiclass.yaml

uv run python scripts/ds004504_rbp_paper/train.py \
  cfgs/ds004504_rbp_paper/label_swap_80_20/ftd_vs_healthy.yaml
```

`make_report_csv.py` 會把這兩個 runs 以 `ours_label_swap` source 寫入：

- `classification_metrics.csv`
- `support_comparison.csv`
- Table 3 Markdown
- Table 6 Markdown

### 6. Standard RBP vs Modified RBP

狀態：資料處理與 training config 入口已補，結果 artifact 尚未實際產生。

新增 processed dataset id：

- `ds004504_standard_rbp_paper`

Standard RBP artifact command：

```bash
uv run process_raw_dataset \
  --dataset ds004504_standard_rbp_paper \
  --raw-dir data/raw/ds004504 \
  --output data/processed_raw_dataset/ds004504_standard_rbp_paper.h5 \
  --manifest data/processed_raw_dataset/ds004504_standard_rbp_paper_manifest.json
```

Table 12-13 training commands：

```bash
uv run python scripts/ds004504_rbp_paper/train.py \
  cfgs/ds004504_rbp_paper/standard_rbp/multiclass.yaml

uv run python scripts/ds004504_rbp_paper/train.py \
  cfgs/ds004504_rbp_paper/standard_rbp/ad_vs_healthy.yaml
```

`make_report_csv.py` 已會讀取：

- `data/runs/ds004504_rbp_paper/standard_rbp/multiclass`
- `data/runs/ds004504_rbp_paper/standard_rbp/ad_vs_healthy`

目前已有 `ds004504_rbp_paper` modified RBP 與 `ds004504_standard_rbp_paper` standard RBP 處理入口。

仍需要執行正式 pipeline 產生：

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
