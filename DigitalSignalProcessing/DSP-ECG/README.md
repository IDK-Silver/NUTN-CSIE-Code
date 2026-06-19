# EEG深度學習失智症分類

原始 Paper 

[An explainable and efficient deep learning framework for EEG-based diagnosis of Alzheimer's disease and frontotemporal dementia](https://www.frontiersin.org/journals/medicine/articles/10.3389/fmed.2025.1590201/full)

表格復現對照請見 [docs/論文表格復現對照.md](docs/論文表格復現對照.md)。
正式復現執行順序請見 [docs/論文復現執行清單.md](docs/論文復現執行清單.md)。
完成條件請見 [docs/復現完成條件.md](docs/復現完成條件.md)。

目前狀態：Table 1-14 的復現 pipeline、configs、reporting、plotting 與 audit 已建立；正式復現 artifacts 尚未保證完成，需執行完整 pipeline 並通過 audit。


## 下載資料集

```bash
uv run download_raw_dataset \
  --dataset ds004504 \
  --tag 1.0.5 \
  --target-dir data/raw/ds004504
```

這個 command 會把 OpenNeuro `ds004504` 下載到 `data/raw/ds004504`。論文 Data availability statement 指向 `openneuro.ds004504.v1.0.5`，所以嚴格復現應優先使用 `--tag 1.0.5`。目前本機資料若是 `v1.0.8`，需在報告中明確註記版本差異。
目前只有 `ds004504` handler，其他 dataset 會回傳不支援。

## 產生 RBP H5

```bash
uv run process_raw_dataset \
  --dataset ds004504_rbp_paper \
  --raw-dir data/raw/ds004504 \
  --output data/processed_raw_dataset/ds004504_rbp_paper.h5 \
  --manifest data/processed_raw_dataset/ds004504_rbp_paper_manifest.json
```

目前資料處理 handler 是針對 `ds004504_rbp_paper` 寫的，不是通用 BIDS/EEG 轉換器。

輸出：

```text
data/processed_raw_dataset/ds004504_rbp_paper.h5
data/processed_raw_dataset/ds004504_rbp_paper_manifest.json
```

同一組資料產物路徑保存在 `cfgs/ds004504_rbp_paper/base.yaml`，但 dataset processing command 不會自動讀取 YAML。

若要重現論文 Section 4.5 / Table 12-13 的 standard RBP 對照實驗，需另外產生五頻帶 standard RBP artifact：

```bash
uv run process_raw_dataset \
  --dataset ds004504_standard_rbp_paper \
  --raw-dir data/raw/ds004504 \
  --output data/processed_raw_dataset/ds004504_standard_rbp_paper.h5 \
  --manifest data/processed_raw_dataset/ds004504_standard_rbp_paper_manifest.json
```

## 訓練

訓練 script 由 YAML 控制，只接收一個 config path。

三分類：

```bash
uv run python scripts/ds004504_rbp_paper/train.py cfgs/ds004504_rbp_paper/multiclass.yaml
```

AD + FTD vs Healthy：

```bash
uv run python scripts/ds004504_rbp_paper/train.py cfgs/ds004504_rbp_paper/ad_ftd_vs_healthy.yaml
```

AD vs Healthy：

```bash
uv run python scripts/ds004504_rbp_paper/train.py cfgs/ds004504_rbp_paper/ad_vs_healthy.yaml
```

FTD vs Healthy：

```bash
uv run python scripts/ds004504_rbp_paper/train.py cfgs/ds004504_rbp_paper/ftd_vs_healthy.yaml
```

Standard RBP 對照實驗只對應論文 Table 12-13 的兩個任務：

```bash
uv run python scripts/ds004504_rbp_paper/train.py cfgs/ds004504_rbp_paper/standard_rbp/multiclass.yaml
uv run python scripts/ds004504_rbp_paper/train.py cfgs/ds004504_rbp_paper/standard_rbp/ad_vs_healthy.yaml
```

論文 Table 10-11 的 5-fold validation 需要使用 k-fold runner：

```bash
uv run python scripts/ds004504_rbp_paper/train_kfold.py cfgs/ds004504_rbp_paper/kfold/multiclass.yaml
uv run python scripts/ds004504_rbp_paper/train_kfold.py cfgs/ds004504_rbp_paper/kfold/ad_vs_healthy.yaml
```

論文 Table 8-9 的 SMOTE balancing 正式以論文文字的 `80/10/10` split 重現：

```bash
uv run python scripts/ds004504_rbp_paper/train_smote.py cfgs/ds004504_rbp_paper/smote/multiclass.yaml
uv run python scripts/ds004504_rbp_paper/train_smote.py cfgs/ds004504_rbp_paper/smote/ad_vs_healthy.yaml
```

若要檢查論文表中的 reported support 是否比較接近 paper-inferred `80/20 validation-as-test` scenario，使用：

```bash
uv run python scripts/ds004504_rbp_paper/train_smote.py cfgs/ds004504_rbp_paper/val_as_test_80_20/smote/multiclass.yaml
uv run python scripts/ds004504_rbp_paper/train_smote.py cfgs/ds004504_rbp_paper/val_as_test_80_20/smote/ad_vs_healthy.yaml
```

訓練輸出會寫到 `data/runs/ds004504_rbp_paper/`。`cfgs/ds004504_rbp_paper/*.yaml` 使用 `extends` 合併設定，`src/` 不讀 YAML。`split.test_fraction` 會和 `evaluation.test_source` 一起決定 final reported metrics 的來源；例如 `test_source: split` 需要獨立 test split，`test_source: val` 則表示沒有獨立 test split、最後用 validation metrics 回報。

如果要檢查論文是否把 80/20 validation metrics 當作 reported test metrics，可以改跑 `val_as_test_80_20` scenario：

```bash
uv run python scripts/ds004504_rbp_paper/train.py cfgs/ds004504_rbp_paper/val_as_test_80_20/multiclass.yaml
uv run python scripts/ds004504_rbp_paper/train.py cfgs/ds004504_rbp_paper/val_as_test_80_20/ad_ftd_vs_healthy.yaml
uv run python scripts/ds004504_rbp_paper/train.py cfgs/ds004504_rbp_paper/val_as_test_80_20/ad_vs_healthy.yaml
uv run python scripts/ds004504_rbp_paper/train.py cfgs/ds004504_rbp_paper/val_as_test_80_20/ftd_vs_healthy.yaml
```

若要檢查 Table 3 與 Table 6 的 FTD/Healthy label-swap 疑點，可以跑：

```bash
uv run python scripts/ds004504_rbp_paper/train.py cfgs/ds004504_rbp_paper/label_swap_80_20/multiclass.yaml
uv run python scripts/ds004504_rbp_paper/train.py cfgs/ds004504_rbp_paper/label_swap_80_20/ftd_vs_healthy.yaml
```

若是在 paper-literal `80/10/10` scenario 下檢查同一個疑點，使用：

```bash
uv run python scripts/ds004504_rbp_paper/train.py cfgs/ds004504_rbp_paper/label_swap/multiclass.yaml
uv run python scripts/ds004504_rbp_paper/train.py cfgs/ds004504_rbp_paper/label_swap/ftd_vs_healthy.yaml
```

## 產生報表表格

完整 Table 1-14 pipeline 可以先列出 command plan：

```bash
uv run python scripts/ds004504_rbp_paper/run_reproduction_pipeline.py \
  --scenario paper_literal_80_10_10
```

這是照論文文字描述的 `80/10/10` split 跑完整 pipeline。若不指定 `--scenario`，目前預設也是 `paper_literal_80_10_10`。

若要檢查論文 reported support 比較接近 `80/20 validation-as-test` 的疑點，再列出 inferred scenario：

```bash
uv run python scripts/ds004504_rbp_paper/run_reproduction_pipeline.py \
  --scenario val_as_test_80_20
```

快速 smoke test 可用 fixture scenario 跑完整報表鏈，不會處理 EEG 或訓練模型：

```bash
uv run python scripts/ds004504_rbp_paper/run_reproduction_pipeline.py \
  --scenario fixture_smoke \
  --execute
```

`fixture_smoke` 的 audit 會把真實 H5/manifest 標成 `not_applicable`；它只檢查報表鏈，不代表論文復現數值。
`fixture_smoke` 每次執行都會重建 fixture runs；正式資料處理與訓練步驟才會依 marker 跳過。

確認後再執行：

```bash
uv run python scripts/ds004504_rbp_paper/run_reproduction_pipeline.py \
  --scenario paper_literal_80_10_10 \
  --execute
```

完整 `--execute` 會進行資料處理與多組模型訓練，耗時較長，且需要本機已具備資料集與適合的 PyTorch runtime。

或執行 inferred `80/20 validation-as-test` scenario：

```bash
uv run python scripts/ds004504_rbp_paper/run_reproduction_pipeline.py \
  --scenario val_as_test_80_20 \
  --execute
```

若尚未下載 OpenNeuro 資料，加入 `--include-download`。
完整 pipeline 的最後一步會以 `--fail-on-missing` 產生 artifact audit；若正式 scenario 仍有 missing/stale/not_applicable 項目，pipeline 會失敗。
Audit 不只檢查檔案存在，也會檢查 `table_summary.csv` 是否涵蓋 Table 1-14、`paper_vs_ours.csv` 是否涵蓋可比較的 Table 3-13，以及 `issue_summary.csv` 是否保留既定論文疑點。
`run_reproduction_pipeline.py` 預設會跳過已存在 marker 的資料處理與訓練步驟，但 report stage 會每次重跑，確保 CSV、Markdown、SVG 與 audit 反映最新 runs。
若 training output dir 已存在但 marker 檔不存在，pipeline 會標成 `partial-output` 並在 `--execute` 時停止；這通常代表前一次 run 中斷，需先移除該 output dir 或補完該 run。
每次列出或執行 pipeline 都會寫出 `pipeline_execution_manifest.json`，記錄 command、marker 與 plan-time status。

先把各個 run output 整理成 normalized CSV：

```bash
uv run python scripts/ds004504_rbp_paper/make_report_csv.py \
  --scenario paper_literal_80_10_10
```

再把 CSV render 成接近論文 Table 1-14 形狀的 Markdown：

```bash
uv run python scripts/ds004504_rbp_paper/render_report_tables.py \
  --scenario paper_literal_80_10_10
```

也可以從 CSV 與 prediction CSV 產生 SVG 圖：

```bash
uv run python scripts/ds004504_rbp_paper/plot_report.py \
  --scenario paper_literal_80_10_10
```

Markdown 會輸出到：

```text
data/reports/ds004504_rbp_paper/paper_literal_80_10_10/paper_tables/
```

其中 `paper_tables/report.md` 是 protocol manifest 與 Table 1-14 的合併版人讀報告。

SVG 圖會輸出到：

```text
data/reports/ds004504_rbp_paper/paper_literal_80_10_10/plots/
```

其中會包含 training/validation 判讀用圖與資料，例如：

```text
data/reports/ds004504_rbp_paper/paper_literal_80_10_10/plots/confusion_matrices/*.svg
data/reports/ds004504_rbp_paper/paper_literal_80_10_10/plots/history/*_accuracy.svg
data/reports/ds004504_rbp_paper/paper_literal_80_10_10/plots/roc/roc_auc.csv
```

最後可稽核 Table 1-14 所需 artifacts 是否齊全：

```bash
uv run python scripts/ds004504_rbp_paper/audit_reproduction_artifacts.py \
  --scenario paper_literal_80_10_10
```

稽核輸出：

```text
data/reports/ds004504_rbp_paper/paper_literal_80_10_10/audit/audit.md
data/reports/ds004504_rbp_paper/paper_literal_80_10_10/audit/audit.json
```

若同時產生了 `paper_literal_80_10_10` 和 `val_as_test_80_20` 報告，可比較兩套 scenario。這個步驟不在單一 scenario pipeline 內，需在兩套 scenario 都完成後另外執行：

```bash
uv run python scripts/ds004504_rbp_paper/compare_report_scenarios.py
```

輸出包含：

```text
data/reports/ds004504_rbp_paper/scenario_comparison/scenario_table_summary.csv
data/reports/ds004504_rbp_paper/scenario_comparison/scenario_coverage.csv
data/reports/ds004504_rbp_paper/scenario_comparison/scenario_protocol_summary.csv
data/reports/ds004504_rbp_paper/scenario_comparison/scenario_paper_vs_ours.csv
data/reports/ds004504_rbp_paper/scenario_comparison/scenario_issue_summary.csv
data/reports/ds004504_rbp_paper/scenario_comparison/scenario_comparison.md
```

`scenario_coverage.csv` 檢查每個 scenario 是否涵蓋 Table 1-14、paper-vs-ours Table 3-13、protocol components 與既定 issue ids。
其中 `scenario_paper_vs_ours.csv` 是逐 Table、逐 class、逐 metric 的 paper value vs ours value 比較。
`scenario_issue_summary.csv` 保留 split/support/label-swap/SMOTE 等論文疑點對各 scenario 的影響說明。

若要把缺少任一 scenario report 視為失敗：

```bash
uv run python scripts/ds004504_rbp_paper/compare_report_scenarios.py \
  --fail-on-missing
```

若要用單一入口列出完整 paper-literal 復現、80/20 support-audit scenario 與 scenario comparison gate：

```bash
uv run python scripts/ds004504_rbp_paper/run_full_comparison.py
```

這只會列出 command plan，輸出會標示 `Mode: PLAN ONLY`，每行 command 會以前綴 `would run:` 顯示，不會啟動訓練。

確認 command plan 後執行：

```bash
uv run python scripts/ds004504_rbp_paper/run_full_comparison.py \
  --execute
```

若加上 `--skip-val-as-test`，只會執行正式 paper-literal 復現，不會產生兩個 scenario 的總比較。
因此 `--skip-val-as-test --execute` 成功只代表正式 `80/10/10` 復現完成；若目標包含「再比較兩者」，不要加 `--skip-val-as-test`。
Wrapper 會寫出完整比較層級的 command/execution manifest，並在 full mode 的最後自動執行 `verify_full_reproduction.py`：

```text
data/reports/ds004504_rbp_paper/full_comparison_execution_manifest.json
```

Manifest 會記錄 final verifier 使用的 coverage CSV path，預設是 `data/reports/ds004504_rbp_paper/scenario_comparison/scenario_coverage.csv`。
只有 `workflow_status` 與 `overall_status` 都是 `completed`，且非 verifier 的 workflow steps 全部為 `status: completed` 時，這份 manifest 才能作為 wrapper 層級成功證據。若 manifest 中已記錄 verifier step，該 step 也必須是 `completed`。
若 `workflow_status` 是 `completed` 但 `overall_status` 是 `failed`，代表前置 workflow 完成但 final verifier 沒通過。
完整比較還需要確認 `data/reports/ds004504_rbp_paper/scenario_comparison/scenario_coverage.csv` 沒有 `missing` 或 `stale`。
可用以下 command 做最後完成證據檢查：

```bash
uv run python scripts/ds004504_rbp_paper/verify_full_reproduction.py
```

手動執行 verifier 通常只需要在你要重新檢查既有 artifacts 時使用；手動模式預設只接受 `overall_status: completed`。wrapper 內部會用 `--allow-verifying` 處理 verifier 執行中的中間 manifest。

輸出：

```text
data/reports/ds004504_rbp_paper/scenario_comparison/
```

## 程式結構

```text
cfgs/ds004504_rbp_paper/      # explicit paper reproduction configs for scripts
src/ecg/commands/             # console script entrypoints
src/ecg/data/raw_ds004504.py  # raw ds004504 download handler
src/ecg/data/ds004504_rbp_paper.py  # ds004504 paper-style modified/standard RBP processing handler
src/ecg/datasets/             # PyTorch Dataset definitions
src/ecg/models/               # model definitions
src/ecg/training/             # training loops, metrics, and paper reproduction helpers
scripts/ds004504_rbp_paper/   # paper reproduction training script
```
