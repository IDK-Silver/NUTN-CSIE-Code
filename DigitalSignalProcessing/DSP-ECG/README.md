# EEG深度學習失智症分類

原始 Paper 

[An explainable and efficient deep learning framework for EEG-based diagnosis of Alzheimer's disease and frontotemporal dementia](https://www.frontiersin.org/journals/medicine/articles/10.3389/fmed.2025.1590201/full)


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

訓練輸出會寫到 `data/runs/ds004504_rbp_paper/`。`cfgs/ds004504_rbp_paper/*.yaml` 使用 `extends` 合併設定，`src/` 不讀 YAML。`split.test_fraction` 會和 `evaluation.test_source` 一起決定 final reported metrics 的來源；例如 `test_source: split` 需要獨立 test split，`test_source: val` 則表示沒有獨立 test split、最後用 validation metrics 回報。

如果要檢查論文是否把 80/20 validation metrics 當作 reported test metrics，可以改跑 `val_as_test_80_20` scenario：

```bash
uv run python scripts/ds004504_rbp_paper/train.py cfgs/ds004504_rbp_paper/val_as_test_80_20/multiclass.yaml
uv run python scripts/ds004504_rbp_paper/train.py cfgs/ds004504_rbp_paper/val_as_test_80_20/ad_ftd_vs_healthy.yaml
uv run python scripts/ds004504_rbp_paper/train.py cfgs/ds004504_rbp_paper/val_as_test_80_20/ad_vs_healthy.yaml
uv run python scripts/ds004504_rbp_paper/train.py cfgs/ds004504_rbp_paper/val_as_test_80_20/ftd_vs_healthy.yaml
```

## 程式結構

```text
cfgs/ds004504_rbp_paper/      # explicit paper reproduction configs for scripts
src/ecg/commands/             # console script entrypoints
src/ecg/data/raw_ds004504.py  # raw ds004504 download handler
src/ecg/data/ds004504_rbp_paper.py  # ds004504 paper-style RBP processing handler
src/ecg/datasets/             # PyTorch Dataset definitions
src/ecg/models/               # model definitions
src/ecg/training/             # training loops, metrics, and paper reproduction helpers
scripts/ds004504_rbp_paper/   # paper reproduction training script
```
