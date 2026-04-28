# EEG深度學習失智症分類

原始 Paper 

[An explainable and efficient deep learning framework for EEG-based diagnosis of Alzheimer's disease and frontotemporal dementia](https://www.frontiersin.org/journals/medicine/articles/10.3389/fmed.2025.1590201/full)


## 下載資料集

```bash
uv run download_raw_dataset \
  --dataset ds004504 \
  --tag 1.0.8 \
  --target-dir data/raw/ds004504
```

這個 command 會把 OpenNeuro `ds004504` 下載到 `data/raw/ds004504`。
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

同一組參數保存在 `cfgs/recipes/ds004504_rbp_paper.yaml`，但 command 不會自動讀取 recipe。

## 程式結構

```text
cfgs/recipes/                 # explicit command recipes, not implicit defaults
src/ecg/commands/             # console script entrypoints
src/ecg/data/raw_ds004504.py  # raw ds004504 download handler
src/ecg/data/ds004504_rbp_paper.py  # ds004504 paper-style RBP processing handler
src/ecg/datasets/             # PyTorch Dataset definitions
src/ecg/models/               # model definitions
```
