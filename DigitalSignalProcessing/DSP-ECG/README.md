# EEG深度學習失智症分類

原始 Paper 

[An explainable and efficient deep learning framework for EEG-based diagnosis of Alzheimer's disease and frontotemporal dementia](https://www.frontiersin.org/journals/medicine/articles/10.3389/fmed.2025.1590201/full)


## 下載資料集

```bash
uv run download_raw_dataset
```

這個 command 會把 OpenNeuro `ds004504` 下載到 `data/raw/ds004504`。

## 產生 RBP H5

```bash
uv run process_raw_dataset
```

輸出：

```text
data/processed_raw_dataset/rbp_epochs.h5
data/processed_raw_dataset/rbp_epochs_manifest.json
```
