# Diabetes Binary Classifier

糖尿病二元分類模型，用於預測是否患有糖尿病（含糖尿病前期）。

## 環境需求

- Python 3.10+
- CUDA 11.8+（GPU 訓練需要）
- [uv](https://docs.astral.sh/uv/) 套件管理工具

    uv 安裝方式請參考[官方文件](https://docs.astral.sh/uv/getting-started/installation/)。

## 快速開始

取得原始碼：

```bash
git clone https://github.com/IDK-Silver/NUTN-CSIE-Code.git
cd NUTN-CSIE-Code/DataMining/final-project
```

安裝相依套件：

```bash
uv sync
```

下載資料集：

```bash
uv run python scripts/setup/download_dataset.py
```

資料集會儲存至 `blobs/raw/`。

## 使用方式

訓練完成後，模型與 `info.json` 儲存於 `blobs/models/<model>/`，提交檔案輸出至 `blobs/submit/<model>/`。

### AutoGluon

AutoML 框架，自動嘗試多種模型並選擇最佳組合。訓練時間較長（數十小時或天）。

```bash
uv run python scripts/train/train_autogluon.py
uv run python scripts/predict/predict_autogluon.py
```

### LightGBM

梯度提升決策樹，訓練速度快。

```bash
uv run python scripts/train/train_lightgbm.py
uv run python scripts/predict/predict_lightgbm.py
```

### CatBoost

支援類別特徵的梯度提升，不需手動編碼類別變數。

```bash
uv run python scripts/train/train_catboost.py
uv run python scripts/predict/predict_catboost.py
```

### TabularNet

PyTorch 實作的神經網路，使用 K-Fold 交叉驗證。

```bash
uv run python scripts/train/train_tabular_net.py
uv run python scripts/predict/predict_tabular_net.py
```

### TabularNet + Optuna

TabularNet 搭配 Optuna 超參數搜索。

```bash
uv run python scripts/train/train_tabular_net_optuna.py
uv run python scripts/predict/predict_tabular_net_optuna.py
```

### TabularNet Multi-Seed

TabularNet 使用多組隨機種子訓練後平均，提升穩定性。

```bash
uv run python scripts/train/train_tabular_net_multi_seed.py
uv run python scripts/predict/predict_tabular_net_multi_seed.py
```

### 多模型 Ensemble

結合多個模型的預測結果。

```bash
# 需先執行 TabularNet + LightGBM 的訓練與預測
uv run python scripts/ensemble/ensemble_tabular_lightgbm.py

# 需先執行 TabularNet + TabularNet Optuna 的訓練與預測
uv run python scripts/ensemble/ensemble_optuna_tabular.py
```

## 專案結構

```
.
├── blobs/
│   ├── raw/                          # 原始資料集
│   ├── models/                       # 訓練完成的模型
│   └── submit/                       # 提交用 CSV 檔案
├── scripts/
│   ├── setup/                        # 環境與資料集設定
│   ├── train/                        # 訓練腳本
│   ├── predict/                      # 預測腳本
│   ├── ensemble/                     # 多模型組合
│   ├── postprocess/                  # 後處理
│   └── baseline/                     # 基準線模型
├── src/
│   └── diabetes_binary_classifier/   # 共用模組
├── docs/                             # 文件
└── pyproject.toml
```

## 模型

| 模型 | 說明 |
|------|------|
| AutoGluon | AutoML 框架 |
| LightGBM | 梯度提升 |
| CatBoost | 支援類別特徵的梯度提升 |
| TabularNet | PyTorch 神經網路 |
| TabularNet Optuna | TabularNet + 超參數搜索 |
| TabularNet Multi-Seed | TabularNet + 多種子集成 |

## 文件

- [競賽資訊](docs/competition.md)
- [資料集說明](docs/dataset.md)

## 授權

作者：黃毓峰 (Yu Feng)

此專案為國立臺南大學資工系 114-1 資料探勘期末專案。

可自由使用，無需授權。若能標註出處我會很開心，覺得有幫助的話歡迎給顆 Star。
