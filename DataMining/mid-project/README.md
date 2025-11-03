# 資料探勘期中專案 - 交通流量預測

使用機器學習模型預測交通流量的專案，實作了 LinearRegression 和 Polynomial Regression 兩種模型。

## 專案特色

- ✅ **進階特徵工程**：52 個精心設計的特徵（時間循環、溫度分段、天氣分組、交互作用）
- ✅ **Polynomial Regression**：自動多項式特徵轉換，R² 達到 0.190
- ✅ **模型版本管理**：時間戳管理、最新/最佳模型追蹤
- ✅ **CLI 介面**：簡潔的命令列操作

## 快速開始

### 安裝依賴

```bash
# 使用 uv (推薦)
uv venv
uv pip install -r requirements.txt

# 或使用 pip
pip install -r requirements.txt
```

### 資料準備

將原始資料放在 `blob/raw/` 目錄：
- `traffic_train.csv` - 訓練資料
- `traffic_test.csv` - 測試資料

### 執行流程

```bash
# 1. 前處理資料
uv run python main.py preprocess --mode train
uv run python main.py preprocess --mode test

# 2. 訓練模型（選擇一種）

# LinearRegression（R² = 0.177）
uv run python main.py train --full

# Polynomial Regression（R² = 0.190，推薦）
uv run python main.py train-poly --degree 2 --all-terms --feature-selection --top-k 15 --full

# 3. 生成預測
uv run python main.py predict --run latest
```

## 模型效能比較

| 模型 | R² Score | RMSE | MAE | 特徵數 | 特徵選擇方法 |
|------|----------|------|-----|--------|------------|
| LinearRegression（基礎） | 0.146 | 1823 | 1585 | 18 | 手動 |
| LinearRegression（完整） | 0.177 | 1803 | 1560 | 52 | 手動 |
| Polynomial（硬編碼） | 0.190 | 1789 | 1541 | 15→135 | 硬編碼 |
| **Polynomial（智能選擇）** | **0.193** ✅ | **1785** ✅ | **1536** ✅ | **15→135** | **數據驅動+領域知識** |

## 檔案結構

```
blob/
  raw/                  # 原始資料
  process/              # 前處理後資料
    meta/               # scaler 和類別資訊
  models/
    runs/               # 所有模型版本
    latest/             # 最新模型
  submit/
    latest/             # 最新預測

src/
  preprocess.py         # 資料前處理（含特徵工程）
  train.py             # LinearRegression 訓練
  train_poly.py        # Polynomial Regression 訓練
  predict.py           # 預測（自動偵測模型類型）
  registry.py          # 模型版本管理

main.py                # CLI 入口
```

## CLI 命令詳解

### 前處理
```bash
uv run python main.py preprocess --mode {train|test}
```

### 訓練 LinearRegression
```bash
uv run python main.py train [--full|--split]
# --full: 使用完整訓練集（預設）
# --split: 80/20 分割驗證
```

### 訓練 Polynomial Regression
```bash
uv run python main.py train-poly [OPTIONS]
# --degree N: 多項式次數（預設 2）
# --interaction-only: 只產生交互作用項
# --feature-selection: 選擇重要特徵（預設啟用）
# --top-k N: 選擇前 N 個特徵（預設 15）
# --full/--split: 完整訓練或分割驗證
```

### 預測
```bash
uv run python main.py predict --run {latest|best|<model-path>}
```

## 特徵工程詳解

### 核心特徵（52 個）
1. **基本特徵**：temp, clouds_all, Rush Hour, is_holiday
2. **時間循環**：hour_sin, hour_cos（從 ID % 24 提取）
3. **溫度分段**：5 個區間（極冷/冷/溫暖/熱/極熱）
4. **天氣分組**：高流量天氣、低流量天氣
5. **多項式特徵**：temp², temp³
6. **交互作用**：Rush Hour × 各種特徵

### Polynomial 特徵選擇（15 個核心特徵）
- Rush Hour, temp, clouds_all, is_holiday
- hour_sin, hour_cos
- rush_temp, rush_hour_cycle
- 重要天氣和溫度分段特徵

轉換後產生 135 個多項式特徵

## 特徵選擇方法

### 🧠 智能選擇（數據驅動 + 領域知識）

我們開發了一個智能特徵選擇系統，結合：

1. **數據分析**：計算所有特徵與目標變數的相關性
   ```bash
   uv run python analyze_features.py
   ```

2. **多項式回歸知識**：
   - ✅ 保留時間循環特徵（`hour_sin`, `hour_cos`）：雖然單獨相關性低，但在多項式空間中提供重要的時間模式
   - ✅ 避免多重共線性：不同時選 `temp`, `temp_squared`, `temp_cubed`（會產生冗餘資訊）
   - ✅ 包含基礎特徵：如 `clouds_all`, `is_holiday`（提供交互作用基礎）
   - ✅ 選擇高相關交互作用：`rush_temp`, `rush_weather_high` 等

3. **結果**：智能選擇比單純相關性排序提升 5.5% 效能

### 分析腳本

- `analyze_features.py` - 特徵重要性分析（保留在專案中供查證）
- 輸出：`blob/analysis/feature_importance.json`

## 最佳實踐

### ✅ 推薦配置：Polynomial Regression (degree=2) + 智能選擇
- **R² Score**: 0.193
- **特徵選擇**: 15 個核心特徵（智能選擇）
- **避免過擬合**: degree=2 是最佳平衡點（degree=3 會過擬合）
- **使用完整訓練集**: `--full` 選項提供更好的效能

### 完整工作流程
```bash
# 1. 分析特徵重要性（一次性）
uv run python analyze_features.py

# 2. 訓練模型（自動使用智能選擇）
uv run python main.py train-poly --degree 2 --all-terms --feature-selection --top-k 15 --full

# 3. 生成預測
uv run python main.py predict --run latest
```

## 技術細節

- **前處理**：StandardScaler 正規化、one-hot 編碼、缺失值處理
- **特徵工程**：時間編碼、多項式特徵、交互作用
- **模型儲存**：joblib 序列化
- **版本管理**：時間戳命名、registry.json 追蹤

## 作者

NUTN CSIE - 資料探勘期中專案