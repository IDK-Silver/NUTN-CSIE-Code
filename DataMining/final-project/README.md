# 糖尿病二元分類器 (Diabetes Binary Classifier)

## 專案總覽 (Overview)

本專案實作了一個 **二元分類模型 (Binary Classification Model)**，用於預測患者是否患有糖尿病（包含糖尿病前期）。這是國立臺南大學資工系 (CSIE.NUTN) 114-1 學期 **資料探勘 (Data Mining)** 課程的期末專案。

## 競賽資訊

**競賽主辦人：** I-Fang Su
**參賽狀況：** 27 位報名，25 位參與，25 支隊伍
**總提交次數：** 335 次

### 評分標準

1. 成功上傳結果、超越 **基準線 (Baseline)** 並出現在排行榜上：**60%**
2. **私有排行榜 (Private Leaderboard)** 排名（排名越高分數越高）：**20%**
3. **程式碼 (Code)** 品質與分析深度：**20%**

### 需求 / 規定

* 排行榜顯示名稱必須包含學號。
* 程式碼必須在截止期限前上傳至 E-course。
* 逾期提交將不予受理。

### 評估指標 (Evaluation Metric)

模型效能使用 **F1-score** 進行評估。

## 資料集 (Dataset)

### 目標變數 (Target Variable)

* `target` (**二元/Binary**)：0 = 無糖尿病，1 = 糖尿病前期或糖尿病

### 特徵 (Features)

* `ID` (整數)：病患 ID
* `HighBP` (二元)：高血壓 (0 = 否，1 = 是)
* `HighChol` (二元)：高膽固醇 (0 = 否，1 = 是)
* `CholCheck` (二元)：過去 5 年內是否有做過膽固醇檢查 (0 = 否，1 = 是)
* `BMI` (整數)：身體質量指數 (Body Mass Index)
* `Smoker` (二元)：一生中是否吸過至少 100 支菸 (0 = 否，1 = 是)
* `Stroke` (二元)：是否曾中風 (0 = 否，1 = 是)
* `HeartDiseaseorAttack` (二元)：冠狀動脈心臟病或心肌梗塞 (0 = 否，1 = 是)
* `PhysActivity` (二元)：過去 30 天內是否有體能活動，不含工作 (0 = 否，1 = 是)
* `Fruits` (二元)：每天食用水果 1 次以上 (0 = 否，1 = 是)
* `Veggies` (二元)：每天食用蔬菜 1 次以上 (0 = 否，1 = 是)
* `HvyAlcoholConsump` (二元)：酗酒 / 重度飲酒 (0 = 否，1 = 是)
* 男性：每週 >14 杯，女性：每週 >7 杯


* `AnyHealthcare` (二元)：擁有醫療保險/健保覆蓋 (0 = 否，1 = 是)
* `NoDocbcCost` (二元)：過去 12 個月內是否因費用問題無法看醫生 (0 = 否，1 = 是)
* `GenHlth` (整數)：一般健康狀況 (1-5 分量表)
* 1 = 極佳 (excellent)，2 = 很好 (very good)，3 = 好 (good)，4 = 普通 (fair)，5 = 差 (poor)


* `MentHlth` (整數)：過去 30 天內心理健康不佳的天數 (0-30)
* `PhysHlth` (整數)：過去 30 天內身體健康不佳的天數 (0-30)
* `DiffWalk` (二元)：行走或爬樓梯是否有困難 (0 = 否，1 = 是)
* `Sex` (二元)：0 = 女性，1 = 男性
* `Age` (整數)：13 個年齡層級距 (1-13)
* 1 = 18-24 歲，9 = 60-64 歲，13 = 80 歲或以上


* `Education` (整數)：教育程度 (1-6 分量表)
* 1 = 未受過教育/僅幼稚園
* 2 = 1-8 年級 (國小/國中)
* 3 = 9-11 年級 (高中肄業)
* 4 = 12 年級或 GED (高中畢業)
* 5 = 大學 1-3 年 (大學肄業/專科學校)
* 6 = 大學 4 年以上 (大學畢業)


* `Income` (整數)：收入等級 (1-8 分量表)
* 1 = 低於 $10,000 美元
* 5 = 低於 $35,000 美元
* 8 = $75,000 美元或更多



## 環境設定 (Setup)

### 前置需求 (Prerequisites)

* Python 3.12 或更高版本
* [uv](https://github.com/astral-sh/uv) **套件管理器 (Package Manager)**

### 安裝 (Installation)

1. **複製 (Clone)** 此 **儲存庫 (Repository)**
2. 使用 uv 初始化專案：

```bash
uv init
```

3. 同步 **相依套件 (Dependencies)**：

```bash
uv sync
```

這將會安裝 `pyproject.toml` 中指定的所有必要套件。

## 使用方法 (Usage)

### 下載資料集

下載並解壓縮資料集至適當目錄：

```bash
uv run python scripts/setup/download_dataset.py
```

此 **腳本 (Script)** 將會：

* 下載資料集 zip 檔至 `blobs/unzip/`
* 解壓縮內容至 `blobs/raw/`
* 顯示下載與解壓縮進度條

### 執行模型

```bash
# TabularNet
uv run python scripts/train/tabular_net.py

# LightGBM
uv run python scripts/train/train_lightgbm.py

# Ensemble (TabularNet + LightGBM)
uv run python scripts/train/ensemble_tabular_lightgbm.py

# AutoGluon
uv run python scripts/train/train_autogluon.py

# AutoGluon predict
uv run python scripts/predict/predict_autogluon.py

# Ensemble (Optuna + TabularNet)
uv run python scripts/train/ensemble_optuna_tabular.py

# Postprocess Optuna (nested threshold + calibration)
uv run python scripts/train/postprocess_optuna_threshold.py
```

### 其他訓練腳本 (Optional)

```bash
# Multi-seed TabularNet
uv run python scripts/train/train_tabular_net_multi_seed.py

# Optuna search for TabularNet
uv add optuna
uv run python scripts/train/train_tabular_net_optuna.py

# CatBoost (requires catboost)
uv add catboost
uv run python scripts/train/train_catboost.py
```

### 執行測試

```bash
uv run pytest

```

## 提交格式 (Submission Format)

提交檔案必須包含 **測試集 (Test Set)** 中每個 ID 的預測結果，格式如下：

```csv
ID,TARGET
2,0
5,0
6,0

```

每一列應包含：

* `ID`：測試集中的病患 ID
* `TARGET`：預測機率 (0 或 1)

### 提交檔輸出位置

* TabularNet：`blobs/submit/tabular/submission_tabular_net.csv`
* LightGBM：`blobs/submit/lightgbm/submission_lightgbm.csv`
* Ensemble：`blobs/submit/ensemble/submission_ensemble_tabular_lightgbm.csv`
* TabularNet Multi-Seed：`blobs/submit/tabular_multi_seed/submission_tabular_net_multi_seed.csv`
* TabularNet Optuna：`blobs/submit/tabular_optuna/submission_tabular_net_optuna.csv`
* CatBoost：`blobs/submit/catboost/submission_catboost.csv`
* AutoGluon：`blobs/submit/autogluon/submission_autogluon.csv` (from `scripts/predict/predict_autogluon.py`)
* Ensemble Optuna + TabularNet：`blobs/submit/ensemble_optuna_tabular/submission_ensemble_optuna_tabular.csv`
* Optuna Postprocess (nested): `blobs/submit/tabular_optuna_post/submission_tabular_net_optuna_nested.csv`
* Optuna Postprocess (calibrated): `blobs/submit/tabular_optuna_post/submission_tabular_net_optuna_calibrated.csv`

## 專案結構 (Project Structure)

```
.
├── blobs/
│   ├── unzip/          # 下載的 zip 檔案
│   ├── raw/            # 解壓縮後的資料集檔案
│   └── submit/         # 提交檔案輸出
├── scripts/
│   ├── setup/
│   │   └── download_dataset.py
│   ├── predict/
│   │   └── predict_autogluon.py
│   └── train/
│       ├── base_line.py
│       ├── tabular_net.py
│       ├── train_tabular_net_multi_seed.py
│       ├── train_tabular_net_optuna.py
│       ├── train_lightgbm.py
│       ├── train_catboost.py
│       └── ensemble_tabular_lightgbm.py
│       └── ensemble_optuna_tabular.py
│       └── postprocess_optuna_threshold.py
│       └── train_autogluon.py
├── src/
├── tests/
├── main.py
├── pyproject.toml
└── README.md

```

## 開發 (Development)

所有 Python 指令應使用 `uv run` 執行，以確保 **環境隔離 (Environment Isolation)** 正確：

```bash
# 執行任何 Python 腳本
uv run python <script_name>.py

# 執行 Python 模組 (Module)
uv run python -m <module_name>

# 安裝額外套件
uv add <package_name>

```

## 備註 (Notes)

* 酗酒 / 重度飲酒的定義因性別而異。
* 年齡類別遵循 `_AGEG5YR` **編碼簿 (Codebook)**。
* 教育程度與收入分別使用 `EDUCA` 和 `INCOME2` codebook 量表。
