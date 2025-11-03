# 專案狀態

## ✅ 已完成

### 核心功能
- [x] 資料前處理（含進階特徵工程，52 個特徵）
- [x] LinearRegression 模型（R² = 0.177）
- [x] Polynomial Regression 模型（R² = 0.190）
- [x] 自動模型偵測預測系統
- [x] CLI 介面

### 最佳模型
- **類型**: Polynomial Regression (degree=2)
- **效能**: R² = 0.193, RMSE = 1785, MAE = 1536
- **特徵選擇**: 智能選擇（數據驅動 + 多項式回歸知識）
- **位置**: `blob/models/latest/`
- **預測**: `blob/submit/latest/submission.csv`

## 快速執行

```bash
# 完整流程
uv run python main.py preprocess --mode train
uv run python main.py preprocess --mode test

# 特徵重要性分析（一次性，已完成）
uv run python analyze_features.py

# 訓練最佳模型（自動使用智能特徵選擇）
uv run python main.py train-poly --degree 2 --all-terms --feature-selection --top-k 15 --full

# 生成預測
uv run python main.py predict --run latest
```

## 檔案結構

```
專案根目錄/
├── main.py                  # CLI 入口
├── analyze_features.py      # 特徵重要性分析（數據驅動選擇）
├── README.md               # 完整文檔
├── PROJECT_STATUS.md       # 本檔案
├── docs/
│   └── design.md           # 設計規範
├── src/
│   ├── preprocess.py       # 前處理（含特徵工程）
│   ├── train.py           # LinearRegression
│   ├── train_poly.py      # Polynomial Regression（智能特徵選擇）
│   ├── predict.py         # 預測（自動偵測模型類型）
│   ├── registry.py        # 版本管理
│   └── utils.py           # 工具函數
└── blob/
    ├── raw/               # 原始資料
    ├── process/           # 處理後資料
    ├── analysis/          # 特徵分析結果
    │   └── feature_importance.json
    ├── models/            # 訓練模型
    └── submit/            # 預測結果
```

## 提交檔案

最終提交檔案位於：`blob/submit/latest/submission.csv`
- 包含 14,462 筆預測
- 格式：ID, traffic_volume
