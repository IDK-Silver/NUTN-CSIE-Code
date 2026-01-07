# ANN + PSO for XOR Problem

使用粒子群演算法 (PSO) 或梯度下降 (SGD) 訓練類神經網路解決 XOR 問題。

## 網路架構

- **結構**: 2-2-1 (2 輸入 → 2 隱藏層 → 1 輸出)
- **激勵函數**: Sigmoid
- **損失函數**: MSE = 0.5 × Σ(y - ŷ)²
- **參數數量**: 9 個權重

## 快速開始

```bash
# 1. PSO 訓練
cargo run --example train_pso

# 2. SGD 訓練
cargo run --example train_gd -- sgd

# 3. 預測 (使用 PSO 模型)
cargo run --example predict -- pso

# 3. 預測 (使用 SGD 模型)
cargo run --example predict -- sgd
```

## 目錄結構

```
ANN_PSO/
├── config/                         # 配置檔
│   ├── pso/
│   │   └── default.yaml           # PSO 參數
│   └── gradient-descent/
│       └── sgd/
│           └── default.yaml       # SGD 參數
├── blob/                          # 輸出目錄
│   ├── train/                     # 訓練產出
│   │   ├── pso/
│   │   │   ├── model.json        # 模型權重
│   │   │   ├── loss.csv          # 損失歷史
│   │   │   └── loss.png          # 損失曲線圖
│   │   └── gradient-descent/
│   │       └── sgd/
│   │           └── ...
│   └── predict/                   # 預測產出
│       ├── pso/
│       │   └── results.csv
│       └── gradient-descent/
│           └── sgd/
│               └── results.csv
├── src/                           # 核心程式碼
│   ├── lib.rs
│   ├── mat.rs                    # 矩陣運算
│   ├── layer/                    # 神經網路層
│   ├── optimizer/                # 優化器 (PSO, SGD)
│   ├── loss.rs                   # 損失函數
│   ├── network.rs                # XorNetwork
│   ├── config.rs                 # 配置讀取
│   └── utils.rs                  # 工具函數
└── examples/                      # 執行程式
    ├── train_pso.rs              # PSO 訓練
    ├── train_gd.rs               # 梯度下降訓練
    └── predict.rs                # 預測
```

## 配置說明

### PSO 參數 (config/pso/default.yaml)

```yaml
num_particles: 100    # 粒子數量
w: 0.729              # 慣性權重 (Clerc & Kennedy)
c1: 1.49445           # 認知係數
c2: 1.49445           # 社會係數
pos_min: -10.0        # 位置下限
pos_max: 10.0         # 位置上限
vel_max: 4.0          # 速度上限
max_iter: 5000        # 最大迭代次數
target_loss: 0.01     # 目標損失
```

### SGD 參數 (config/gradient-descent/sgd/default.yaml)

```yaml
lr: 0.5               # 學習率
max_iter: 10000       # 最大迭代次數
target_loss: 0.01     # 目標損失
```

## 測試案例

根據作業要求，預測程式會測試以下輸入:

| X1  | X2  |
|-----|-----|
| 0.7 | 0.3 |
| 0.6 | 0.4 |
| 0.5 | 0.5 |

## 輸出格式

### 模型檔案 (model.json)

```json
{
  "architecture": "2-2-1",
  "optimizer": "pso",
  "final_loss": 0.00234,
  "iterations": 1523,
  "weights": {
    "linear1_weight": [...],
    "linear1_bias": [...],
    "linear2_weight": [...],
    "linear2_bias": [...]
  }
}
```

### 損失歷史 (loss.csv)

```csv
iteration,loss
1,0.5
2,0.48
...
```

### 預測結果 (results.csv)

```csv
x1,x2,prediction
0.7,0.3,0.8765
0.6,0.4,0.7654
0.5,0.5,0.5432
```

## 執行測試

```bash
cargo test
```

## 依賴

- `rand` - 隨機數生成
- `plotters` - 圖表繪製
- `serde` - 序列化
- `serde_yaml` - YAML 解析
- `serde_json` - JSON 解析
