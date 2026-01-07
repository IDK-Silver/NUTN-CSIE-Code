# 開發指南

## 目錄結構

```
ANN_PSO/
├── config/                              # 配置檔
│   ├── xor/
│   │   ├── pso/default.yaml
│   │   └── gradient-descent/sgd/default.yaml
│   └── mnist/
│       ├── pso/default.yaml
│       └── gradient-descent/sgd/default.yaml
│
├── blob/                                # 輸出目錄 (gitignore)
│   ├── xor/
│   │   ├── train/{pso,gradient-descent/sgd}/
│   │   └── predict/{pso,gradient-descent/sgd}/
│   └── mnist/
│       ├── data/                        # MNIST 資料集
│       ├── train/{pso,gradient-descent/sgd}/
│       └── predict/{pso,gradient-descent/sgd}/
│
├── src/
│   ├── lib.rs
│   ├── bin/
│   │   ├── xor/
│   │   │   ├── train_pso.rs
│   │   │   ├── train_gd.rs
│   │   │   └── predict.rs
│   │   └── mnist/
│   │       ├── download.rs              # 下載資料集
│   │       ├── train_pso.rs
│   │       ├── train_gd.rs
│   │       ├── predict.rs
│   │       └── plot.rs                  # 圖片生成
│   ├── dataset/
│   │   ├── mod.rs                       # Dataset trait
│   │   ├── xor.rs
│   │   └── mnist.rs
│   ├── model/
│   │   ├── mod.rs                       # Model, GradientModel traits
│   │   ├── xor.rs                       # 2-2-1 MLP
│   │   └── mnist.rs                     # 784-128-10 MLP
│   ├── layer/
│   │   ├── linear.rs
│   │   ├── sigmoid.rs
│   │   ├── relu.rs
│   │   └── softmax.rs
│   ├── optimizer/
│   │   ├── pso.rs
│   │   └── sgd.rs
│   ├── loss.rs                          # MSE, Cross-Entropy
│   ├── config.rs
│   └── utils.rs
│
└── README.md
```

## 核心抽象

### Dataset Trait

```rust
pub trait Dataset {
    fn train_data(&self) -> DataSplit;
    fn val_data(&self) -> Option<DataSplit>;
    fn test_data(&self) -> Option<DataSplit>;
    fn name(&self) -> &str;
}
```

### Model Trait

```rust
// 基礎 Model (PSO 使用)
pub trait Model {
    fn forward(&self, x: &Mat) -> Mat;
    fn param_count(&self) -> usize;
    fn get_params(&self) -> Vec<f64>;
    fn set_params(&mut self, params: &[f64]);
}

// 梯度 Model (SGD/Adam 使用)
pub trait GradientModel: Model {
    fn forward_with_cache(&self, x: &Mat) -> (Mat, Self::Cache);
    fn backward(&mut self, cache: &Self::Cache, grad: &Mat);
    fn apply_grads(&mut self, lr: f64);
}
```

## 配置說明

### XOR PSO (config/xor/pso/default.yaml)

```yaml
num_particles: 100
w: 0.729
c1: 1.49445
c2: 1.49445
pos_min: -10.0
pos_max: 10.0
vel_max: 4.0
max_iter: 5000
target_loss: 0.01
```

### MNIST SGD (config/mnist/gradient-descent/sgd/default.yaml)

```yaml
hidden_size: 128
lr: 0.1
max_iter: 50
batch_size: 64
```

## 依賴

- `rand` - 隨機數生成
- `plotters` - 圖表繪製
- `serde` - 序列化
- `serde_yaml` - YAML 解析
- `serde_json` - JSON 解析
- `reqwest` - HTTP 下載 (MNIST)
- `flate2` - Gzip 解壓 (MNIST)
- `byteorder` - IDX 格式解析 (MNIST)
