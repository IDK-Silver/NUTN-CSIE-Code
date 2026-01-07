# ANN-PSO 設計文件

## 1. 概述

### 1.1 目標

建立一個模組化的 Rust 神經網路函式庫，支援：
1. PSO（粒子群最佳化）進行權重最佳化
2. 梯度下降法進行權重最佳化
3. 使用者定義網路拓撲的靜態分派

### 1.2 範圍

本次作業：使用 PSO 的 2-2-1 網路解決 XOR 問題。
設計目標：可擴展至任意拓撲結構和新的層類型。

### 1.3 設計原則

- 簡單易讀優於巧妙複雜
- 使用者透過結構組合控制網路拓撲
- Layer trait 提供標準介面
- 最佳化器與網路結構解耦

---

## 2. 架構

```
src/
├── lib.rs              # 重新匯出
├── mat.rs              # 矩陣運算
├── layer/
│   ├── mod.rs          # Layer trait
│   ├── linear.rs       # 全連接層
│   └── sigmoid.rs      # Sigmoid 激活函數
├── loss.rs             # 損失函數
└── optimizer/
    ├── mod.rs          # 重新匯出
    ├── pso.rs          # 粒子群最佳化
    └── sgd.rs          # 隨機梯度下降

examples/
└── xor.rs              # 使用 PSO 和 SGD 的 XOR 問題
```

### 2.1 依賴關係圖

```
main.rs / examples/xor.rs
    │
    ├── XorNetwork（使用者定義）
    │       │
    │       ├── Linear（層）
    │       └── Sigmoid（層）
    │
    ├── mse / mse_grad（損失）
    │
    └── PSO / SGD（最佳化器）
            │
            └── Mat（全域使用）
```

---

## 3. 模組規格

### 3.1 `mat.rs` - 矩陣

#### 結構

```rust
#[derive(Debug, Clone)]
pub struct Mat {
    pub data: Vec<f64>,
    pub rows: usize,
    pub cols: usize,
}
```

儲存方式：列優先（Row-major）。位於 (i, j) 的元素索引為 `i * cols + j`。

#### 建構子

```rust
impl Mat {
    /// 使用給定資料建立矩陣
    pub fn new(rows: usize, cols: usize, data: Vec<f64>) -> Self;

    /// 建立填滿零的矩陣
    pub fn zeros(rows: usize, cols: usize) -> Self;

    /// 從二維切片建立矩陣（便利方法）
    pub fn from_slice(data: &[&[f64]]) -> Self;

    /// 建立值在 [min, max] 範圍內的隨機矩陣
    pub fn random(rows: usize, cols: usize, min: f64, max: f64) -> Self;
}
```

#### 存取器

```rust
impl Mat {
    /// 取得 (row, col) 位置的元素
    pub fn get(&self, row: usize, col: usize) -> f64;

    /// 設定 (row, col) 位置的元素
    pub fn set(&mut self, row: usize, col: usize, value: f64);

    /// 取得列的切片
    pub fn row(&self, row: usize) -> &[f64];
}
```

#### 運算

```rust
impl Mat {
    /// 矩陣乘法：self(m×k) × other(k×n) → result(m×n)
    /// 維度不匹配時會 panic
    pub fn matmul(&self, other: &Mat) -> Mat;

    /// 轉置：self(m×n) → result(n×m)
    pub fn transpose(&self) -> Mat;

    /// 元素級加法（支援偏置的廣播）
    /// 若 other 為 (1, cols)，則廣播到所有列
    pub fn add(&self, other: &Mat) -> Mat;

    /// 元素級減法
    pub fn sub(&self, other: &Mat) -> Mat;

    /// 元素級乘法（Hadamard 乘積）
    pub fn mul(&self, other: &Mat) -> Mat;

    /// 純量乘法
    pub fn scale(&self, scalar: f64) -> Mat;

    /// 沿軸求和
    /// axis=0：對行求和 → (1, cols)
    /// axis=1：對列求和 → (rows, 1)
    pub fn sum_axis(&self, axis: usize) -> Mat;

    /// 元素級套用函數
    pub fn map<F>(&self, f: F) -> Mat
    where
        F: Fn(f64) -> f64;
}
```

#### 數學細節

**矩陣乘法**（`matmul`）：

$$C_{ij} = \sum_{k=0}^{K-1} A_{ik} \cdot B_{kj}$$

其中 A 為 (M×K)，B 為 (K×N)，C 為 (M×N)。

**廣播加法**：當 (M×N) + (1×N) 時，將 (1×N) 矩陣複製 M 次。

---

### 3.2 `layer/mod.rs` - Layer Trait

```rust
pub trait Layer {
    /// 前向傳播
    /// input：(batch_size, in_features)
    /// output：(batch_size, out_features)
    fn forward(&self, input: &Mat) -> Mat;

    /// 反向傳播
    /// input：傳遞給 forward 的輸入
    /// grad_output：來自上游的梯度 (batch_size, out_features)
    /// 返回：相對於輸入的梯度 (batch_size, in_features)
    /// 副作用：內部儲存參數的梯度
    fn backward(&mut self, input: &Mat, grad_output: &Mat) -> Mat;

    /// 可訓練參數數量
    fn param_count(&self) -> usize;

    /// 將參數展平為向量
    /// 順序必須是確定性的且有文件說明
    fn get_params(&self) -> Vec<f64>;

    /// 從切片載入參數
    /// 返回消耗的元素數量
    fn set_params(&mut self, params: &[f64]) -> usize;

    /// 取得梯度（與 get_params 順序相同）
    fn get_grads(&self) -> Vec<f64>;

    /// 更新參數：param -= lr * grad
    fn apply_grads(&mut self, lr: f64);
}
```

---

### 3.3 `layer/linear.rs` - 線性層

#### 結構

```rust
pub struct Linear {
    /// 權重矩陣 (in_features, out_features)
    pub weight: Mat,
    /// 偏置向量 (1, out_features)
    pub bias: Mat,
    /// 權重的梯度（與 weight 形狀相同）
    grad_weight: Mat,
    /// 偏置的梯度（與 bias 形狀相同）
    grad_bias: Mat,
}
```

#### 建構子

```rust
impl Linear {
    /// 建立權重在 [-1, 1] 範圍內的隨機層
    pub fn new(in_features: usize, out_features: usize) -> Self;

    /// 建立指定權重範圍的層
    pub fn new_with_range(in_features: usize, out_features: usize, min: f64, max: f64) -> Self;
}
```

#### 前向傳播

$$Y = X \cdot W + B$$

- $X$：輸入 (batch\_size, in\_features)
- $W$：權重 (in\_features, out\_features)
- $B$：偏置 (1, out\_features)，廣播到所有列
- $Y$：輸出 (batch\_size, out\_features)

```rust
fn forward(&self, input: &Mat) -> Mat {
    input.matmul(&self.weight).add(&self.bias)
}
```

#### 反向傳播

給定 $\frac{\partial L}{\partial Y}$（grad\_output），計算：

$$\frac{\partial L}{\partial X} = \frac{\partial L}{\partial Y} \cdot W^T$$

$$\frac{\partial L}{\partial W} = X^T \cdot \frac{\partial L}{\partial Y}$$

$$\frac{\partial L}{\partial B} = \sum_{\text{batch}} \frac{\partial L}{\partial Y}$$

```rust
fn backward(&mut self, input: &Mat, grad_output: &Mat) -> Mat {
    // grad_input = grad_output @ weight.T
    let grad_input = grad_output.matmul(&self.weight.transpose());

    // grad_weight = input.T @ grad_output
    self.grad_weight = input.transpose().matmul(grad_output);

    // grad_bias = 對批次求和 (axis=0)
    self.grad_bias = grad_output.sum_axis(0);

    grad_input
}
```

#### 參數順序

`get_params()` 返回：`[weight 列優先..., bias...]`

對於 (2, 3) 線性層：
```
[w00, w01, w02, w10, w11, w12, b0, b1, b2]
```

總計：`in_features * out_features + out_features`

---

### 3.4 `layer/sigmoid.rs` - Sigmoid 激活函數

#### 結構

```rust
pub struct Sigmoid;
```

無參數，無狀態。

#### 前向傳播

$$\sigma(x) = \frac{1}{1 + e^{-x}}$$

```rust
fn forward(&self, input: &Mat) -> Mat {
    input.map(|x| 1.0 / (1.0 + (-x).exp()))
}
```

#### 反向傳播

$$\frac{\partial L}{\partial X} = \frac{\partial L}{\partial Y} \odot \sigma(X) \odot (1 - \sigma(X))$$

其中 $\odot$ 為元素級乘法。

```rust
fn backward(&mut self, input: &Mat, grad_output: &Mat) -> Mat {
    let sig = self.forward(input);  // 重新計算 sigmoid
    let one_minus_sig = sig.map(|s| 1.0 - s);
    grad_output.mul(&sig).mul(&one_minus_sig)
}
```

#### Layer Trait（參數方法為空操作）

```rust
fn param_count(&self) -> usize { 0 }
fn get_params(&self) -> Vec<f64> { vec![] }
fn set_params(&mut self, _params: &[f64]) -> usize { 0 }
fn get_grads(&self) -> Vec<f64> { vec![] }
fn apply_grads(&mut self, _lr: f64) {}
```

---

### 3.5 `loss.rs` - 損失函數

#### MSE（均方誤差）

依據作業要求：

$$L = \frac{1}{2} \sum_{i} (y_i - \hat{y}_i)^2$$

注意：這是求和，不是平均。$\frac{1}{2}$ 簡化了梯度計算。

```rust
/// 計算 MSE 損失（0.5 * 平方誤差和）
pub fn mse(pred: &Mat, target: &Mat) -> f64 {
    assert_eq!(pred.rows, target.rows);
    assert_eq!(pred.cols, target.cols);

    let mut sum = 0.0;
    for i in 0..pred.data.len() {
        let diff = pred.data[i] - target.data[i];
        sum += diff * diff;
    }
    0.5 * sum
}
```

#### MSE 梯度

$$\frac{\partial L}{\partial \hat{y}_i} = \hat{y}_i - y_i$$

```rust
/// 計算 MSE 損失相對於預測值的梯度
pub fn mse_grad(pred: &Mat, target: &Mat) -> Mat {
    pred.sub(target)
}
```

---

### 3.6 `optimizer/pso.rs` - 粒子群最佳化

#### 結構

```rust
pub struct PsoConfig {
    pub num_particles: usize,
    pub dim: usize,
    pub w: f64,           // 慣性權重
    pub c1: f64,          // 認知係數
    pub c2: f64,          // 社會係數
    pub pos_min: f64,     // 位置邊界
    pub pos_max: f64,
    pub vel_max: f64,     // 速度限制
}

impl Default for PsoConfig {
    fn default() -> Self {
        Self {
            num_particles: 30,
            dim: 9,
            w: 0.7,
            c1: 1.5,
            c2: 1.5,
            pos_min: -10.0,
            pos_max: 10.0,
            vel_max: 1.0,
        }
    }
}

struct Particle {
    position: Vec<f64>,
    velocity: Vec<f64>,
    best_position: Vec<f64>,
    best_fitness: f64,
}

pub struct Pso {
    particles: Vec<Particle>,
    global_best_position: Vec<f64>,
    global_best_fitness: f64,
    config: PsoConfig,
}
```

#### 建構子

```rust
impl Pso {
    pub fn new(config: PsoConfig) -> Self {
        // 初始化粒子，位置在 [pos_min, pos_max] 範圍內隨機
        // 初始化速度為 0 或小的隨機值
        // 評估初始適應度並設定個人/全域最佳
    }
}
```

#### 步驟

```rust
impl Pso {
    /// 執行一次 PSO 迭代
    /// fitness_fn：越小越好（最小化）
    pub fn step<F>(&mut self, fitness_fn: F)
    where
        F: Fn(&[f64]) -> f64,
    {
        for particle in &mut self.particles {
            // 更新速度
            for d in 0..self.config.dim {
                let r1: f64 = rand::random();
                let r2: f64 = rand::random();

                particle.velocity[d] = self.config.w * particle.velocity[d]
                    + self.config.c1 * r1 * (particle.best_position[d] - particle.position[d])
                    + self.config.c2 * r2 * (self.global_best_position[d] - particle.position[d]);

                // 限制速度
                particle.velocity[d] = particle.velocity[d]
                    .clamp(-self.config.vel_max, self.config.vel_max);
            }

            // 更新位置
            for d in 0..self.config.dim {
                particle.position[d] += particle.velocity[d];
                // 限制位置
                particle.position[d] = particle.position[d]
                    .clamp(self.config.pos_min, self.config.pos_max);
            }

            // 評估適應度
            let fitness = fitness_fn(&particle.position);

            // 更新個人最佳
            if fitness < particle.best_fitness {
                particle.best_fitness = fitness;
                particle.best_position = particle.position.clone();
            }

            // 更新全域最佳
            if fitness < self.global_best_fitness {
                self.global_best_fitness = fitness;
                self.global_best_position = particle.position.clone();
            }
        }
    }

    pub fn best_position(&self) -> &[f64] {
        &self.global_best_position
    }

    pub fn best_fitness(&self) -> f64 {
        self.global_best_fitness
    }
}
```

---

### 3.7 `optimizer/sgd.rs` - 隨機梯度下降

不含動量的簡單 SGD。

```rust
pub struct Sgd {
    pub lr: f64,
}

impl Sgd {
    pub fn new(lr: f64) -> Self {
        Self { lr }
    }
}
```

使用方式為手動：使用者為每一層呼叫 `layer.apply_grads(sgd.lr)`。
這使 SGD 與網路結構保持解耦。

---

## 4. 使用者定義網路

使用者透過組合層來定義自己的網路結構。

### 4.1 XorNetwork 範例

```rust
pub struct XorNetwork {
    pub linear1: Linear,   // 2 -> 2
    pub sigmoid1: Sigmoid,
    pub linear2: Linear,   // 2 -> 1
    pub sigmoid2: Sigmoid,
}

impl XorNetwork {
    pub fn new() -> Self {
        Self {
            linear1: Linear::new(2, 2),
            sigmoid1: Sigmoid,
            linear2: Linear::new(2, 1),
            sigmoid2: Sigmoid,
        }
    }

    pub fn forward(&self, x: &Mat) -> Mat {
        let x = self.linear1.forward(x);
        let x = self.sigmoid1.forward(&x);
        let x = self.linear2.forward(&x);
        self.sigmoid2.forward(&x)
    }
}
```

### 4.2 PSO 用途：扁平參數介面

```rust
impl XorNetwork {
    pub fn param_count(&self) -> usize {
        self.linear1.param_count() + self.linear2.param_count()
        // sigmoid 有 0 個參數
    }

    pub fn get_params(&self) -> Vec<f64> {
        let mut params = self.linear1.get_params();
        params.extend(self.linear2.get_params());
        params
    }

    pub fn set_params(&mut self, params: &[f64]) {
        let consumed = self.linear1.set_params(params);
        self.linear2.set_params(&params[consumed..]);
    }
}
```

2-2-1 網路的參數順序：
```
linear1.weight: [w00, w01, w10, w11]  (2x2, 列優先)
linear1.bias:   [b0, b1]              (1x2)
linear2.weight: [w0, w1]              (2x1)
linear2.bias:   [b0]                  (1x1)

總計：4 + 2 + 2 + 1 = 9 個參數
```

### 4.3 梯度下降用途：帶快取的反向傳播

```rust
pub struct XorCache {
    x: Mat,      // 原始輸入
    z1: Mat,     // linear1 輸出（sigmoid 之前）
    h1: Mat,     // sigmoid1 輸出
    z2: Mat,     // linear2 輸出（sigmoid 之前）
}

impl XorNetwork {
    pub fn forward_with_cache(&self, x: &Mat) -> (Mat, XorCache) {
        let z1 = self.linear1.forward(x);
        let h1 = self.sigmoid1.forward(&z1);
        let z2 = self.linear2.forward(&h1);
        let y = self.sigmoid2.forward(&z2);

        let cache = XorCache {
            x: x.clone(),
            z1,
            h1,
            z2,
        };
        (y, cache)
    }

    pub fn backward(&mut self, cache: &XorCache, grad_output: &Mat) {
        let grad = self.sigmoid2.backward(&cache.z2, grad_output);
        let grad = self.linear2.backward(&cache.h1, &grad);
        let grad = self.sigmoid1.backward(&cache.z1, &grad);
        let _ = self.linear1.backward(&cache.x, &grad);
    }

    pub fn apply_grads(&mut self, lr: f64) {
        self.linear1.apply_grads(lr);
        self.linear2.apply_grads(lr);
    }
}
```

---

## 5. 資料流

### 5.1 前向傳播（推論）

```
輸入 X (4×2)
    │
    ▼
Linear1: Z1 = X @ W1 + B1     (4×2) @ (2×2) + (1×2) → (4×2)
    │
    ▼
Sigmoid1: H1 = σ(Z1)          (4×2) → (4×2)
    │
    ▼
Linear2: Z2 = H1 @ W2 + B2    (4×2) @ (2×1) + (1×1) → (4×1)
    │
    ▼
Sigmoid2: Y = σ(Z2)           (4×1) → (4×1)
    │
    ▼
輸出 Y (4×1)
```

### 5.2 反向傳播（梯度下降）

```
損失 L = 0.5 * sum((Y - T)^2)
    │
    ▼
dL/dY = Y - T                 (4×1)
    │
    ▼
Sigmoid2.backward:
    dL/dZ2 = dL/dY ⊙ σ(Z2) ⊙ (1-σ(Z2))    (4×1)
    │
    ▼
Linear2.backward:
    dL/dH1 = dL/dZ2 @ W2.T                 (4×1) @ (1×2) → (4×2)
    dL/dW2 = H1.T @ dL/dZ2                 (2×4) @ (4×1) → (2×1)
    dL/dB2 = sum(dL/dZ2, axis=0)           (1×1)
    │
    ▼
Sigmoid1.backward:
    dL/dZ1 = dL/dH1 ⊙ σ(Z1) ⊙ (1-σ(Z1))   (4×2)
    │
    ▼
Linear1.backward:
    dL/dX = dL/dZ1 @ W1.T                  (4×2) @ (2×2) → (4×2)
    dL/dW1 = X.T @ dL/dZ1                  (2×4) @ (4×2) → (2×2)
    dL/dB1 = sum(dL/dZ1, axis=0)           (1×2)
```

### 5.3 PSO 流程

```
┌─────────────────────────────────────────────────────┐
│                    PSO 迴圈                          │
│                                                     │
│  對每個粒子：                                        │
│      1. network.set_params(particle.position)       │
│      2. pred = network.forward(X)                   │
│      3. fitness = mse(pred, Y)                      │
│      4. 若更好則更新個人最佳                          │
│      5. 若更好則更新全域最佳                          │
│                                                     │
│  對每個粒子：                                        │
│      6. 更新速度                                     │
│      7. 更新位置                                     │
│                                                     │
└─────────────────────────────────────────────────────┘
```

---

## 6. 訓練迴圈範例

### 6.1 PSO 訓練

```rust
fn main() {
    // 資料
    let x = Mat::from_slice(&[
        &[0.0, 0.0],
        &[0.0, 1.0],
        &[1.0, 0.0],
        &[1.0, 1.0],
    ]);
    let y = Mat::from_slice(&[&[0.0], &[1.0], &[1.0], &[0.0]]);

    // 網路
    let mut network = XorNetwork::new();

    // PSO
    let config = PsoConfig {
        num_particles: 30,
        dim: network.param_count(),
        ..Default::default()
    };
    let mut pso = Pso::new(config);

    // 訓練
    for iter in 0..1000 {
        pso.step(|params| {
            network.set_params(params);
            let pred = network.forward(&x);
            mse(&pred, &y)
        });

        if iter % 100 == 0 {
            println!("迭代 {}: 損失 = {}", iter, pso.best_fitness());
        }

        if pso.best_fitness() < 0.01 {
            break;
        }
    }

    // 套用最佳權重
    network.set_params(pso.best_position());
}
```

### 6.2 SGD 訓練

```rust
fn main() {
    let x = Mat::from_slice(&[
        &[0.0, 0.0],
        &[0.0, 1.0],
        &[1.0, 0.0],
        &[1.0, 1.0],
    ]);
    let y = Mat::from_slice(&[&[0.0], &[1.0], &[1.0], &[0.0]]);

    let mut network = XorNetwork::new();
    let sgd = Sgd::new(0.5);

    for iter in 0..10000 {
        // 前向傳播
        let (pred, cache) = network.forward_with_cache(&x);
        let loss = mse(&pred, &y);

        // 反向傳播
        let grad = mse_grad(&pred, &y);
        network.backward(&cache, &grad);

        // 更新
        network.apply_grads(sgd.lr);

        if iter % 1000 == 0 {
            println!("迭代 {}: 損失 = {}", iter, loss);
        }
    }
}
```

---

## 7. 測試清單

### 7.1 Mat 測試

- [ ] `matmul`：(2×3) × (3×4) = (2×4)
- [ ] `transpose`：(2×3) → (3×2)
- [ ] `add` 廣播：(4×2) + (1×2)
- [ ] `sum_axis`：axis=0 和 axis=1

### 7.2 Layer 測試

- [ ] Linear 前向傳播維度
- [ ] Linear 反向傳播維度
- [ ] Linear 梯度正確性（數值梯度檢查）
- [ ] Sigmoid 前向傳播：值在 (0, 1) 範圍內
- [ ] Sigmoid 反向傳播：梯度正確性

### 7.3 整合測試

- [ ] PSO 在 XOR 上收斂（損失 < 0.01）
- [ ] SGD 在 XOR 上收斂（損失 < 0.01）
- [ ] 測試輸入 (0.7, 0.3)、(0.6, 0.4)、(0.5, 0.5)

---

## 8. 依賴套件

```toml
[dependencies]
rand = "0.8"
```

僅 `rand` 用於 PSO 和權重初始化中的隨機數生成。

---

## 9. 作業對應表

| 作業需求 | 實作 |
|----------|------|
| 9 個權重 | `XorNetwork.param_count() == 9` |
| PSO 速度公式 | `Pso::step()` |
| PSO 位置公式 | `Pso::step()` |
| MSE 損失 | `loss::mse()` |
| Sigmoid 激活函數 | `Sigmoid::forward()` |
| 收斂曲線 | 每次迭代記錄 `pso.best_fitness()` |
| 測試 (0.7, 0.3) 等 | `network.forward(&test_input)` |
| [選用] GD 比較 | `XorNetwork::backward()` + `Sgd` |
