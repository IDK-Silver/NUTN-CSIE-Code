# ANN + PSO/SGD Framework

通用深度學習框架，支援 PSO (粒子群最佳化) 和 SGD (梯度下降) 訓練方法。

## 環境需求

本專案使用 Rust 開發，請先安裝 Rust：

- **Windows**: 下載並執行 [rustup-init.exe](https://win.rustup.rs/)
- **Linux/macOS**:
  ```bash
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
  ```

詳細請參考 [Rust 官方安裝指南](https://www.rust-lang.org/tools/install)。

## XOR 問題

### PSO

```bash
cargo run --bin xor-train-pso
cargo run --bin xor-predict -- pso
```

### 梯度下降 (SGD)

```bash
cargo run --bin xor-train-gd -- sgd
cargo run --bin xor-predict  -- sgd
```

## MNIST 手寫數字辨識

先下載資料集：
```bash
cargo run --bin mnist-download
```

### PSO

```bash
cargo run --release --bin mnist-train-pso
cargo run --release --bin mnist-predict -- pso
cargo run --bin           mnist-plot    -- pso
```

### 梯度下降 (SGD)

```bash
cargo run --release --bin mnist-train-gd -- sgd
cargo run --release --bin mnist-predict  -- sgd
cargo run --bin mnist-plot -- sgd
```

### 圖片生成
修改圖片樣式不需重新訓練。
```bash
# 梯度下降版本
cargo run --bin mnist-plot -- sgd             # 生成所有圖片
cargo run --bin mnist-plot -- sgd loss        # Loss 曲線
cargo run --bin mnist-plot -- sgd accuracy    # Accuracy 曲線
cargo run --bin mnist-plot -- sgd confusion   # Confusion Matrix

cargo run --bin mnist-plot -- pso             # PSO 版本，後面參數以此類推
```


## 網路架構

| 問題 | 架構 | 激活函數 | 損失函數 |
|------|------|----------|----------|
| XOR | 2-2-1 | Sigmoid | MSE |
| MNIST | 784-128-10 | ReLU + Softmax | Cross-Entropy |

## 測試

```bash
cargo test
```

## 開發文件

詳細開發指南請參考 [docs/dev/README.md](docs/dev/README.md)
