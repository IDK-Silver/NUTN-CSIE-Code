# CUDA CMake

這個資料夾建立在 `../openmp_cmake` 的結構之上，提供一個最小可執行的 CUDA 專案骨架，用來驗證你的 CUDA 環境。

## 內容

- `vector_add`: GPU 向量加法 benchmark 與結果正確性驗證（同時計算 CPU 版本作為參考）
- `add_by_block`: 使用 `blockIdx.x`，launch config 為 `<<<N, 1>>>`
- `add_by_thread`: 使用 `threadIdx.x`，launch config 為 `<<<1, N>>>`
- `add_block_size_benchmark`: 使用 `blockIdx.x * blockDim.x + threadIdx.x`，比較 CPU sequential 與不同 CUDA block size
- `add_arbitrary_size`: 使用 `if (index < n)` 和 `<<<(N + M - 1) / M, M>>>` 處理任意向量大小，並比較 CPU sequential 與不同 CUDA block size
- `parallel_kernels`: 比較 default stream 依序執行 add/mul kernel，以及兩個 CUDA stream 平行啟動 add/mul kernel 的時間
- `stencil_serial`: CPU 序列版 1D stencil，作為速度 baseline
- `stencil_global`: GPU 1D stencil，不使用 shared memory，測不同 block size
- `stencil_shared`: GPU 1D stencil，使用 shared memory 載入 tile 與 halo，測不同 block size
- `device_query`: 列出 CUDA 裝置數量與主要硬體屬性
- `hello_world_magic`: 在 GPU 上將字元陣列逐一加一並輸出 `Hello World!`

## Build

```bash
cmake -S . -B build
cmake --build build
```

## Run

```bash
./build/vector_add 1000000 20
./build/add_by_block 1024 1000
./build/add_by_thread 1024 1000
./build/add_block_size_benchmark 655350 100
./build/add_arbitrary_size 4194304 100
./build/add_arbitrary_size 4194305 100
./build/parallel_kernels 4194304 256 100
./build/stencil_serial 16777216 1
./build/stencil_global 16777216 3
./build/stencil_shared 16777216 3
./build/add_by_block --help
./build/device_query
./build/hello_world_magic
```

參數:

- `add_by_block [N] [benchmark_runs] [seed]`: 預設 `N=1024`，測 `<<<N, 1>>>`
- `add_by_thread [N] [benchmark_runs] [seed]`: 預設 `N=1024`，測 `<<<1, N>>>`；`N` 不能超過 `maxThreadsPerBlock`
- `add_block_size_benchmark [N] [benchmark_runs] [seed]`: 預設 `N=655350`，比較 block size `1, 32, 64, 128, 256, 512, 1024`
- `add_arbitrary_size [N] [benchmark_runs] [seed]`: 預設 `N=4194305`、`benchmark_runs=100`，比較 block size `1, 32, 64, 128, 256, 512, 1024` 的時間與邊界保護；可用 `4194304` 和 `4194305` 做大尺寸可整除/不可整除對照
- `parallel_kernels [array_size] [block_size] [benchmark_runs]`: 預設 `array_size=4194304`、`block_size=256`、`benchmark_runs=100`，比較不指定 stream 與兩個 explicit streams 的 add/mul kernel 平均時間
- `stencil_serial [N] [benchmark_runs] [seed]`: 預設 `N=16777216`、`RADIUS=256`、`benchmark_runs=3`，CPU 序列 stencil 平均時間；大 N 建議先用 `benchmark_runs=1`
- `stencil_global [N] [benchmark_runs] [seed]`: 同一份資料，不使用 shared memory，測 block size `16, 32, 64, 128, 256, 512, 1024`
- `stencil_shared [N] [benchmark_runs] [seed]`: 同一份資料，使用 dynamic shared memory，測 block size `16, 32, 64, 128, 256, 512, 1024`
- 加法、stream、stencil target 都支援 `-h` / `--help`。

## 注意

- 若系統未安裝 CUDA Toolkit，`cmake` 會在設定時直接失敗。
