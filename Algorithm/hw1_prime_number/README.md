# 質數測試演算法實作與分析

## 目錄

- [專案簡介](#專案簡介)
- [目錄結構](#目錄結構)
- [安裝與設定](#安裝與設定)
  - [前置條件](#前置條件)
  - [安裝步驟](#安裝步驟)
- [使用方法](#使用方法)
  - [生成 Fermat & Miller-Rabin 機率測試數據](#生成-fermat--miller-rabin-機率測試數據)
  - [生成 Fermat & Miller-Rabin 機率測試數據圖表](#生成-fermat--miller-rabin-機率測試數據圖表)
  - [生成演算法效能數據](#生成演算法效能數據)
  - [生成演算法效能數據圖表](#生成效能圖表)
  - [尋找梅森素數](#尋找梅森素數)
- [腳本說明](#腳本說明)
- [參考文獻](#參考文獻)

---

## 專案簡介

本專案旨在設計與實作多種質數測試演算法，並對其進行理論驗證與實驗分析。主要涵蓋的演算法包括：

- Basic Prime Testing
- Fermat Primality Testing
- Miller-Rabin Primality Testing
- Sieve of Eratosthenes
- Mersenne Prime Number Discovery (2ᵖ⁻¹)

透過本專案，使用者可以比較不同質數測試演算法的效能，並探索梅森素數的發現。

---

## 目錄結構

---

## 安裝與設定

### 前置條件

請確保您的系統已安裝以下軟體：

- **Git**：版本控制工具
- **CMake**：跨平台的構建系統
- **Make**：構建自動化工具
- **g++**：GNU C++ 編譯器
- **Python 3**：程式語言
- **Wget**：網路下載工具

### 安裝步驟

#### 克隆本專案並執行安裝腳本

```bash
git clone https://github.com/IDK-Silver/NUTN-CSIE-Code.git
cd NUTN-CSIE-Code/Algorithm/hw1_prime_number
```

執行以下指令下載並安裝所需的套件與依賴項：

```bash:/path/to/setup.sh
./setup.sh
```

此腳本將完成如下操作：

- 檢查 git、cmake、make 和 g++
- Clone big-int 庫
- 生成並編譯專案
- 建立 Python 虛擬環境並安裝必要的 Python 套件
- 下載所需字體檔案

安裝腳本已自動為您建立 `build` 目錄並編譯專案。如果需要手動編譯，請執行：

```bash:/path/to/manual_build.sh
mkdir -p build
cd build
cmake ..
cmake --build .
cd ..
```

---

## 使用方法
### 生成 Fermat & Miller-Rabin 機率測試數據

```bash:/path/to/generate_cost_time_csv.sh
./script/data/generate_fermat_vs_miller_data.sh
```

**說明：**

- `N_LIST`：要測試的數字列表（預設為 "10,100,1000,10000,100000,1000000"）
- `TRY_TIME_START`：嘗試次數的起始值（預設為 1）
- `TRY_TIME_END`：嘗試次數的結束值（預設為 15）

輸出結果將保存在 `result/fermat_vs_miller` 資料夾內。

### 生成 Fermat & Miller-Rabin 機率測試數據圖表

此腳本將運行 Python 腳本來生成 Fermat & Miller-Rabin 機率測試數據圖表。

```bash:/path/to/generate_cost_time_figur.sh
./script/data/generate_fermat_vs_miller_figur.sh
```

**說明：**

Python 腳本會讀取生成的 `basic_${n}.txt` 以及 `n_${n}_result.txt`，並利用 `pandas`、`matplotlib` 和 `seaborn` 庫生成相關的圖表。

輸出的圖表將保存在 `result/fermat_vs_miller/plots` 資料夾內。

### 生成演算法效能數據

此腳本將運行所有質數測試演算法，並生成運行時間數據。

```bash:/path/to/generate_cost_time_csv.sh
./script/data/generate_cost_time_csv.sh
```

**說明：**

- `N_START`：開始的數字（預設為 2）
- `N_END`：結束的數字（預設為 1,000,000）
- `FERMAT_TRY_TIME`：Fermat 測試的嘗試次數（預設為 1）
- `MILLER_RABIN_TRY_TIME`：Miller-Rabin 測試的嘗試次數（預設為 1）

輸出結果將保存在 `result/all_algorithm` 資料夾內。

### 生成效能圖表

此腳本將運行 Python 腳本來生成效能圖表。

```bash:/path/to/generate_cost_time_figur.sh
./script/data/generate_cost_time_figur.sh
```

**說明：**

Python 腳本會讀取生成的 `cost_time.csv`，並利用 `pandas`、`matplotlib` 和 `seaborn` 庫生成相關的圖表。

輸出的圖表將保存在 `result/all_algorithm` 資料夾內。

### 尋找梅森素數

此腳本將執行尋找梅森素數的程式。

```bash:/path/to/find_mersenne_prime_number.sh
#!/bin/bash
./script/data/find_mersenne_prime_number.sh
```

**說明：**

- `RUN_MIN`：設定執行時間的最小值（單位：分鐘，預設為 720 分鐘，即 12 小時）

輸出結果將保存在 `result/find_mersenne_prime_number` 資料夾內。

---

## 腳本說明

以下是本專案中各 Shell 腳本的詳細說明：

- **setup.sh**
  - **功能**：安裝並設定所有必要的依賴項，包括Clone big-int 庫、建立並編譯專案、建立 Python 虛擬環境並安裝必要的 Python 套件、下載所需字體檔案。
  
- **generate_cost_time_csv.sh**
  - **功能**：執行 `all_algorithm_cost_time` 程式，生成不同質數測試演算法在指定範圍內的運行時間數據。
  
- **generate_cost_time_figur.sh**
  - **功能**：執行 Python 腳本 `cost_time.py`，根據 `cost_time.csv` 生成效能圖表。
  
- **find_mersenne_prime_number.sh**
  - **功能**：執行 `find_mersenne_prime_number` 程式，尋找並記錄梅森素數。
  
- **generate_fermat_vs_miller_data.sh**
  - **功能**：執行 `fermat_vs_miller` 程式，生成 Fermat 與 Miller-Rabin 演算法在不同嘗試次數下的準確率數據。
  
- **generate_fermat_vs_miller_figur.sh**
  - **功能**：執行 Python 腳本 `try_times_effect.py`，根據 `fermat_vs_miller` 資料夾中的數據生成準確率圖表。

---

## 參考文獻

- Thomas H. Cormen, Charles E. Leiserson, Ronald L. Rivest and Clifford Stein, *Introduction to Algorithms*, Third Edition, The MIT Press, 2009.
- R.C.T. Lee, S.S. Tseng, R.C. Chang, and Y.T. Tsai, *Introduction to the Design and Analysis of Algorithms*, McGraw-Hill, 2005.
- Anany V. Levitin, *Introduction to the Design and Analysis of Algorithms*, 3rd Edition, Addison Wesley, 2012.
- Richard Neapolitan and Kumarss Naimipour, *Foundations of Algorithms*, Fourth Edition, Jones and Bartlett Publishers, 2010.
- [Hackerearth Primality Tests Tutorial](https://www.hackerearth.com/primality-tests-tutorial)


