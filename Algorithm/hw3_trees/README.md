
# Trees

## 目錄
- [安裝與設定](#安裝與設定)
  - [前置條件](#前置條件)
  - [安裝步驟](#安裝步驟)
- [使用方法](#使用方法)
  - [執行主程式](#執行主程式)

---

## 安裝與設定

### 前置條件

請確保您的系統已安裝以下軟體：

- **C++ 編譯器**（如 `g++`）
- **CMake**（用於構建專案）
- **Make**（構建自動化工具）

### 安裝步驟

1. **克隆本專案**

   ```bash
   git clone https://github.com/IDK-Silver/NUTN-CSIE-Code.git
   cd NUTN-CSIE-Code/Algorithm/hw3_trees/
   ```

2. **建立並進入建構目錄**

   ```bash
   mkdir build
   cd build
   ```

3. **運行 CMake 並編譯專案**

   ```bash
   cmake ..
   make
   ```

   這將生成可執行檔 `TreeSearch`。


4. **複製測試資料 & 複製執行腳本 & 賦予執行權限**

   ```bash
    cp -r ../dataset ./
    cp -r ../script/* ./
    chmod +x acc_fail.bash
    chmod +x acc_full.bash
   ```
5. **執行腳本**

   ```bash
    ./acc_fail.bash
    ./acc_full.bash
   ```
   這將生成 `acc_fail.csv (有查詢失敗的)` 以及 `acc_full.csv(沒有查詢失敗的)`

---

## 使用方法

### 執行主程式

```bash:/src/main.cpp
./TreeSearch [選項]
```

**選項說明：**

- `--help`  
  顯示幫助訊息。

- `--input <檔案路徑>`  
  指定建構字典的輸入檔。

- `--test <檔案路徑>`  
  輸入到字典查詢的範例檔案

- `--output <檔案路徑>`  
  報告檔 - csv format (可以累加)

- `--target <標記>` `option`  
  報告檔 - 的批次標記

**範例：**

```bash
./TreeSearch --input dataset/fail/FailTest_in.txt --test dataset/fail/FailTest_1.txt --output acc_fail.csv --target acc_fail_1
```
