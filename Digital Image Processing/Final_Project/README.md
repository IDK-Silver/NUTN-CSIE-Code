# 影像處理期末專案：古文獻影像強化系統

本專案旨在開發一個整合性的桌面應用程式，用於處理劣化的歷史文獻掃描圖檔。
系統透過一系列自動化的影像處理技術，強化影像品質、提升文字可讀性，並為後續的光學字元辨識（OCR）提供最佳化的輸入。

---

## 主要功能

* **圖形化使用者介面 (GUI)**：提供直觀的介面，方便使用者載入圖片、觸發處理並查看結果。
* **多階段影像診斷**：能自動分析輸入影像，判斷其可能存在的問題，如曝光問題、低對比度、椒鹽雜訊與影像模糊。
* **動態處理管線**：根據診斷結果，智慧地啟用對應的處理演算法，以最適切的方式處理影像。
* **並排即時比對**：在 GUI 中同時顯示「處理前」與「處理後」的影像，方便使用者立即評估處理成效。
* **結果儲存**：自動將處理後的影像，依照專案要求的格式儲存至本機。
* **中央化專案設定**：將學號、姓名、檔名格式等固定資訊集中於 `+env/config.m` 中，方便統一管理與交付。
* **Python 批次驗證**：提供獨立的 Python 腳本，可批次處理多張圖片並產生詳細的視覺化比較報告。

---

## 影像處理流程與方法

本系統的處理核心 (`functions.process_image`) 採用一個三階段的處理管線，所有演算法參數皆直接定義在對應的處理函式中，以利於快速開發與測試。

### **第一階段：影像診斷 (Diagnosis)**

系統首先對輸入的灰階影像進行一系列的分析，以判斷其品質問題。

1. **光線診斷 (`diagnose_lighting`)**:
    * **方法**: 分析影像像素的平均亮度 (`mean`) 與標準差 (`std`)。
    * **判斷**:
        * 若平均亮度 < 80，則判斷為 **`underexposed`** (曝光不足)
        * 若平均亮度 > 180，則判斷為 **`overexposed`** (過度曝光)
        * 若亮度正常但標準差 < 30，則判斷為 **`low_contrast`** (低對比度)
        * 其他情況則為 `ok`

2. **雜訊診斷 (`diagnose_noise`)**:
    * **方法**: 計算影像中純黑（值=0）與純白（值=255）像素所佔的比例。
    * **判斷**: 若比例高於一設定閾值 (0.001)，則判斷影像含有**椒鹽雜訊 (`salt-and-pepper`)**。

3. **模糊診斷 (`diagnose_blur`)**:
    * **方法**: 使用**拉普拉斯濾波 (Laplacian Filter)** 強化邊緣後，計算影像的**變異數 (Variance)**。
    * **判斷**: 若變異數低於一設定閾值 (100)，則判斷為**模糊影像**。

### **第二階段：動態前處理 (Dynamic Pre-processing)**

根據診斷報告，系統會依序（光線 -> 雜訊 -> 銳化）啟用對應的處理模組。

* **光線校正 (`correct_lighting`)**:
  * **策略**: 採用 `switch` 結構應對不同光線問題。
  * **曝光問題**: 使用**伽瑪校正 (Gamma Correction, `imadjust`)** 來提亮或壓暗影像，此方法對雜訊較不敏感。
  * **低對比度**: 使用**限制對比度之自適應直方圖均化 (CLAHE, `adapthisteq`)**，在提升局部對比度的同時能有效抑制雜訊放大。
  * **後處理**: 校正後，若影像本身雜訊較多，會使用**引導濾波器 (Guided Filter, `imguidedfilter`)** 進行一次邊緣保護的平滑處理。

* **去雜訊 (`remove_noise`)**: 若診斷出椒鹽雜訊，則使用 **3x3 的中值濾波器 (Median Filter, `medfilt2`)** 來移除雜點。

* **影像銳化 (`apply_sharpening`)**: 若診斷為模糊，則使用**遮罩銳化 (Unsharp Masking, `imsharpen`)**，並設定較高的強度 (`Amount: 5`) 來大幅強化邊緣。

### **第三階段：最終二值化與清理 (Final Binarization)**

這是產生最終 OCR 輸入的關鍵步驟，採用了複雜的形態學運算鏈以取得最乾淨的文字輪廓。

1. **自適應二值化**: 使用 `adaptthresh` 取得動態閾值圖，並透過 `imbinarize` 產生初步的黑白影像。
2. **多重形態學運算**:
    * 首先進行一次**閉合 (`imclose`)**，用於連接文字中可能斷裂的筆劃。
    * 接著進行一次**開啟 (`imopen`)**，用於移除細小的背景雜點。
    * 重複此過程以進一步精煉字體形狀。
3. **影像反相與擴張**:
    * 將影像反相，使文字為白色，背景為黑色。
    * 執行一次**擴張 (`imdilate`)**，將文字筆劃稍微加粗，使其在 OCR 辨識中更為穩固。

---

## 專案結構

```
Final_Project/
│
├── 📂 +env/
│   └── 📜 config.m             # 專案設定檔
│
├── 📂 +functions/
│   ├── 📂 +diagnose/
│   │   ├── 📜 diagnose_lighting.m
│   │   ├── 📜 diagnose_noise.m
│   │   └── 📜 diagnose_blur.m
│   ├── 📂 +process/
│   │   ├── 📜 correct_lighting.m
│   │   ├── 📜 remove_noise.m
│   │   ├── 📜 apply_sharpening.m
│   │   └── 📜 binarize_adaptive.m
│   └── 📜 process_image.m      # 核心處理管線函式
│
├── 📜 FinalProjectApp.mlapp    # GUI 主程式檔案
├── 📜 S11159005.m              # 專案啟動腳本
├── 📜 valid_script.py          # Python 批次驗證腳本
├── 📜 main_test.py             # Python 測試腳本
├── 📜 pyproject.toml           # Python 專案配置檔
└── 📜 README.md                # 本文件
```

---

## 系統需求

### MATLAB 環境

* **MATLAB**: R2024b 或更新版本
* **Toolboxes**:
  * Image Processing Toolbox™
  * Computer Vision Toolbox™ (for `imguidedfilter`)

### Python 環境 (用於批次驗證)

* **Python**: 3.8+
* **套件管理**: uv (推薦) 或 pip
* **主要依賴**:
  * `matlab.engine` - MATLAB 引擎接口
  * `pillow` - 影像處理
  * `numpy` - 數值計算
  * `matplotlib` - 視覺化
  * `pytesseract` - OCR 引擎
  * `python-levenshtein` - 字串相似度計算

---

## 安裝與設定

### 1. MATLAB GUI 版本

1. **下載專案**：將此專案的所有檔案複製到您的電腦。
2. **修改設定檔**：打開 `+env/config.m` 檔案，將 `STUDENT_ID` 和 `STUDENT_NAME` 修改為您自己的資訊。
3. **確認啟動腳本**：確保您的 `.mlapp` 檔案名稱與您的 `學號.m` 腳本中呼叫的名稱一致。

### 2. Python 批次驗證版本

使用 `uv` 套件管理器進行快速安裝：

```bash
# 安裝 uv (如果尚未安裝)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 進入專案目錄
cd Final_Project

# 安裝 Python 依賴
uv sync

# 或使用傳統方式
uv pip install matlab-engine pillow numpy matplotlib pytesseract python-levenshtein
```

**額外設定**：
* 安裝 Tesseract OCR 引擎：

  ```bash
  # macOS
  brew install tesseract
  
  # 或下載語言包
  brew install tesseract-lang
  ```

---

## 使用說明

### 1. MATLAB GUI 模式

1. 開啟 MATLAB，並將「目前資料夾」切換至本專案的根目錄。
2. 在命令視窗中輸入您的啟動腳本名稱（例如 `S11159005`）並執行。
3. 在開啟的 GUI 介面中，點擊「載入與處理」按鈕並選擇三張圖片。
4. 程式將自動執行處理，並在介面上顯示結果，同時將處理後的圖片儲存於專案根目錄。

### 2. Python 批次驗證模式

這個模式適合批次處理大量圖片，並產生包含 OCR 評估的詳細報告：

```bash
# 使用 uv 執行
uv run python valid_script.py --input_dir /path/to/test_images --output_dir output_results

# 或傳統方式執行
python valid_script.py --input_dir /path/to/test_images --output_dir output_results
```

**輸入資料夾結構**：

```
test_images/
├── image1.png
├── image2.jpg
├── image3.bmp
└── info.json    # 包含標準答案的 JSON 檔案
```

**info.json 格式範例**：

```json
{
  "image1.png": {
    "ground_truth_text": "Hello World"
  },
  "image2.jpg": {
    "ground_truth_text": "古文獻文字"
  }
}
```

**輸出結果**：

```
output_results/
├── comparison_image1.png     # 包含完整處理流程的視覺化比較圖
├── comparison_image2.png
└── step/                     # 每一步驟的個別圖片
    ├── image1/
    │   ├── 01_original.png
    │   ├── 02_grayscale.png
    │   ├── 03_after_lighting.png
    │   ├── 04_after_noise.png
    │   ├── 05_after_sharpening.png
    │   └── 06_final_binary.png
    └── image2/
        └── ...
```

### 3. 快速測試

```bash
# 快速測試單張圖片
uv run python main_test.py

# 查看詳細日誌
uv run python valid_script.py --input_dir test_images --output_dir results 2>&1 | tee processing.log
```

---

## 開發者說明

### 新增處理演算法

1. 在 `+functions/+process/` 中建立新的 `.m` 檔案
2. 在 `+functions/+diagnose/` 中建立對應的診斷函式
3. 在 `+functions/process_image.m` 中整合新的處理步驟

### Python 驗證腳本自定義

* 修改 `valid_script.py` 中的 OCR 參數
* 調整 `create_comparison_image()` 函式的視覺化設定
* 擴展 `calculate_cer()` 函式以支援其他評估指標

---

## 疑難排解

### MATLAB 相關

- **函式找不到**：確認所有 `+` 資料夾都在正確位置
* **記憶體不足**：處理大圖片時可能需要調整 MATLAB 記憶體設定

### Python 相關

- **MATLAB 引擎啟動失敗**：確認已正確安裝 MATLAB 並設定環境變數
* **OCR 無結果**：檢查 Tesseract 是否正確安裝及語言包設定
* **套件安裝問題**：使用 `uv pip list` 檢查已安裝套件
