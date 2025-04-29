# 113-2 影像處理期中專案

**作者:** Yu-Feng

**程式碼連結:** [點擊此處查看專案程式碼](https://github.com/IDK-Silver/NUTN-CSIE-Code/tree/main/Digital%20Image%20Processing/Midterm_Project/)

## 專案描述

在攝影過程中，受限於拍照設備、環境光線、拍攝角度等因素，照片成像時常會出現失真、細節丟失、雜訊干擾等問題。為了解決這些問題，影像處理中的影像強化（Enhancement）與濾波（Filtering）等功能扮演了關鍵角色，能協助修復這些不完美的照片。

本專案旨在開發一個智能圖像處理系統，此系統能夠自動分析讀入圖片的特性與潛在問題，並根據分析結果，應用適當的影像處理技術進行修復與增強。系統會首先識別圖像的基本屬性（如彩色/灰階、解析度、色彩深度等），接著診斷影像中可能存在的缺陷（如不同類型的雜訊、模糊程度與類型、曝光問題、色彩偏差等）。最後，系統會根據診斷出的問題，自動選擇最適合的處理技術進行疊代處理，直到影像品質不再顯著提升或達到預設的處理上限。處理完成後，系統會將分析結果與處理後的影像檔案寫入指定的資料夾，並提供處理前後的比較資訊。

## 系統功能詳述

本系統包含以下主要功能模組與處理流程：

### 1. 影像分析模組 (Image Analysis Module)

* **屬性識別 (Attribute Identification):** 利用 MATLAB 的 `imfinfo` 函數及影像本身的維度資訊，判斷輸入影像為彩色（Color）或灰階（Grayscale），並記錄其解析度（寬 x 高）與色彩深度（每個像素的位元數）。
* **清晰度分析 (Clarity Analysis):** 在 `detect_blur_type` 函數中，透過計算影像灰階版本的**拉普拉斯響應變異數 (Variance of Laplacian Response)**，得到一個量化的清晰度指標 `clarity_value`。此數值越低，表示影像可能越模糊。影像的對比度則主要透過後續的曝光修正與灰階增強步驟來改善。

### 2. 問題診斷模組 (Problem Diagnostics Module)

* **雜訊偵測 (`detect_noise_type`):** 運用啟發式方法（Heuristics）估計影像中主要的雜訊類型：
  * **椒鹽噪聲 (Salt & Pepper):** 檢查影像中極端亮（接近全白）與極端暗（接近全黑）像素所佔的比例，若超過閾值 (`salt_pepper_frac_thresh`)，則判定為椒鹽噪聲。
  * **高斯噪聲 (Gaussian):** 計算原始影像與經過高斯濾波器平滑後影像的差值（高頻成分），若此差值影像的標準差（轉換回 0-255 範圍）超過閾值 (`gaussian_noise_std_thresh`)，則判定為高斯噪聲。
  * **泊松噪聲 (Poisson):** （簡化檢測）檢查標準化後（0-1 範圍）影像的像素值變異數（Variance）是否約略等於其平均值（Mean），若兩者相對差異在容許範圍內 (`poisson_rel_diff_thresh`)，則可能判定為泊松噪聲。
  * **無明顯噪聲 (None):** 若以上條件皆不滿足。

* **模糊類型偵測 (`detect_blur_type`):** 估計影像模糊的類型：
  * 首先檢查清晰度指標 `clarity_value` 是否低於閾值 (`blur_clarity_threshold`)，若低於則認為影像有模糊。
  * 若影像模糊，則進一步分析其水平與垂直方向的**梯度總和**。如果兩個方向的梯度總和差異比例超過閾值 (`motion_gradient_ratio_threshold`)，則判定為 **運動模糊 (Motion)**，並根據梯度較弱的方向判斷運動是水平或垂直。
  * 若梯度差異不大，則判定為 **高斯模糊 (Gaussian)** 或一般性的失焦模糊。
  * 若清晰度指標高於閾值，則判定為 **無模糊 (None)**。

* **曝光問題偵測 (`detect_exposure_problem`):** 判斷影像是否有曝光問題：
  * **曝光不足 (Underexposed):** 若影像的平均亮度低於閾值 (`underexposure_mean_thresh`)，或者極暗像素的比例超過飽和閾值 (`saturation_percent_thresh`)。
  * **過度曝光 (Overexposed):** 若影像的平均亮度高於閾值 (`overexposure_mean_thresh`)，或者極亮像素的比例超過飽和閾值 (`saturation_percent_thresh`)。
  * **正常曝光 (Normal):** 若以上條件皆不滿足。

* **色彩問題偵測 (`detect_color_problem`):** 判斷影像的色彩狀態：
  * **灰階 (Grayscale):** 若影像為二維矩陣或第三維度為 1。
  * **色偏 (Color Cast):** 若影像為三維彩色影像，且其 R, G, B 三個色彩通道的像素平均值之間的相對偏差超過閾值 (`color_cast_rel_dev_threshold`)。
  * **正常色彩 (Normal):** 若為彩色影像且未檢測到明顯色偏。

### 3. 自動化疊代處理流程 (Automated Iterative Processing)

* **核心邏輯:** 系統採用疊代（Iteration）方式處理影像。在每一輪疊代開始時，都會重新分析當前影像的狀態（雜訊、模糊、曝光、色彩）。
* **修正策略:** 在每一輪疊代中，系統會**嘗試修正所有當前偵測到的問題**。修正會依照固定的優先順序執行（這個順序有助於避免處理間的互相干擾）：**去噪 -> 去模糊/銳化 -> 修正曝光 -> 修正色彩/灰階增強**。
* **停止條件:** 疊代處理會在以下任一條件滿足時停止：
  * 在新一輪疊代開始時，未偵測到任何需要修正的問題。
  * 在完成一整輪的所有修正嘗試後，影像與該輪開始時相比**沒有任何變化**（表示處理已收斂或無法進一步改善）。
  * 達到了設定的**最大疊代次數 (`max_iterations`)**。

* **採用的修正技術詳述 (Processing Functions):**
  * **`denoise_image` (去噪):**
    * 若偵測到 `saltpepper` 或 `poisson` 噪聲，則使用 **中值濾波器 (`medfilt2`)** 進行處理。對於泊松噪聲，中值濾波是基於課程文件可選方法的替代方案，效果可能不如專用演算法。
    * 若偵測到 `gaussian` 噪聲，則使用 **高斯濾波器 (`imgaussfilt`)** 進行平滑去噪。
  
  * **`deblur_image` (去模糊/銳化):**
    * 若偵測到 `motion` 或 `gaussian` 模糊，系統**不執行**複雜的反卷積（Deconvolution）去模糊，而是採用課程文件中提到的基於 **拉普拉斯算子 (`fspecial('laplacian', ...)` + `imfilter`)** 的 **影像銳化 (Sharpening)** 技術。這類似於高提升濾波（High Boost Filtering）的概念，透過增強影像的邊緣和高頻細節來主觀上對抗模糊感。銳化強度可透過 `sharpening_strength` 參數調整。
  
  * **`fix_exposure` (修正曝光):**
    * 若偵測到 `Underexposed`，則應用 **Gamma 校正 (Gamma Correction)**（指數小於 1）來提升整體亮度與暗部細節。Gamma 值可在函數內調整 (`gamma_underexposed`)。
    * 若偵測到 `Overexposed`，則使用 **強度調整 (`imadjust`)** 函數來壓縮亮部像素的範圍，減少過曝區域的細節損失。
  
  * **`fix_color_problem` (修正色彩/灰階增強):**
    * 若影像為 `grayscale`，則在**第一輪**疊代時應用 **直方圖等化 (`histeq`)** 來提升對比度與視覺清晰度。
    * 若偵測到 `color_cast`，則使用簡易的 **通道延展 (Channel Stretching)** 方法進行色彩校正：將 R, G, B 三個通道的像素值分別線性延展（正規化）到 [0, 1] 範圍，以嘗試消除色偏，達到簡易的自動白平衡效果。

### 4. 輸出檔案 (Output Generation)

* **處理後影像:** 將最終處理完成的影像儲存為 **JPEG 格式**，檔名為 `A<原始檔名>.jpg`（例如：`A01.jpg`），儲存於 `result_subdir` 指定的目錄中。JPEG 的儲存品質設定為 95。
* **比較說明文件:** 為每張圖片生成一個同名的 `.txt` 檔案，檔名為 `A<原始檔名>.txt`（例如：`A01.txt`），儲存於 `result_subdir` 目錄。此文件內容包含：
  * 原始影像的分析結果（雜訊、模糊、曝光、色彩狀態）。
  * 疊代過程中實際應用的處理步驟列表。
  * 最終處理完成後影像的分析結果。
* **比較圖 (可選):** 如果設定 `generate_comparison_png = true`，則會額外儲存一個 **PNG 格式** 的比較圖，檔名為 `<原始檔名>_comparison.png`。圖中會並列顯示原始影像與最終處理後的影像，並在標題中簡要顯示分析結果與處理步驟摘要。灰階影像在此比較圖中會以灰階色彩映射顯示。

## 系統需求 (Requirements)

* MATLAB R2019b 或更新版本 (因程式碼中使用了 `im2gray` 函數)。
* 已安裝 MATLAB Image Processing Toolbox™。

## 使用說明 (Usage)

1. **放置影像檔案:** 將要處理的輸入影像檔案 (`01.jpg`, `02.jpg`, `03.jpg`) 放置在與 `.m` 腳本相同的目錄下，或者放置在 `search_dirs` 變數中定義的任何一個子目錄內（例如 `res`、`images` 等）。

2. **設定組態參數:** 開啟 `.m` 腳本檔案，找到 "CONFIGURATION SECTION"，根據需要修改以下參數：
   * `image_names`: 指定要處理的影像檔名列表。
   * `search_dirs`: 指定搜尋輸入影像的子目錄列表。
   * `result_subdir`: 指定儲存輸出結果的子目錄名稱（設為 `.` 表示儲存在腳本所在的當前目錄）。
   * `max_iterations`: 設定每張圖片進行疊代處理的最大次數。
   * `generate_comparison_png`: 設定為 `true` 以生成比較圖檔，設為 `false` 則不生成。

3. **執行腳本:** 在 MATLAB 環境中執行此 `.m` 腳本檔案。

4. **檢查輸出結果:** 執行完畢後，檢查 `result_subdir` 所指定的目錄。您應該會找到處理後的 `A*.jpg` 影像檔案和 `A*.txt` 比較說明文件。如果 `generate_comparison_png` 設為 `true`，還會找到 `*_comparison.png` 比較圖檔。同時，留意 MATLAB 命令視窗中輸出的處理過程日誌與可能的警告訊息。
