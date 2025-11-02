# Mid-Project Data Pipeline Design (KISS)

本文件說明以 KISS 原則設計的資料前處理、訓練、與提交產物管理規劃。重點：簡單可用、可重現、易於擴充，但避免過度工程。

## 目標
- 僅處理必要欄位，保持簡單與穩定。
- 前處理一致化（train 決定、test 對齊）。
- 只使用回歸模型；訓練成果只保存「模型本體」。
- 結果與模型皆以時間戳管理多版本；提供 latest/best 的快捷副本與索引。

## 資料假設（輸入欄位）
- CSV 欄位（順序可能不同，但名稱一致）：
  - `ID, holiday, temp, rain_1h, snow_1h, clouds_all, weather_main, Rush Hour[, traffic_volume]`
- `holiday` 只有空值或節日字串。
- `weather_main` 為多個不同天氣字串。
- `traffic_volume` 僅存在於訓練資料。

> CSV 內有對齊空白，讀檔時需去除欄名與字串值前後空白（如 `skipinitialspace` 與 `str.strip()`）。

## 目錄結構
```
blob/
  raw/
    traffic_train.csv
    traffic_test.csv
  process/
    train_processed.csv
    test_processed.csv
    meta/
      weather_categories.json
  models/
    runs/
      <YYYYMMDD-HHMMSS>-<model>/
        model.joblib
    latest/
      model.joblib
    best/
      model.joblib
    registry.json
  submit/
    runs/
      <YYYYMMDD-HHMMSS>/
        submission.csv
    latest/
      submission.csv   # 可選，作為方便存取的拷貝
```

## 前處理規範（train/test 共用邏輯）
- 清理：
  - 欄名去前後空白；所有字串欄位值去前後空白。
  - 將空字串視為缺值（NA）。
  - 移除包含缺失值的資料列（確保訓練資料完整）。
- `holiday` → `is_holiday`：
  - 非空字串 = 1；空/NA = 0。
  - 預設移除原 `holiday` 欄，只保留 `is_holiday`（0/1）。
- `weather_main` → one-hot：
  - train：從資料收集完整類別，另加保留類別 `Unknown`，輸出至 `blob/process/meta/weather_categories.json`。
  - test：讀取上述類別，未見過的字串映射為 `Unknown`，確保欄位一致。
  - 轉換為整數型別（0/1）而非布林值。
- 數值欄位轉型與正規化：
  - `temp, rain_1h, snow_1h, clouds_all, Rush Hour[, traffic_volume]` 以「可容錯的方式」轉為數值（不丟例外，無法解析者→NA）。
  - **正規化（StandardScaler）**：
    - train：對數值特徵 `temp, rain_1h, snow_1h, clouds_all, Rush Hour` 進行標準化（均值0、標準差1），儲存 scaler 至 `blob/process/meta/scaler.joblib`。
    - test：使用訓練時的 scaler 進行相同轉換，確保一致性。
    - `traffic_volume`（目標變數）不正規化。
- 欄位與順序（輸出）：
  - 共通：`ID, temp, rain_1h, snow_1h, clouds_all, Rush Hour, is_holiday, weather_*...`
  - 訓練多 `traffic_volume` 於最後。
  - `weather_*` 欄位順序以 `weather_categories.json` 為準，穩定一致。

## 模型訓練（Regression Only）
- 模型：僅使用 `LinearRegression`（最簡單的線性迴歸模型）。
- 資料已於前處理階段完成正規化，直接訓練即可。
- 輸入：`blob/process/train_processed.csv`。
- 輸出：
  - `blob/models/runs/<YYYYMMDD-HHMMSS>-linear/model.joblib`（僅此單一檔）。
  - 複製一份至 `blob/models/latest/model.joblib` 方便下游使用。

## registry.json（最小索引）
- 位置：`blob/models/registry.json`
- 目的：追蹤 `latest`、`best` 與簡易歷史，不在 run 目錄保存 metrics/config。
- 建議最小結構：
```json
{
  "latest": "blob/models/runs/20251102-154530-linear/model.joblib",
  "best": "blob/models/runs/20251102-154530-linear/model.joblib",
  "history": [
    { "run": "blob/models/runs/20251102-154530-linear/model.joblib", "ts": "20251102-154530", "model": "linear" }
  ]
}
```
- 更新策略：
  - `latest`：每次訓練成功後更新。
  - `best`：僅在明確標記（mark-best）時更新；不自動等於 latest。

## 預測與提交
- 模型選擇：
  - 預設使用 `latest`；也可指定使用 `best` 或某個具體 run 路徑。
- 輸入：`blob/process/test_processed.csv`。
- 輸出：
  - `blob/submit/runs/<YYYYMMDD-HHMMSS>/submission.csv`（欄位：`ID, traffic_volume`）。
  - 可選同步複製至 `blob/submit/latest/submission.csv` 以方便存取（不影響歷史保留）。

## 工作流程（CLI 規劃草案）
- `preprocess --mode {train,test} --drop-holiday`：
  - train：產出 `train_processed.csv`、`meta/weather_categories.json` 與 `meta/scaler.joblib`。
  - test：讀取 `meta/weather_categories.json` 與 `meta/scaler.joblib` 產出 `test_processed.csv`。
- `train`：
  - 使用 LinearRegression 訓練模型，保存 `model.joblib`，更新 `latest` 與 `registry.json`。
- `mark-best --run <path-to-model.joblib>`：
  - 將指定 run 設為 `best`，並複製至 `blob/models/best/model.joblib`、更新 `registry.json`。
- `predict --run {latest|best|<run-path>}`：
  - 依時間戳建立輸出資料夾並寫出 `submission.csv`；可選更新 `submit/latest` 拷貝。

> 現階段不引入 Snakemake。若未來流程擴張（多資料源、特徵版本、CV、上傳），再補簡易 Snakefile。

## 時間戳與命名
- 時間戳格式：`YYYYMMDD-HHMMSS`（本地時間）。
- run 目錄：`<timestamp>-linear`，如：`20251102-154530-linear/`。
- 提交輸出：`blob/submit/runs/<timestamp>/submission.csv`。

## 總結（決策）
- 前處理：
  - `holiday`→`is_holiday`（移除原欄）
  - `weather_main` one-hot（未知→`Unknown`，類別由 train 決定並固化，轉為整數 0/1）
  - 移除包含缺失值的資料列
  - **數值特徵正規化（StandardScaler）**，scaler 儲存供測試集使用
- 模型：僅使用 `LinearRegression`，run 僅保存 `model.joblib`。
- 版本：所有產出以時間戳管理；保留 `latest` 快捷；`best` 需顯式 `mark-best` 才會更新。
- 提交：每次生成一個時間戳版本，可選維護 `submit/latest` 以便使用。

以上規劃符合 KISS 原則，使用最簡單的線性迴歸模型搭配標準資料前處理流程。
