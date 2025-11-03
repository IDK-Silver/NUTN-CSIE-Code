"""資料前處理模組"""
import json
import joblib
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """清理 DataFrame：去除欄名與字串值的前後空白，將空字串視為 NA

    Args:
        df: 原始 DataFrame

    Returns:
        pd.DataFrame: 清理後的 DataFrame
    """
    # 清理欄名
    df.columns = df.columns.str.strip()

    # 清理所有字串欄位的值
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].str.strip()
            # 將空字串轉為 NaN
            df[col] = df[col].replace('', np.nan)

    return df


def encode_holiday(df: pd.DataFrame, drop_original: bool = True) -> pd.DataFrame:
    """將 holiday 欄位編碼為 is_holiday (0/1)

    Args:
        df: 包含 holiday 欄位的 DataFrame
        drop_original: 是否移除原始 holiday 欄位

    Returns:
        pd.DataFrame: 新增 is_holiday 欄位的 DataFrame
    """
    # 非空字串 = 1，空/NA = 0
    df['is_holiday'] = df['holiday'].notna().astype(int)

    if drop_original:
        df = df.drop(columns=['holiday'])

    return df


def encode_weather(df: pd.DataFrame, categories: list[str] | None = None,
                   drop_original: bool = True) -> tuple[pd.DataFrame, list[str]]:
    """將 weather_main 欄位進行 one-hot 編碼（整數型別）

    Args:
        df: 包含 weather_main 欄位的 DataFrame
        categories: 已知的天氣類別列表（用於測試集）；若為 None 則從資料中收集（訓練集）
        drop_original: 是否移除原始 weather_main 欄位

    Returns:
        tuple: (編碼後的 DataFrame, 天氣類別列表)
    """
    if categories is None:
        # 訓練模式：收集所有類別並加上 Unknown
        unique_weather = sorted(df['weather_main'].dropna().unique().tolist())
        categories = unique_weather + ['Unknown']

    # 將未見過的天氣映射為 Unknown
    df['weather_main'] = df['weather_main'].apply(
        lambda x: x if pd.notna(x) and x in categories else 'Unknown'
    )

    # One-hot 編碼
    weather_dummies = pd.get_dummies(df['weather_main'], prefix='weather', dtype=int)

    # 確保所有類別都存在（即使某些類別在當前資料中沒有出現）
    for cat in categories:
        col_name = f'weather_{cat}'
        if col_name not in weather_dummies.columns:
            weather_dummies[col_name] = 0

    # 按照類別順序排序欄位
    weather_cols = [f'weather_{cat}' for cat in categories]
    weather_dummies = weather_dummies[weather_cols]

    # 合併回原 DataFrame
    df = pd.concat([df, weather_dummies], axis=1)

    if drop_original:
        df = df.drop(columns=['weather_main'])

    return df, categories


def convert_numeric_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """將指定欄位轉換為數值型別（容錯方式，無法解析者→NA）

    Args:
        df: DataFrame
        columns: 要轉換的欄位列表

    Returns:
        pd.DataFrame: 轉換後的 DataFrame
    """
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    return df


def add_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """加入特徵工程：時間循環、溫度分段、天氣分組、交互作用

    包含 52 個特徵：
    - 時間循環特徵（hour_sin, hour_cos）
    - 溫度分段（5 個區間）
    - 天氣分組（高流量/低流量）
    - 多項式特徵（temp², temp³）
    - 二階交互作用（Rush Hour × 其他特徵）

    Args:
        df: 包含基本特徵的 DataFrame

    Returns:
        pd.DataFrame: 加入工程特徵的 DataFrame
    """
    # ========== 1. 從 ID 提取循環時間特徵 ==========
    # ID % 24 可能代表小時循環
    df['hour_cycle'] = df['ID'] % 24
    # 將小時編碼為 sin/cos（保持循環性）
    df['hour_sin'] = np.sin(2 * np.pi * df['hour_cycle'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour_cycle'] / 24)

    # ========== 2. 溫度分段特徵（捕捉非線性）==========
    # 根據分析：極熱(300+)流量最高，極冷(<260)流量最低
    df['temp_extreme_cold'] = (df['temp'] < 260).astype(int)  # <260K
    df['temp_cold'] = ((df['temp'] >= 260) & (df['temp'] < 270)).astype(int)  # 260-270K
    df['temp_moderate'] = ((df['temp'] >= 270) & (df['temp'] < 290)).astype(int)  # 270-290K
    df['temp_warm'] = ((df['temp'] >= 290) & (df['temp'] < 300)).astype(int)  # 290-300K
    df['temp_extreme_hot'] = (df['temp'] >= 300).astype(int)  # >=300K

    # ========== 3. 天氣分組特徵 ==========
    # 高流量天氣：Smoke, Clouds, Haze, Rain, Drizzle
    df['weather_high_traffic'] = (
        df.get('weather_Smoke', 0) +
        df.get('weather_Clouds', 0) +
        df.get('weather_Haze', 0) +
        df.get('weather_Rain', 0) +
        df.get('weather_Drizzle', 0)
    ).astype(int)

    # 低流量天氣：Mist, Fog, Squall
    df['weather_low_traffic'] = (
        df.get('weather_Mist', 0) +
        df.get('weather_Fog', 0) +
        df.get('weather_Squall', 0)
    ).astype(int)

    # ========== 4. 多項式特徵（溫度）==========
    df['temp_squared'] = df['temp'] ** 2
    df['temp_cubed'] = df['temp'] ** 3

    # ========== 5. 二階交互作用 ==========
    # Rush Hour × temp
    df['rush_temp'] = df['Rush Hour'] * df['temp']

    # Rush Hour × 溫度分段
    df['rush_extreme_hot'] = df['Rush Hour'] * df['temp_extreme_hot']
    df['rush_extreme_cold'] = df['Rush Hour'] * df['temp_extreme_cold']

    # Rush Hour × 天氣分組
    df['rush_weather_high'] = df['Rush Hour'] * df['weather_high_traffic']
    df['rush_weather_low'] = df['Rush Hour'] * df['weather_low_traffic']

    # Rush Hour × hour_cycle
    df['rush_hour_cycle'] = df['Rush Hour'] * df['hour_cycle']

    # 溫度分段 × 天氣分組
    df['hot_high_weather'] = df['temp_extreme_hot'] * df['weather_high_traffic']
    df['cold_low_weather'] = df['temp_extreme_cold'] * df['weather_low_traffic']

    # ========== 6. 原有交互作用（保留重要的）==========
    # Rush Hour × weather（所有天氣欄位）
    weather_cols = [col for col in df.columns if col.startswith('weather_')
                   and not col.endswith('_traffic')]
    for weather_col in weather_cols:
        df[f'rush_{weather_col}'] = df['Rush Hour'] * df[weather_col]

    # temp × 重要天氣
    important_weather = ['weather_Clouds', 'weather_Clear', 'weather_Mist',
                        'weather_Rain', 'weather_Fog']
    for weather_col in important_weather:
        if weather_col in df.columns:
            df[f'temp_{weather_col}'] = df['temp'] * df[weather_col]

    # ========== 7. 移除無用特徵 ==========
    useless_features = ['rain_1h', 'snow_1h', 'hour_cycle']  # hour_cycle 已轉為 sin/cos
    df = df.drop(columns=useless_features, errors='ignore')

    return df


def preprocess_train(raw_path: str = "blob/raw/traffic_train.csv",
                    output_path: str = "blob/process/train_processed.csv",
                    meta_path: str = "blob/process/meta/weather_categories.json",
                    scaler_path: str = "blob/process/meta/scaler.joblib",
                    drop_holiday: bool = True) -> None:
    """前處理訓練資料（含正規化）

    Args:
        raw_path: 原始訓練資料路徑
        output_path: 輸出處理後資料路徑
        meta_path: 輸出天氣類別 JSON 路徑
        scaler_path: 輸出 StandardScaler 路徑
        drop_holiday: 是否移除原始 holiday 欄位
    """
    print("=== 訓練資料前處理 ===")

    # 讀取資料（使用 skipinitialspace 處理對齊空白）
    df = pd.read_csv(raw_path, skipinitialspace=True)
    print(f"原始資料形狀: {df.shape}")

    # 清理資料
    df = clean_dataframe(df)

    # 編碼 holiday
    df = encode_holiday(df, drop_original=drop_holiday)

    # 編碼 weather_main（訓練模式）
    df, weather_categories = encode_weather(df, categories=None, drop_original=True)

    # 轉換數值欄位
    numeric_cols = ['temp', 'rain_1h', 'snow_1h', 'clouds_all', 'Rush Hour', 'traffic_volume']
    df = convert_numeric_columns(df, numeric_cols)

    # 移除包含缺失值的資料列
    df_before = len(df)
    df = df.dropna()
    df_after = len(df)
    print(f"移除缺失值: {df_before} → {df_after} (移除 {df_before - df_after} 筆)")

    # ⭐ 特徵工程：加入多項式與交互作用（在正規化前）
    df = add_feature_engineering(df)
    print(f"✓ 特徵工程完成")

    # 正規化數值特徵（不包含 traffic_volume 和 ID）
    # 包含連續數值特徵，不包含 0/1 的分類特徵
    feature_cols = ['temp', 'clouds_all', 'Rush Hour',
                   'hour_sin', 'hour_cos',
                   'temp_squared', 'temp_cubed', 'rush_temp',
                   'rush_hour_cycle']
    scaler = StandardScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])
    print(f"✓ 數值特徵已正規化（共 {len(feature_cols)} 個）")

    # 調整欄位順序（簡化：按原有順序，只確保 ID 和 traffic_volume 在首尾）
    # 將 ID 移到最前，traffic_volume 移到最後
    cols = df.columns.tolist()
    cols.remove('ID')
    cols.remove('traffic_volume')
    final_cols = ['ID'] + cols + ['traffic_volume']
    df = df[final_cols]

    # 確保輸出目錄存在
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(meta_path).parent.mkdir(parents=True, exist_ok=True)
    Path(scaler_path).parent.mkdir(parents=True, exist_ok=True)

    # 儲存處理後的資料
    df.to_csv(output_path, index=False)

    # 儲存天氣類別
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(weather_categories, f, ensure_ascii=False, indent=2)

    # 儲存 Scaler
    joblib.dump(scaler, scaler_path)

    print(f"✓ 訓練資料前處理完成")
    print(f"  - 輸出: {output_path}")
    print(f"  - 天氣類別: {meta_path}")
    print(f"  - Scaler: {scaler_path}")
    print(f"  - 資料形狀: {df.shape}")
    print(f"  - 天氣類別數: {len(weather_categories)}")


def preprocess_test(raw_path: str = "blob/raw/traffic_test.csv",
                   output_path: str = "blob/process/test_processed.csv",
                   meta_path: str = "blob/process/meta/weather_categories.json",
                   scaler_path: str = "blob/process/meta/scaler.joblib",
                   drop_holiday: bool = True) -> None:
    """前處理測試資料（使用訓練時的 scaler）

    Args:
        raw_path: 原始測試資料路徑
        output_path: 輸出處理後資料路徑
        meta_path: 天氣類別 JSON 路徑
        scaler_path: StandardScaler 路徑
        drop_holiday: 是否移除原始 holiday 欄位
    """
    print("=== 測試資料前處理 ===")

    # 讀取天氣類別
    with open(meta_path, 'r', encoding='utf-8') as f:
        weather_categories = json.load(f)

    # 讀取 Scaler
    scaler = joblib.load(scaler_path)
    print(f"✓ 載入 Scaler: {scaler_path}")

    # 讀取資料
    df = pd.read_csv(raw_path, skipinitialspace=True)
    print(f"原始資料形狀: {df.shape}")

    # 清理資料
    df = clean_dataframe(df)

    # 編碼 holiday
    df = encode_holiday(df, drop_original=drop_holiday)

    # 編碼 weather_main（測試模式，使用既有類別）
    df, _ = encode_weather(df, categories=weather_categories, drop_original=True)

    # 轉換數值欄位（測試集沒有 traffic_volume）
    numeric_cols = ['temp', 'rain_1h', 'snow_1h', 'clouds_all', 'Rush Hour']
    df = convert_numeric_columns(df, numeric_cols)

    # 處理測試集的缺失值（用 0 填充以避免資料遺失）
    feature_cols_before = ['temp', 'rain_1h', 'snow_1h', 'clouds_all', 'Rush Hour']
    df[feature_cols_before] = df[feature_cols_before].fillna(0)
    print(f"✓ 缺失值已填充為 0")

    # ⭐ 特徵工程：加入多項式與交互作用（與訓練集相同）
    df = add_feature_engineering(df)
    print(f"✓ 特徵工程完成")

    # 使用訓練時的 Scaler 正規化（與訓練集相同的特徵）
    feature_cols = ['temp', 'clouds_all', 'Rush Hour',
                   'hour_sin', 'hour_cos',
                   'temp_squared', 'temp_cubed', 'rush_temp',
                   'rush_hour_cycle']
    df[feature_cols] = scaler.transform(df[feature_cols])
    print(f"✓ 數值特徵已正規化（共 {len(feature_cols)} 個）")

    # 調整欄位順序（ID 在最前）
    cols = df.columns.tolist()
    cols.remove('ID')
    final_cols = ['ID'] + cols
    df = df[final_cols]

    # 確保輸出目錄存在
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # 儲存處理後的資料
    df.to_csv(output_path, index=False)

    print(f"✓ 測試資料前處理完成")
    print(f"  - 輸出: {output_path}")
    print(f"  - 資料形狀: {df.shape}")
