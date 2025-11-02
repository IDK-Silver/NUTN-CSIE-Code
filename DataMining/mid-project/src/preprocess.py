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

    # 正規化數值特徵（不包含 traffic_volume 和 ID）
    feature_cols = ['temp', 'rain_1h', 'snow_1h', 'clouds_all', 'Rush Hour']
    scaler = StandardScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])
    print(f"✓ 數值特徵已正規化: {feature_cols}")

    # 調整欄位順序：ID, 數值欄位, is_holiday, weather_*..., traffic_volume
    weather_cols = [col for col in df.columns if col.startswith('weather_')]
    base_cols = ['ID', 'temp', 'rain_1h', 'snow_1h', 'clouds_all', 'Rush Hour', 'is_holiday']
    target_col = ['traffic_volume']

    final_cols = base_cols + weather_cols + target_col
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
    feature_cols = ['temp', 'rain_1h', 'snow_1h', 'clouds_all', 'Rush Hour']
    df[feature_cols] = df[feature_cols].fillna(0)
    print(f"✓ 缺失值已填充為 0")

    # 使用訓練時的 Scaler 正規化
    df[feature_cols] = scaler.transform(df[feature_cols])
    print(f"✓ 數值特徵已正規化: {feature_cols}")

    # 調整欄位順序：ID, 數值欄位, is_holiday, weather_*...
    weather_cols = [col for col in df.columns if col.startswith('weather_')]
    base_cols = ['ID', 'temp', 'rain_1h', 'snow_1h', 'clouds_all', 'Rush Hour', 'is_holiday']

    final_cols = base_cols + weather_cols
    df = df[final_cols]

    # 確保輸出目錄存在
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # 儲存處理後的資料
    df.to_csv(output_path, index=False)

    print(f"✓ 測試資料前處理完成")
    print(f"  - 輸出: {output_path}")
    print(f"  - 資料形狀: {df.shape}")
