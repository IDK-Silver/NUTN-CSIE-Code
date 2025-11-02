"""模型訓練模組"""
import json
import joblib
from pathlib import Path
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import numpy as np

from .utils import get_timestamp, ensure_dir
from .registry import update_registry


def train_model(input_path: str = 'blob/process/train_processed.csv',
                runs_dir: str = 'blob/models/runs',
                latest_dir: str = 'blob/models/latest',
                registry_path: str = 'blob/models/registry.json',
                test_size: float = 0.2,
                random_state: int = 42) -> str:
    """訓練 LinearRegression 模型並儲存（含指標評估）

    Args:
        input_path: 前處理後的訓練資料路徑
        runs_dir: 時間戳模型儲存目錄
        latest_dir: 最新模型儲存目錄
        registry_path: 模型註冊檔路徑
        test_size: 驗證集比例（預設 0.2 = 20%）
        random_state: 隨機種子（確保可重現）

    Returns:
        str: 模型檔案路徑
    """
    # 讀取訓練資料
    df = pd.read_csv(input_path)

    # 分離特徵與目標
    X = df.drop(columns=['ID', 'traffic_volume'])
    y = df['traffic_volume']

    print(f"原始資料形狀: X={X.shape}, y={y.shape}")

    # 分割訓練集和驗證集
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    print(f"訓練集: {X_train.shape}, 驗證集: {X_val.shape}")

    # 建立 LinearRegression 模型
    model = LinearRegression()
    print("使用模型: LinearRegression")

    # 訓練模型
    print("開始訓練...")
    model.fit(X_train, y_train)
    print("✓ 訓練完成")

    # 計算訓練集指標
    y_train_pred = model.predict(X_train)
    train_r2 = r2_score(y_train, y_train_pred)
    train_mse = mean_squared_error(y_train, y_train_pred)
    train_rmse = np.sqrt(train_mse)
    train_mae = mean_absolute_error(y_train, y_train_pred)

    # 計算驗證集指標
    y_val_pred = model.predict(X_val)
    val_r2 = r2_score(y_val, y_val_pred)
    val_mse = mean_squared_error(y_val, y_val_pred)
    val_rmse = np.sqrt(val_mse)
    val_mae = mean_absolute_error(y_val, y_val_pred)

    # 顯示指標
    print("\n=== 訓練集指標 ===")
    print(f"  R² Score:  {train_r2:.6f}")
    print(f"  RMSE:      {train_rmse:.2f}")
    print(f"  MAE:       {train_mae:.2f}")
    print(f"  MSE:       {train_mse:.2f}")

    print("\n=== 驗證集指標 ===")
    print(f"  R² Score:  {val_r2:.6f}")
    print(f"  RMSE:      {val_rmse:.2f}")
    print(f"  MAE:       {val_mae:.2f}")
    print(f"  MSE:       {val_mse:.2f}")

    # 準備儲存路徑
    timestamp = get_timestamp()
    run_name = f"{timestamp}-linear"
    run_dir = Path(runs_dir) / run_name
    ensure_dir(run_dir)

    model_path = run_dir / 'model.joblib'
    metrics_path = run_dir / 'metrics.json'

    # 儲存模型
    joblib.dump(model, model_path)
    print(f"\n✓ 模型已儲存: {model_path}")

    # 儲存指標
    metrics = {
        'train': {
            'r2': float(train_r2),
            'rmse': float(train_rmse),
            'mae': float(train_mae),
            'mse': float(train_mse),
            'samples': int(len(X_train))
        },
        'validation': {
            'r2': float(val_r2),
            'rmse': float(val_rmse),
            'mae': float(val_mae),
            'mse': float(val_mse),
            'samples': int(len(X_val))
        },
        'config': {
            'test_size': test_size,
            'random_state': random_state
        }
    }

    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"✓ 指標已儲存: {metrics_path}")

    # 複製到 latest
    ensure_dir(latest_dir)
    latest_path = Path(latest_dir) / 'model.joblib'
    latest_metrics_path = Path(latest_dir) / 'metrics.json'
    joblib.dump(model, latest_path)
    with open(latest_metrics_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"✓ 最新模型副本: {latest_path}")

    # 更新註冊檔
    update_registry(
        registry_path=registry_path,
        model_path=str(model_path),
        timestamp=timestamp,
        model_name='linear',
        metrics=metrics
    )
    print(f"✓ 註冊檔已更新: {registry_path}")

    return str(model_path)
