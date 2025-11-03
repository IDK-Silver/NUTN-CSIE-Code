"""Polynomial Regression 訓練模組"""
import json
import joblib
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from .utils import get_timestamp, ensure_dir
from .registry import update_registry


def select_important_features(X: pd.DataFrame, top_k: int = 15) -> list:
    """選擇最重要的特徵（基於數據分析的相關性排序）

    從 blob/analysis/feature_importance.json 讀取特徵重要性分析結果，
    選擇與 traffic_volume 相關性最高的前 k 個特徵。

    Args:
        X: 特徵 DataFrame
        top_k: 選擇前 k 個特徵

    Returns:
        list: 選中的特徵名稱列表
    """
    import json
    from pathlib import Path

    # 讀取特徵重要性分析結果
    analysis_path = Path('blob/analysis/feature_importance.json')

    if not analysis_path.exists():
        print("⚠️  未找到特徵重要性分析結果，使用預設特徵選擇")
        print("   請先執行: uv run python analyze_features.py")

        # 降級方案：使用基本核心特徵
        fallback_features = [
            'Rush Hour', 'rush_temp', 'rush_hour_cycle',
            'temp', 'clouds_all', 'is_holiday'
        ]
        selected = [f for f in fallback_features if f in X.columns]
        return selected[:top_k]

    # 從分析結果載入推薦特徵
    with open(analysis_path, 'r', encoding='utf-8') as f:
        analysis = json.load(f)

    # 優先使用智能選擇（針對多項式回歸優化）
    if f'smart_{top_k}' in analysis['recommendations']:
        recommended = analysis['recommendations'][f'smart_{top_k}']
        print(f"✓ 使用智能特徵選擇（針對 Polynomial Regression 優化）")
    elif f'top_{top_k}' in analysis['recommendations']:
        recommended = analysis['recommendations'][f'top_{top_k}']
        print(f"✓ 使用數據驅動的特徵選擇（基於相關性分析）")
    else:
        # 如果沒有對應的推薦，從 correlations 取前 k 個
        recommended = [item['feature'] for item in analysis['correlations'][:top_k]]
        print(f"✓ 使用相關性排序的前 {top_k} 個特徵")

    # 過濾出實際存在於 X 中的特徵
    selected = [feat for feat in recommended if feat in X.columns]

    print(f"  選擇前 {top_k} 個特徵: {', '.join(selected[:5])}...")

    return selected[:top_k]


def train_polynomial_model(input_path: str = 'blob/process/train_processed.csv',
                          runs_dir: str = 'blob/models/runs',
                          latest_dir: str = 'blob/models/latest',
                          registry_path: str = 'blob/models/registry.json',
                          degree: int = 2,
                          interaction_only: bool = False,
                          use_feature_selection: bool = True,
                          top_k_features: int = 15,
                          use_full_train: bool = True,
                          test_size: float = 0.2,
                          random_state: int = 42) -> str:
    """訓練 Polynomial Regression 模型

    Args:
        input_path: 前處理後的訓練資料路徑
        runs_dir: 時間戳模型儲存目錄
        latest_dir: 最新模型儲存目錄
        registry_path: 模型註冊檔路徑
        degree: 多項式次數（預設 2）
        interaction_only: 是否只產生交互作用項（不包含高次項）
        use_feature_selection: 是否先選擇重要特徵（避免維度爆炸）
        top_k_features: 選擇前 k 個重要特徵
        use_full_train: 是否使用完整訓練集
        test_size: 驗證集比例
        random_state: 隨機種子

    Returns:
        str: 模型檔案路徑
    """
    print("=== 訓練 Polynomial Regression 模型 ===")
    print(f"參數設定:")
    print(f"  - Degree: {degree}")
    print(f"  - Interaction only: {interaction_only}")
    print(f"  - Feature selection: {use_feature_selection}")
    if use_feature_selection:
        print(f"  - Top K features: {top_k_features}")
    print()

    # 讀取訓練資料
    df = pd.read_csv(input_path)

    # 分離特徵與目標
    X = df.drop(columns=['ID', 'traffic_volume'])
    y = df['traffic_volume']

    print(f"原始資料形狀: X={X.shape}, y={y.shape}")

    # 特徵選擇（避免多項式轉換後維度爆炸）
    if use_feature_selection:
        selected_features = select_important_features(X, top_k=top_k_features)
        X_selected = X[selected_features]
        print(f"選擇特徵: {len(selected_features)} 個")
        print(f"  特徵: {', '.join(selected_features[:5])}...")
    else:
        X_selected = X
        selected_features = X.columns.tolist()

    # 建立多項式特徵轉換器
    poly = PolynomialFeatures(
        degree=degree,
        interaction_only=interaction_only,
        include_bias=False  # 不包含偏差項（LinearRegression 會自動加）
    )

    # 轉換特徵
    print(f"\n轉換多項式特徵...")
    X_poly = poly.fit_transform(X_selected)
    print(f"多項式特徵形狀: {X_poly.shape}")
    print(f"  原始特徵數: {X_selected.shape[1]}")
    print(f"  轉換後特徵數: {X_poly.shape[1]}")

    # 分割資料或使用完整訓練集
    if use_full_train:
        print("\n使用完整訓練集（不分割驗證集）")
        X_train, y_train = X_poly, y
        X_val, y_val = None, None
        print(f"訓練集: {X_train.shape}")
    else:
        X_train, X_val, y_train, y_val = train_test_split(
            X_poly, y, test_size=test_size, random_state=random_state
        )
        print(f"\n訓練集: {X_train.shape}, 驗證集: {X_val.shape}")

    # 建立並訓練 LinearRegression 模型
    model = LinearRegression()
    print("\n開始訓練...")
    model.fit(X_train, y_train)
    print("✓ 訓練完成")

    # 計算訓練集指標
    y_train_pred = model.predict(X_train)
    train_r2 = r2_score(y_train, y_train_pred)
    train_mse = mean_squared_error(y_train, y_train_pred)
    train_rmse = np.sqrt(train_mse)
    train_mae = mean_absolute_error(y_train, y_train_pred)

    print("\n=== 訓練集指標 ===")
    print(f"  R² Score:  {train_r2:.6f}")
    print(f"  RMSE:      {train_rmse:.2f}")
    print(f"  MAE:       {train_mae:.2f}")

    # 準備指標字典
    metrics = {
        'train': {
            'r2': float(train_r2),
            'rmse': float(train_rmse),
            'mae': float(train_mae),
            'mse': float(train_mse),
            'samples': int(len(X_train))
        },
        'config': {
            'model_type': 'polynomial',
            'degree': degree,
            'interaction_only': interaction_only,
            'use_feature_selection': use_feature_selection,
            'selected_features': len(selected_features),
            'polynomial_features': X_poly.shape[1],
            'use_full_train': use_full_train,
            'test_size': test_size if not use_full_train else None,
            'random_state': random_state
        }
    }

    # 如果有驗證集，計算驗證集指標
    if not use_full_train:
        y_val_pred = model.predict(X_val)
        val_r2 = r2_score(y_val, y_val_pred)
        val_mse = mean_squared_error(y_val, y_val_pred)
        val_rmse = np.sqrt(val_mse)
        val_mae = mean_absolute_error(y_val, y_val_pred)

        print("\n=== 驗證集指標 ===")
        print(f"  R² Score:  {val_r2:.6f}")
        print(f"  RMSE:      {val_rmse:.2f}")
        print(f"  MAE:       {val_mae:.2f}")

        metrics['validation'] = {
            'r2': float(val_r2),
            'rmse': float(val_rmse),
            'mae': float(val_mae),
            'mse': float(val_mse),
            'samples': int(len(X_val))
        }

        # 檢查過擬合
        overfit_ratio = (train_r2 - val_r2) / train_r2
        if overfit_ratio > 0.1:
            print(f"\n⚠️ 警告：可能過擬合")
            print(f"   訓練 R²: {train_r2:.4f}")
            print(f"   驗證 R²: {val_r2:.4f}")
            print(f"   差距: {overfit_ratio*100:.1f}%")

    # 準備儲存路徑
    timestamp = get_timestamp()
    run_name = f"{timestamp}-polynomial-d{degree}"
    if interaction_only:
        run_name += "-interact"
    if use_full_train:
        run_name += "-full"

    run_dir = Path(runs_dir) / run_name
    ensure_dir(run_dir)

    model_path = run_dir / 'model.joblib'
    poly_path = run_dir / 'polynomial.joblib'
    features_path = run_dir / 'features.json'
    metrics_path = run_dir / 'metrics.json'

    # 儲存模型和轉換器
    joblib.dump(model, model_path)
    joblib.dump(poly, poly_path)
    print(f"\n✓ 模型已儲存: {model_path}")
    print(f"✓ 多項式轉換器已儲存: {poly_path}")

    # 儲存選中的特徵
    features_info = {
        'selected_features': selected_features,
        'polynomial_features': X_poly.shape[1]
    }
    with open(features_path, 'w', encoding='utf-8') as f:
        json.dump(features_info, f, ensure_ascii=False, indent=2)
    print(f"✓ 特徵資訊已儲存: {features_path}")

    # 儲存指標
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"✓ 指標已儲存: {metrics_path}")

    # 更新註冊檔
    update_registry(
        registry_path=registry_path,
        model_path=str(model_path),
        timestamp=timestamp,
        model_name='polynomial',
        metrics=metrics
    )
    print(f"✓ 註冊檔已更新: {registry_path}")

    return str(model_path)