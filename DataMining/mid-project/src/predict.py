"""預測與提交模組"""
import json
import joblib
from pathlib import Path
import pandas as pd

from .utils import get_timestamp, ensure_dir
from .registry import load_registry


def load_model(run: str,
               registry_path: str = 'blob/models/registry.json',
               latest_dir: str = 'blob/models/latest',
               best_dir: str = 'blob/models/best'):
    """載入模型（支援 linear 和 polynomial）

    Args:
        run: 'latest', 'best', 或具體的模型路徑
        registry_path: 模型註冊檔路徑
        latest_dir: 最新模型目錄
        best_dir: 最佳模型目錄

    Returns:
        tuple: (model, poly_transformer, selected_features)
               poly_transformer 和 selected_features 僅對 polynomial model 有值

    Raises:
        ValueError: 若模型路徑不存在或無效
    """
    if run == 'latest':
        model_path = Path(latest_dir) / 'model.joblib'
    elif run == 'best':
        model_path = Path(best_dir) / 'model.joblib'
    else:
        model_path = Path(run)

    if not model_path.exists():
        raise ValueError(f"模型檔案不存在: {model_path}")

    print(f"載入模型: {model_path}")
    model = joblib.load(model_path)

    # 檢查是否為 polynomial model
    model_dir = model_path.parent
    poly_path = model_dir / 'polynomial.joblib'
    features_path = model_dir / 'features.json'

    poly_transformer = None
    selected_features = None

    if poly_path.exists() and features_path.exists():
        print("檢測到 Polynomial Regression 模型")
        poly_transformer = joblib.load(poly_path)
        print(f"✓ 載入多項式轉換器: {poly_path}")

        with open(features_path, 'r', encoding='utf-8') as f:
            features_info = json.load(f)
            selected_features = features_info['selected_features']
        print(f"✓ 載入特徵資訊: {len(selected_features)} 個選中特徵")

    return model, poly_transformer, selected_features


def predict(run: str = 'latest',
           input_path: str = 'blob/process/test_processed.csv',
           output_dir: str = 'blob/submit/runs',
           latest_dir: str = 'blob/submit/latest',
           registry_path: str = 'blob/models/registry.json') -> str:
    """執行預測並產生提交檔（支援 linear 和 polynomial）

    Args:
        run: 使用的模型 ('latest', 'best', 或具體路徑)
        input_path: 前處理後的測試資料路徑
        output_dir: 提交檔輸出目錄
        latest_dir: 最新提交檔目錄
        registry_path: 模型註冊檔路徑

    Returns:
        str: 提交檔路徑
    """
    # 載入模型和相關元件
    model, poly_transformer, selected_features = load_model(run, registry_path=registry_path)

    # 讀取測試資料
    df = pd.read_csv(input_path)
    test_ids = df['ID']
    X_test = df.drop(columns=['ID'])

    print(f"測試資料形狀: {X_test.shape}")

    # 如果是 polynomial model，需要進行特徵選擇和多項式轉換
    if poly_transformer is not None and selected_features is not None:
        print(f"\n應用多項式轉換...")

        # 選擇特徵
        missing_features = [f for f in selected_features if f not in X_test.columns]
        if missing_features:
            print(f"⚠️ 警告：測試資料缺少特徵: {missing_features}")
            # 使用可用的特徵
            available_features = [f for f in selected_features if f in X_test.columns]
            X_test_selected = X_test[available_features]
            print(f"使用 {len(available_features)} 個可用特徵")
        else:
            X_test_selected = X_test[selected_features]
            print(f"選擇 {len(selected_features)} 個特徵")

        # 多項式轉換
        X_test_poly = poly_transformer.transform(X_test_selected)
        print(f"轉換後形狀: {X_test_poly.shape}")
        X_test_final = X_test_poly
    else:
        # Linear model，直接使用原始特徵
        print("使用 Linear Regression 模型")
        X_test_final = X_test

    # 預測
    print("\n開始預測...")
    predictions = model.predict(X_test_final)
    print("✓ 預測完成")

    # 建立提交 DataFrame
    submission = pd.DataFrame({
        'ID': test_ids,
        'traffic_volume': predictions
    })

    # 準備輸出路徑
    timestamp = get_timestamp()
    run_dir = Path(output_dir) / timestamp
    ensure_dir(run_dir)

    submission_path = run_dir / 'submission.csv'

    # 儲存提交檔
    submission.to_csv(submission_path, index=False)
    print(f"✓ 提交檔已儲存: {submission_path}")

    # 複製到 latest
    ensure_dir(latest_dir)
    latest_path = Path(latest_dir) / 'submission.csv'
    submission.to_csv(latest_path, index=False)
    print(f"✓ 最新提交副本: {latest_path}")

    print(f"預測筆數: {len(submission)}")

    return str(submission_path)