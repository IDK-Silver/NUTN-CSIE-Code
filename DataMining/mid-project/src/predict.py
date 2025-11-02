"""預測與提交模組"""
import joblib
from pathlib import Path
import pandas as pd

from .utils import get_timestamp, ensure_dir
from .registry import load_registry


def load_model(run: str,
               registry_path: str = 'blob/models/registry.json',
               latest_dir: str = 'blob/models/latest',
               best_dir: str = 'blob/models/best'):
    """載入模型

    Args:
        run: 'latest', 'best', 或具體的模型路徑
        registry_path: 模型註冊檔路徑
        latest_dir: 最新模型目錄
        best_dir: 最佳模型目錄

    Returns:
        載入的模型

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
    return model


def predict(run: str = 'latest',
           input_path: str = 'blob/process/test_processed.csv',
           output_dir: str = 'blob/submit/runs',
           latest_dir: str = 'blob/submit/latest',
           registry_path: str = 'blob/models/registry.json') -> str:
    """執行預測並產生提交檔

    Args:
        run: 使用的模型 ('latest', 'best', 或具體路徑)
        input_path: 前處理後的測試資料路徑
        output_dir: 提交檔輸出目錄
        latest_dir: 最新提交檔目錄
        registry_path: 模型註冊檔路徑

    Returns:
        str: 提交檔路徑
    """
    # 載入模型
    model = load_model(run, registry_path=registry_path)

    # 讀取測試資料
    df = pd.read_csv(input_path)
    test_ids = df['ID']
    X_test = df.drop(columns=['ID'])

    print(f"測試資料形狀: {X_test.shape}")

    # 預測
    print("開始預測...")
    predictions = model.predict(X_test)
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

    # 複製到 latest（可選）
    ensure_dir(latest_dir)
    latest_path = Path(latest_dir) / 'submission.csv'
    submission.to_csv(latest_path, index=False)
    print(f"✓ 最新提交副本: {latest_path}")

    print(f"預測筆數: {len(submission)}")

    return str(submission_path)
