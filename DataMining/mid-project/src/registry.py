"""模型註冊檔管理模組"""
import json
import shutil
from pathlib import Path


def load_registry(registry_path: str = 'blob/models/registry.json') -> dict:
    """載入模型註冊檔

    Args:
        registry_path: 註冊檔路徑

    Returns:
        dict: 註冊檔內容；若檔案不存在則返回空結構
    """
    registry_path = Path(registry_path)

    if not registry_path.exists():
        return {
            'latest': None,
            'best': None,
            'history': []
        }

    with open(registry_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_registry(registry: dict, registry_path: str = 'blob/models/registry.json') -> None:
    """儲存模型註冊檔

    Args:
        registry: 註冊檔內容
        registry_path: 註冊檔路徑
    """
    registry_path = Path(registry_path)
    registry_path.parent.mkdir(parents=True, exist_ok=True)

    with open(registry_path, 'w', encoding='utf-8') as f:
        json.dump(registry, f, ensure_ascii=False, indent=2)


def update_registry(registry_path: str,
                   model_path: str,
                   timestamp: str,
                   model_name: str,
                   metrics: dict | None = None) -> None:
    """更新註冊檔（新增訓練記錄並更新 latest）

    Args:
        registry_path: 註冊檔路徑
        model_path: 模型檔案路徑
        timestamp: 時間戳
        model_name: 模型名稱
        metrics: 訓練指標（可選）
    """
    registry = load_registry(registry_path)

    # 更新 latest
    registry['latest'] = model_path

    # 新增歷史記錄
    history_entry = {
        'run': model_path,
        'ts': timestamp,
        'model': model_name
    }

    # 加入指標（如果有）
    if metrics:
        history_entry['metrics'] = metrics

    registry['history'].append(history_entry)

    # 儲存
    save_registry(registry, registry_path)


def mark_best(run_path: str,
             registry_path: str = 'blob/models/registry.json',
             best_dir: str = 'blob/models/best') -> None:
    """標記指定的模型為最佳模型

    Args:
        run_path: 要標記的模型路徑
        registry_path: 註冊檔路徑
        best_dir: 最佳模型目錄

    Raises:
        ValueError: 若模型檔案不存在
    """
    run_path = Path(run_path)

    if not run_path.exists():
        raise ValueError(f"模型檔案不存在: {run_path}")

    # 載入註冊檔
    registry = load_registry(registry_path)

    # 更新 best
    registry['best'] = str(run_path)

    # 複製模型到 best 目錄
    best_dir = Path(best_dir)
    best_dir.mkdir(parents=True, exist_ok=True)
    best_path = best_dir / 'model.joblib'

    shutil.copy2(run_path, best_path)

    # 儲存註冊檔
    save_registry(registry, registry_path)

    print(f"✓ 已標記為最佳模型: {run_path}")
    print(f"✓ 最佳模型副本: {best_path}")
    print(f"✓ 註冊檔已更新: {registry_path}")
