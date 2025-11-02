"""共用工具函式"""
from datetime import datetime
from pathlib import Path


def get_timestamp() -> str:
    """產生 YYYYMMDD-HHMMSS 格式的時間戳

    Returns:
        str: 格式化的時間戳字串
    """
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def ensure_dir(path: str | Path) -> Path:
    """確保目錄存在，若不存在則建立

    Args:
        path: 目錄路徑

    Returns:
        Path: 目錄的 Path 物件
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path
