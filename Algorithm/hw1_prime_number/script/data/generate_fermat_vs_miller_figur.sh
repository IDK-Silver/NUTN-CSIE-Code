#!/bin/bash
# 啟動虛擬環境
source venv/bin/activate

# 執行 Python 腳本
python3 script/report/try_times_effect.py

# 退出虛擬環境
deactivate
