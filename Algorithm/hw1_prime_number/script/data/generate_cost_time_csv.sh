#!/bin/bash

# 設置變數
N_START=2
N_END=100000
FERMAT_TRY_TIME=5
MILLER_RABIN_TRY_TIME=5

echo "開始執行程序..."

# 檢查可執行文件是否存在
if [ ! -f "./build/all_algorithm_cost_time" ]; then
    echo "錯誤：可執行文件 './build/all_algorithm_cost_time' 不存在"
    exit 1
fi

# 執行程序
./build/all_algorithm_cost_time \
    --n_start $N_START \
    --n_end $N_END \
    --fermat_try_time $FERMAT_TRY_TIME \
    --miller_rabin_try_time $MILLER_RABIN_TRY_TIME

RESULT=$?

if [ $RESULT -eq 0 ]; then
    echo "程序執行完成。"
else
    echo "程序執行失敗，錯誤碼：$RESULT"
fi

echo "腳本執行結束。"