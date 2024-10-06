#!/bin/bash

# 設置變數
N_LIST="10,100,1000,10000,100000,1000000"
TRY_TIME_START=1
TRY_TIME_END=15

echo "開始執行程序..."

# 檢查可執行文件是否存在
if [ ! -f "./build/fermat_vs_miller" ]; then
    echo "錯誤：可執行文件 './build/fermat_vs_miller' 不存在"
    exit 1
fi

# 執行程序
./build/fermat_vs_miller \
    --n_list $N_LIST \
    --try_time_start $TRY_TIME_START \
    --try_time_end $TRY_TIME_END

RESULT=$?

if [ $RESULT -eq 0 ]; then
    echo "程序執行完成。"
else
    echo "程序執行失敗，錯誤碼：$RESULT"
fi

echo "腳本執行結束。"
