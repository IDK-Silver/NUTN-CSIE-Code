#!/bin/bash

# 設置變數
RUN_MIN=720


echo "開始執行程序..."

# 檢查可執行文件是否存在
if [ ! -f "./build/find_mersenne_prime_number" ]; then
    echo "錯誤：可執行文件 './build/find_mersenne_prime_number' 不存在"
    exit 1
fi

# 執行程序
./build/find_mersenne_prime_number \
    --run_min $RUN_MIN

RESULT=$?

if [ $RESULT -eq 0 ]; then
    echo "程序執行完成。"
else
    echo "程序執行失敗，錯誤碼：$RESULT"
fi

echo "腳本執行結束。"