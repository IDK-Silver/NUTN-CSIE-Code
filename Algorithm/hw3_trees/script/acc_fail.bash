#!/bin/bash

# Define the list of numbers for input sizes
nums=(1 2 3 4 5)  # Add more sizes if needed

# Loop through each number in the nums array
for num in "${nums[@]}"; do
    ./TreeSearch --input dataset/fail/FailTest_in.txt --test dataset/fail/FailTest_${num}.txt --output acc_fail.csv --target acc_fail_${num}
done

