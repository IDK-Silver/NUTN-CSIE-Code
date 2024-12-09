#!/bin/bash

# Define the list of numbers for input sizes
nums=(512 2048 8192 32768 131072)  # Add more sizes if needed

# Loop through each number in the nums array
for num_i in "${nums[@]}"; do
    for num_j in "${nums[@]}"; do
        ./TreeSearch --input ./dataset/full_acc/InputSize${num_i}.txt --test ./dataset/full_acc/InputSize${num_j}_1.txt --output acc_full.csv --target acc_full_${num_i}_${num_j}
    done
done

