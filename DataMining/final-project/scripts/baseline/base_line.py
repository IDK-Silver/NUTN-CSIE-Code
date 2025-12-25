import pandas as pd
import numpy as np
import os

# Ensure output directory exists
os.makedirs('blobs/submit/base_line', exist_ok=True)

test = pd.read_csv('blobs/raw/test.csv')

# All zeros
sub_zero = pd.DataFrame({'ID': test['ID'], 'TARGET': 0})
sub_zero.to_csv('blobs/submit/base_line/submission_all_zero.csv', index=False)

# All ones
sub_one = pd.DataFrame({'ID': test['ID'], 'TARGET': 1})
sub_one.to_csv('blobs/submit/base_line/submission_all_one.csv', index=False)

# Random generation based on class distribution (13.9% positive class)
sub_random = pd.DataFrame({
    'ID': test['ID'], 
    'TARGET': np.random.choice([0, 1], size=len(test), p=[0.861, 0.139])
})
sub_random.to_csv('blobs/submit/base_line/submission_random.csv', index=False)

print("Baseline submissions have been generated and saved in 'blobs/submit/base_line/'.")
