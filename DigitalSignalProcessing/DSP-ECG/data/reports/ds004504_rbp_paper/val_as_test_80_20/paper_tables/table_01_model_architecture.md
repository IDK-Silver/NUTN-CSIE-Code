# Table 1. Model architecture summary

| Order | Layer type | Output shape | Parameters | Connected to |
| --- | --- | --- | --- | --- |
| 1 | Input layer | (None, 6, 1) | 0 | - |
| 2 | Conv 1D | (None, 6, 32) | 256 | Input layer |
| 3 | Batch normalization | (None, 6, 32) | 128 | Conv1D |
| 4 | Activation | (None, 6, 32) | 0 | Batch normalization |
| 5 | Spatial dropout 1D | (None, 6, 32) | 0 | Activation |
| 6 | Conv1D | (None, 6, 32) | 7200 | Spatial dropout 1D |
| 7 | Batch normalization | (None, 6, 32) | 128 | Conv1D |
| 8 | Activation | (None, 6, 32) | 0 | Batch normalization |
| 9 | Conv 1D residual | (None, 6, 32) | 64 | Input layer |
| 10 | Spatial dropout 1D | (None, 6, 32) | 0 | Activation |
| 11 | Add | (None, 6, 32) | 0 | Conv1D + Spatial dropout |
| 12 | Conv 1D | (None, 6, 32) | 7200 | Add |
| 13 | Batch normalization | (None, 6, 32) | 128 | Conv1D |
| 14 | Activation | (None, 6, 32) | 0 | Batch normalization |
| 15 | Spatial dropout 1D | (None, 6, 32) | 0 | Activation |
| 16 | Conv 1D | (None, 6, 32) | 7200 | Spatial dropout 1D |
| 17 | Batch normalization | (None, 6, 32) | 128 | Conv 1D |
| 18 | Activation | (None, 6, 32) | 0 | Batch normalization |
| 19 | Conv 1D residual | (None, 6, 32) | 1056 | Add |
| 20 | Spatial dropout 1D | (None, 6, 32) | 0 | Activation |
| 21 | Add | (None, 6, 32) | 0 | Conv1D + Spatial dropout |
| 22 | LSTM | (None, 64) | 24832 | Add |
| 23 | Dense | (None, 128) | 8320 | LSTM |
| 24 | Dropout | (None, 128) | 0 | Dense |
| 25 | Dense | (None, 192) | 24768 | Dropout |
| 26 | Dropout | (None, 192) | 0 | Dense |
| 27 | Dense | (None, 256) | 49408 | Dropout |
| 28 | Dropout | (None, 256) | 0 | Dense |
| 29 | Dense output | (None, 3) | 771 | Dropout |
