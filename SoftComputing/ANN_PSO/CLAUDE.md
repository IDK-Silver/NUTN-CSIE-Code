# ANN_PSO Project

Use PSO (Particle Swarm Optimization) to find optimal weights for a neural network solving the 2-input XOR problem.

## Problem Summary

- **Input**: 2 binary inputs (X1, X2)
- **Output**: XOR result (Y)
- **Network**: 2-2-1 architecture (2 input, 2 hidden, 1 output)
- **Activation**: Sigmoid
- **Total weights**: 9 (w11, w12, w21, w22, wb1, wb2, w31, w32, wb3)

## Loss Function

```
Loss = (1/2) * sum((y_desired - y_predicted)^2)
```

## Deliverables

1. Source code
2. Report with:
   - Language/compiler/library versions
   - PSO parameters (encoding, particle count, velocity formula, termination)
   - Best weights found
   - Loss vs iteration plot
   - Test results for (0.7, 0.3), (0.6, 0.4), (0.5, 0.5)
   - Optional: Gradient descent comparison

## Code Style

- Python preferred
- Keep it simple, no over-engineering
- Comments in English
