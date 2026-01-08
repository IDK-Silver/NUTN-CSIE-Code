#!/usr/bin/env python3
"""
MNIST Model Analysis Script
分析 MNIST 模型的權重統計資訊
"""

import json
import sys
from pathlib import Path
import numpy as np

def analyze_model(model_path: Path) -> dict:
    """分析模型權重統計"""
    with open(model_path, 'r') as f:
        model = json.load(f)

    weights = model['weights']

    # 計算各層權重統計
    stats = {
        'architecture': model['architecture'],
        'optimizer': model['optimizer'],
        'final_loss': model['final_loss'],
        'iterations': model['iterations'],
        'hidden_size': weights.get('hidden_size', 'N/A'),
        'layers': {}
    }

    for key in ['linear1_weight', 'linear1_bias', 'linear2_weight', 'linear2_bias']:
        arr = np.array(weights[key])
        stats['layers'][key] = {
            'shape': arr.shape,
            'count': len(arr),
            'mean': float(np.mean(arr)),
            'std': float(np.std(arr)),
            'min': float(np.min(arr)),
            'max': float(np.max(arr)),
        }

    # 計算總參數量
    total_params = sum(len(weights[k]) for k in ['linear1_weight', 'linear1_bias', 'linear2_weight', 'linear2_bias'])
    stats['total_params'] = total_params

    return stats

def print_stats(stats: dict, name: str):
    """印出統計資訊"""
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    print(f"  Architecture: {stats['architecture']}")
    print(f"  Optimizer: {stats['optimizer']}")
    print(f"  Hidden Size: {stats['hidden_size']}")
    print(f"  Final Loss: {stats['final_loss']:.6f}")
    print(f"  Iterations: {stats['iterations']}")
    print(f"  Total Parameters: {stats['total_params']:,}")
    print()

    for layer_name, layer_stats in stats['layers'].items():
        print(f"  {layer_name}:")
        print(f"    Count: {layer_stats['count']:,}")
        print(f"    Mean: {layer_stats['mean']:.6f}")
        print(f"    Std:  {layer_stats['std']:.6f}")
        print(f"    Range: [{layer_stats['min']:.4f}, {layer_stats['max']:.4f}]")
        print()

def main():
    base_path = Path(__file__).parent.parent / 'blob' / 'mnist'

    # 分析 SGD 模型
    sgd_path = base_path / 'train' / 'gradient-descent' / 'sgd' / 'model.json'
    if sgd_path.exists():
        sgd_stats = analyze_model(sgd_path)
        print_stats(sgd_stats, "MNIST SGD Model")

    # 分析 PSO 模型
    pso_path = base_path / 'train' / 'pso' / 'model.json'
    if pso_path.exists():
        pso_stats = analyze_model(pso_path)
        print_stats(pso_stats, "MNIST PSO Model")

    # 比較
    if sgd_path.exists() and pso_path.exists():
        print(f"\n{'='*60}")
        print("  Comparison Summary")
        print(f"{'='*60}")
        print(f"  {'Metric':<20} {'SGD':>15} {'PSO':>15}")
        print(f"  {'-'*50}")
        print(f"  {'Final Loss':<20} {sgd_stats['final_loss']:>15.6f} {pso_stats['final_loss']:>15.6f}")
        print(f"  {'Iterations':<20} {sgd_stats['iterations']:>15} {pso_stats['iterations']:>15}")
        print(f"  {'Total Params':<20} {sgd_stats['total_params']:>15,} {pso_stats['total_params']:>15,}")

if __name__ == '__main__':
    main()
