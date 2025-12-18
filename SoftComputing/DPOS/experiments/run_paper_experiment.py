"""
重現論文實驗結果

這個腳本用來驗證 HDPSO 和 MPSO 演算法是否能達到論文報告的效果。

論文：The Discrete Particle Swarm Optimization Algorithms For 
Multi-Objective Permutation Flowshop Scheduling Problem

預期結果：
- FCFS (先來先服務): 1000.29
- 最佳解: 291.05
"""

import time
import random
from typing import List, Tuple
from dataclasses import dataclass

from dpos import (
    FlowShop,
    HDPSO,
    MPSO,
    create_paper_flowshop,
    get_fcfs_sequence,
    get_paper_optimal_fitness,
    get_fcfs_fitness,
)


@dataclass
class ExperimentResult:
    """實驗結果"""
    algorithm: str
    population: int
    iterations: int
    avg_fitness: float
    std_fitness: float
    min_fitness: float
    avg_time: float
    best_sequence: List[int]


def run_experiment(
    algorithm: str,
    flowshop: FlowShop,
    num_particles: int,
    max_iterations: int,
    num_replications: int = 10,
    seed: int = None
) -> ExperimentResult:
    """
    執行實驗
    
    Args:
        algorithm: 演算法名稱 ("HDPSO" or "MPSO")
        flowshop: FlowShop 物件
        num_particles: 粒子數量
        max_iterations: 最大迭代次數
        num_replications: 重複次數
        seed: 隨機種子（如果設定，每次重複會用 seed + i）
        
    Returns:
        ExperimentResult
    """
    fitness_values = []
    times = []
    best_fitness = float('inf')
    best_sequence = []
    
    for i in range(num_replications):
        if seed is not None:
            random.seed(seed + i)
        
        start_time = time.time()
        
        if algorithm == "HDPSO":
            algo = HDPSO(
                flowshop=flowshop,
                num_particles=num_particles,
                max_iterations=max_iterations,
                c1=1.0,
                c2=1.0,
                theta_max=0.9,
                theta_min=0.4
            )
        else:  # MPSO
            algo = MPSO(
                flowshop=flowshop,
                num_particles=num_particles,
                max_iterations=max_iterations,
                c1=1.0,
                c2=1.0,
                theta_max=0.9,
                theta_min=0.4
            )
        
        sequence, fitness = algo.run(verbose=False)
        
        elapsed_time = time.time() - start_time
        
        fitness_values.append(fitness)
        times.append(elapsed_time)
        
        if fitness < best_fitness:
            best_fitness = fitness
            best_sequence = sequence
    
    import statistics
    
    return ExperimentResult(
        algorithm=algorithm,
        population=num_particles,
        iterations=max_iterations,
        avg_fitness=statistics.mean(fitness_values),
        std_fitness=statistics.stdev(fitness_values) if len(fitness_values) > 1 else 0,
        min_fitness=min(fitness_values),
        avg_time=statistics.mean(times),
        best_sequence=best_sequence
    )


def print_experiment_table(results: List[ExperimentResult]):
    """印出實驗結果表格"""
    print()
    print(f"{'Algorithm':<10} {'N':<5} {'Iter':<6} {'Avg':<10} {'Std':<10} {'Min':<10} {'Time(s)':<10}")

    for r in results:
        print(f"{r.algorithm:<10} {r.population:<5} {r.iterations:<6} "
              f"{r.avg_fitness:<10.2f} {r.std_fitness:<10.2f} {r.min_fitness:<10.2f} "
              f"{r.avg_time:<10.3f}")

    print()


def main():
    """主程式"""
    print()
    print("DPOS 論文實驗重現")
    print()

    # 建立論文案例
    flowshop = create_paper_flowshop()

    # 計算 FCFS 的適應值
    fcfs_sequence = get_fcfs_sequence()
    fcfs_fitness = flowshop.compute_fitness(fcfs_sequence)

    print(f"\n基準測試")
    print(f"FCFS 排序: {fcfs_sequence}")
    print(f"FCFS 適應值: {fcfs_fitness:.2f}")
    print(f"論文報告的 FCFS 適應值: {get_fcfs_fitness()}")
    print(f"論文報告的最佳適應值: {get_paper_optimal_fitness()}")

    # 實驗參數（根據論文 Table 2）
    # 論文測試了 N=10,50,100 和 Iter=50,250,500 的組合
    experiment_configs = [
        (10, 50),
        (10, 250),
        (10, 500),
        (50, 50),
        (50, 250),
        (50, 500),
        (100, 50),
        (100, 250),
        (100, 500),
    ]

    print(f"\n執行實驗（每組 10 次重複）...")
    print(f"這可能需要幾分鐘...")
    
    results = []
    
    for num_particles, max_iterations in experiment_configs:
        print(f"\n測試 N={num_particles}, Iter={max_iterations}...")
        
        # HDPSO
        hdpso_result = run_experiment(
            algorithm="HDPSO",
            flowshop=flowshop,
            num_particles=num_particles,
            max_iterations=max_iterations,
            num_replications=10,
            seed=42
        )
        results.append(hdpso_result)
        print(f"  HDPSO: avg={hdpso_result.avg_fitness:.2f}, min={hdpso_result.min_fitness:.2f}")
        
        # MPSO
        mpso_result = run_experiment(
            algorithm="MPSO",
            flowshop=flowshop,
            num_particles=num_particles,
            max_iterations=max_iterations,
            num_replications=10,
            seed=42
        )
        results.append(mpso_result)
        print(f"  MPSO:  avg={mpso_result.avg_fitness:.2f}, min={mpso_result.min_fitness:.2f}")
    
    # 印出結果表格
    print_experiment_table(results)
    
    # 找出最佳結果
    best_hdpso = min([r for r in results if r.algorithm == "HDPSO"], key=lambda x: x.min_fitness)
    best_mpso = min([r for r in results if r.algorithm == "MPSO"], key=lambda x: x.min_fitness)

    print(f"\n最佳結果")
    print(f"HDPSO 最佳: {best_hdpso.min_fitness:.2f} (N={best_hdpso.population}, Iter={best_hdpso.iterations})")
    print(f"  排序: {best_hdpso.best_sequence}")

    print(f"MPSO  最佳: {best_mpso.min_fitness:.2f} (N={best_mpso.population}, Iter={best_mpso.iterations})")
    print(f"  排序: {best_mpso.best_sequence}")

    print(f"\n改善幅度")
    improvement_hdpso = (fcfs_fitness - best_hdpso.min_fitness) / fcfs_fitness * 100
    improvement_mpso = (fcfs_fitness - best_mpso.min_fitness) / fcfs_fitness * 100
    print(f"HDPSO vs FCFS: {improvement_hdpso:.1f}% 改善")
    print(f"MPSO  vs FCFS: {improvement_mpso:.1f}% 改善")

    # 顯示最佳解的詳細排程
    print(f"\n最佳解詳細排程（HDPSO）")
    schedule_results = flowshop.compute_schedule(best_hdpso.best_sequence)
    
    total_earliness = 0
    total_tardiness = 0
    
    print(f"{'Job':<5} {'完成':<10} {'交期':<10} {'提早':<10} {'延遲':<10}")

    for r in schedule_results:
        print(f"{r.job_id:<5} {r.completion_time:<10.2f} {r.due_date:<10.2f} "
              f"{r.earliness:<10.2f} {r.tardiness:<10.2f}")
        total_earliness += r.earliness
        total_tardiness += r.tardiness

    print()
    print(f"{'總計':<5} {'':<10} {'':<10} {total_earliness:<10.2f} {total_tardiness:<10.2f}")
    print(f"目標函數值: {total_earliness + total_tardiness:.2f}")


if __name__ == "__main__":
    main()
