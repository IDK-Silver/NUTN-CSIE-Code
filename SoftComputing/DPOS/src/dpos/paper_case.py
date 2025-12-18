"""
論文案例數據

來源：Amallynda, I. (2019). The Discrete Particle Swarm Optimization Algorithms 
For Permutation Flowshop Scheduling Problem.

案例：13 個工作，7 台機器
"""

from .job import Job
from .flow_shop import FlowShop


def create_paper_flowshop() -> FlowShop:
    """
    建立論文中的案例
    
    13 個工作、7 台機器的流水車間排程問題
    
    Returns:
        FlowShop 物件
    """
    shop = FlowShop(num_machines=7)
    
    # 論文 Table 1 的數據
    # (job_id, due_date, [M1, M2, M3, M4, M5, M6, M7])
    jobs_data = [
        (1,  31, [7.62, 10.99, 25.97, 4.76, 12.99, 0.67, 5.00]),
        (2,  10, [1.90,  2.75,  6.49, 1.19,  3.25, 0.17, 1.25]),
        (3,  15, [3.81,  5.49, 12.99, 2.38,  6.49, 0.33, 2.50]),
        (4,   3, [0.38,  0.55,  1.30, 0.24,  0.65, 0.03, 0.25]),
        (5,  10, [1.90,  2.75,  6.49, 1.19,  3.25, 0.17, 1.25]),
        (6,   6, [0.95,  1.37,  3.25, 0.60,  1.62, 0.08, 0.63]),
        (7,  10, [1.90,  2.75,  6.49, 1.19,  3.25, 0.17, 1.25]),
        (8,   9, [1.30,  1.87,  4.42, 0.81,  2.21, 0.11, 0.85]),
        (9,   9, [1.30,  1.87,  4.43, 0.81,  2.21, 0.11, 0.85]),
        (10,  9, [1.30,  1.88,  4.44, 0.81,  2.22, 0.11, 0.86]),
        (11,  6, [0.95,  1.37,  3.25, 0.60,  1.62, 0.08, 0.63]),
        (12,  4, [0.67,  0.96,  2.27, 0.42,  1.14, 0.06, 0.44]),
        (13,  1, [0.19,  0.27,  0.65, 0.12,  0.32, 0.02, 0.13]),
    ]
    
    for job_id, due_date, processing_times in jobs_data:
        shop.add_job(Job(
            job_id=job_id,
            processing_times=processing_times,
            due_date=due_date
        ))
    
    return shop


def get_fcfs_sequence() -> list:
    """
    取得先來先服務 (FCFS) 的排序
    
    Returns:
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
    """
    return list(range(1, 14))


def get_paper_optimal_fitness() -> float:
    """
    論文中報告的最佳適應值
    
    Returns:
        291.05
    """
    return 291.05


def get_fcfs_fitness() -> float:
    """
    論文中 FCFS 的適應值
    
    Returns:
        1000.29
    """
    return 1000.29
