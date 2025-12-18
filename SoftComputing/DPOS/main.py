"""
DPOS - Discrete Particle Swarm Optimization Demo

示範 HDPSO 和 MPSO 兩種演算法解決流水車間排程問題
"""

from dpos import Job, FlowShop, HDPSO, MPSO
import time

def create_paper_flowshop() -> FlowShop:
    """建立論文中的案例（13 個工作、7 台機器）"""
    shop = FlowShop(num_machines=7)
    
    # 論文 Table 1 的資料
    jobs_data = [
        # (job_id, due_date, [M1, M2, M3, M4, M5, M6, M7])
        (1,  31, [7.62, 10.99, 25.97, 4.76, 12.99, 0.67, 5.00]),
        (2,  10, [1.90, 2.75, 6.49, 1.19, 3.25, 0.17, 1.25]),
        (3,  15, [3.81, 5.49, 12.99, 2.38, 6.49, 0.33, 2.50]),
        (4,   3, [0.38, 0.55, 1.30, 0.24, 0.65, 0.03, 0.25]),
        (5,  10, [1.90, 2.75, 6.49, 1.19, 3.25, 0.17, 1.25]),
        (6,   6, [0.95, 1.37, 3.25, 0.60, 1.62, 0.08, 0.63]),
        (7,  10, [1.90, 2.75, 6.49, 1.19, 3.25, 0.17, 1.25]),
        (8,   9, [1.30, 1.87, 4.42, 0.81, 2.21, 0.11, 0.85]),
        (9,   9, [1.30, 1.87, 4.43, 0.81, 2.21, 0.11, 0.85]),
        (10,  9, [1.30, 1.88, 4.44, 0.81, 2.22, 0.11, 0.86]),
        (11,  6, [0.95, 1.37, 3.25, 0.60, 1.62, 0.08, 0.63]),
        (12,  4, [0.67, 0.96, 2.27, 0.42, 1.14, 0.06, 0.44]),
        (13,  1, [0.19, 0.27, 0.65, 0.12, 0.32, 0.02, 0.13]),
    ]
    
    for job_id, due_date, processing_times in jobs_data:
        shop.add_job(Job(
            job_id=job_id,
            processing_times=processing_times,
            due_date=due_date
        ))
    
    return shop


def print_schedule_details(shop: FlowShop, sequence: list, title: str = "排程詳細結果"):
    """印出排程的詳細結果"""
    print()
    print(f"{title}")
    print()

    results = shop.compute_schedule(sequence)
    
    total_earliness = 0
    total_tardiness = 0
    
    for r in results:
        status = ""
        if r.earliness > 0:
            status = f"(提早 {r.earliness:.1f})"
        elif r.tardiness > 0:
            status = f"(延遲 {r.tardiness:.1f})"
        else:
            status = "(準時)"
        
        print(f"Job {r.job_id:2d}: 完成={r.completion_time:6.1f}, 交期={r.due_date:6.1f} {status}")
        total_earliness += r.earliness
        total_tardiness += r.tardiness

    print()
    print(f"排序: {sequence}")
    print(f"總提早: {total_earliness:.2f}")
    print(f"總延遲: {total_tardiness:.2f}")
    print(f"目標函數值: {total_earliness + total_tardiness:.2f}")



def demo_paper_case():
    """示範論文案例"""
    print()
    print("論文案例：13 個工作、7 台機器")
    print()

    shop = create_paper_flowshop()

    # 測試 FCFS（先來先做）- 論文中的原始方法
    fcfs_sequence = list(range(1, 14))  # [1, 2, 3, ..., 13]
    fcfs_fitness = shop.compute_fitness(fcfs_sequence)
    print(f"\nFCFS (先來先做): {fcfs_sequence}")
    print(f"   適應值: {fcfs_fitness:.2f}")
    print(f"   （論文中原公司的適應值: 1000.29）")

    # 測試 HDPSO
    print()
    print("執行 HDPSO...")
    hdpso = HDPSO(
        flowshop=shop,
        num_particles=50,
        max_iterations=250,
        c1=1.0,
        c2=1.0
    )
    start_time = time.time()
    hdpso_seq, hdpso_fit = hdpso.run(verbose=True)
    hdpso_time = time.time() - start_time
    print(f"   計算時間: {hdpso_time:.4f} 秒")

    # 測試 MPSO
    print()
    print("執行 MPSO...")
    mpso = MPSO(
        flowshop=shop,
        num_particles=50,
        max_iterations=250,
        c1=1.0,
        c2=1.0
    )
    start_time = time.time()
    mpso_seq, mpso_fit = mpso.run(verbose=True)
    mpso_time = time.time() - start_time
    print(f"   計算時間: {mpso_time:.4f} 秒")

    # 比較結果
    print()
    print("結果比較")
    print()
    print(f"{'方法':<15} {'適應值':<15} {'時間(秒)':<15} {'改善率':<15}")
    print(f"{'FCFS':<15} {fcfs_fitness:<15.2f} {'-':<15} {'-':<15}")
    
    hdpso_improve = (fcfs_fitness - hdpso_fit) / fcfs_fitness * 100
    print(f"{'HDPSO':<15} {hdpso_fit:<15.2f} {hdpso_time:<15.4f} {hdpso_improve:<14.1f}%")
    
    mpso_improve = (fcfs_fitness - mpso_fit) / fcfs_fitness * 100
    print(f"{'MPSO':<15} {mpso_fit:<15.2f} {mpso_time:<15.4f} {mpso_improve:<14.1f}%")
    
    print(f"\n論文中的最佳解適應值: 291.05")


def main():
    """主程式"""
    print("DPOS - Discrete Particle Swarm Optimization")
    print("流水車間排程問題求解器")

    # 示範論文案例
    print()
    print("執行論文案例")
    demo_paper_case()


if __name__ == "__main__":
    main()
