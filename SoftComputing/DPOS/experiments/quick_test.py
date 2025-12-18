"""
快速測試腳本

用於快速驗證 HDPSO 和 MPSO 是否正常運作
"""

from dpos import (
    Job, FlowShop, HDPSO, MPSO,
    create_paper_flowshop, get_fcfs_sequence
)


def quick_test():
    """快速測試"""
    print()
    print("DPOS 快速測試")
    print()

    # 使用論文案例
    shop = create_paper_flowshop()

    # 測試 FCFS
    fcfs_seq = get_fcfs_sequence()
    fcfs_fit = shop.compute_fitness(fcfs_seq)
    print(f"\nFCFS 基準: {fcfs_fit:.2f}")

    # 測試 HDPSO
    print("\n測試 HDPSO...")
    hdpso = HDPSO(shop, num_particles=30, max_iterations=100)
    hdpso_seq, hdpso_fit = hdpso.run(verbose=False)
    print(f"   結果: {hdpso_fit:.2f}")
    print(f"   排序: {hdpso_seq}")

    # 測試 MPSO
    print("\n測試 MPSO...")
    mpso = MPSO(shop, num_particles=30, max_iterations=100)
    mpso_seq, mpso_fit = mpso.run(verbose=False)
    print(f"   結果: {mpso_fit:.2f}")
    print(f"   排序: {mpso_seq}")

    # 比較
    print("\n改善幅度")
    print(f"   HDPSO: {(fcfs_fit - hdpso_fit) / fcfs_fit * 100:.1f}%")
    print(f"   MPSO:  {(fcfs_fit - mpso_fit) / fcfs_fit * 100:.1f}%")

    print("\n測試完成！")


if __name__ == "__main__":
    quick_test()
