"""MPSO 測試"""

import pytest
from dpos import Job, FlowShop, MPSO, MPSOParticle, MPSOOperations


class TestMPSOOperations:
    """測試 MPSO 運算操作"""

    def test_position_to_sequence_basic(self):
        """測試基本的位置轉排序"""
        # position = [0.1067, 0.8687, 0.4314, 0.1361, 0.8530]
        # 按值排序：0.1067(0) < 0.1361(3) < 0.4314(2) < 0.8530(4) < 0.8687(1)
        # 1-based: [1, 4, 3, 5, 2]
        position = [0.1067, 0.8687, 0.4314, 0.1361, 0.8530]
        sequence = MPSOOperations.position_to_sequence(position)
        assert sequence == [1, 4, 3, 5, 2]

    def test_position_to_sequence_with_job_ids(self):
        """測試帶有 job_ids 的位置轉排序"""
        position = [0.7749, 1.0, 0.2638, 0.5499, 0.0]
        # 按值排序：0.0(4) < 0.2638(2) < 0.5499(3) < 0.7749(0) < 1.0(1)
        # 索引順序：[4, 2, 3, 0, 1]
        job_ids = [10, 20, 30, 40, 50]  # 自定義 job_ids
        sequence = MPSOOperations.position_to_sequence(position, job_ids)
        assert sequence == [50, 30, 40, 10, 20]

    def test_normalize_position(self):
        """測試正規化"""
        position = [0.5, 1.5, -0.5, 2.0, 0.0]
        normalized = MPSOOperations.normalize_position(position)
        
        # 檢查範圍
        assert min(normalized) == 0.0
        assert max(normalized) == 1.0
        
        # 檢查相對順序保持不變
        # 原本：-0.5 < 0.0 < 0.5 < 1.5 < 2.0
        # 索引：  2    4    0    1    3
        original_order = sorted(range(len(position)), key=lambda i: position[i])
        normalized_order = sorted(range(len(normalized)), key=lambda i: normalized[i])
        assert original_order == normalized_order

    def test_clip_position(self):
        """測試裁剪"""
        position = [0.5, 1.5, -0.5, 2.0, 0.0]
        clipped = MPSOOperations.clip_position(position)
        
        assert clipped == [0.5, 1.0, 0.0, 1.0, 0.0]


class TestMPSOParticle:
    """測試 MPSO 粒子"""

    def test_particle_initialization(self):
        """測試粒子初始化"""
        particle = MPSOParticle(num_jobs=5)
        
        assert len(particle.position) == 5
        assert len(particle.velocity) == 5
        assert all(0 <= x <= 1 for x in particle.position)
        assert all(v == 0.0 for v in particle.velocity)
        assert particle.fitness == float('inf')


class TestMPSO:
    """測試 MPSO 演算法"""

    @pytest.fixture
    def simple_flowshop(self):
        """建立簡單的測試用 FlowShop"""
        shop = FlowShop(num_machines=2)
        shop.add_job(Job(job_id=1, processing_times=[3, 2], due_date=10))
        shop.add_job(Job(job_id=2, processing_times=[2, 3], due_date=8))
        shop.add_job(Job(job_id=3, processing_times=[4, 1], due_date=12))
        shop.add_job(Job(job_id=4, processing_times=[1, 2], due_date=5))
        shop.add_job(Job(job_id=5, processing_times=[2, 2], due_date=7))
        return shop

    def test_mpso_initialization(self, simple_flowshop):
        """測試 MPSO 初始化"""
        mpso = MPSO(
            flowshop=simple_flowshop,
            num_particles=10,
            max_iterations=50
        )
        
        assert mpso.num_jobs == 5
        assert mpso.job_ids == [1, 2, 3, 4, 5]
        assert len(mpso.particles) == 0  # 尚未初始化

    def test_mpso_run(self, simple_flowshop):
        """測試 MPSO 執行"""
        mpso = MPSO(
            flowshop=simple_flowshop,
            num_particles=20,
            max_iterations=50
        )
        
        best_sequence, best_fitness = mpso.run(verbose=False)
        
        # 檢查結果是有效的排序
        assert len(best_sequence) == 5
        assert set(best_sequence) == {1, 2, 3, 4, 5}
        
        # 檢查適應值是合理的
        assert best_fitness >= 0
        assert best_fitness < float('inf')
        
        # 檢查歷史記錄
        assert len(mpso.fitness_history) == 51  # 初始 + 50 次迭代

    def test_mpso_convergence(self, simple_flowshop):
        """測試 MPSO 收斂性"""
        mpso = MPSO(
            flowshop=simple_flowshop,
            num_particles=30,
            max_iterations=100
        )
        
        best_sequence, best_fitness = mpso.run(verbose=False)
        
        # 檢查最終適應值應該 <= 初始適應值
        assert best_fitness <= mpso.fitness_history[0]
        
        # 檢查適應值歷史是非遞增的（最佳解不會變差）
        for i in range(1, len(mpso.fitness_history)):
            assert mpso.fitness_history[i] <= mpso.fitness_history[i-1]

    def test_mpso_theta_decay(self, simple_flowshop):
        """測試慣性權重衰減"""
        mpso = MPSO(
            flowshop=simple_flowshop,
            theta_max=0.9,
            theta_min=0.4,
            max_iterations=100
        )
        
        # 檢查 theta 從 0.9 衰減到 0.4
        assert mpso.get_theta(0) == 0.9
        assert mpso.get_theta(100) == 0.4
        assert mpso.get_theta(50) == pytest.approx(0.65, rel=0.01)


class TestMPSOVsHDPSO:
    """比較 MPSO 和 HDPSO"""

    @pytest.fixture
    def simple_flowshop(self):
        """建立簡單的測試用 FlowShop"""
        shop = FlowShop(num_machines=2)
        shop.add_job(Job(job_id=1, processing_times=[3, 2], due_date=10))
        shop.add_job(Job(job_id=2, processing_times=[2, 3], due_date=8))
        shop.add_job(Job(job_id=3, processing_times=[4, 1], due_date=12))
        shop.add_job(Job(job_id=4, processing_times=[1, 2], due_date=5))
        shop.add_job(Job(job_id=5, processing_times=[2, 2], due_date=7))
        return shop

    def test_both_find_good_solution(self, simple_flowshop):
        """測試兩種演算法都能找到好的解"""
        from dpos import HDPSO
        
        # 執行 HDPSO
        hdpso = HDPSO(
            flowshop=simple_flowshop,
            num_particles=30,
            max_iterations=100
        )
        hdpso_seq, hdpso_fit = hdpso.run(verbose=False)
        
        # 執行 MPSO
        mpso = MPSO(
            flowshop=simple_flowshop,
            num_particles=30,
            max_iterations=100
        )
        mpso_seq, mpso_fit = mpso.run(verbose=False)
        
        # 已知最佳解的適應值大約是 6
        # 兩種演算法都應該能找到接近最佳的解
        assert hdpso_fit <= 10  # 應該比隨機解好很多
        assert mpso_fit <= 10
        
        print(f"\nHDPSO: {hdpso_seq}, fitness={hdpso_fit}")
        print(f"MPSO:  {mpso_seq}, fitness={mpso_fit}")
