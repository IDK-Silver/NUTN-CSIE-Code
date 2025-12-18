"""HDPSO 演算法的 pytest 測試"""

import pytest
from dpos.job import Job
from dpos.flow_shop import FlowShop
from dpos.hdpso.particle import HDPSOParticle, HDPSOOperations, HDPSO


@pytest.fixture
def sample_flowshop():
    """建立測試用的 FlowShop"""
    shop = FlowShop(num_machines=2)

    shop.add_job(Job(job_id=1, processing_times=[3, 2], due_date=10))
    shop.add_job(Job(job_id=2, processing_times=[2, 3], due_date=8))
    shop.add_job(Job(job_id=3, processing_times=[4, 1], due_date=12))
    shop.add_job(Job(job_id=4, processing_times=[1, 2], due_date=5))
    shop.add_job(Job(job_id=5, processing_times=[2, 2], due_date=7))

    return shop


@pytest.fixture
def job_ids():
    """測試用的工作 ID 列表"""
    return [1, 2, 3, 4, 5]


class TestHDPSOOperations:
    """測試 HDPSO 的基本運算"""

    def test_subtract_positions(self):
        """測試位置相減運算"""
        source = [4, 2, 5, 1, 3]
        target = [2, 4, 5, 1, 3]

        velocity = HDPSOOperations.subtract_positions(target, source)

        # 驗證：source + velocity 應該等於 target
        result = HDPSOOperations.apply_velocity(source, velocity)
        assert result == target, f"Expected {target}, got {result}"

    def test_subtract_positions_same(self):
        """測試相同位置相減應該產生空速度"""
        position = [1, 2, 3, 4, 5]
        velocity = HDPSOOperations.subtract_positions(position, position)
        assert velocity == [], "相同位置相減應該產生空速度"

    def test_multiply_coefficient_zero(self):
        """測試係數為 0 的情況"""
        v = [(1, 2), (3, 5), (1, 4)]
        result = HDPSOOperations.multiply_coefficient(0, v)
        assert result == [], "係數為 0 應該返回空速度"

    def test_multiply_coefficient_positive_fraction(self):
        """測試係數為正分數（0 < c <= 1）"""
        v = [(1, 2), (3, 5), (1, 4), (2, 3)]

        # 0.5 應該保留前 2 個（ceil(0.5 * 4) = 2）
        result = HDPSOOperations.multiply_coefficient(0.5, v)
        assert len(result) == 2
        assert result == v[:2]

        # 0.3 應該保留前 2 個（ceil(0.3 * 4) = 2）
        result = HDPSOOperations.multiply_coefficient(0.3, v)
        assert len(result) == 2

    def test_multiply_coefficient_greater_than_one(self):
        """測試係數大於 1 的情況"""
        v = [(1, 2), (3, 5)]

        # 2.5 = 2 * v + 0.5 * v
        result = HDPSOOperations.multiply_coefficient(2.5, v)
        # 應該是 2 次完整的 v + 0.5 * v（保留 1 個）
        expected_length = 2 * len(v) + 1  # 2*2 + 1 = 5
        assert len(result) == expected_length

    def test_multiply_coefficient_negative(self):
        """測試負係數"""
        v = [(1, 2), (3, 5), (1, 4)]

        # -0.5 應該先反轉再取 0.5
        result = HDPSOOperations.multiply_coefficient(-0.5, v)
        reversed_v = v[::-1]
        expected = HDPSOOperations.multiply_coefficient(0.5, reversed_v)
        assert result == expected

    def test_add_velocities(self):
        """測試速度相加"""
        v1 = [(1, 2), (3, 4)]
        v2 = [(2, 3), (4, 5)]

        result = HDPSOOperations.add_velocities(v1, v2)
        assert result == v1 + v2
        assert len(result) == len(v1) + len(v2)

    def test_apply_velocity(self):
        """測試應用速度到位置"""
        position = [1, 2, 3, 4, 5]
        velocity = [(1, 3), (2, 4)]  # 交換位置 1 和 3，然後交換位置 2 和 4

        result = HDPSOOperations.apply_velocity(position, velocity)

        # 驗證結果是有效的排列
        assert len(result) == len(position)
        assert set(result) == set(position)

    def test_apply_empty_velocity(self):
        """測試應用空速度"""
        position = [1, 2, 3, 4, 5]
        velocity = []

        result = HDPSOOperations.apply_velocity(position, velocity)
        assert result == position


class TestHDPSOParticle:
    """測試 HDPSO 粒子"""

    def test_particle_initialization(self, job_ids):
        """測試粒子初始化"""
        particle = HDPSOParticle(job_ids)

        # 檢查位置是否包含所有工作 ID
        assert set(particle.position) == set(job_ids)
        assert len(particle.position) == len(job_ids)

        # 檢查初始速度為空
        assert particle.velocity == []

        # 檢查 pBest 初始化
        assert particle.pbest_position == particle.position
        assert particle.pbest_fitness == float('inf')
        assert particle.fitness == float('inf')

    def test_particle_randomization(self, job_ids):
        """測試多個粒子的初始位置是隨機的"""
        particles = [HDPSOParticle(job_ids) for _ in range(10)]
        positions = [tuple(p.position) for p in particles]

        # 至少應該有一些不同的初始位置
        unique_positions = set(positions)
        assert len(unique_positions) > 1, "粒子的初始位置應該是隨機的"


class TestHDPSO:
    """測試 HDPSO 演算法"""

    def test_hdpso_initialization(self, sample_flowshop):
        """測試 HDPSO 初始化"""
        hdpso = HDPSO(
            flowshop=sample_flowshop,
            num_particles=10,
            max_iterations=5
        )

        assert hdpso.num_particles == 10
        assert hdpso.max_iterations == 5
        assert len(hdpso.job_ids) == 5
        assert hdpso.gbest_fitness == float('inf')

    def test_hdpso_particle_initialization(self, sample_flowshop):
        """測試 HDPSO 粒子群初始化"""
        hdpso = HDPSO(
            flowshop=sample_flowshop,
            num_particles=10,
            max_iterations=5
        )

        hdpso.initialize()

        # 檢查粒子數量
        assert len(hdpso.particles) == 10

        # 檢查每個粒子都有適應值
        for particle in hdpso.particles:
            assert particle.fitness != float('inf')
            assert particle.pbest_fitness != float('inf')

        # 檢查全域最佳被更新
        assert hdpso.gbest_fitness != float('inf')
        assert len(hdpso.gbest_position) == 5

    def test_theta_calculation(self, sample_flowshop):
        """測試慣性權重計算"""
        hdpso = HDPSO(
            flowshop=sample_flowshop,
            theta_max=0.9,
            theta_min=0.4,
            max_iterations=100
        )

        # 第一次迭代應該接近 theta_max
        theta_1 = hdpso.get_theta(1)
        assert 0.85 < theta_1 <= 0.9

        # 最後一次迭代應該接近 theta_min
        theta_100 = hdpso.get_theta(100)
        assert 0.4 <= theta_100 < 0.45

        # 中間應該在兩者之間
        theta_50 = hdpso.get_theta(50)
        assert 0.4 < theta_50 < 0.9

    def test_hdpso_run_basic(self, sample_flowshop):
        """測試 HDPSO 基本執行"""
        hdpso = HDPSO(
            flowshop=sample_flowshop,
            num_particles=10,
            max_iterations=5,
            c1=1.0,
            c2=1.0
        )

        best_sequence, best_fitness = hdpso.run(verbose=False)

        # 檢查返回值
        assert len(best_sequence) == 5
        assert set(best_sequence) == {1, 2, 3, 4, 5}
        assert best_fitness > 0
        assert best_fitness != float('inf')

    def test_hdpso_improves_over_iterations(self, sample_flowshop):
        """測試 HDPSO 隨迭代改進"""
        hdpso = HDPSO(
            flowshop=sample_flowshop,
            num_particles=20,
            max_iterations=30
        )

        best_sequence, best_fitness = hdpso.run(verbose=False)

        # 檢查適應值歷史
        assert len(hdpso.fitness_history) > 0

        # 適應值應該是單調遞減的（或至少不增加）
        for i in range(1, len(hdpso.fitness_history)):
            assert hdpso.fitness_history[i] <= hdpso.fitness_history[i-1]

    def test_hdpso_solution_validity(self, sample_flowshop):
        """測試 HDPSO 解的有效性"""
        hdpso = HDPSO(
            flowshop=sample_flowshop,
            num_particles=20,
            max_iterations=50
        )

        best_sequence, best_fitness = hdpso.run(verbose=False)

        # 驗證解是有效的排列
        assert set(best_sequence) == {1, 2, 3, 4, 5}
        assert len(best_sequence) == 5

        # 驗證適應值計算正確
        computed_fitness = sample_flowshop.compute_fitness(best_sequence)
        assert abs(computed_fitness - best_fitness) < 1e-10


class TestHDPSOIntegration:
    """整合測試"""

    def test_full_workflow(self, sample_flowshop):
        """測試完整工作流程"""
        hdpso = HDPSO(
            flowshop=sample_flowshop,
            num_particles=20,
            max_iterations=50,
            c1=1.0,
            c2=1.0,
            theta_max=0.9,
            theta_min=0.4
        )

        best_sequence, best_fitness = hdpso.run(verbose=False)

        # 計算詳細結果
        results = sample_flowshop.compute_schedule(best_sequence)

        assert len(results) == 5

        total_earliness = sum(r.earliness for r in results)
        total_tardiness = sum(r.tardiness for r in results)
        total = total_earliness + total_tardiness

        # 驗證目標函數值
        assert abs(total - best_fitness) < 1e-10

        # 驗證每個工作都有結果
        job_ids_in_results = {r.job_id for r in results}
        assert job_ids_in_results == {1, 2, 3, 4, 5}

    def test_deterministic_with_seed(self, sample_flowshop):
        """測試使用隨機種子的確定性（需要手動設置種子）"""
        import random

        random.seed(42)
        hdpso1 = HDPSO(flowshop=sample_flowshop, num_particles=10, max_iterations=10)
        _, fitness1 = hdpso1.run(verbose=False)

        random.seed(42)
        hdpso2 = HDPSO(flowshop=sample_flowshop, num_particles=10, max_iterations=10)
        _, fitness2 = hdpso2.run(verbose=False)

        # 相同種子應該產生相同結果
        assert fitness1 == fitness2


@pytest.mark.parametrize("num_particles,max_iterations", [
    (5, 10),
    (10, 20),
    (20, 50),
])
def test_different_parameters(sample_flowshop, num_particles, max_iterations):
    """測試不同參數組合"""
    hdpso = HDPSO(
        flowshop=sample_flowshop,
        num_particles=num_particles,
        max_iterations=max_iterations
    )

    best_sequence, best_fitness = hdpso.run(verbose=False)

    assert len(best_sequence) == 5
    assert best_fitness > 0
    assert best_fitness != float('inf')


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
