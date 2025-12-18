
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import random
import math

# 交換動作的型別：(位置a, 位置b)
Swap = Tuple[int, int]

# 速度 = 一連串交換動作
Velocity = List[Swap]

# 位置 = 一個排列（工作順序）
Position = List[int]


class HDPSOParticle:
    """HDPSO 的粒子"""

    def __init__(self, job_ids: List[int]):
        """
        初始化粒子

        Args:
            job_ids: 所有工作的 ID 列表，例如 [1, 2, 3, 4, 5]
        """
        # 隨機打亂作為初始位置
        self.position: Position = job_ids.copy()
        random.shuffle(self.position)

        # 初始速度為空
        self.velocity: Velocity = []

        # pBest 初始化為目前位置
        self.pbest_position: Position = self.position.copy()
        self.pbest_fitness: float = float('inf')

        # 目前的適應值
        self.fitness: float = float('inf')

    def __repr__(self):
        return f"Particle(pos={self.position}, fit={self.fitness:.2f})"


class HDPSOOperations:
    """HDPSO 的各種運算操作"""

    @staticmethod
    def subtract_positions(target: Position, source: Position) -> Velocity:
        """
        位置相減：找出從 source 變成 target 需要的交換動作

        target - source = velocity
        意思是：source + velocity = target
        """
        # 複製一份來操作
        current = source.copy()
        velocity = []

        for i in range(len(target)):
            if current[i] != target[i]:
                # 找到 target[i] 在 current 中的位置
                j = current.index(target[i])
                # 交換
                current[i], current[j] = current[j], current[i]
                # 記錄這個交換（用 1-based index，符合論文）
                velocity.append((i + 1, j + 1))

        return velocity

    @staticmethod
    def multiply_coefficient(c: float, velocity: Velocity) -> Velocity:
        """
        係數乘以速度

        根據論文的規則：
        - c = 0: 回傳空的
        - 0 < c <= 1: 從前面保留 ceil(c * len) 組
        - c > 1: 重複 + 截斷
        - c < 0: 先反轉再處理
        """
        if len(velocity) == 0:
            return []

        if c == 0:
            return []

        if c < 0:
            # 反轉速度順序
            reversed_v = velocity[::-1]
            return HDPSOOperations.multiply_coefficient(abs(c), reversed_v)

        if 0 < c <= 1:
            # 截斷：保留前 ceil(c * len) 組
            keep_count = math.ceil(c * len(velocity))
            return velocity[:keep_count]

        # c > 1 的情況
        # 拆成整數部分 k 和小數部分 c'
        k = int(c)  # 整數部分
        c_prime = c - k  # 小數部分

        # 重複 k 次完整的速度
        result = velocity * k

        # 再加上 c' * velocity
        if c_prime > 0:
            extra = HDPSOOperations.multiply_coefficient(c_prime, velocity)
            result = result + extra

        return result

    @staticmethod
    def add_velocities(v1: Velocity, v2: Velocity) -> Velocity:
        """
        速度相加：串接兩個速度

        v1 ⊕ v2 = v1 的所有交換 + v2 的所有交換
        """
        return v1 + v2

    @staticmethod
    def apply_velocity(position: Position, velocity: Velocity) -> Position:
        """
        位置加上速度：執行所有交換動作

        position + velocity = new_position
        """
        result = position.copy()

        for (a, b) in velocity:
            # 轉成 0-based index
            i, j = a - 1, b - 1
            # 交換
            result[i], result[j] = result[j], result[i]

        return result


class HDPSO:
    """HDPSO 演算法"""

    def __init__(
        self,
        flowshop,  # FlowShop 物件
        num_particles: int = 30,
        max_iterations: int = 100,
        c1: float = 1.0,
        c2: float = 1.0,
        theta_max: float = 0.9,
        theta_min: float = 0.4
    ):
        self.flowshop = flowshop
        self.num_particles = num_particles
        self.max_iterations = max_iterations
        self.c1 = c1
        self.c2 = c2
        self.theta_max = theta_max
        self.theta_min = theta_min

        # 取得所有工作的 ID
        self.job_ids = [job.job_id for job in flowshop.jobs]

        # 初始化粒子群
        self.particles: List[HDPSOParticle] = []

        # 全域最佳
        self.gbest_position: Position = []
        self.gbest_fitness: float = float('inf')

        # 記錄每次迭代的最佳適應值（用於畫圖）
        self.fitness_history: List[float] = []

    def initialize(self):
        """初始化粒子群"""
        self.particles = []

        for _ in range(self.num_particles):
            particle = HDPSOParticle(self.job_ids)

            # 計算初始適應值
            particle.fitness = self.flowshop.compute_fitness(particle.position)
            particle.pbest_fitness = particle.fitness
            particle.pbest_position = particle.position.copy()

            self.particles.append(particle)

            # 更新全域最佳
            if particle.fitness < self.gbest_fitness:
                self.gbest_fitness = particle.fitness
                self.gbest_position = particle.position.copy()

    def get_theta(self, iteration: int) -> float:
        """計算當前迭代的慣性權重"""
        # θ_t = θ_max - (θ_max - θ_min) / i_max * i
        theta = self.theta_max - (self.theta_max - self.theta_min) / self.max_iterations * iteration
        return theta

    def update_particle(self, particle: HDPSOParticle, iteration: int):
        """更新單個粒子的速度和位置"""
        ops = HDPSOOperations

        # 取得參數
        theta = self.get_theta(iteration)
        r1 = random.random()
        r2 = random.random()

        # ===== 計算新速度 =====
        # v_new = θ * v_old + c1*r1*(pBest - x) + c2*r2*(gBest - x)

        # 第一項：慣性項 θ * v_old
        inertia = ops.multiply_coefficient(theta, particle.velocity)

        # 第二項：認知項 c1*r1*(pBest - x)
        diff_pbest = ops.subtract_positions(particle.pbest_position, particle.position)
        cognitive = ops.multiply_coefficient(self.c1 * r1, diff_pbest)

        # 第三項：社會項 c2*r2*(gBest - x)
        diff_gbest = ops.subtract_positions(self.gbest_position, particle.position)
        social = ops.multiply_coefficient(self.c2 * r2, diff_gbest)

        # 合併三項
        new_velocity = ops.add_velocities(inertia, cognitive)
        new_velocity = ops.add_velocities(new_velocity, social)

        # ===== 更新位置 =====
        # x_new = x_old + v_new
        new_position = ops.apply_velocity(particle.position, new_velocity)

        # 更新粒子狀態
        particle.velocity = new_velocity
        particle.position = new_position

        # 計算新的適應值
        particle.fitness = self.flowshop.compute_fitness(particle.position)

        # 更新 pBest
        if particle.fitness < particle.pbest_fitness:
            particle.pbest_fitness = particle.fitness
            particle.pbest_position = particle.position.copy()

    def run(self, verbose: bool = True) -> Tuple[Position, float]:
        """
        執行 HDPSO 演算法

        Returns:
            (最佳排序, 最佳適應值)
        """
        # 初始化
        self.initialize()
        self.fitness_history = [self.gbest_fitness]

        if verbose:
            print(f"初始化完成，初始最佳適應值: {self.gbest_fitness:.2f}")
            print(f"初始最佳排序: {self.gbest_position}")
            print()

        # 迭代
        for iteration in range(1, self.max_iterations + 1):
            # 更新每個粒子
            for particle in self.particles:
                self.update_particle(particle, iteration)

                # 更新全域最佳
                if particle.fitness < self.gbest_fitness:
                    self.gbest_fitness = particle.fitness
                    self.gbest_position = particle.position.copy()

            # 記錄歷史
            self.fitness_history.append(self.gbest_fitness)

            # 輸出進度
            if verbose and (iteration % 10 == 0 or iteration == 1):
                theta = self.get_theta(iteration)
                print(f"迭代 {iteration:3d}: 最佳適應值 = {self.gbest_fitness:.2f}, θ = {theta:.3f}")

        if verbose:
            print()
            print(f"最終最佳排序: {self.gbest_position}")
            print(f"最終最佳適應值: {self.gbest_fitness:.2f}")

        return self.gbest_position, self.gbest_fitness
