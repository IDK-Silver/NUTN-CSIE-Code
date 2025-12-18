"""
MPSO - Modified Particle Swarm Optimization

MPSO 使用機率轉移矩陣來表示粒子位置，透過排序轉換成實際的工作排序。
這樣可以直接使用標準 PSO 的連續值更新公式。

論文參考：
- Santosa, B., & Siswanto, N. (2018). Discrete particle swarm optimization 
  to solve multi-objective limited-wait hybrid flow shop scheduling problem.
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional
import random

# 位置 = 機率向量（連續值）
ProbabilityPosition = List[float]

# 排序 = 工作順序（離散值）
Sequence = List[int]

# 速度 = 連續值向量
Velocity = List[float]


class MPSOParticle:
    """MPSO 的粒子"""

    def __init__(self, num_jobs: int):
        """
        初始化粒子

        Args:
            num_jobs: 工作數量
        """
        self.num_jobs = num_jobs
        
        # 位置：隨機生成 [0, 1] 之間的機率值
        self.position: ProbabilityPosition = [random.random() for _ in range(num_jobs)]
        
        # 速度：初始為 0
        self.velocity: Velocity = [0.0] * num_jobs
        
        # pBest 初始化為目前位置
        self.pbest_position: ProbabilityPosition = self.position.copy()
        self.pbest_fitness: float = float('inf')
        
        # 目前的適應值
        self.fitness: float = float('inf')

    def __repr__(self):
        seq = MPSOOperations.position_to_sequence(self.position)
        return f"MPSOParticle(seq={seq}, fit={self.fitness:.2f})"


class MPSOOperations:
    """MPSO 的各種運算操作"""

    @staticmethod
    def position_to_sequence(position: ProbabilityPosition, job_ids: Optional[List[int]] = None) -> Sequence:
        """
        將機率位置轉換成工作排序
        
        方法：按照機率值的大小排序，取得索引順序
        
        Args:
            position: 機率向量，例如 [0.1067, 0.8687, 0.4314, 0.1361, 0.8530]
            job_ids: 工作 ID 列表，如果為 None 則使用 1-based index
            
        Returns:
            工作排序，例如 [1, 4, 3, 5, 2]（按機率值由小到大排序的工作 ID）
            
        Example:
            position = [0.1067, 0.8687, 0.4314, 0.1361, 0.8530]
            
            索引和值的對應：
            index 0: 0.1067 (最小)
            index 1: 0.8687 (最大)
            index 2: 0.4314 (中間)
            index 3: 0.1361 (第二小)
            index 4: 0.8530 (第二大)
            
            按值排序後的索引順序：[0, 3, 2, 4, 1]
            轉成 1-based: [1, 4, 3, 5, 2]
        """
        # 取得排序後的索引（由小到大）
        sorted_indices = sorted(range(len(position)), key=lambda i: position[i])
        
        if job_ids is None:
            # 使用 1-based index
            return [i + 1 for i in sorted_indices]
        else:
            # 使用提供的 job_ids
            return [job_ids[i] for i in sorted_indices]

    @staticmethod
    def normalize_position(position: ProbabilityPosition) -> ProbabilityPosition:
        """
        正規化位置，確保所有值都在 [0, 1] 之間
        
        Args:
            position: 可能超出範圍的機率向量
            
        Returns:
            正規化後的機率向量
        """
        min_val = min(position)
        max_val = max(position)
        
        # 避免除以零
        if max_val == min_val:
            return [0.5] * len(position)
        
        # 線性正規化到 [0, 1]
        return [(x - min_val) / (max_val - min_val) for x in position]

    @staticmethod
    def clip_position(position: ProbabilityPosition, 
                      min_val: float = 0.0, 
                      max_val: float = 1.0) -> ProbabilityPosition:
        """
        裁剪位置，將超出範圍的值限制在 [min_val, max_val]
        
        Args:
            position: 可能超出範圍的機率向量
            min_val: 最小值
            max_val: 最大值
            
        Returns:
            裁剪後的機率向量
        """
        return [max(min_val, min(max_val, x)) for x in position]


class MPSO:
    """MPSO 演算法"""

    def __init__(
        self,
        flowshop,  # FlowShop 物件
        num_particles: int = 30,
        max_iterations: int = 100,
        c1: float = 1.0,
        c2: float = 1.0,
        theta_max: float = 0.9,
        theta_min: float = 0.4,
        use_normalization: bool = True  # 是否使用正規化（否則用裁剪）
    ):
        """
        初始化 MPSO 演算法
        
        Args:
            flowshop: FlowShop 物件
            num_particles: 粒子數量
            max_iterations: 最大迭代次數
            c1: 認知學習因子
            c2: 社會學習因子
            theta_max: 最大慣性權重
            theta_min: 最小慣性權重
            use_normalization: 是否使用正規化處理超出範圍的值
        """
        self.flowshop = flowshop
        self.num_particles = num_particles
        self.max_iterations = max_iterations
        self.c1 = c1
        self.c2 = c2
        self.theta_max = theta_max
        self.theta_min = theta_min
        self.use_normalization = use_normalization
        
        # 取得工作數量和 ID
        self.num_jobs = len(flowshop.jobs)
        self.job_ids = [job.job_id for job in flowshop.jobs]
        
        # 初始化粒子群
        self.particles: List[MPSOParticle] = []
        
        # 全域最佳
        self.gbest_position: ProbabilityPosition = []
        self.gbest_fitness: float = float('inf')
        
        # 記錄每次迭代的最佳適應值（用於畫圖）
        self.fitness_history: List[float] = []

    def _evaluate_particle(self, particle: MPSOParticle) -> float:
        """
        評估粒子的適應值
        
        Args:
            particle: 粒子
            
        Returns:
            適應值（總提早 + 總延遲）
        """
        # 將機率位置轉換成工作排序
        sequence = MPSOOperations.position_to_sequence(particle.position, self.job_ids)
        # 計算適應值
        return self.flowshop.compute_fitness(sequence)

    def initialize(self):
        """初始化粒子群"""
        self.particles = []
        self.gbest_fitness = float('inf')
        
        for _ in range(self.num_particles):
            particle = MPSOParticle(self.num_jobs)
            
            # 計算初始適應值
            particle.fitness = self._evaluate_particle(particle)
            particle.pbest_fitness = particle.fitness
            particle.pbest_position = particle.position.copy()
            
            self.particles.append(particle)
            
            # 更新全域最佳
            if particle.fitness < self.gbest_fitness:
                self.gbest_fitness = particle.fitness
                self.gbest_position = particle.position.copy()

    def get_theta(self, iteration: int) -> float:
        """
        計算當前迭代的慣性權重
        
        θ_t = θ_max - (θ_max - θ_min) / i_max * i
        
        Args:
            iteration: 當前迭代次數
            
        Returns:
            慣性權重
        """
        theta = self.theta_max - (self.theta_max - self.theta_min) / self.max_iterations * iteration
        return theta

    def update_particle(self, particle: MPSOParticle, iteration: int):
        """
        更新單個粒子的速度和位置
        
        速度更新公式（標準 PSO）：
        v_new = θ * v_old + c1 * r1 * (pBest - x) + c2 * r2 * (gBest - x)
        
        位置更新公式：
        x_new = x_old + v_new
        
        Args:
            particle: 粒子
            iteration: 當前迭代次數
        """
        # 取得參數
        theta = self.get_theta(iteration)
        r1 = random.random()
        r2 = random.random()
        
        # 更新每個維度的速度和位置
        new_velocity = []
        new_position = []
        
        for i in range(self.num_jobs):
            # ===== 計算新速度 =====
            # v_new = θ * v_old + c1 * r1 * (pBest - x) + c2 * r2 * (gBest - x)
            
            inertia = theta * particle.velocity[i]
            cognitive = self.c1 * r1 * (particle.pbest_position[i] - particle.position[i])
            social = self.c2 * r2 * (self.gbest_position[i] - particle.position[i])
            
            v_new = inertia + cognitive + social
            new_velocity.append(v_new)
            
            # ===== 計算新位置 =====
            # x_new = x_old + v_new
            x_new = particle.position[i] + v_new
            new_position.append(x_new)
        
        # 處理超出範圍的位置值
        if self.use_normalization:
            new_position = MPSOOperations.normalize_position(new_position)
        else:
            new_position = MPSOOperations.clip_position(new_position)
        
        # 更新粒子狀態
        particle.velocity = new_velocity
        particle.position = new_position
        
        # 計算新的適應值
        particle.fitness = self._evaluate_particle(particle)
        
        # 更新 pBest
        if particle.fitness < particle.pbest_fitness:
            particle.pbest_fitness = particle.fitness
            particle.pbest_position = particle.position.copy()

    def run(self, verbose: bool = True) -> Tuple[Sequence, float]:
        """
        執行 MPSO 演算法
        
        Args:
            verbose: 是否輸出過程資訊
            
        Returns:
            (最佳排序, 最佳適應值)
        """
        # 初始化
        self.initialize()
        self.fitness_history = [self.gbest_fitness]
        
        if verbose:
            best_sequence = MPSOOperations.position_to_sequence(self.gbest_position, self.job_ids)
            print(f"初始化完成，初始最佳適應值: {self.gbest_fitness:.2f}")
            print(f"初始最佳排序: {best_sequence}")
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
        
        # 取得最終最佳排序
        best_sequence = MPSOOperations.position_to_sequence(self.gbest_position, self.job_ids)
        
        if verbose:
            print()
            print(f"最終最佳排序: {best_sequence}")
            print(f"最終最佳適應值: {self.gbest_fitness:.2f}")
        
        return best_sequence, self.gbest_fitness

    def get_best_sequence(self) -> Sequence:
        """取得目前的最佳排序"""
        return MPSOOperations.position_to_sequence(self.gbest_position, self.job_ids)
