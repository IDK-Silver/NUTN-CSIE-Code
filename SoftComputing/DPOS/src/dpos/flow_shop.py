from typing import List, Tuple
from dataclasses import dataclass
from .job import Job

@dataclass
class ScheduleResult:
    """排程結果"""
    job_id: int
    completion_time: float  # 完成時間
    due_date: float         # 交期
    earliness: float        # 提早
    tardiness: float        # 延遲

class FlowShop:
    """流水車間"""

    def __init__(self, num_machines: int):
        self.num_machines = num_machines
        self.jobs: List[Job] = []

    def add_job(self, job: Job):
        """新增一個工作"""
        if len(job.processing_times) != self.num_machines:
            raise ValueError(
                f"Job {job.job_id} 的處理時間數量 ({len(job.processing_times)}) "
                f"與機器數量 ({self.num_machines}) 不符"
            )
        self.jobs.append(job)

    def compute_schedule(self, sequence: List[int]) -> List[ScheduleResult]:
        """
        給定一個工作順序，計算每個工作的完成時間、提早、延遲

        Args:
            sequence: 工作的處理順序，例如 [3, 1, 2] 表示先做job3，再做job1，最後job2

        Returns:
            每個工作的排程結果
        """
        # 建立 job_id 到 Job 的對應
        job_dict = {job.job_id: job for job in self.jobs}

        # 記錄每台機器目前被佔用到什麼時候
        machine_available = [0.0] * self.num_machines

        # 記錄每個工作在上一台機器的完成時間
        job_completion = {}

        results = []

        for job_id in sequence:
            job = job_dict[job_id]

            # 這個工作要依序經過每台機器
            for m in range(self.num_machines):
                # 開始時間 = max(這台機器空閒的時間, 上一台機器完成的時間)
                if m == 0:
                    # 第一台機器：只需要等機器空閒
                    start_time = machine_available[m]
                else:
                    # 後續機器：要等機器空閒，也要等工作從上一台機器出來
                    start_time = max(machine_available[m], job_completion[job_id])

                # 完成時間 = 開始時間 + 處理時間
                end_time = start_time + job.processing_times[m]

                # 更新機器的佔用時間
                machine_available[m] = end_time

                # 記錄這個工作在這台機器的完成時間
                job_completion[job_id] = end_time

            # 最後一台機器的完成時間就是這個工作的總完成時間
            completion_time = job_completion[job_id]

            # 計算提早和延遲
            earliness = max(0, job.due_date - completion_time)
            tardiness = max(0, completion_time - job.due_date)

            results.append(ScheduleResult(
                job_id=job_id,
                completion_time=completion_time,
                due_date=job.due_date,
                earliness=earliness,
                tardiness=tardiness
            ))

        return results

    def compute_fitness(self, sequence: List[int]) -> float:
        """計算目標函數值（總提早 + 總延遲）"""
        results = self.compute_schedule(sequence)
        total = sum(r.earliness + r.tardiness for r in results)
        return total
