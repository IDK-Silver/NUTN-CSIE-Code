from dataclasses import dataclass
from typing import List

@dataclass
class Job:
    """ 一個工作 / 訂單 """
    job_id: int
    processing_times: List[float]  # 在每台機器上的處理時間
    due_date: float                 # 交期

    def __repr__(self):
        return f"Job({self.job_id})"
