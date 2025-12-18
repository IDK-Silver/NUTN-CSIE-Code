from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class TaskInfo:
    # the unique task id
    task_id: int

    # a array store each pipe process time
    cost_times: List[int]


class Machine:
    def __init__(self, num_of_pipe: int):
        self.num_of_pipe = num_of_pipe
        self.__task_list: List[TaskInfo] = []

    def add_task(self, task_info: TaskInfo):

        if len(task_info.cost_times) != self.num_of_pipe:
            raise Exception("num of pipe size is not same.")

        self.__task_list.append(task_info)


    def compute_cost_time(self):

        pass
