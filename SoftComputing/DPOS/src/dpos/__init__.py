"""DPOS - Discrete Particle Swarm Optimization for Flow Shop Scheduling"""

__version__ = "0.1.0"

from .job import Job
from .flow_shop import FlowShop, ScheduleResult
from .hdpso import HDPSO, HDPSOParticle, HDPSOOperations
from .mpso import MPSO, MPSOParticle, MPSOOperations
from .paper_case import (
    create_paper_flowshop,
    get_fcfs_sequence,
    get_paper_optimal_fitness,
    get_fcfs_fitness,
)

__all__ = [
    "Job",
    "FlowShop",
    "ScheduleResult",
    "HDPSO",
    "HDPSOParticle",
    "HDPSOOperations",
    "MPSO",
    "MPSOParticle",
    "MPSOOperations",
    "create_paper_flowshop",
    "get_fcfs_sequence",
    "get_paper_optimal_fitness",
    "get_fcfs_fitness",
]
