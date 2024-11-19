from dataclasses import dataclass, field
from numpy import ndarray

@dataclass
class ExperimentValues:
    true_labels: list[int] = field(default_factory=list)
    data: ndarray = None
    permutations: list[int] = field(default_factory=list)
    shuffled_data: ndarray = None
