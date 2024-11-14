from dataclasses import dataclass, field
from numpy import ndarray

@dataclass
class ExperimentValues:
    true_labels: list[int] = field(default_factory=list)
    '''Just a test to see what happens if I put stuff down here.'''
    data: ndarray = None
    permutations: list[int] = field(default_factory=list)
    shuffled_data: ndarray = None
