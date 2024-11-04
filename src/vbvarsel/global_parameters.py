from dataclasses import dataclass, field

#Changed this over to a dataclass versus the previous function-based implementaton.
#Seemed to make more sense this way since I'm not really doing anything
#To the params once they are set, plus I don't like named tuples plus this
#Seems more easier to pass around, comes with as many dunder attribs as wanted
#And just seems like an overall better idea.
@dataclass(frozen=True, order=True)
class Hyperparameters:
    '''Class representing the hyperparameters for the simulation.
    
    Attributes
    ----------
    threshold: float
        The threshold for simulation convergence. (default 1e-1)
    k1: int
        Maximum number of clusters to simulate for. (default 5)
    alpha0: float
        Prior coefficient count, also known as the concentration parameter for
        Dirichelet prior on the mixture proportions. This field is calculated
        from 1/k1. (Default 0.2)
    a0: int
        Degrees of freedom for the Gamma prior on the cluster precision, which
        controls the shape of the Gamma distribution. A higher number results
        in a more peaked distribution. (Default 3)
    beta0: float
        Shrinkage parameter of the Gaussian conditional prior on the cluster
        mean. This influences the tightness and spread of the cluster, smaller
        shrinkage leads to tighter clusters. (Default 1e-3)
    d0: int
        Shape parameter of the Beta distribution on the probability. A value of
        1 results in a uniform distribution. (Default 1)
    t_max: int
        Maximum starting annealing temperature. Value of 1 has no annealing.
        (Default 1)
    max_itr: int
        Maximum number of iterations. (Default 25)
    max_models: int
        Maximum number of models to run for averaging (Default 10)
    '''
    threshold: float = 1e-1
    k1: int = 5
    alpha0: float = field(init=False)
    a0: int = 3
    beta0: float = 1e-3
    d0: int = 1
    t_max: int = 1
    max_itr: int = 25
    max_models: int = 10

    def __post_init__(self):
        # nifty little hack to temporarily unfreeze the class and set alpha0
        object.__setattr__(self, 'alpha0', 1/self.k1)

@dataclass(order=True)
class SimulationParameters:
    '''Class representing the simulation parameters.
    
    These parameters are the "settings" for the simulation experiment. For more
    information regarding the simulation, see [INSERT PAPER HERE]. Mixture 
    proportions must be numbers between 0 and 1. The n_relevants array must
    be all numbers that are less than the n_variables value, as it is not
    possible to have a higher number of relevant variables than total variables.
    

    Attributes
    ----------
    n_observations: list[int]
        The number of observations to observe in the simulation. 
        (Default [100,1000])
    n_variables: int
        The number of variables to consider. The value must exceed the largest
        number in `n_relevants`. (Default 200)
    n_relevants: list[int]
        A list of integer values of different quantities of relevant variables
        to test for. These numbers should not exceed `n_variables`.
        (Default [10, 20, 50, 100])
    mixture_proportions: list[float]
        A list of float values for ~ proportion of observations in each cluster.
        The length of the array influence the number of simulated clusters. All
        values should be between 0 and 1, and all values must sum to 1. 
        (Default [0.5, 0.3, 0.2])
    means: list[int]
        List of integers of Gaussian distributions for each cluster. (Default
        [-2, 0, 2])
    '''
    n_observations: list[int] = field(default_factory=lambda:[100,1000])
    n_variables: int = 200
    n_relevants: list[int] = field(default_factory=lambda:[10, 20, 50, 100])
    mixture_proportions: list[float] = field(default_factory=lambda:[0.2, 0.3, 0.5])
    means: list[int] = field(default_factory=lambda:[-2, 0, 2])
