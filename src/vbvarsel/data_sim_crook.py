import numpy as np
from .experiment_data import ExperimentValues

class SimulateCrookData:
    # """
    # A class to represent simulated data as described by Crook et al. These 
    # attributes mirror the simulation parameters, which are set via the
    # `SimulationParameters()` function in `global_parameters.py`.

    # Attributes
    #     observation : int
    #         Number of observations to simulate.
    #     n_variables : int
    #         Number of variables to simulate.
    #     n_relevant : int
    #         Number of variables that are relevant.
    #     mixture_proportions : list[float]
    #         Proportion of observations in each cluster, length of the array defines 
    #         number of simulated clusters.
    #     means : list[int]
    #         Mean of the Gaussian distribution for each cluster.
    #     variance_covariance_matrix : np.ndarray
    #         Matrix of variance and covariance for simulation.

    # """

    def __init__(
        self,
        observation:int,
        n_variables: int,
        n_relevant: int,
        mixture_proportions: list,
        means: list,
        variance_covariance_matrix: np.ndarray,
    ):
        self.observation = observation
        self.n_variables = n_variables
        self.n_relevant = n_relevant
        self.mixture_proportions = mixture_proportions
        self.means = means
        self.variance_covariance_matrix = variance_covariance_matrix
        self.ExperimentValues = ExperimentValues()


    def relevant_vars(self) -> np.ndarray:
        """Returns array of relevant variables for use in simulation."""
        samples = []
        true_labels = []  # Store the true labels
        for _ in range(self.observation):
            # Select mixture component based on proportions
            component = np.random.choice(list(range(len(self.mixture_proportions))), p=self.mixture_proportions)
            true_labels.append(component)  # Store the true label
            
            mean_vector = np.full(self.n_relevant, self.means[component])
            sample = np.random.multivariate_normal(
                mean_vector, self.variance_covariance_matrix
            )
            samples.append(sample)

        # Convert list of samples to numpy array
        self.ExperimentValues.true_labels = true_labels
        return np.array(samples)


    def irrelevant_vars(self) -> np.ndarray:
        """Returns array of irrelevant variables in simulation."""
        n_irrelevant = self.n_variables - self.n_relevant
        return np.random.randn(self.observation, n_irrelevant)
         

    def data_sim(self) -> np.ndarray:
        """Returns simulated data array."""
        # Combine relevant and irrelevant variables
        relevant_variables = self.relevant_vars()
        irrelevant_variables = self.irrelevant_vars()
        data = np.hstack((relevant_variables, irrelevant_variables))
        self.ExperimentValues.data = data
        return data


    def permutation(self) -> np.ndarray:
        """Returns permutations for simulation."""
        permutations = np.random.permutation(self.n_variables)
        self.ExperimentValues.permutations = permutations
        return permutations


    def shuffle_sim_data(self, data, permutation) -> np.ndarray:
        """Shuffles randomised data for simulation.

        Params
            data: np.ndarray
                Array of data generated from `self.data_sim()`
            permutation: np.ndarray
                Array of permutations generated from `self.permutations()`
        """
        shuffled_data = data[:, permutation]
        self.ExperimentValues.shuffled_data = shuffled_data

