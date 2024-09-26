import numpy as np
from experiment_data import SimulatedValues

class SimulateCrookData:
    """
    A class to represent simulated data as described by Crook et al.

    Attributes:
    n_observations : list[int]
        Number of observations to simulate.
    n_variables : int
        Number of variables to simulate.
    n_relevant : int
        Number of variables that are relevant.
    mixture_proportions : np.ndarray
        Proportion of observations in each cluster, length of the array defines 
        number of simulated clusters.
    means : np.ndarray
        Mean of the Gaussian distribution for each cluster.
    variance_covariance_matrix : np.ndarray
        Matrix of variance and covariance for simulation.
    simulation_object : dict
        Object containing results of data attributes from simulation.

    These attributes mirror the simulation parameters, which are set via the
    `establish_sim_params` function in `simulation.py`.
    """

    def __init__(
        self,
        n_observations:int,
        n_variables: int,
        n_relevant: int,
        mixture_proportions: list,
        means: list,
        variance_covariance_matrix: np.ndarray,
    ):
        self.n_observations = n_observations
        self.n_variables = n_variables
        self.n_relevant = n_relevant
        self.mixture_proportions = mixture_proportions
        self.means = means
        self.variance_covariance_matrix = variance_covariance_matrix
        self.SimulatedValues = SimulatedValues()


    def relevant_vars(self):
        """Returns array of relevant variables for use in simulation."""
        samples = []
        true_labels = []  # Store the true labels
        for _ in range(self.n_observations):
            # Select mixture component based on proportions
            component = np.random.choice([0, 1, 2], p=self.mixture_proportions)
            true_labels.append(component)  # Store the true label
            mean_vector = np.full(self.n_relevant, self.means[component])
            sample = np.random.multivariate_normal(
                mean_vector, self.variance_covariance_matrix
            )
            samples.append(sample)

        # Convert list of samples to numpy array
        self.SimulatedValues.true_labels = true_labels
        return np.array(samples)


    def irrelevant_vars(self):
        """Returns array of irrelevant variables in simulation."""
        n_irrelevant = self.n_variables - self.n_relevant
        return np.random.randn(self.n_observations, n_irrelevant)
         

    def data_sim(self):
        """Returns simulated data array."""
        # Combine relevant and irrelevant variables
        relevant_variables = self.relevant_vars()
        irrelevant_variables = self.irrelevant_vars()
        data = np.hstack((relevant_variables, irrelevant_variables))
        self.SimulatedValues.data = data
        return data


    def permutation(self):
        """Returns permutations for simulation."""
        permutations = np.random.permutation(self.n_variables)
        self.SimulatedValues.permutations = permutations
        return permutations


    def shuffle_sim_data(self, data, permutation):
        """Shuffles randomised data for simulation.

        Params:
            data:np.ndarray
                Array of data generated from `self.data_sim()`
            permutation:np.ndarray
                Array of permutations generated from `self.permutations()`
        """
        shuffled_data = data[:, permutation]
        self.SimulatedValues.shuffled_data = shuffled_data

    
if __name__ == '__main__':
    test = []
    for n in range(100):
        test.append(np.random.choice([0, 1, 2], p=[0.2, 0.3, 0.5]))
    one = 0
    two = 0
    zero = 0
    for t in test:
        if t == 0:
            zero += 1
        if t == 1:
            one += 1
        else:
            two += 1
    
    print(f"zeroes: {zero/100}")
    print(f"ones: {one/100}")
    print(f"twos: {two/100}")



    # scd = SimulateCrookData(10, 100, 10, [0.5, 0.3, 0.2], [0, 2, -2],  np.identity(10))
    # scd.relevant_vars()
    # print(scd.SimulatedValues.true_labels)

    