import numpy as np

class SimulateData:
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
        Proportion of observations in each cluster, length of the array defines number of simulated clusters.
    means : np.ndarray
        Mean of the Gaussian distribution for each cluster.
    variance_covariance_matrix : np.ndarray
        Matrix of variance and covariance for simulation.
    simulation_object : dict
        Object containing results of data attributes from simulation.
    """

    def __init__(
        self,
        n_observations: list[int],
        n_variables: int,
        n_relevant: list,
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
        self.simulation_object = {}

    # Simulate relevant variables
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
        self.simulation_object["true_labels"] = true_labels
        relevant_variables = np.array(samples)
        return relevant_variables

        # Simulate irrelevant variables

    def irrelevant_vars(self):
        """Returns array of irrelevant variables in simulation."""
        n_irrelevant = self.n_variables - self.n_relevant
        print(n_irrelevant)
        irrelevant_variables = np.random.randn(self.n_observations, n_irrelevant)
        return irrelevant_variables

    def data_sim(self):
        """Returns simulated data array."""
        # Combine relevant and irrelevant variables
        relevant_variables = self.relevant_vars()
        irrelevant_variables = self.irrelevant_vars()
        data = np.hstack((relevant_variables, irrelevant_variables))
        self.simulation_object["data"] = data
        return data

    # Make the permutations for the simulation
    def permutation(self):
        """Returns permutations for simulation."""
        permutations = np.random.permutation(self.n_variables)
        self.simulation_object["permutation"] = permutations
        return permutations

    # Shuffle the variables
    def shuffle_sim_data(self, data, permutation):
        """Shuffles randomised data for simulation.

        Params:
            data:np.ndarray
                Array of data generated from `self.data_sim()`
            permutation:np.ndarray
                Array of permutations generated from `self.permutations()`
        """
        shuffled_data = data[:, permutation]
        self.simulation_object["shuffled_data"] = shuffled_data

    # Now data contains 100 observations with 200 variables, where the first n_relevant are drawn
    # from the Gaussian mixtures and the rest are irrelevant variables from standard Gaussian.

    # permutation is just np.random.permutation(n_variables) <- this can be another function
    # data is just an np function on a couple of supplied variables, one of which is made in this for loop
    # shuffled data comes from data
    # true_labels is created in the for loop as well

    # i would rather return a dictionary of values or try to break these out into disparate functions
    # having functions returns more than one thing feels a) very unpythonic and b) imo is a bit confusing

    # return data, shuffled_data, true_labels, permutation