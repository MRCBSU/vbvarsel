import numpy as np
import pandas as pd
import typing



# This doesn't seem to be utilised
def normalise_data(data: pd.DataFrame) -> list:

    # Compute the mean and standard deviation of each column
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)

    # Subtract the mean and divide by the standard deviation
    array_normalized = (data - mean) / std

    return array_normalized


# Nor does this
def shuffle_data(
    data: pd.DataFrame, out: int = 0
) -> typing.Union[pd.Series, np.ndarray]:

    # Number of columns
    num_columns = np.shape(data)[1] - out

    # Generate a permutation of column indices
    shuffled_indices = np.random.permutation(num_columns)

    # Concatenate with the indices of the last 10 columns
    shuffled_indices = np.concatenate(
        (shuffled_indices, np.arange(num_columns, np.shape(data)[1]))
    )

    # Shuffle the columns of the matrix
    shuffled_data = data[:, shuffled_indices]

    return shuffled_data, shuffled_indices



if __name__ == "__main__":
    import dataSimCrook
    n_observations = [10, 50]
    n_variables = 200
    n_relevants = [
        10,
        20,
        50,
        100,
    ]  # For example, change this to vary the number of relevant variables
    mixture_proportions = [0.5, 0.3, 0.2]
    means = [0, 2, -2]

    # MODEL AVERAGING

    # setting the hyperparameters

    # convergence threshold
    threshold = 1e-1

    K1 = 5  # num components in inference

    # alpha0 = 0.01 #prior coefficient count (for Dir)
    alpha0 = 1 / (K1)  # cabassi

    beta0 = (1e-3) * 1.0
    a0 = 3.0

    # variable selection
    d = 1

    T_max = 1.0

    max_itr = 25

    max_models = 10
    convergence_ELBO = []
    convergence_itr = []
    clust_predictions = []
    variable_selected = []
    times = []
    ARIs = []
    n_relevant_var = []
    n_obs = []

    n_rel_rel = []  # correct relevant

    n_irr_irr = []  # correct irrelevant

    for p in range(len(n_observations)):
        for n_rel in range(len(n_relevants)):
            for i in range(max_models):

                print("Model " + str(i))
                print("obs " + str(p))
                print("rel " + str(n_rel))
                print()
                n_relevant_var.append(n_relevants[n_rel])
                n_obs.append(n_observations[p])

                variance_covariance_matrix = np.identity(n_relevants[n_rel])

                data_crook = dataSimCrook.SimulateData(
                    n_observations[p],
                    n_variables,
                    n_relevants[n_rel],
                    mixture_proportions,
                    means,
                    variance_covariance_matrix,
                )
                d = data_crook.data_sim()
                perms = data_crook.permutation()
                data_crook.shuffle_sim_data(d, perms)
                print(data_crook.simulation_object)
