import numpy as np
import pandas as pd
import typing
import os


# This doesn't seem to be utilised
def normalise_data(data: pd.DataFrame) -> list:
    '''Function that returns a normalised dataframe from a non-normalised input.'''
    # Compute the mean and standard deviation of each column
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)

    # Subtract the mean and divide by the standard deviation
    array_normalized = (data - mean) / std

    return array_normalized


# Nor does this
def shuffle_data(
    normalised_data: pd.DataFrame, out: int = 0
) -> typing.Union[pd.Series, np.ndarray]:
    '''Shuffles a DataFrame of normalised data.

    Params:
        normalised_data: pd.DataFrame -> A normalisd dataframe.
        out: int -> how many columns to exclude

    Returns:
        shuffled_data, shuffled_indices: pd.Series, np.ndarray -> a tuple of
            shuffled data and their corresponding indices.
    '''

    # Number of columns
    num_columns = np.shape(normalised_data)[1] - out

    # Generate a permutation of column indices
    shuffled_indices = np.random.permutation(num_columns)

    # Concatenate with the indices of the last 10 columns
    shuffled_indices = np.concatenate(
        (shuffled_indices, np.arange(num_columns, np.shape(normalised_data)[1]))
    )

    # Shuffle the columns of the matrix
    shuffled_data = normalised_data[:, shuffled_indices]

    return shuffled_data, shuffled_indices


def load_data(data_loc: str | os.PathLike, clean_too: bool = False) -> list:
    """Loads data to be be used in simulations with option to clean data.

    Parameters:

    data_loc: str|os.Pathlike
        The file location of the spreadsheet, must be in CSV format.
    clean_too: bool, optional
        A flag to enable pre-determined cleaning. Should only be set to TRUE if
        using PAM50 datasets, data reformatting operations may not apply to
        other datasets, in which case a user should take the returned dataframe
        from this function and reformat their data as needed.

    Returns:

    raw_data|shuffled_data: list
        An array of data.
    """

    raw_data = pd.DataFrame(data_loc)

    if clean_too:

        normalised_data = normalise_data(raw_data)
        shuffled_data = shuffle_data(normalised_data)
        return shuffled_data
    return raw_data
