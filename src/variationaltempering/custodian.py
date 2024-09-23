import numpy as np
import pandas as pd
import typing
import os
import pathlib
from experiment_data import SimulatedValues

class UserDataHandler:

    def __init__(self):
        self.SimulatedValues = SimulatedValues()

    def normalise_data(self, data: pd.DataFrame) -> list:
        '''Function that returns a normalised dataframe from a non-normalised input.'''
        # Compute the mean and standard deviation of each column
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)

        # Subtract the mean and divide by the standard deviation
        array_normalized = (data - mean) / std
        self.SimulatedValues.data = array_normalized
        return array_normalized

    def shuffle_normalised_data(self,
        normalised_data: pd.DataFrame, 
        out: int = 0
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
        self.SimulatedValues.shuffled_data = shuffled_data
        self.SimulatedValues.permutations = shuffled_indices
        return shuffled_data, shuffled_indices


    def load_data(self, data_loc: str | os.PathLike, 
                normalise: bool = False) -> list:
        """Loads data to be be used in simulations with option to clean data.

        Parameters:

        data_loc: str|os.Pathlike
            The file location of the spreadsheet, must be in CSV format.
        normalise: bool, optional
            A flag to enable pre-determined cleaning.

        Returns:

        raw_data|shuffled_data: list
            An array of data.
        """

        raw_data = pd.read_csv(data_loc, header=None,)

        if normalise:

            normalised_data = self.normalise_data(raw_data)
            shuffled_data = self.shuffle_normalised_data(normalised_data)
            return shuffled_data
        else:
            self.SimulatedValues.data = raw_data

    def save_data(data: pd.DataFrame, filename: str, save_path: pathlib.Path, with_index:bool=False):

        data.to_csv(path_or_buf=os.path.join(save_path, filename), index=with_index)
        print(f"{filename} saved to {save_path}.")


# if __name__ == '__main__':
#     df = pd.DataFrame(data={'col1': [1,2,3], 'col2':[4,5,6]})
#     save_data(data=df, filename="test.csv", save_path=r"C:\Users\Alan\Desktop\dev\variationalTempering_beta")