import numpy as np
import pandas as pd
import typing
import os
from sklearn.preprocessing import LabelEncoder
from .experiment_data import SimulatedValues

class UserDataHandler:

    def __init__(self):
        self.SimulatedValues = SimulatedValues()

    def normalise_data(self, data: pd.DataFrame) -> list:
        '''Function that returns a normalised dataframe from a non-normalised input.'''
        #in the sample datasets, the 0th column was the DNA sequences, so we save that
        # to a list as the True Labels

        true_labels = data[data.columns[0]].to_list()
        encoded_labels = LabelEncoder.transform(true_labels)
        self.SimulatedValues.true_labels = encoded_labels
        data.drop(data.columns[0], axis=1, inplace=True)
        # Compute the mean and standard deviation of each column
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)

        # Subtract the mean and divide by the standard deviation
        array_normalized = (data - mean) / std
        array_normalized = array_normalized.to_numpy()
        self.SimulatedValues.data = array_normalized
        return array_normalized

    def shuffle_normalised_data(self,
        normalised_data: np.ndarray, 
        out: int = 0
    ) -> typing.Union[pd.Series, np.ndarray]:
        '''Shuffles a DataFrame of normalised data.

        Params
    ------
            normalised_data: pd.DataFrame -> A normalisd dataframe.
            out: int -> how many columns to exclude

        Returns
    -------
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

        Returns
    -------

        raw_data|shuffled_data: list
            An array of data.
        """

        raw_data = pd.read_csv(data_loc, header=0)
        if normalise:

            normalised_data = self.normalise_data(raw_data)
            shuffled_data = self.shuffle_normalised_data(normalised_data)
        else:
            self.SimulatedValues.data = raw_data

    # def save_data(data: pd.DataFrame, filename: str, save_path: pathlib.Path, with_index:bool=False):

    #     data.to_csv(path_or_buf=os.path.join(save_path, filename), index=with_index)
    #     print(f"{filename} saved to {save_path}.")


if __name__ == '__main__':
    UserDataHandler().load_data(r"C:\Users\Alan\Desktop\dev\\test importing\9_BRCA.pam50.50.csv",
                                normalise=True)