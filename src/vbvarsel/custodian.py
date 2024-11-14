import numpy as np
import pandas as pd
import typing
import os
from sklearn.preprocessing import LabelEncoder
from .experiment_data import ExperimentValues

class UserDataHandler:

    def __init__(self):
        '''Initalizer to create the ExperimentValues object.'''
        self.ExperimentValues = ExperimentValues()

    def normalise_data(self, data: pd.DataFrame) -> np.ndarray:
        '''Function that returns a normalised dataframe from a non-normalised input.
        
        Params
           data: pd.DataFrame
            DataFrame of unsorted, unshuffled and non-normalised data to be used.
        Returns
            array_normalised: np.ndarray
                2-D array of normalised data

        '''

        # Compute the mean and standard deviation of each column
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)

        # Subtract the mean and divide by the standard deviation
        array_normalized = (data - mean) / std
        array_normalized = array_normalized.to_numpy()
        self.ExperimentValues.data = array_normalized
        return array_normalized

    def shuffle_normalised_data(self,
        normalised_data: np.ndarray
    ) -> typing.Union[pd.Series, np.ndarray]:
        '''Shuffles a DataFrame of normalised data.

        Params
            normalised_data: pd.DataFrame
                A normalisd dataframe of numerical data.
        Returns
            Tuple:
                shuffled_data: pd.Series, shuffled_indices: np.ndarray
                    A tuple of shuffled data and their corresponding indices.

        '''

        # Number of columns
        num_columns = np.shape(normalised_data)[1]

        # Generate a permutation of column indices
        shuffled_indices = np.random.permutation(num_columns)
        # Concatenate with the indices of the last 10 columns
        shuffled_indices = np.concatenate(
            (shuffled_indices, np.arange(num_columns, np.shape(normalised_data)[1]))
        )
        # Shuffle the columns of the matrix
        shuffled_data = normalised_data[:, shuffled_indices]
        
        self.ExperimentValues.shuffled_data = shuffled_data
        self.ExperimentValues.permutations = shuffled_indices
        return shuffled_data, shuffled_indices
    
    def load_data(
                self, 
                data_source: str | os.PathLike,
                cols_to_ignore: list[str] = None,
                labels: str | list[str] = None,
                header: int = 0,
                index_col: bool = False,
                ) -> None:
        """Loads data to be be used in simulations with option to clean data.

        Params
            data_loc: str | os.Pathlike
                The file location of the spreadsheet, must be in CSV format.
                IMPORTANT Format note: Columns should be variables, and rows should be observation.
                Header rows and index column will not be loaded. CSV must only have numerical values.
                Non-numerical values can be passed, via the `cols_to_ignore` parameter. 
            cols_to_ignore: list[str] (Optional) (Default: None)
                Any columns which are irrelevant or non-numerical to be dropped. 
            labels: str | list[str] (Optional) (Default: None)
                Labels to be used to calculate the ARI to check clustering accuracy.
                This parameter is optional but strongly encouraged. If a string value
                is passed, logic assumes that is the name of a column to be used as
                labels. Alternatively, a list of strings may be passed separately.
                Label column can be included in the `cols_to_drop` parameter, labels
                are extracted before the ignored columns are dropped. 
            header: int (Optional) (Default: 0)
                Parameter to determine the header of the incoming dataframe. 
            index_col: bool (Optional) (Default: False)
                Parameter to determine whether to include an index col. 
        Returns
            None

        """

        raw_data = pd.read_csv(data_source, header=header, index_col=index_col)

        if isinstance(labels, str):
            self.ExperimentValues.true_labels = raw_data[labels].to_numpy()
        else:
            self.ExperimentValues.true_labels = np.array(labels)
        
        raw_data = raw_data.drop(cols_to_ignore, axis=1)
        normalised_data = self.normalise_data(raw_data)
        self.shuffle_normalised_data(normalised_data)