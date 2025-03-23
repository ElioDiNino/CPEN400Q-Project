from numpy import ndarray
from abc import ABC, abstractmethod
import csv
import numpy as np


class ForecastingMethod(ABC):
    def __init__(self):
        """Basic initialization stuff"""
        pass

    def load_data(
        self, filepath: str, windows_size: int = 12
    ) -> tuple[ndarray, ndarray]:
        """
        Load data from a CSV file and convert it to a format that can be used
        for time-series forecasting. No preprocessing is done here.

        Args:
            - filepath (str): Path to the CSV file. File should have a single
              column of data where the first row is the header.
            - windows_size (int): Number of data points to use for each
              prediction. Default is 12.

        Returns:
            tuple:
            - X (ndarray): 2D array of shape (n_examples, window_size) where
              n_examples is the number of examples.
            - y (ndarray): 1D array of shape (n_examples,) where
              n_examples is the number of examples.
        """
        with open(filepath, mode="r") as file:
            reader = csv.reader(file)
            data = list(reader)[1:]  # don't include the header

        # convert data to float
        data = [float(datapoint[0]) for datapoint in data]
        if windows_size >= len(data):
            raise ValueError(
                f"Window size {windows_size} must be smaller "
                f"than the data size {len(data)}"
            )

        # create X and y
        X = np.zeros((len(data) - windows_size, windows_size))
        y = np.zeros(len(data) - windows_size)

        # stop window right before the last value so it can be used in y
        for i in range(len(data) - windows_size):
            X[i] = data[i : i + windows_size]
            y[i] = data[i + windows_size]

        return X, y

    def preprocess_data(self, X: ndarray, y: ndarray) -> dict[str, ndarray]:
        """Applies differencing and normalization to the input data"""
        pass

    @abstractmethod
    def train(self, train_X: ndarray, train_y: ndarray) -> None:
        """Run model training with training and validation data"""
        pass

    @abstractmethod
    def save_weights(self, filepath: str) -> bool:
        """
        Save the computed weights from training. Returns an error if training
        has not yet run or no weights have been loaded
        """
        pass

    @abstractmethod
    def load_weights(self, filepath: str) -> bool:
        """Load saved weights from a file or data structure"""
        pass

    @abstractmethod
    def predict(self, X: ndarray) -> ndarray:
        """Predict future values based on input data"""
        pass
