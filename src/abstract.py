from numpy import ndarray
from abc import ABC, abstractmethod
import csv
import numpy as np


class ForecastingMethod(ABC):
    """
    Abstract base class for forecasting methods.

    This class defines the interface for all forecasting methods. It includes
    methods for loading data, preprocessing data, training the model, saving
    and loading weights, and making predictions. The actual implementation of
    these methods is be provided by subclasses.
    """

    @staticmethod
    def load_data(
        filepath: str, windows_size: int = 12
    ) -> tuple[ndarray, ndarray]:
        """
        Load data from a CSV file and convert it to a format that can be used
        for time-series forecasting. No preprocessing is done here.

        Args:
            filepath: Path to the desired CSV file. File should have a single
              column of data where the first row is the header.
            windows_size: Number of data points to use for each
              prediction. Default is 12.

        Returns:
            (X, y):
            - X: 2D array of shape (n_examples, window_size) where
              n_examples is the number of examples.
            - y: 1D array of shape (n_examples,) where
              n_examples is the number of examples.

        Raises:
            ValueError: If window_size >= len(data)
            OSError: If the file does not exist
        """
        with open(filepath, mode="r") as file:
            reader = csv.reader(file)
            data = list(reader)[1:]  # don't include the header row

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

    @staticmethod
    def preprocess_data(X: ndarray, y: ndarray) -> tuple[ndarray, ndarray]:
        """
        Applies differencing and normalization to the input data.

        Args:
            X: Input data
            y: Target data

        Returns:
            (X, y): Tuple containing the preprocessed data
        """
        pass

    @abstractmethod
    def train(self, train_X: ndarray, train_y: ndarray) -> None:
        """
        Train the forecasting model using the given data.

        Args:
            train_X: Training data
            train_y: Training labels
        """
        pass

    @abstractmethod
    def save_weights(self, filepath: str) -> bool:
        """
        Save the computed weights from training. Requires that the model has
        been trained or weights have been loaded before saving successfully.

        Args:
            filepath: Path to the weights file to save

        Returns:
            bool: True if weights are saved successfully, False otherwise
        """
        pass

    @abstractmethod
    def load_weights(self, filepath: str) -> bool:
        """
        Load saved weights from a file.

        Args:
            filepath: Path to the weights file to load

        Returns:
            bool: True if weights are loaded successfully, False otherwise
        """
        pass

    @abstractmethod
    def predict(self, X: ndarray) -> ndarray:
        """
        Predict future values based on input data.

        Args:
            X: Input data

        Returns:
            ndarray: Predicted values (y-hat)
        """
        pass
