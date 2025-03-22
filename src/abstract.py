from numpy import ndarray
from abc import ABC, abstractmethod
import csv
import numpy as np


class ForecastingMethod(ABC):
    def __init__():
        """Basic initialization stuff"""
        pass

    def load_data(
        self, filepath: str, windows_size: int = 12
    ) -> dict[str, ndarray]:
        with open(filepath, mode="r") as file:
            reader = csv.reader(file)
            data = list(reader)[1:]  # don't include the header

        # convert data to float
        data = [float(datapoint[0]) for datapoint in data]
        X = np.zeros((len(data) - windows_size, windows_size))
        y = np.zeros(len(data) - windows_size)

        # stop window right before the last value so it can be used in y
        for i in range(len(data) - windows_size):
            X[i] = data[i : i + windows_size]
            y[i] = data[i + windows_size]

        return {"X": X, "y": y}

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
