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
    def preprocess_data(
        filepath: str,
        difference: bool,
        scale_to_range: bool,
        training_data_cutoff: float = 2 / 3,
    ) -> tuple[str, float, float, float]:
        """
        Optionally applies differencing and scaling to the data, writing
        the result into a new CSV file with "-processed" appended.

        Args:
            filepath: Path to the CSV file containing the data
            difference: Whether to apply differencing
            scale_to_range: Whether to scale the data to a specific range
            training_data_cutoff: Proportion of data to use for scaling

        Returns:
            (newFilepath, min_value_prescale, max_value_prescale,
            initial_value_predifference):
            - newFilepath: Path to the new CSV file with processed data
            - min_value_prescale: Minimum value used for scaling
            - max_value_prescale: Maximum value used for scaling
            - initial_value_predifference: Initial value used for
                differencing
        """
        min_value_prescale = None
        max_value_prescale = None
        initial_value_predifference = None

        # read the data from the CSV file
        with open(filepath, mode="r") as file:
            reader = csv.reader(file)
            data = list(reader)[1:]

        # convert data to float
        data = np.array([float(datapoint[0]) for datapoint in data])

        # apply differencing
        if difference:
            # save the initial data for inverse differencing
            initial_value_predifference = data[0]
            # apply differencing
            data = np.diff(data, n=1)
            # convert to list
            data = data.tolist()

        # apply scaling
        if scale_to_range:
            training_data = data[: int(len(data) * training_data_cutoff)]
            # find the min and max values
            min_value = np.min(training_data)
            max_value = np.max(training_data)
            # scale the data to the range of 0 to 1
            data = (data - min_value) / (max_value - min_value)
            # normalize to range of -0.25 to 0.25 (based on paper)
            data = data * 0.5 - 0.25
            # convert to list
            data = data.tolist()

            # save the min and max values for inverse scaling
            min_value_prescale = min_value
            max_value_prescale = max_value

        # create and write to a CSV with original name + "-processed"
        new_filepath = filepath.split("/")
        new_filepath[-1] = (
            new_filepath[-1].split(".")[0] + "-processed.csv"
        )  # append "-processed" to the filename
        new_filepath = "/".join(new_filepath)  # join the list back together
        with open(new_filepath, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["data"])
            for datapoint in data:
                writer.writerow([datapoint])

        return (
            new_filepath,
            min_value_prescale,
            max_value_prescale,
            initial_value_predifference,
        )

    @staticmethod
    def post_process_data(
        processed_data: ndarray,
        min_value: float,
        max_value: float,
        initial_value: float,
        difference: bool = False,
        scale_to_range: bool = False,
    ) -> ndarray:
        """
        Post-process the data after prediction. This includes inverse
        differencing and inverse scaling.

        Args:
            processed_data: Data after prediction
            min_value: Minimum value used for scaling
            max_value: Maximum value used for scaling
            initial_data: Initial data used for differencing
            difference: Whether differencing was applied
            scale_to_range: Whether scaling was applied

        Returns:
            ndarray: Post-processed data
        """

        # Convert the input data to a NumPy array.
        data = np.array(processed_data)

        # Inverse scaling
        if scale_to_range:
            if min_value is None or max_value is None:
                raise ValueError(
                    "Saved min and max values (min_value_prescale, "
                    "max_value_prescale) are required to invert scaling."
                )

            # Inverse of the range adjustment:
            normalized = (data + 0.25) / 0.5
            # Inverse of the normalization:
            data = normalized * (max_value - min_value) + min_value

        # Inverse differencing
        if difference:
            if initial_value is None:
                raise ValueError(
                    "initial_value must be provided to reverse differencing."
                )
            # Prepend the original first value and perform a cumulative sum
            data = np.concatenate(([initial_value], data))
            data = np.cumsum(data)

        return data

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
