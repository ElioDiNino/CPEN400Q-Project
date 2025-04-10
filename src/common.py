import numpy as np
from datetime import date
from numpy import ndarray
from sklearn.model_selection import train_test_split

from abstract import ForecastingMethod

START_DATE = date(2019, 1, 1)  # Start date for the data
TRAIN_TEST_RATIO = (
    6.75 / 10
)  # Ratio of training to testing data determined from the paper
WINDOW_SIZE = 12
MAX_ITERATIONS = 1000
VQLS_WIRES = 2  # Training takes too long with anything larger


def get_paper_data(
    train_test_ratio: float = TRAIN_TEST_RATIO, window_size: int = WINDOW_SIZE
) -> tuple[
    ndarray,
    ndarray,
    ndarray,
    ndarray,
    ndarray,
    ndarray,
    int,
    float,
    float,
    float,
]:
    """
    Load and preprocess the data for training and testing.
    The data is split into training and testing sets based on the
    ratio argument.
    The data is loaded from a CSV file, and the preprocessing
    includes normalization and scaling.

    Args:
        train_test_ratio: The ratio of training to testing data.
        window_size: The size of the window for the time series data.

    Returns:
        (X, y, X_train, X_test, y_train, y_test, train_test_split_index,
        min_value, max_value, initial_value):
        - X: The input data
        - y: The target data
        - X_train: The training data
        - X_test: The testing data
        - y_train: The training labels
        - y_test: The testing labels
        - train_test_split_index: The index where the data is split
        - min_value: The minimum value for scaling
        - max_value: The maximum value for scaling
        - initial_value: The initial value for scaling
    """
    # Preprocess the data: apply normalization and scaling like the paper does
    new_filepath, min_value, max_value, initial_value = (
        ForecastingMethod.preprocess_data(
            filepath="../data/paper-data.csv",
            difference=False,
            scale_to_range=True,
            training_data_cutoff=train_test_ratio,
        )
    )

    # Load the data
    X, y = ForecastingMethod.load_data(
        filepath=new_filepath, windows_size=window_size
    )
    train_test_split_index = int(len(X) * train_test_ratio)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=(1 - train_test_ratio), shuffle=False
    )

    return (
        X,
        y,
        X_train,
        X_test,
        y_train,
        y_test,
        train_test_split_index,
        min_value,
        max_value,
        initial_value,
    )


def rescale_paper_data(
    X: ndarray,
    y: ndarray,
    min_value: float,
    max_value: float,
    initial_value: float,
    train_test_split_index: int,
    window_size: int = WINDOW_SIZE,
) -> tuple[ndarray, int]:
    """
    Rescale the data for plotting purposes.

    Args:
        X: The input data
        y: The target data
        min_value: The minimum value for scaling
        max_value: The maximum value for scaling
        initial_value: The initial value for scaling
        train_test_split_index: The index where the data is split
        window_size: The size of the window for the time series data.

    Returns:
        (ORIG_SCALED, GRAPH_SPLIT_INDEX):
        - ORIG_SCALED: The original data scaled for plotting
        - GRAPH_SPLIT_INDEX: The index where the data is split for plotting
    """
    # Post process original data (for plots only)
    y_scaled = ForecastingMethod.post_process_data(
        y, min_value, max_value, initial_value, scale_to_range=True
    )
    X_scaled = ForecastingMethod.post_process_data(
        X, min_value, max_value, initial_value, scale_to_range=True
    )
    ORIG_SCALED = np.concatenate((X_scaled[0:window_size, 0], y_scaled))
    GRAPH_SPLIT_INDEX = train_test_split_index + window_size

    return ORIG_SCALED, GRAPH_SPLIT_INDEX
