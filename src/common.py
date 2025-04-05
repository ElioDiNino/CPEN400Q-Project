from datetime import date
from sklearn.model_selection import train_test_split

from abstract import ForecastingMethod

START_DATE = date(2019, 1, 1)  # Start date for the data
TRAIN_TEST_RATIO = 6.75 / 10
WINDOW_SIZE = 12


def get_paper_data(
    train_test_ratio: float = TRAIN_TEST_RATIO, window_size: int = WINDOW_SIZE
):
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
        (X, X_train, X_test, y_train, y_test, train_test_split_index):
        - X: The input data
        - X_train: The training data
        - X_test: The testing data
        - y_train: The training labels
        - y_test: The testing labels
        - train_test_split_index: The index where the data is split
    """
    # Preprocess the data: apply normalization and scaling like the paper does
    new_filepath, _, _, _ = ForecastingMethod.preprocess_data(
        filepath="../data/paper-data.csv",
        difference=False,
        scale_to_range=True,
        training_data_cutoff=train_test_ratio,
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

    return X, X_train, X_test, y_train, y_test, train_test_split_index
