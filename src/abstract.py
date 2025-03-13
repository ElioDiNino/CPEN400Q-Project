from numpy import ndarray
from abc import ABC, abstractmethod


class ForecastingMethod(ABC):
    def __init__():
        """Basic initialization stuff"""
        pass

    def load_data(filepath: str) -> dict[str, ndarray]:
        pass

    def preprocess_data(X: ndarray, y: ndarray) -> dict[str, ndarray]:
        """Note: Implemented within Abstract Class. Applies differencing and normalization to the input data"""
        pass

    @abstractmethod
    def train(
        train_X: ndarray, train_y: ndarray, val_X: ndarray, val_y: ndarray
    ) -> None:
        """Run model training with training and validation data"""
        pass

    @abstractmethod
    def save_weights(filepath: str) -> bool:
        """Save the computed weights from training. Returns an error if training has not yet run or no weights have been loaded"""
        pass

    @abstractmethod
    def load_weights(filepath: str) -> bool:
        """Load saved weights from a file or data structure"""
        pass

    @abstractmethod
    def predict(X: ndarray) -> ndarray:
        """Predict future values based on input data"""
        pass
