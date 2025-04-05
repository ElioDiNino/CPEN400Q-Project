import keras
import numpy as np
from numpy import ndarray

from common import get_paper_data
from abstract import ForecastingMethod


class NeuralNetwork(ForecastingMethod):
    """
    A neural network implemented according to the specifications in the paper.
    """

    def __init__(self, epochs: int = 200, batch_size: int = 8):
        """
        Initialize the neural network with the specified parameters.

        Args:
            epochs: Number of epochs for training
            batch_size: Size of each training batch
        """
        super().__init__()
        self.epochs = epochs
        self.batch_size = batch_size
        self.model: keras.Sequential | None = None

    def __build_model(self) -> keras.Sequential:
        """
        Use keras API to build neural net and compile it with
        mean squared error loss as specified in paper
        """
        model = keras.Sequential(
            [
                keras.layers.Dense(12, activation="relu"),
                keras.layers.Dense(12, activation="relu"),
                keras.layers.Dense(1),
            ]
        )
        model.compile(optimizer="adam", loss="mse")
        return model

    def train(self, train_X: ndarray, train_y: ndarray) -> None:
        self.model = self.__build_model()
        self.model.fit(
            train_X, train_y, epochs=self.epochs, batch_size=self.batch_size
        )

    def save_weights(self, filepath: str) -> bool:
        try:
            self.model.save(filepath + ".keras")
            return True
        except Exception:
            return False

    def load_weights(self, filepath: str) -> bool:
        try:
            self.model = keras.saving.load_model(filepath)
            return True
        except Exception:
            return False

    def predict(self, X: ndarray) -> ndarray:
        return self.model.predict(X).flatten()


def train():
    """
    Train the neural network on the paper data
    """
    print("\nTraining Neural Network...")

    X, X_train, _, y_train, y_test, split_idx = get_paper_data()

    # Train the model
    nn = NeuralNetwork()
    nn.train(X_train, y_train)
    nn.save_weights("../models/neural_network")

    # Evaluate the model
    predictions = nn.predict(X)

    train_mse = np.mean((predictions[:split_idx] - y_train) ** 2)
    test_mse = np.mean((predictions[split_idx:] - y_test) ** 2)

    print(f"Training Loss (MSE): {train_mse}")
    print(f"Testing Loss (MSE): {test_mse}")


if __name__ == "__main__":
    train()
