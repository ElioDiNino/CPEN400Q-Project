import keras
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
        self.__mse_iterations: list[float] = []

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

    @property
    def mse_iterations(self) -> list[float]:
        return self.__mse_iterations

    def train(self, train_X: ndarray, train_y: ndarray) -> None:
        self.model = self.__build_model()
        self.model.fit(
            train_X, train_y, epochs=self.epochs, batch_size=self.batch_size
        )
        self.__mse_iterations = self.model.history.history["loss"]

    def predict(self, X: ndarray) -> ndarray:
        return self.model.predict(X).flatten()


def train():
    """
    Train the neural network on the paper data
    """
    print("\nTraining Neural Network...")

    _, _, X_train, X_test, y_train, y_test, _, _, _, _ = get_paper_data()

    # Train the model
    nn = NeuralNetwork(epochs=300)
    nn.train(X_train, y_train)
    nn.save_model("../models/neural_network")

    # Evaluate the model
    print(f"Training Loss (MSE): {nn.score(X_train, y_train)}")
    print(f"Testing Loss (MSE): {nn.score(X_test, y_test)}")


if __name__ == "__main__":
    train()
