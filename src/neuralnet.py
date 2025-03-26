"""
The neural network implementation

If you run the python file directly,
it trains a new model and tests using paper data
"""

import time
from sklearn.model_selection import train_test_split
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

from abstract import ForecastingMethod


class NeuralNet(ForecastingMethod):
    """
    A neural network implemented according to the specifications in the paper.
    """

    def __init__(self):
        self.model = None

    def __build_model(self):
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

    def train(self, train_X, train_y):
        self.model = self.__build_model()
        self.model.fit(train_X, train_y, epochs=200, batch_size=8)

    def save_weights(self, filepath):
        try:
            self.model.save(filepath + ".keras")
            return True
        except:
            return False

    def load_weights(self, filepath):
        try:
            self.model = keras.saving.load_model(filepath)
            return True
        except:
            return False

    def predict(self, X):
        return self.model.predict(np.expand_dims(X, axis=0))


if __name__ == "__main__":
    X, y = ForecastingMethod.load_data("../data/paper-data.csv")
    train_X, _, train_y, _ = train_test_split(
        X, y, test_size=0.1, shuffle=False
    )
    # Training
    nn = NeuralNet()
    nn.train(train_X, train_y)
    training_timestamp = str(int(time.time()))
    nn.save_weights("neural_net_" + training_timestamp)
    # Testing
    predictions = nn.model.predict(X)
    x_axis = range(len(y))
    plt.plot(x_axis, predictions, label="Predicted")
    plt.plot(x_axis, y, label="Correct")
    plt.legend()
    plt.savefig("neural_net_" + training_timestamp + ".png")
