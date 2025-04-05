import copy
import pennylane as qml
import numpy as np
from scipy.optimize import minimize
from numpy import ndarray

from common import get_paper_data
from abstract import ForecastingMethod


class PQC(ForecastingMethod):
    """
    A class that implements a parameterized quantum circuit (PQC) for
    forecasting.
    """

    def __init__(self, optimizer: str | None, n_wires: int, n_layers: int):
        """
        Initialize the Parameterized Quantum Circuit (PQC) model.

        Args:
            optimizer: The optimization algorithm to use for training.
            n_wires: Number of qubits (wires) in the quantum circuit.
            n_layers: Number of layers in the quantum circuit.
        """
        super().__init__()
        self.n_wires = n_wires
        self.n_layers = n_layers
        self.dev = qml.device("default.qubit", wires=n_wires)
        self.theta_ary = np.zeros((self.n_layers, self.n_wires * 2))
        self.optimizer = optimizer

    def __pqc_qnode(self, phi_ary: ndarray, theta_ary: ndarray):
        qnode = qml.QNode(self.__pqc_circuit, self.dev)
        return qnode(phi_ary, theta_ary)

    def __pqc_circuit(self, phi_ary: ndarray, theta_ary: ndarray):
        """
        A quantum circuit that encodes the data before applying 2 layers of
        RX and RY rotations followed by a series of CNOT gates.

        Args:
            phi_ary: Array of angles for encoding the data.
            theta_ary: 2D array of angles theta_ary[layer][gate]
            for both RX and RY gates.

        Returns:
            np.ndarray: The final state vector of the quantum circuit.
        """

        def encode_circuit(phi_ary: ndarray):
            for wire in range(self.n_wires):
                qml.RY(phi_ary[wire], wires=wire)

        def apply_layer(layer_theta_ary: ndarray):
            """
            A function that applies a layer of RX and RY rotations

            Args:
                theta_ary (np.ndarray): 1D array of angles for both RX
                and RY gates.
            """
            theta_index = 0
            for wire in range(self.n_wires):
                qml.RX(layer_theta_ary[theta_index], wires=wire)
                theta_index += 1
                qml.RY(layer_theta_ary[theta_index], wires=wire)
                theta_index += 1

        # Encode the data
        encode_circuit(phi_ary)

        # Apply the first layer
        # Layer 1 Entanglements
        for wire in range(0, self.n_wires - 1, 2):
            qml.CNOT(wires=[wire, wire + 1])
        # Layer 1 RX and RY rotations
        apply_layer(theta_ary[0])

        # Apply the second layer
        # Layer 2 Entanglements
        for wire in range(1, self.n_wires - 1, 2):
            qml.CNOT(wires=[wire, wire + 1])
        qml.CNOT(wires=[self.n_wires - 1, 0])
        # Layer 2 RX and RY rotations
        apply_layer(theta_ary[1])

        return qml.expval(qml.PauliZ(0))

    def draw_circuit(self):
        """
        Draw the quantum circuit.
        """
        return qml.draw_mpl(self.__pqc_circuit, decimals=3)(
            np.zeros(self.n_wires), self.theta_ary
        )

    def train(self, train_X: ndarray, train_y: ndarray) -> None:
        # deep copy train_X and train_y
        train_X = copy.deepcopy(train_X)
        train_y = copy.deepcopy(train_y)

        # use scipy.optimize.minimize to optimize the theta_ary
        def cost_function(theta_ary_flattened):
            mean_squared_error = 0

            # Reshape the flattened theta_ary to its original shape
            # rows = n_layers, columns = n_wires * 2(1 for RX and 1 for RY)
            theta_ary = theta_ary_flattened.reshape(
                self.n_layers, self.n_wires * 2
            )

            for i in range(len(train_X)):
                example = train_X[i]
                label = train_y[i]
                phi_ary = example
                prediction = self.__pqc_qnode(phi_ary, theta_ary)
                error = prediction - label
                mean_squared_error += error**2
            return mean_squared_error

        # Initial guess for theta_ary(flattened to single array)
        initial_theta_ary = np.random.rand(self.n_layers * self.n_wires * 2)
        # Optimize the cost function
        result = minimize(
            cost_function, initial_theta_ary, method=self.optimizer
        )
        # Update the theta_ary with the optimized values
        self.theta_ary = result.x.reshape(self.n_layers, self.n_wires * 2)

    def save_weights(self, filepath: str) -> bool:
        try:
            np.save(filepath, self.theta_ary)
            return True
        except Exception as e:
            print(f"Error saving weights: {e}")
            return False

    def load_weights(self, filepath: str) -> bool:
        try:
            self.theta_ary = np.load(filepath)
            return True
        except Exception as e:
            print(f"Error loading weights: {e}")
            return False

    def predict(self, test_X: ndarray) -> ndarray:
        predictions = []
        for example in test_X:
            phi_ary = example
            prediction = self.__pqc_qnode(phi_ary, self.theta_ary)
            predictions.append(prediction)
        return np.array(predictions)


def train():
    """
    Train the PQC model on the paper data
    """
    N_WIRES = 12
    N_LAYERS = 2

    print("\nTraining PQC...")

    X, X_train, _, y_train, y_test, split_idx = get_paper_data()

    # Train the models
    pqc_model_lbfgsb = PQC(
        n_wires=N_WIRES, n_layers=N_LAYERS, optimizer="L-BFGS-B"
    )
    pqc_model_cobyla = PQC(
        n_wires=N_WIRES, n_layers=N_LAYERS, optimizer="COBYLA"
    )
    pqc_model_lbfgsb.train(X_train, y_train)
    pqc_model_cobyla.train(X_train, y_train)

    pqc_model_lbfgsb.save_weights("../models/pqc_lbfgsb.npy")
    pqc_model_cobyla.save_weights("../models/pqc_cobyla.npy")

    # Evaluate the model
    predictions_L = pqc_model_lbfgsb.predict(X)
    predictions_C = pqc_model_cobyla.predict(X)

    # Compute the training loss with processed data
    train_mse_L = np.mean((predictions_L[:split_idx] - y_train) ** 2)
    train_mse_C = np.mean((predictions_C[:split_idx] - y_train) ** 2)

    # Compute testing loss
    test_mse_L = np.mean((predictions_L[split_idx:] - y_test) ** 2)
    test_mse_C = np.mean((predictions_C[split_idx:] - y_test) ** 2)

    print("\nPQC with L-BFGS-B Optimizer")
    print(f"Training Loss (MSE): {train_mse_L}")
    print(f"Test Loss (MSE): {test_mse_L}")
    print("\nPQC with COBYLA Optimizer")
    print(f"Training Loss (MSE): {train_mse_C}")
    print(f"Test Loss (MSE): {test_mse_C}")


if __name__ == "__main__":
    train()
