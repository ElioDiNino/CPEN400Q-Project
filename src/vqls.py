import numpy as np
from numpy import ndarray
import pennylane as qml
import scipy.optimize

from common import get_paper_data
from abstract import ForecastingMethod


class VQLS(ForecastingMethod):
    """
    This class implements a Variational Quantum Linear Solver (VQLS)
    model.
    """

    def __init__(self, n_wires: int = 2):
        """
        Initialize the Variational Quantum Linear Solver (VQLS) model.

        Args:
            n_wires: Number of wires/qubits to use in the model.
        """
        super().__init__()
        if n_wires <= 0:
            raise ValueError("n_wires must be positive")
        self.n_wires = n_wires
        self.weights = None

    def __apply_paulis(self, start, wires):
        """
        Apply the Pauli gates to the provided wires based on the
        starting value.

        Args:
            start: The starting value to apply gates based on.
            wires: The wires to which the gates will be applied.
        """
        pauli_gates = {
            0: qml.Identity,
            1: qml.PauliX,
            2: qml.PauliY,
            3: qml.PauliZ,
        }

        n = len(wires)
        i = start
        for x in range(n):
            gate = pauli_gates[i % 4]
            gate(wires[x])
            i = i // 4

    def __unitary(self, b, i, psi, wires):
        # TODO: docstring

        # Encode psi
        qml.AmplitudeEmbedding(psi, wires=wires)

        self.__apply_paulis(i, wires)

        # Encode b
        qml.adjoint(qml.AmplitudeEmbedding)(b, wires=wires)
        return

    def __unitary_normalization(self, i1, i2, psi, wires):
        # TODO: docstring

        # Encode psi
        qml.AmplitudeEmbedding(psi, wires=wires)

        self.__apply_paulis(i1, wires)
        self.__apply_paulis(i2, wires)

        # Encode b
        qml.adjoint(qml.AmplitudeEmbedding)(psi, wires=wires)

    def __compute_product(self, b, U, psi):
        # TODO: docstring
        n = self.n_wires

        # Hadamard Test
        def HtestReal(b, i, psi):
            qml.Hadamard(wires=0)
            b = b / np.linalg.norm(b)
            psi = psi / np.linalg.norm(psi)

            length = int(np.ceil(np.log2(len(b))))

            qml.ctrl(self.__unitary, control=0)(
                b, i, psi, list(range(1, length + 1))
            )
            qml.Hadamard(wires=0)

            return qml.expval(qml.Z(0))

        # Hadamard test to compute the solution norm
        def HtestRealNorm(i1, i2, psi):
            qml.Hadamard(wires=0)
            psi = psi / np.linalg.norm(psi)

            length = int(np.ceil(np.log2(len(psi))))

            qml.ctrl(self.__unitary_normalization, control=0)(
                i1, i2, psi, list(range(1, length + 1))
            )
            qml.Hadamard(wires=0)

            return qml.expval(qml.Z(0))

        # QNodes
        dev = qml.device("default.qubit", wires=self.n_wires + 1)
        HTEST = qml.QNode(HtestReal, dev)
        HTESTNORM = qml.QNode(HtestRealNorm, dev)

        # Expectation value
        exp = 0

        # Normalization of the expectation value
        normalize = 0

        # Compute expectation value
        for i in range(4**n):
            # Compute trace
            Ui = np.array([1])
            ii = i
            for x in range(n):
                if ii % 4 == 0:
                    Ui = np.kron(Ui, np.array([[1, 0], [0, 1]]))
                if ii % 4 == 1:
                    Ui = np.kron(Ui, np.array([[0, 1], [1, 0]]))
                if ii % 4 == 2:
                    Ui = np.kron(Ui, np.array([[0, -1j], [1j, 0]]))
                if ii % 4 == 3:
                    Ui = np.kron(Ui, np.array([[1, 0], [0, -1]]))
                ii = np.floor(ii / 4)

            trace = np.trace(np.matmul(Ui, U))
            alpha = 2.0**-n * trace

            # Do Hadamard test
            real = 0
            if abs(alpha) ** 2 > 0:
                real = HTEST(b, i, psi)
            exp = exp + alpha * real

            for i2 in range(4**n):
                # Compute coefficient
                Ui = [1]
                ii = i2
                for x in range(n):
                    if ii % 4 == 0:
                        Ui = np.kron(Ui, np.array([[1, 0], [0, 1]]))
                    if ii % 4 == 1:
                        Ui = np.kron(Ui, np.array([[0, 1], [1, 0]]))
                    if ii % 4 == 2:
                        Ui = np.kron(Ui, np.array([[0, -1j], [1j, 0]]))
                    if ii % 4 == 3:
                        Ui = np.kron(Ui, np.array([[1, 0], [0, -1]]))
                    ii = np.floor(ii / 4)

                trace = np.trace(np.matmul(Ui, U))
                alpha2 = 2.0**-n * trace

                # Do Hadamard test
                if abs(alpha) ** 2 > 0:
                    real = HTESTNORM(i, i2, psi)
                normalize = normalize + alpha * np.conj(alpha2) * real

        return exp / np.sqrt(normalize)

    def __angles_to_vector(self, psi_angles):  # hypersphere coordinates
        # TODO: docstring
        n = len(psi_angles)

        psi = []

        for i in range(n):
            val = np.cos(psi_angles[i])
            for k in range(i):
                val = val * np.sin(psi_angles[k])
            psi.append(val)

        val = 1
        for i in range(n):
            val = val * np.sin(psi_angles[i])
        psi.append(val)
        return psi

    def __compute_product_angles(self, b, U, psi_angles):
        # TODO: docstring
        psi = self.__angles_to_vector(psi_angles)
        return self.__compute_product(b, U, psi)

    @property
    def mse_iterations(self) -> list[float]:
        pass

    def train(self, train_X: ndarray, train_y: ndarray) -> None:
        M = train_X.T @ train_X
        b = train_X.T @ train_y

        x0 = np.zeros(self.n_wires)

        cost_history = []

        def cost(x):
            return 1 - np.real(self.__compute_product_angles(b, M, x))

        def callback(xk):
            c = cost(xk)
            cost_history.append(c)
            print(f"Iteration {len(cost_history)} - cost: {c}")

        result = scipy.optimize.minimize(cost, x0, callback=callback, tol=1e-8)

        wv = np.array(self.__angles_to_vector(result.x))

        print("Ansatz vector:", wv)
        Aw = M @ wv
        w = np.linalg.norm(b) * wv / np.linalg.norm(Aw)

        print("Computed weights:", w)
        print("Exact b:", M @ np.linalg.inv(M) @ b)
        print("Computed b:", M @ w)

        self.weights = w

    def predict(self, X: ndarray) -> ndarray:
        return X @ self.weights


def train():
    """
    Train the VQLS model on the paper data
    """
    print("\nTraining VQLS...")

    _, _, X_train, X_test, y_train, y_test, _, _, _, _ = get_paper_data(
        window_size=2**2
    )

    # Train the model
    vqls = VQLS(n_wires=2)
    vqls.train(X_train, y_train)
    vqls.save_model("../models/vqls")

    # Evaluate the model
    print(f"Training Loss (MSE): {vqls.score(X_train, y_train)}")
    print(f"Testing Loss (MSE): {vqls.score(X_test, y_test)}")


if __name__ == "__main__":
    train()
