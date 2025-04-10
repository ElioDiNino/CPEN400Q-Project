import numpy as np
from numpy import ndarray
import pennylane as qml
import scipy.optimize
from sklearn.metrics import mean_squared_error

from common import VQLS_WIRES, get_paper_data
from abstract import ForecastingMethod


class VQLS(ForecastingMethod):
    """
    This class implements a Variational Quantum Linear Solver (VQLS) model.
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
        self.__mse_iterations: list[float] = []

    def __apply_paulis(self, start, wires):
        """
        Apply the Pauli gates to the provided wires based on the starting
        value.

        Args:
            start: The first pauli matrix to apply, encoded in base 4.
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
            i //= 4

    def __unitary(self, b: ndarray, i: int, psi: ndarray, wires: list[int]):
        """
        Apply amplitude embedding, paulis and de-amplitude embedding for VQLS.
        Applies: M_b^t * Pauli * M_psi, where M is amplitude embedding.

        Args:
            b: the training vector.
            i: the first pauli matrix to apply, encoded in base 4.
            psi: the ansatz vector (w).
            wires: The wires to which the gates will be applied.
        """

        # Encode psi
        qml.AmplitudeEmbedding(psi, wires=wires)

        self.__apply_paulis(i, wires)

        # Encode b
        qml.adjoint(qml.AmplitudeEmbedding)(b, wires=wires)

    def __unitary_normalization(
        self, i1: int, i2: int, psi: ndarray, wires: list[int]
    ):
        """
        Apply the unitary for normalization:
        M_psi^t * Pauli_i2 * Pauli_i1 * M_psi, where M is amplitude embedding.

        Args:
            i1: the first pauli matrix to apply, encoded in base 4.
            i2: the second pauli matrix to apply, encoded in base 4.
            psi: the ansatz vector (w).
            wires: The wires to which the gates will be applied.
        """

        # Encode psi
        qml.AmplitudeEmbedding(psi, wires=wires)

        self.__apply_paulis(i1, wires)
        self.__apply_paulis(i2, wires)

        # De-encode psi
        qml.adjoint(qml.AmplitudeEmbedding)(psi, wires=wires)

    def __compute_product(self, b: ndarray, M: ndarray, psi: ndarray) -> float:
        """
        Computes the normalized expectation <b|M|psi> / |m|psi>|, for use in
        VQLS training.

        Args:
            b: training data embedding vector.
            M: training data matrix.
            psi: weight ansatz vector (w).

        Returns:
            The normalized expectation value <b|M|psi> / |m|psi>|.
        """

        n = self.n_wires
        dev = qml.device("default.qubit", wires=self.n_wires + 1)

        @qml.qnode(dev)
        def HtestReal(b: ndarray, i: int, psi: ndarray) -> float:
            """
            Computes the expectation <b|M_i|psi> using the Hadamard test.

            Args:
                b: training data embedding vector.
                i: the first pauli matrix to apply, encoded in base 4.
                psi: weight ansatz vector (w).

            Returns:
                The expectation value <b|M_i|psi>.
            """

            qml.Hadamard(wires=0)
            b = b / np.linalg.norm(b)
            psi = psi / np.linalg.norm(psi)

            length = int(np.ceil(np.log2(len(b))))

            qml.ctrl(self.__unitary, control=0)(
                b, i, psi, list(range(1, length + 1))
            )
            qml.Hadamard(wires=0)

            return qml.expval(qml.Z(0))

        @qml.qnode(dev)
        def HtestRealNorm(i1: int, i2: int, psi: ndarray) -> float:
            """
            Computes the expectation <psi|M_i2|M_i1|psi> using the Hadamard
            test.

            Args:
                i1: the first pauli matrix to apply, encoded in base 4.
                i2: the second pauli matrix to apply, encoded in base 4.
                psi: weight ansatz vector (w).

            Returns:
                The expectation value <psi|M_i2|M_i1|psi>.
            """

            qml.Hadamard(wires=0)
            psi = psi / np.linalg.norm(psi)

            length = int(np.ceil(np.log2(len(psi))))

            qml.ctrl(self.__unitary_normalization, control=0)(
                i1, i2, psi, list(range(1, length + 1))
            )
            qml.Hadamard(wires=0)

            return qml.expval(qml.Z(0))

        def compute_kron(i: int, n: int) -> ndarray:
            """
            Computes the Kronecker product of the Pauli matrices starting from
            the provided index.

            Args:
                i: The first pauli gate Kronecker product to apply, encoded
                  in base 4.
                n: The number of wires/qubits.

            Returns:
                The Kronecker product as a NumPy ndarray.
            """
            pauli_matrices = [
                [[1, 0], [0, 1]],  # Identity
                [[0, 1], [1, 0]],  # Pauli-X
                [[0, -1j], [1j, 0]],  # Pauli-Y
                [[1, 0], [0, -1]],  # Pauli-Z
            ]

            Ui = [1]
            for _ in range(n):
                Ui = np.kron(Ui, pauli_matrices[i % 4])
                i //= 4

            return Ui

        # Expectation value
        exp = 0

        # Normalization of the expectation value
        normalize = 0

        # Compute expectation value
        for i in range(4**n):
            # Compute trace
            Ui = compute_kron(i, n)
            trace = np.trace(np.matmul(Ui, M))
            alpha = 2.0**-n * trace

            # Do Hadamard test
            real = 0
            if abs(alpha) ** 2 > 0:
                real = HtestReal(b, i, psi)
            exp += alpha * real

            # Compute weight for this i
            for i2 in range(4**n):
                # Compute coefficient
                Ui = compute_kron(i2, n)
                trace = np.trace(np.matmul(Ui, M))
                alpha2 = 2.0**-n * trace

                # Do Hadamard test
                if abs(alpha) ** 2 > 0:
                    real = HtestRealNorm(i, i2, psi)
                normalize += alpha * np.conj(alpha2) * real

        return exp / np.sqrt(normalize)

    def __angles_to_vector(self, psi_angles: ndarray) -> list[float]:
        """
        Converts n angles to n + 1 vector magnitudes with a norm of 1.

        Args:
            psi_angles: n angles which encode n + 1 vector magnitudes.

        Returns:
            n + 1 vector magnitudes with a norm of 1.
        """

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

    def __compute_product_angles(self, b, M, psi_angles):
        """
        Wrapper of self.__compute_product() that takes in angles instead
        of vector magnitude. Used for training using n - 1 inputs.

        Args:
            b: training data embedding vector.
            M: training data matrix.
            psi: weight ansatz angles (w).

        Returns:
            The normalized expectation value <b|M|psi> / |m|psi>|.
        """
        psi = self.__angles_to_vector(psi_angles)
        return self.__compute_product(b, M, psi)

    def __compute_weights(self, x: ndarray, M: ndarray, b: ndarray) -> ndarray:
        """
        Computes the weights for the VQLS model.

        Args:
            x: the ansatz vector (w).
            M: training data matrix.
            b: training data embedding vector.

        Returns:
            The computed weights.
        """
        wv = np.array(self.__angles_to_vector(x))
        Aw = M @ wv
        return np.linalg.norm(b) * wv / np.linalg.norm(Aw)

    @property
    def mse_iterations(self) -> list[float]:
        return self.__mse_iterations

    def train(self, train_X: ndarray, train_y: ndarray) -> None:
        M = train_X.T @ train_X
        b = train_X.T @ train_y

        x0 = np.zeros(2**self.n_wires - 1)

        cost_history = []
        self.__mse_iterations = []

        def cost(x):
            return 1 - np.real(self.__compute_product_angles(b, M, x))

        def callback(xk):
            c = cost(xk)
            cost_history.append(c)
            mse = mean_squared_error(
                train_y, train_X @ self.__compute_weights(xk, M, b)
            )
            self.__mse_iterations.append(mse)
            print(f"Iteration {len(cost_history)} - cost: {c}, mse: {mse}")

        result = scipy.optimize.minimize(cost, x0, callback=callback, tol=1e-8)

        wv = np.array(self.__angles_to_vector(result.x))
        print("Ansatz vector:", wv)

        w = self.__compute_weights(result.x, M, b)

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
        window_size=2**VQLS_WIRES
    )

    # Train the model
    vqls = VQLS(n_wires=VQLS_WIRES)
    vqls.train(X_train, y_train)
    vqls.save_model("../models/vqls")

    # Evaluate the model
    print(f"Training Loss (MSE): {vqls.score(X_train, y_train)}")
    print(f"Testing Loss (MSE): {vqls.score(X_test, y_test)}")


if __name__ == "__main__":
    train()
