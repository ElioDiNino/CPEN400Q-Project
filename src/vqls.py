import numpy as np
from numpy import ndarray
import pennylane as qml
import scipy

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
        self.n_wires = n_wires
        self.weights = None

    def unitary(self, b, i, psi, wires):

        n = len(wires)

        # encode psi
        qml.AmplitudeEmbedding(psi, wires=wires)

        # for each qubit apply a pauli
        ii = i
        for x in range(n):

            if ii % 4 == 0:
                qml.Identity(wires[x])
            if ii % 4 == 1:
                qml.PauliX(wires[x])
            if ii % 4 == 2:
                qml.PauliY(wires[x])
            if ii % 4 == 3:
                qml.PauliZ(wires[x])
            ii = np.floor(ii / 4)

        # encode b
        qml.adjoint(qml.AmplitudeEmbedding)(b, wires=wires)
        return

    def unitary_normalization(self, i1, i2, psi, wires):
        qml.AmplitudeEmbedding(psi, wires=wires)
        # qml.MottonenStatePreparation(psi,wires=wires)
        # qml.MottonenStatePreparation(features=psi,wires=wires,normalize=True)
        n = len(wires)
        ii = i1
        for x in range(n):

            if ii % 4 == 0:
                qml.Identity(wires[x])
            if ii % 4 == 1:
                qml.PauliX(wires[x])
            if ii % 4 == 2:
                qml.PauliY(wires[x])
            if ii % 4 == 3:
                qml.PauliZ(wires[x])
            ii = np.floor(ii / 4)

        ii = i2
        for x in range(n):

            if ii % 4 == 0:
                qml.Identity(wires[x])
            if ii % 4 == 1:
                qml.PauliX(wires[x])
            if ii % 4 == 2:
                qml.PauliY(wires[x])
            if ii % 4 == 3:
                qml.PauliZ(wires[x])
            ii = np.floor(ii / 4)

        # qml.adjoint(qml.MottonenStatePreparation)(psi,wires=wires)
        qml.adjoint(qml.AmplitudeEmbedding)(psi, wires=wires)

        return

    def compute_product(self, b, U, psi):

        # number of qubits
        n = round(np.log2(len(U)))

        # Hadamard Test
        def HtestReal(b, i, psi):
            qml.Hadamard(wires=0)
            b = b / np.linalg.norm(b)
            psi = psi / np.linalg.norm(psi)

            length = int(np.ceil(np.log2(len(b))))

            qml.ctrl(self.unitary, control=0)(
                b, i, psi, list(range(1, length + 1))
            )
            qml.Hadamard(wires=0)

            return qml.expval(qml.Z(0))

        # hadamard test to compute the solution norm
        def HtestRealNorm(i1, i2, psi):
            qml.Hadamard(wires=0)
            psi = psi / np.linalg.norm(psi)

            length = int(np.ceil(np.log2(len(psi))))

            qml.ctrl(self.unitary_normalization, control=0)(
                i1, i2, psi, list(range(1, length + 1))
            )
            qml.Hadamard(wires=0)

            return qml.expval(qml.Z(0))

        # qnodes
        dev = qml.device("default.qubit", wires=n + 1)
        HTEST = qml.QNode(HtestReal, dev)
        HTESTNORM = qml.QNode(HtestRealNorm, dev)

        # expectation value
        exp = 0

        # normalization of the expectation value
        normalize = 0

        # compute expectation value
        for i in range(4**n):

            # compute trace
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

            # do Htest
            real = 0
            if abs(alpha) ** 2 > 0:
                real = HTEST(b, i, psi)
            exp = exp + alpha * real

            for i2 in range(4**n):

                # compute coefficent
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

                # do Htest
                if abs(alpha) ** 2 > 0:
                    real = HTESTNORM(i, i2, psi)
                normalize = normalize + alpha * np.conj(alpha2) * real

        return exp / np.sqrt(normalize)

    def angles_to_vector(self, psi_angles):  # hypersphere coordinates
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

    def compute_product_angles(self, b, U, psi_angles):
        psi = self.angles_to_vector(psi_angles)
        return self.compute_product(b, U, psi)

    def mse_iterations(self) -> list[float]:
        pass

    def train(self, train_X: ndarray, train_y: ndarray) -> None:
        # b=[1,-10]
        # M=[[22,-10],[-10,97]]
        # TODO: check over this
        M = train_X.T @ train_X
        b = train_X.T @ train_y

        x0 = [1]  # TODO: maybe randomize?
        cost_history = []

        def cost(x):
            return 1 - np.real(self.compute_product_angles(b, M, x))

        def callback(xk):
            c = cost(xk)
            cost_history.append(c)
            print("step #", len(cost_history), self.angles_to_vector(xk), c)

        result = scipy.optimize.minimize(cost, x0, callback=callback, tol=1e-4)

        print(result)

        wv = np.array(self.angles_to_vector(result.x))

        # TODO: save w to class variable for predicting (self.weights)
        # TODO: don't need all the print statements afterwards

        print("ansatz vector", wv)
        Aw = np.matmul(M, wv)
        w = np.linalg.norm(b) * wv / np.linalg.norm(Aw)

        print(" computed weights", w)
        print("exact b", np.matmul(np.matmul(M, np.linalg.inv(M)), b))
        print("computed b", np.matmul(M, w))

    def predict(self, X: ndarray) -> ndarray:
        # TODO: predict using the weights. Should just be X @ self.weights
        pass


def train():
    """
    Train the linear regression model on the paper data
    """
    print("\nTraining VQLS...")

    _, _, X_train, X_test, y_train, y_test, _, _, _, _ = get_paper_data()

    # Train the model
    vqls = VQLS()
    vqls.train(X_train, y_train)
    vqls.save_model("../models/vqls")

    # Evaluate the model
    print(f"Training Loss (MSE): {vqls.score(X_train, y_train)}")
    print(f"Testing Loss (MSE): {vqls.score(X_test, y_test)}")


if __name__ == "__main__":
    train()
