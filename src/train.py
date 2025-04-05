from linear_regression import train as lr_train
from neural_network import train as nn_train
from pqc import train as pqc_train
import time

if __name__ == "__main__":
    training_functions = [
        ("Linear Regression", lr_train),
        ("Neural Network", nn_train),
        ("PQC", pqc_train),
    ]

    for name, func in training_functions:
        start_time = time.time()
        func()
        execution_time = time.time() - start_time
        print(f"{name} training completed in {execution_time:.2f} seconds.")

    print("\nTraining complete.")
