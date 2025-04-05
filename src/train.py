from linear_regression import train as lr_train
from neural_network import train as nn_train
from pqc import train as pqc_train

if __name__ == "__main__":
    lr_train()
    nn_train()
    pqc_train()
    print("\nTraining complete.")
