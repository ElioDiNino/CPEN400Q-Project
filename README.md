# CPEN 400Q Final Project

An implementation of "[Exploring Hybrid Quantum-Classical Methods for Practical Time-Series Forecasting](https://arxiv.org/abs/2412.05615v1)"

## Overview

The goal of this project is to implement and compare both classical and hybrid quantum-classical models for time-series forecasting. The model specifics are based off the paper linked above, with reasonable assumptions made where necessary.

You can check out our final report and in-class presentation in the [`documents`](documents/) directory. The report contains a detailed explanation of our models, their performance, and a reflection of the source paper.

The repository is structured as follows:
```
├─ .flake8                        # Flake8 (Python formatting check) configuration
├─ .gitignore                     # Git ignore file
├─ .pre-commit-config.yaml        # Pre-commit configuration (automated formatting and other checks)
├─ README.md
├─ requirements.txt               # Required packages
├─ .github
│  └─ workflows
│     └─ pre-commit.yml           # GitHub Actions workflow to enforce pre-commit hooks
├─ data                           # Datasets
│  ├─ paper-data.csv
│  └─ paper-data-processed.csv
├─ documents
│  └─ presentation.pdf            # In-class presentation
│  ├─ report.pdf                  # Final report
├─ models                         # Trained models
│  ├─ linear_regression_vqls.pkl
│  ├─ linear_regression.pkl
│  ├─ neural_network.pkl
│  ├─ pqc_cobyla.pkl
│  ├─ pqc_lbfgsb.pkl
│  └─ vqls.npy
├─ plots                          # Plots for the report
│  ├─ linear_regression_vqls.png
│  ├─ linear_regression.png
│  ├─ losses_linear_no_cobyla.png
│  ├─ losses_linear.png
│  ├─ losses_log.png
│  ├─ neural_network.png
│  ├─ pqc_cobyla.png
│  ├─ pqc_lbfgsb.png
│  └─ vqls.png
└─ src
   ├─ abstract.py                 # Abstract base class for all the models
   ├─ common.py                   # Common functions and variables for all models
   ├─ demo.ipynb                  # Jupyter notebook to demonstrate the models
   ├─ helpers.py                  # Helper functions for graphing
   ├─ linear_regression.py        # Linear regression model
   ├─ neural_network.py           # Neural network model
   ├─ plots.py                    # Final plot generation for our report
   ├─ pqc.py                      # Parameterized Quantum Circuit (PQC) model
   ├─ train.py                    # Training script for all models
   └─ vqls.py                     # Variational Quantum Linear Solver (VQLS) model
```

### Framework Choices

#### Forecasting Models

- **Linear Regression**: We used `scikit-learn` for the linear regression model. It is widely used and provides a simple interface for training and evaluating linear regression models (as well as many others). It also allowed us to experiment with different regularization techniques, such as Lasso and Ridge regression, to improve the model's performance.
- **Neural Network**: We used `Keras` for the neural network model, which is a high-level API for building and training deep learning models. We used the `Sequential` model to define a simple feedforward neural network with two hidden layers.
- **Parameterized Quantum Circuit (PQC)**: We used `PennyLane` for the quantum parts of PQC because it is well-documented and we are familiar with it from class. As for the classical optimizer, we used `SciPy`'s `optimize` module to perform the optimization of the parameters. We used the `L-BFGS-B` and `COBYLA` optimizers, as was done in the paper.
- **Variational Quantum Linear Solver (VQLS)**: We used `PennyLane` for the quantum parts of VQLS, for the same reasons as PQC. The classical optimizer was again `SciPy`'s `optimize` module, but we used just the `COBYLA` optimizer.

#### Miscellaneous

- `NumPy` for the linear algebra operations and matrix manipulations since it is powerful and used by many of the other libraries we are using.
- `Matplotlib` for plotting results and exporting report figures since we are familiar with it and has all the features we need.
- `TensorFlow` to use Keras (this is a requirement for Keras installation).
- `Pre-commit` for automated formatting and linting checks. This is a great tool to ensure that our code is consistently formatted and adheres to best practices. We used `black` for code formatting and `flake8` for linting. The pre-commit hooks are configured to run automatically before each commit, ensuring that our code is always in a clean state.

### Design Decisions

#### Model Definitions

Since we had to implement four models, we decided to use an abstract class to define a common interface for all the models. This ensures that all the models can be defined and used in the same way, greatly simplifying various areas of the codebase. The abstract class contains method specifications for training, evaluating, and predicting with the models. Each model then inherits from this base class and implements these required methods.

Additionally, the base class contains static methods for loading, preprocessing, and postprocessing datasets. This allows us to keep the data handling code in one place and ensures that all models use the same data processing pipeline. Initially we considered having these in a different class or in distinct functions, but decided that it was better to keep everything together since it would greatly simplify things from a usage point of view.

Lastly, there are methods for saving and loading the models implemented in the base class, which allows us to easily save and load precomputed models without having to duplicate code in each model. We went through a few iterations of this design, initially having each model save and load what it wanted, but this led to a lot of duplicated code and made it difficult to keep track of what was being saved and loaded. By having a common interface, we were able to simplify this process and make it more consistent across all models.

Accompanying each model is a training function that handles the training process specific to the paper's dataset. We kept these outside of the classes but in the same file since they are closely related but we don't want to force only the paper's dataset to be used with the models as part of the design. Doing it this way allows us to have each model trained separately, but still allow for training all models at once if desired.

#### Model Usage

We decided to use a Jupyter notebook for the demo and examples of how to use the models. This allows us to easily demonstrate the models and their usage, as well as provide examples of how to generate plots all in one place.

To assist our demo notebook and training functions, we created a common file that contains helper functions for loading preprocessing, and postprocessing the paper's dataset. It also contains a few constants that are used throughout the codebase (but outside of the model classes). This was done again to not couple the models to the paper's dataset, but still provide a convenient way to load and preprocess the data.

Another helper file was created for generating plots that emulate the ones in the paper. This was done for consistency and to reduce duplication.

Finally, we created a plotting script that generates the final plots for our report. This script loads the precomputed models and generates the plots using the helper functions. This allows us to easily generate the plots without having to run the training process again, which would take a long time.

### Limitations

- For linear regression, `scikit-learn` does not give any way for us to get the mean squared error (MSE) when it uses gradient descent to optimize the weights. Luckily, this is only applicable to L1 regularization (since direct solve is used for L2 and no regularization) and we found that L1 only used one iteration on the paper's dataset anyway, but this is something to keep in mind if we were to use a different dataset.
- For VQLS, we initially used `SciPy`'s default `L-BFGS-B` optimizer, but it converged way too slowly (roughly 25 minutes for 2 qubits). We switched to `COBYLA`, which is a gradient-free optimizer, and it converged much faster (roughly 10 minutes for 2 qubits), but with a small decrease in accuracy since we had to increase the tolerance at the same time. Regardless though, the training time is still very long and grows exponentially with the number of qubits due to the algorithm and complexity of simulating more qubits. Since the paper did not get VQLS working, we decided to only train it with 2 qubits and compare it against linear regression with the same window size to prove correctness.

## Instructions

Our project requires Python and is compatible with **version 3.12 only**. It is recommended to create a virtual environment prior to running anything.

### Installation

Please install the required packages using the following command:

```bash
pip install -r requirements.txt
```

### Usage

For an overview of the models and their usage, please refer to the Jupyter notebook [`demo.ipynb`](src/demo.ipynb). The notebook contains examples of how to use the models and generate plots.

#### Generating Plots

To generate the final plots for our report, run the following command inside the [`src`](src/) directory:

```bash
python plots.py
```
This will generate the plots using the models from the [`models`](models/) directory and save them in the [`plots`](plots/) directory.

#### Training the Models

If you want to train the models, from inside the the [`src`](src/) directory you can either run each model's respective Python file (e.g. `python neural_network.py`) or run:

```bash
python train.py
```
This will train all the models and save them in the [`models`](models/) directory. Note that this will take a very long time to run.

## Contributions

- **Cihan Bosnali**
  - [Abstract base class for models](src/abstract.py)
  - [Neural network implementation](src/neural_network.py)
  - [Presentation slides](documents/presentation.pdf)
  - [Final report](documents/report.pdf)
- **Dante Prins**
  - [Data collection](data/paper-data.csv)
  - [Variational Quantum Linear Solver (VQLS) implementation](src/vqls.py)
  - [Presentation slides](documents/presentation.pdf)
  - [Final report](documents/report.pdf)
- **Elio Di Nino**
  - Repository setup
    - [`gitignore`](.gitignore)
    - [Pre-commit configuration](.pre-commit-config.yaml)
    - [GitHub Actions workflow to enforce pre-commit hooks](.github/workflows/pre-commit.yml)
    - Ruleset to require pull requests to main
  - [Abstract base class for models](src/abstract.py)
  - [Linear regression model](src/linear_regression.py)
  - [Helper functions for graphs](src/helpers.py)
  - [Demo notebook](src/demo.ipynb)
  - [Training script](src/train.py) and per-model functions
  - [Common functions and variables](src/common.py)
  - [Integrating VQLS into the class framework](src/vqls.py)
  - [Trained linear regression, neural network, and VQLS models](models/)
  - [Plot generation script](src/plots.py)
  - [All README documentation](README.md)
  - [Presentation slides](documents/presentation.pdf)
  - [Final report](documents/report.pdf)
- **Richard Sun**
  - [Abstract base class for models](src/abstract.py)
  - [Dataset loading, preprocessing, and postprocessing](src/abstract.py)
  - [Parameterized Quantum Circuit (PQC) Implementation](src/pqc.py)
  - [Load and save model functions](src/abstract.py)
  - [Trained PQC models](models/)
  - [Demo notebook](src/demo.ipynb)
  - [Presentation slides](documents/presentation.pdf)
  - [Final report](documents/report.pdf)
