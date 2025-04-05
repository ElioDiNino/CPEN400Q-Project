# CPEN 400Q Final Project

An implementation of "[Exploring Hybrid Quantum-Classical Methods for Practical Time-Series Forecasting](https://arxiv.org/abs/2412.05615v1)"

## Overview

The goal of this project is to implement and compare both classical and hybrid quantum-classical models for time-series forecasting. The models are based off the paper linked above, but are fine-tuned to improve results.

The repository is structured as follows:
```
├─ .flake8                   # Flake8 (Python formatting check) configuration
├─ .gitignore                # Git ignore file
├─ .pre-commit-config.yaml   # Pre-commit configuration (automatic formatting and other checks)
├─ README.md
├─ requirements.txt          # Required packages
├─ .github
│  └─ workflows
│     └─ pre-commit.yml      # GitHub Actions workflow to enforce pre-commit hooks
├─ data                      # Datasets
│  ├─ paper-data.csv
│  └─ paper-data-processed.csv
├─ documents
│  ├─ report.pdf             # Final report
│  └─ presentation.pdf       # In-class presentation
├─ models                    # Trained models/weights
│  ├─ pqc_cobyla.npy
│  └─ pqc_lbfgsb.npy
└─ src
   ├─ abstract.py            # Abstract base class for all the models
   ├─ common.py              # Common functions and variables for all models
   ├─ demo.ipynb             # Jupyter notebook to demonstrate the models
   ├─ helpers.py             # Helper functions for graphing
   ├─ linear_regression.py   # Linear regression model
   ├─ neural_network.py      # Neural network model
   ├─ plots.py               # Final plot generation for our report
   ├─ pqc.py                 # Parameterized Quantum Circuit (PQC) model
   ├─ train.py               # Training script for all models
   └─ vqls.py                # Variational Quantum Linear Regression (VQLS) model
```

### Framework Choices

TODO

### Design Decisions

TODO

### Limitations

TODO

## Instructions

Our project requires Python and is compatible with version 3.12 and above. It is recommended to create a virtual environment prior to running anything.

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
- **Dante Prins**
- **Elio Di Nino**
  - Repository setup
    - Base README
    - [`gitignore`](.gitignore)
    - [Pre-commit configuration](.pre-commit-config.yaml)
    - [GitHub Actions workflow to enforce pre-commit hooks](.github/workflows/pre-commit.yml)
    - Ruleset to require pull requests to main
  - [Abstract base class for models](src/abstract.py)
  - [Linear regression model](src/linear_regression.py)
  - [Helper functions for graphs](src/helpers.py)
  - [Demo notebook](src/demo.ipynb)
  - [Training script](src/train.py) and per-model functions
- **Richard Sun**
  - [Abstract base class for models](src/abstract.py)
  - [Dataset loading, preprocessing, and postprocessing](src/abstract.py)
  - [PQC Implementation](src/pqc.py)
  - [Demo notebook](src/demo.ipynb)
