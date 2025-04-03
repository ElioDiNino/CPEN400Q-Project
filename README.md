# CPEN 400Q Final Project

An implementation of "[Exploring Hybrid Quantum-Classical Methods for Practical Time-Series Forecasting](https://arxiv.org/abs/2412.05615v1)"

## Overview

The goal of this project is to implement and compare both classical and hybrid quantum-classical models for time-series forecasting. The models are based off the paper linked above, but are fine-tuned to improve results.

The repository is structured as follows:
```
Root
├─ .flake8                   # Flake8 (Python formatting check) configuration
├─ .pre-commit-config.yaml   # Pre-commit configuration
├─ README.md
├─ requirements.txt          # Required packages
├─ .github
│  └─ workflows
│     └─ pre-commit.yml      # GitHub Actions workflow to enforce pre-commit hooks
├─ data                      # Datasets
│  ├─ paper-data.csv
│  └─ paper-data_processed.csv
├─ weights                   # Trained weights for the models
│  ├─ pqc_cobyla.npy
│  └─ pqc_lbfgsb.npy
└─ src
   ├─ abstract.py            # Abstract base class for all our models
   ├─ demo.ipynb             # Jupyter notebook to demonstrate the models
   ├─ helpers.py             # Helper functions for graphing
   ├─ linear_regression.py   # Linear regression model
   ├─ neural_network.py      # Neural network model
   ├─ pqc.py                 # Parameterized Quantum Circuit (PQC) model
   └─ plots.py               # Final plot generation for the paper
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

To generate the final plots for the paper, run the following command:

```bash
python src/plots.py
```

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
- **Richard Sun**
  - [Abstract base class for models](src/abstract.py)
  - [Dataset loading, preprocessing, and postprocessing](src/abstract.py)
  - [PQC Implementation](src/pqc.py)
  - [Demo notebook](src/demo.ipynb)
