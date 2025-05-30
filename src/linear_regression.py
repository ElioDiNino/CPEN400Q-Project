from numpy import ndarray
from sklearn.linear_model import (
    LinearRegression as SklearnLinearRegression,
    Ridge,
    Lasso,
)
from sklearn.model_selection import GridSearchCV

from common import WINDOW_SIZE, VQLS_WIRES, get_paper_data
from abstract import ForecastingMethod


class LinearRegression(ForecastingMethod):
    """
    This class implements a linear regression model with optional
    regularization (L1 or L2).
    """

    def __init__(
        self,
        fit_intercept: bool = True,
        regularization: str | None = None,
        cv_folds: int | None = 5,
        alphas: list[float] = [0.01, 0.1, 0.3, 1, 5, 10],
    ):
        """
        Initialize the Linear Regression model.

        Args:
            fit_intercept : Whether to fit an intercept term in the model
            regularization : The type of regularization to apply to the model
              ("l1", "l2", or None)
            cv_folds : Number of cross-validation folds to use for
              hyperparameter tuning (if regularization is used)
            alphas : List of regularization strength values to try
        """
        super().__init__()

        if regularization not in ["l1", "l2", None]:
            raise ValueError(
                "Invalid regularization type. Use 'l1', 'l2', or None."
            )

        self.fit_intercept = fit_intercept
        self.regularization = regularization
        self.cv_folds = cv_folds
        self.alphas = alphas
        self.model = None
        self.__mse_iterations: list[float] = []

    @property
    def mse_iterations(self) -> list[float]:
        return self.__mse_iterations

    def train(self, train_X: ndarray, train_y: ndarray) -> None:
        """
        Train the model, using cross-validation if regularization is
        specified.

        Args:
            train_X: Training data
            train_y: Training labels
        """
        if self.regularization is None:
            self.model = SklearnLinearRegression(
                fit_intercept=self.fit_intercept
            )
            self.model.fit(train_X, train_y)
        else:
            param_grid = {
                "alpha": self.alphas,
            }  # Regularization strength values
            if self.regularization == "l2":
                base_model = Ridge(fit_intercept=self.fit_intercept)
            elif self.regularization == "l1":
                base_model = Lasso(fit_intercept=self.fit_intercept)
            else:
                raise ValueError(
                    "Invalid regularization type. Use 'l1', 'l2', or None."
                )

            # Use GridSearchCV to find the best hyperparameters
            grid_search = GridSearchCV(
                base_model,
                param_grid,
                cv=self.cv_folds,
                scoring="neg_mean_squared_error",
            )
            grid_search.fit(train_X, train_y)
            self.model = grid_search.best_estimator_

        # All the variations only use a single iteration of training
        # See the README for limitation details
        self.__mse_iterations = [self.score(train_X, train_y)]

    def predict(self, X: ndarray) -> ndarray:
        return self.model.predict(X)


def train():
    """
    Train the linear regression models on the paper data
    """
    models = [
        (
            "Linear Regression with Y-Intercept",
            "linear_regression",
            True,
            None,
            WINDOW_SIZE,
        ),
        (
            "Linear Regression with L1 Regularization",
            "linear_regression_l1",
            True,
            "l1",
            WINDOW_SIZE,
        ),
        (
            "Linear Regression with L2 Regularization",
            "linear_regression_l2",
            True,
            "l2",
            WINDOW_SIZE,
        ),
        (
            "Linear Regression with VQLS Window Size",
            "linear_regression_vqls",
            False,
            None,
            2**VQLS_WIRES,
        ),
    ]

    for name, model_file, fit_intercept, regularization, window_size in models:
        print(f"\nTraining {name}...")

        _, _, X_train, X_test, y_train, y_test, _, _, _, _ = get_paper_data(
            window_size=window_size
        )

        # Train the model
        lr = LinearRegression(
            fit_intercept=fit_intercept, regularization=regularization
        )
        lr.train(X_train, y_train)
        lr.save_model(f"../models/{model_file}")

        # Evaluate the model
        print(f"Training Loss (MSE): {lr.score(X_train, y_train)}")
        print(f"Testing Loss (MSE): {lr.score(X_test, y_test)}")


if __name__ == "__main__":
    train()
