from abstract import ForecastingMethod
from numpy import ndarray, fromfile
from sklearn.linear_model import (
    LinearRegression as SklearnLinearRegression,
    Ridge,
    Lasso,
)
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error


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
        alphas: list[float] = [0.01, 0.1, 1, 10, 100],
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

            grid_search = GridSearchCV(
                base_model,
                param_grid,
                cv=self.cv_folds,
                scoring="neg_mean_squared_error",
            )
            grid_search.fit(train_X, train_y)
            self.model = grid_search.best_estimator_

    def save_weights(self, filepath: str) -> bool:
        if self.model is None or self.model.coef_ is None:
            return False

        self.model.coef_.tofile(filepath)
        return True

    def load_weights(self, filepath: str) -> bool:
        if self.regularization == "l2":
            self.model = Ridge(fit_intercept=self.fit_intercept)
        elif self.regularization == "l1":
            self.model = Lasso(fit_intercept=self.fit_intercept)
        else:
            self.model = SklearnLinearRegression(
                fit_intercept=self.fit_intercept
            )

        try:
            self.model.coef_ = fromfile(filepath)
        except FileNotFoundError:
            return False
        return True

    def predict(self, X: ndarray) -> ndarray:
        return self.model.predict(X)

    def score(self, X: ndarray, y: ndarray) -> float:
        """
        Compute the mean squared error of the model.

        Args:
            X: Input data
            y: True labels

        Returns:
            float: The mean squared error of the model on the given data
        """
        predictions = self.predict(X)
        return mean_squared_error(y, predictions)
