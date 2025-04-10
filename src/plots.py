from common import (
    START_DATE,
    WINDOW_SIZE,
    MAX_ITERATIONS,
    VQLS_WIRES,
    get_paper_data,
    rescale_paper_data,
)
from helpers import plot_loss, plot_sales_growth
from abstract import ForecastingMethod
from linear_regression import LinearRegression
from neural_network import NeuralNetwork
from pqc import PQC
from vqls import VQLS


# Model definitions
defs: dict[int, list[tuple[str, str, ForecastingMethod, str, str]]] = {
    WINDOW_SIZE: [
        (
            "Linear Regression",
            "linear_regression",
            LinearRegression,
            "-.",
            "lime",
        ),
        (
            "Linear Regression with L1 Regularization",
            "linear_regression_l1",
            LinearRegression,
            "-.",
            "lime",
        ),
        (
            "Linear Regression with L2 Regularization",
            "linear_regression_l2",
            LinearRegression,
            "-.",
            "lime",
        ),
        ("Neural Network", "neural_network", NeuralNetwork, ":", "firebrick"),
        ("PQC with L-BFGS-B", "pqc_lbfgsb", PQC, "-", "blue"),
        ("PQC with COBYLA", "pqc_cobyla", PQC, "--", "red"),
    ],
    (2**VQLS_WIRES): [
        ("VQLS", "vqls", VQLS, (0, (3, 1, 1, 1)), "purple"),
        (
            "Linear Regression",
            "linear_regression_vqls",
            LinearRegression,
            "-.",
            "lime",
        ),  # To compare with VQLS
    ],
}

losses = []

for window_size, models in defs.items():
    # Load the data
    (
        X,
        y,
        X_train,
        X_test,
        y_train,
        y_test,
        train_test_split_index,
        min_value,
        max_value,
        initial_value,
    ) = get_paper_data(window_size=window_size)

    # Rescale data
    ORIG_SCALED, GRAPH_SPLIT_INDEX = rescale_paper_data(
        X,
        y,
        min_value,
        max_value,
        initial_value,
        train_test_split_index,
        window_size,
    )

    # Plot the sales growth
    for name, model_file, model_class, linestyle, color in models:
        model = model_class.load_model(f"../models/{model_file}.pkl")

        X_pred_scaled = ForecastingMethod.post_process_data(
            model.predict(X),
            min_value,
            max_value,
            initial_value,
            scale_to_range=True,
        )

        plot_sales_growth(
            START_DATE,
            ORIG_SCALED,
            X_pred_scaled,
            pred_linestyle=linestyle,
            pred_color=color,
            train_test_split=GRAPH_SPLIT_INDEX,
            window_size=window_size,
            save_path=f"../plots/{model_file}.png",
        )

        # Save the loss for the next plot
        losses.append(model.mse_iterations)


# Plotting the loss
for scale in ["linear", "log"]:
    plot_loss(
        list(range(MAX_ITERATIONS)),
        losses[0],
        losses[3],
        losses[4],
        losses[5],
        losses[6],
        save_path=f"../plots/losses_{scale}.png",
        yscale=scale,
    )

plot_loss(
    list(range(MAX_ITERATIONS)),
    losses[0],
    losses[3],
    losses[4],
    None,
    losses[6],
    save_path="../plots/losses_linear_no_cobyla.png",
)
