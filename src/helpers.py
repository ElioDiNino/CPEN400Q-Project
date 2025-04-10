import matplotlib.pyplot as plt
from datetime import date
from dateutil.relativedelta import relativedelta
from numpy import ndarray
import numpy as np

from common import WINDOW_SIZE


def plot_loss(
    iterations: list[int],
    linear_reg_loss: list[float] | None,
    neural_net_loss: list[float] | None,
    pqc_lbfgsb_loss: list[float] | None,
    pqc_cobyla_loss: list[float] | None,
    vqls_loss: list[float] | None,
    title: str = "",
    yscale: str = "linear",
    save_path: str | None = None,
):
    """
    Plot the losses of different methods over iterations.

    Args:
        iterations: List of iteration numbers.
        linear_reg_loss: Loss values for Linear Regression method.
        neural_net_loss: Loss values for Neural Network method.
        pqc_lbfgsb_loss: Loss values for PQC with L-BFGS-B method.
        pqc_cobyla_loss: Loss values for PQC with COBYLA method.
        vqls_loss: Loss values for VQLS method.
        title: Title for the plot. Leave empty for no title.
        save_path: Path to save the plot. If None, the plot will be shown
          interactively.
        yscale: Scale for the y-axis. Default is 'linear'. Can be 'log' for
          logarithmic scale.

    Returns:
        None, but displays a plot.
    """
    # Make sure all loss lists are of iteration length and extend if necessary
    length = len(iterations)

    plt.figure(figsize=(10, 4))
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = "Times New Roman"
    plt.rcParams["font.size"] = 18
    plt.title(title)

    if vqls_loss:
        vqls_loss.extend([vqls_loss[-1]] * (length - len(vqls_loss)))
        plt.plot(
            iterations,
            vqls_loss,
            linestyle=(0, (3, 1, 1, 1)),
            color="purple",
            label="VQLS",
        )
    if pqc_lbfgsb_loss:
        pqc_lbfgsb_loss.extend(
            [pqc_lbfgsb_loss[-1]] * (length - len(pqc_lbfgsb_loss))
        )
        plt.plot(
            iterations,
            pqc_lbfgsb_loss,
            linestyle="-",
            color="blue",
            label="PQC with L-BFGS-B",
        )
    if pqc_cobyla_loss:
        pqc_cobyla_loss.extend(
            [pqc_cobyla_loss[-1]] * (length - len(pqc_cobyla_loss))
        )
        plt.plot(
            iterations,
            pqc_cobyla_loss,
            linestyle="--",
            color="red",
            label="PQC with COBYLA",
        )
    if linear_reg_loss:
        linear_reg_loss.extend(
            [linear_reg_loss[-1]] * (length - len(linear_reg_loss))
        )
        plt.plot(
            iterations,
            linear_reg_loss,
            linestyle="-.",
            color="lime",
            label="Linear Regression",
        )
    if neural_net_loss:
        neural_net_loss.extend(
            [neural_net_loss[-1]] * (length - len(neural_net_loss))
        )
        plt.plot(
            iterations,
            neural_net_loss,
            linestyle=":",
            color="firebrick",
            label="Neural Network",
        )

    plt.xlabel("Iteration")
    plt.ylabel("Loss (MSE)")

    plt.xticks(range(0, len(iterations), 100))  # Show every 100th iteration
    plt.xlim([0, len(iterations) - 1])  # Remove padding from either side
    plt.yscale(yscale)  # Set y-axis scale
    plt.legend(loc="upper right")
    plt.grid()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=600)
    else:
        plt.show()


def plot_sales_growth(
    start_date: date,
    sales_growth: list[float] | ndarray,
    sales_growth_pred: list[float] | ndarray,
    pred_label: str = "Predicted",
    pred_linestyle: str | tuple = None,
    pred_color: str = "red",
    title: str = "",
    train_test_split: int = None,
    window_size: int = WINDOW_SIZE,
    save_path: str = None,
):
    """
    Plot sales growth over time.

    Args:
        start_date: The starting date for the sales growth data.
        sales_growth: The actual sales growth data.
        sales_growth_pred: The predicted sales growth data.
        pred_label: Label for the predicted data line.
        pred_linestyle: Line style for the predicted data line.
        pred_color: Color for the predicted data line.
        title: title for the plot. Leave empty for no title.
        train_test_split: Index to indicate the train/test split in the data.
          Leave empty for no split line.
        window_size: The size of the window used for predictions.
          Note: This is used to add empty values to the beginning of
          sales_growth_pred to align with the actual data.
        save_path: Path to save the plot. If None, the plot will be shown
          interactively.

    Returns:
        None, but displays a plot.
    """
    # Create a list of months starting from the given start date
    months = [
        (start_date + relativedelta(months=i)).strftime("%Y-%m")
        for i in range(len(sales_growth))
    ]

    # Add empty values to the beginning of sales_growth_pred
    sales_growth_pred = np.concatenate(
        (np.empty((window_size,), dtype=object), sales_growth_pred)
    )

    plt.figure(figsize=(10, 2.5))
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = "Times New Roman"
    plt.rcParams["font.size"] = 18
    plt.title(title)

    plt.plot(months, sales_growth, label="Data", color="grey")
    plt.plot(
        months,
        sales_growth_pred,
        label=pred_label,
        linestyle=pred_linestyle,
        color=pred_color,
    )

    plt.ylabel("M\u20ac")  # Unicode for Euro sign
    plt.yticks(range(-4, 5, 2))
    plt.xticks(
        range(6, len(months), 6),  # Show every 6th month
        rotation=45,
        ha="right",
    )
    plt.xlim([0, len(months) - 1])  # Remove padding from either side

    if train_test_split is not None:
        plt.axvline(
            x=months[train_test_split],
            color="black",
            linestyle="--",
            label="Train/Test Split",
        )

    plt.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.4),
        frameon=False,
        ncol=3,
    )
    plt.grid()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=600)
    else:
        plt.show()
