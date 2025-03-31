import matplotlib.pyplot as plt
from datetime import date
from dateutil.relativedelta import relativedelta


def plot_loss(
    iterations,
    vqls_loss,
    pqc_lbfgsb_loss,
    pqc_cobyla_loss,
    linear_reg_loss,
    neural_net_loss,
):
    """
    Plot the losses of different methods over iterations.

    Args:
        iterations: List of iteration numbers.
        vqls_loss: Loss values for VQLS method.
        pqc_lbfgsb_loss: Loss values for PQC with L-BFGS-B method.
        pqc_cobyla_loss: Loss values for PQC with COBYLA method.
        linear_reg_loss: Loss values for Linear Regression method.
        neural_net_loss: Loss values for Neural Network method.

    Returns:
        None, but displays a plot.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(iterations, vqls_loss, label="VQLS")
    plt.plot(iterations, pqc_lbfgsb_loss, label="PQC with L-BFGS-B")
    plt.plot(iterations, pqc_cobyla_loss, label="PQC with COBYLA")
    plt.plot(iterations, linear_reg_loss, label="Linear Regression")
    plt.plot(iterations, neural_net_loss, label="Neural Network")
    plt.xlabel("Iteration")
    plt.ylabel("Loss (MSE)")
    plt.legend()
    plt.grid()
    plt.show()


def plot_sales_growth(
    start_date: date,
    sales_growth,
    sales_growth_pred,
    pred_label="Predicted",
    pred_linestyle=None,
    pred_color="red",
    title="",
    train_test_split=None,
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

    Returns:
        None, but displays a plot.
    """
    # Create a list of months starting from the given start date
    months = [
        (start_date + relativedelta(months=i)).strftime("%Y-%m")
        for i in range(len(sales_growth))
    ]
    plt.figure(figsize=(10, 2.5))
    plt.plot(months, sales_growth, label="Data", color="grey")
    plt.plot(
        months,
        sales_growth_pred,
        label=pred_label,
        linestyle=pred_linestyle,
        color=pred_color,
    )
    plt.title(title)
    plt.ylabel("M\u20ac")  # Unicode for Euro sign
    plt.xticks(
        range(0, len(months), max(1, len(months) // 10)),  # Show fewer labels
        rotation=45,
        ha="right",
    )

    if train_test_split is not None:
        plt.axvline(
            x=months[train_test_split],
            color="black",
            linestyle="--",
            label="Train/Test Split",
        )
    plt.legend()
    plt.grid()
    plt.show()
