import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def train_test_split_numpy(X, y, test_size=0.2, random_seed=3513):
    """
    Split X and y into training and testing sets.
    """

    rng = np.random.default_rng(random_seed)

    indices = np.arange(len(X))
    rng.shuffle(indices)

    test_count = int(len(X) * test_size)

    test_indices = indices[:test_count]
    train_indices = indices[test_count:]

    X_train = X[train_indices]
    X_test = X[test_indices]

    y_train = y[train_indices]
    y_test = y[test_indices]

    return X_train, X_test, y_train, y_test


def fit_standard_scaler(X_train):
    """
    Calculate scaling parameters using training data only.
    """

    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)

    # Prevent division by zero
    std[std == 0] = 1.0

    return mean, std


def transform_standard_scaler(X, mean, std):
    """
    Standardize data using previously calculated training parameters.
    """

    return (X - mean) / std


def add_bias_column(X):
    """
    Add a column of ones for the intercept/bias.
    """

    return np.column_stack(
        [np.ones(X.shape[0]), X]
    )


def compute_predictions(X, weights):
    """
    Calculate predictions for linear regression.
    """

    return X @ weights


def compute_mse(y_true, y_pred):
    """
    Mean Squared Error.
    """

    return np.mean((y_true - y_pred) ** 2)


def batch_gradient_descent(
    X,
    y,
    learning_rate=0.01,
    epochs=5000,
):
    """
    Train linear regression using batch gradient descent.

    The entire training dataset is used for every gradient update.
    """

    weights = np.zeros(X.shape[1], dtype=float)

    loss_history = []

    n_samples = X.shape[0]

    for epoch in range(epochs):

        predictions = compute_predictions(X, weights)

        errors = predictions - y

        gradient = (2 / n_samples) * (X.T @ errors)

        weights = weights - learning_rate * gradient

        loss = compute_mse(y, predictions)

        loss_history.append(loss)

    return weights, loss_history


def calculate_mae(y_true, y_pred):
    return float(
        np.mean(np.abs(y_true - y_pred))
    )


def calculate_rmse(y_true, y_pred):
    return float(
        np.sqrt(np.mean((y_true - y_pred) ** 2))
    )


def calculate_r2(y_true, y_pred):

    ss_res = np.sum(
        (y_true - y_pred) ** 2
    )

    ss_tot = np.sum(
        (y_true - np.mean(y_true)) ** 2
    )

    if ss_tot == 0:
        return 0.0

    return float(
        1 - (ss_res / ss_tot)
    )


def save_loss_plot(loss_history, output_path):

    plt.figure(figsize=(8, 5))

    plt.plot(
        range(1, len(loss_history) + 1),
        loss_history,
    )

    plt.xlabel("Epoch")
    plt.ylabel("Mean Squared Error")
    plt.title("Regression Training Loss")

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=150,
    )

    plt.close()


def run_regression(
    X,
    y,
    output_dir,
    random_seed=3513,
):
    """
    Complete regression pipeline.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ---------------------------------------------------------
    # 1. Train/test split
    # ---------------------------------------------------------

    (
        X_train,
        X_test,
        y_train,
        y_test,
    ) = train_test_split_numpy(
        X,
        y,
        test_size=0.2,
        random_seed=random_seed,
    )

    # ---------------------------------------------------------
    # 2. Fit scaler ONLY on training data
    # ---------------------------------------------------------

    train_mean, train_std = fit_standard_scaler(
        X_train
    )

    X_train_scaled = transform_standard_scaler(
        X_train,
        train_mean,
        train_std,
    )

    X_test_scaled = transform_standard_scaler(
        X_test,
        train_mean,
        train_std,
    )

    # ---------------------------------------------------------
    # 3. Add bias/intercept column
    # ---------------------------------------------------------

    X_train_with_bias = add_bias_column(
        X_train_scaled
    )

    X_test_with_bias = add_bias_column(
        X_test_scaled
    )

    # ---------------------------------------------------------
    # 4. Train using batch gradient descent
    # ---------------------------------------------------------

    weights, loss_history = batch_gradient_descent(
        X_train_with_bias,
        y_train,
        learning_rate=0.01,
        epochs=5000,
    )

    # ---------------------------------------------------------
    # 5. Generate predictions
    # ---------------------------------------------------------

    train_predictions = compute_predictions(
        X_train_with_bias,
        weights,
    )

    test_predictions = compute_predictions(
        X_test_with_bias,
        weights,
    )

    # ---------------------------------------------------------
    # 6. Calculate metrics
    # ---------------------------------------------------------

    train_mae = calculate_mae(
        y_train,
        train_predictions,
    )

    test_mae = calculate_mae(
        y_test,
        test_predictions,
    )

    train_rmse = calculate_rmse(
        y_train,
        train_predictions,
    )

    test_rmse = calculate_rmse(
        y_test,
        test_predictions,
    )

    train_r2 = calculate_r2(
        y_train,
        train_predictions,
    )

    test_r2 = calculate_r2(
        y_test,
        test_predictions,
    )

    # ---------------------------------------------------------
    # 7. Save metrics
    # ---------------------------------------------------------

    metrics = {
        "model": "Linear Regression",
        "method": "Batch Gradient Descent",
        "random_seed": random_seed,
        "test_size": 0.2,
        "training_rows": int(len(X_train)),
        "testing_rows": int(len(X_test)),
        "learning_rate": 0.01,
        "epochs": 5000,
        "train": {
            "mae": train_mae,
            "rmse": train_rmse,
            "r2": train_r2,
        },
        "test": {
            "mae": test_mae,
            "rmse": test_rmse,
            "r2": test_r2,
        },
    }

    metrics_path = (
        output_dir / "regression_metrics.json"
    )

    with open(
        metrics_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            metrics,
            file,
            indent=4,
        )

    # ---------------------------------------------------------
    # 8. Save loss graph
    # ---------------------------------------------------------

    loss_path = (
        output_dir / "regression_loss.png"
    )

    save_loss_plot(
        loss_history,
        loss_path,
    )

    print("\nRegression completed.")
    print(f"Training rows: {len(X_train)}")
    print(f"Testing rows: {len(X_test)}")
    print(f"Test MAE: {test_mae:.4f}")
    print(f"Test RMSE: {test_rmse:.4f}")
    print(f"Test R²: {test_r2:.4f}")
    print(
        f"Metrics saved to: {metrics_path}"
    )
    print(
        f"Loss plot saved to: {loss_path}"
    )

    return {
        "weights": weights,
        "train_mean": train_mean,
        "train_std": train_std,
        "loss_history": loss_history,
        "test_predictions": test_predictions,
        "test_actual": y_test,
        "metrics": metrics,
    }