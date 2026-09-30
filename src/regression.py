import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


class StandardScalerManual:
    def __init__(self):
        self.mean_ = None
        self.scale_ = None

    def fit(self, X):
        self.mean_ = np.mean(X, axis=0)
        self.scale_ = np.std(X, axis=0)

        # Prevent division by zero
        self.scale_[self.scale_ == 0] = 1.0

        return self

    def transform(self, X):
        return (X - self.mean_) / self.scale_

    def fit_transform(self, X):
        return self.fit(X).transform(X)


def train_test_split_manual(
    X,
    y,
    test_size=0.2,
    random_seed=3513,
):
    rng = np.random.default_rng(random_seed)

    indices = np.arange(len(X))
    rng.shuffle(indices)

    test_count = int(len(X) * test_size)

    test_indices = indices[:test_count]
    train_indices = indices[test_count:]

    return (
        X[train_indices],
        X[test_indices],
        y[train_indices],
        y[test_indices],
    )


def add_bias(X):
    return np.column_stack(
        [
            np.ones(X.shape[0]),
            X,
        ]
    )


def mean_squared_error(y_true, y_pred):
    return np.mean(
        (y_true - y_pred) ** 2
    )


def train_gradient_descent(
    X,
    y,
    learning_rate=0.01,
    epochs=5000,
):
    weights = np.zeros(X.shape[1])

    loss_history = []

    for _ in range(epochs):

        predictions = X @ weights

        error = predictions - y

        gradient = (
            2
            / len(X)
            * X.T
            @ error
        )

        weights -= learning_rate * gradient

        loss = mean_squared_error(
            y,
            predictions,
        )

        loss_history.append(loss)

    return weights, loss_history


def calculate_metrics(y_true, y_pred):

    mae = np.mean(
        np.abs(y_true - y_pred)
    )

    rmse = np.sqrt(
        np.mean(
            (y_true - y_pred) ** 2
        )
    )

    ss_res = np.sum(
        (y_true - y_pred) ** 2
    )

    ss_tot = np.sum(
        (y_true - np.mean(y_true)) ** 2
    )

    r2 = 1 - (ss_res / ss_tot)

    return mae, rmse, r2


def run_regression(
    X,
    y,
    output_dir,
    random_seed=3513,
    models_dir="models",
):
    output_dir = Path(output_dir)
    models_dir = Path(models_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    models_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Train/test split
    (
        X_train,
        X_test,
        y_train,
        y_test,
    ) = train_test_split_manual(
        X,
        y,
        test_size=0.2,
        random_seed=random_seed,
    )

    # Scale using training data only
    scaler = StandardScalerManual()

    X_train_scaled = scaler.fit_transform(
        X_train
    )

    X_test_scaled = scaler.transform(
        X_test
    )

    # Add bias
    X_train_bias = add_bias(
        X_train_scaled
    )

    X_test_bias = add_bias(
        X_test_scaled
    )

    # Train
    weights, loss_history = train_gradient_descent(
        X_train_bias,
        y_train,
    )

    # Predictions
    y_pred = X_test_bias @ weights

    # Metrics
    mae, rmse, r2 = calculate_metrics(
        y_test,
        y_pred,
    )

    print("\nRegression completed.")
    print(f"Training rows: {len(X_train)}")
    print(f"Testing rows: {len(X_test)}")
    print(f"Test MAE: {mae:.4f}")
    print(f"Test RMSE: {rmse:.4f}")
    print(f"Test R²: {r2:.4f}")

    # ---------------------------------------------------------
    # Save metrics
    # ---------------------------------------------------------

    metrics = {
        "model": "Linear Regression with Batch Gradient Descent",
        "random_seed": random_seed,
        "training_rows": int(len(X_train)),
        "testing_rows": int(len(X_test)),
        "mae": float(mae),
        "rmse": float(rmse),
        "r2": float(r2),
    }

    metrics_path = (
        output_dir
        / "regression_metrics.json"
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
    # Save loss plot
    # ---------------------------------------------------------

    plt.figure(
        figsize=(8, 5)
    )

    plt.plot(
        loss_history
    )

    plt.title(
        "Regression Training Loss"
    )

    plt.xlabel(
        "Epoch"
    )

    plt.ylabel(
        "Mean Squared Error"
    )

    plt.grid(
        True,
        alpha=0.25,
    )

    plt.tight_layout()

    loss_path = (
        output_dir
        / "regression_loss.png"
    )

    plt.savefig(
        loss_path,
        dpi=150,
    )

    plt.close()

    # ---------------------------------------------------------
    # Save regression model
    # ---------------------------------------------------------

    regression_model = {
        "weights": weights.tolist(),
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "feature_count": int(X.shape[1]),
        "model_version": "1.0",
    }

    regression_model_path = (
        models_dir
        / "regression_model.json"
    )

    with open(
        regression_model_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            regression_model,
            file,
            indent=4,
        )

    print(
        f"Metrics saved to: {metrics_path}"
    )

    print(
        f"Loss plot saved to: {loss_path}"
    )

    print(
        f"Regression model saved to: "
        f"{regression_model_path}"
    )