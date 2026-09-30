import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def run_classification(
    X,
    y,
    output_dir,
    random_seed=3513,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Stratified train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=random_seed,
        stratify=y,
    )

    # Scale using training data only
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Logistic Regression
    # class_weight='balanced' helps the classifier pay attention
    # to the minority dispatch_attention=1 class.
    model = LogisticRegression(
        class_weight="balanced",
        random_state=random_seed,
        max_iter=1000,
    )

    model.fit(X_train_scaled, y_train)

    # Predictions
    y_pred = model.predict(X_test_scaled)
    y_probability = model.predict_proba(X_test_scaled)[:, 1]

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0,
    )
    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0,
    )
    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0,
    )

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    print("\nClassification completed.")
    print(f"Training rows: {len(X_train)}")
    print(f"Testing rows: {len(X_test)}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1: {f1:.4f}")

    print("\nConfusion Matrix:")
    print(cm)

    # Save metrics
    metrics = {
        "model": "Logistic Regression",
        "random_seed": random_seed,
        "training_rows": int(len(X_train)),
        "testing_rows": int(len(X_test)),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "class_distribution": {
            "class_0": int(np.sum(y == 0)),
            "class_1": int(np.sum(y == 1)),
        },
        "confusion_matrix": cm.tolist(),
        "costly_error": (
            "False negatives: records requiring dispatch attention "
            "that were predicted as not requiring attention."
        ),
    }

    metrics_path = output_dir / "classification_metrics.json"

    with open(metrics_path, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4)

    # Plot confusion matrix
    fig, ax = plt.subplots(figsize=(6, 5))

    image = ax.imshow(cm)

    ax.set_title("Dispatch Attention Confusion Matrix")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("Actual Label")

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])

    ax.set_xticklabels(["No Attention", "Attention"])
    ax.set_yticklabels(["No Attention", "Attention"])

    # Display values inside cells
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                cm[i, j],
                ha="center",
                va="center",
            )

    fig.colorbar(image, ax=ax)
    fig.tight_layout()

    plot_path = output_dir / "confusion_matrix.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)

    print(f"\nMetrics saved to: {metrics_path}")
    print(f"Confusion matrix saved to: {plot_path}")