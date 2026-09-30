import argparse
import json
import math
from pathlib import Path

import joblib
import numpy as np


GROUP_CODE = "AI_A1_G06"
MODEL_VERSION = "1.0"

FEATURE_NAMES = [
    "plot_area_ha",
    "rainfall_mm",
    "soil_ph",
    "seed_kg",
    "distance_km",
    "arrival_hour",
]


def error(message):
    print(json.dumps({"error": message}))
    raise SystemExit(1)


def validate_record(record):
    if not isinstance(record, dict):
        error("Record must be a JSON object.")

    missing = [name for name in FEATURE_NAMES if name not in record]

    if missing:
        error(f"Missing required fields: {', '.join(missing)}")

    for name in FEATURE_NAMES:
        value = record[name]

        if isinstance(value, bool):
            error(f"Field '{name}' must be numeric.")

        if not isinstance(value, (int, float)):
            error(f"Field '{name}' must be numeric.")

        if not math.isfinite(float(value)):
            error(f"Field '{name}' must be finite.")


def load_regression_model(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def main():
    parser = argparse.ArgumentParser(
        description="Run AI prediction pipeline."
    )

    parser.add_argument(
        "--record",
        required=True,
        help="JSON record containing the six required input features.",
    )

    args = parser.parse_args()

        # Parse JSON
    record_text = args.record.strip()

    # Remove surrounding single quotes if PowerShell passes them through.
    if (
        len(record_text) >= 2
        and record_text.startswith("'")
        and record_text.endswith("'")
    ):
        record_text = record_text[1:-1]

    # If PowerShell removed the JSON double quotes, restore them
    # around the known feature names.
    for name in FEATURE_NAMES:
        record_text = record_text.replace(
            f"{name}:",
            f'"{name}":',
        )

    try:
        record = json.loads(record_text)
    except json.JSONDecodeError as exc:
        error(f"Invalid JSON: {exc}")

    validate_record(record)

    base_dir = Path(__file__).resolve().parent
    models_dir = base_dir / "models"

    regression_path = models_dir / "regression_model.json"
    classification_path = models_dir / "classification_model.joblib"
    clustering_path = models_dir / "clustering_model.joblib"

    # Check model files
    missing_models = [
        str(path)
        for path in [
            regression_path,
            classification_path,
            clustering_path,
        ]
        if not path.exists()
    ]

    if missing_models:
        error(
            "Missing model file(s): "
            + ", ".join(missing_models)
        )

    # ---------------------------------------------------------
    # Regression
    # ---------------------------------------------------------

    regression_model = load_regression_model(regression_path)

    weights = np.array(
        regression_model["weights"],
        dtype=float,
    ).reshape(-1)

    scaler_mean = np.array(
        regression_model["scaler_mean"],
        dtype=float,
    )

    scaler_scale = np.array(
        regression_model["scaler_scale"],
        dtype=float,
    )

    X = np.array(
        [[float(record[name]) for name in FEATURE_NAMES]],
        dtype=float,
    )

    X_scaled = (X - scaler_mean) / scaler_scale

    X_with_bias = np.column_stack(
        [np.ones(X_scaled.shape[0]), X_scaled]
    )

    regression_prediction = float(
        (X_with_bias @ weights).item()
    )

    # ---------------------------------------------------------
    # Classification
    # ---------------------------------------------------------

    classification_bundle = joblib.load(
        classification_path
    )

    classification_model = classification_bundle["model"]
    classification_scaler = classification_bundle["scaler"]

    X_classification = classification_scaler.transform(X)

    classification_prediction = int(
        classification_model.predict(
            X_classification
        )[0]
    )

    classification_probability = float(
        classification_model.predict_proba(
            X_classification
        )[0][1]
    )

    # ---------------------------------------------------------
    # Clustering
    # ---------------------------------------------------------

    clustering_bundle = joblib.load(
        clustering_path
    )

    clustering_model = clustering_bundle["model"]
    clustering_scaler = clustering_bundle["scaler"]

    X_clustering = clustering_scaler.transform(X)

    cluster_label = int(
        clustering_model.predict(
            X_clustering
        )[0]
    )

    # ---------------------------------------------------------
    # Final JSON
    # ---------------------------------------------------------

    result = {
        "regression_prediction": round(
            regression_prediction,
            4,
        ),
        "classification_prediction": classification_prediction,
        "classification_probability": round(
            classification_probability,
            4,
        ),
        "cluster_label": cluster_label,
        "group_code": GROUP_CODE,
        "model_version": MODEL_VERSION,
    }

    print(json.dumps(result))


if __name__ == "__main__":
    main()