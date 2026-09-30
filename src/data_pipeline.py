import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


# Columns required by the assignment
REQUIRED_COLUMNS = [
    "record_id",
    "plot_area_ha",
    "rainfall_mm",
    "soil_ph",
    "seed_kg",
    "distance_km",
    "arrival_hour",
    "actual_yield_kg",
    "dispatch_attention",
]

# Columns used as model input features
FEATURE_COLUMNS = [
    "plot_area_ha",
    "rainfall_mm",
    "soil_ph",
    "seed_kg",
    "distance_km",
    "arrival_hour",
]

REGRESSION_TARGET = "actual_yield_kg"
CLASSIFICATION_TARGET = "dispatch_attention"


def calculate_sha256(file_path):
    """Calculate SHA-256 fingerprint of the CSV file."""
    sha256 = hashlib.sha256()

    with open(file_path, "rb") as file:
        for chunk in iter(lambda: file.read(8192), b""):
            sha256.update(chunk)

    return sha256.hexdigest()


def validate_columns(df):
    """Check that all required columns exist."""

    missing_columns = [
        column for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )


def validate_data_types(df):
    """Check that required numeric columns contain numeric values."""

    numeric_columns = [
        "plot_area_ha",
        "rainfall_mm",
        "soil_ph",
        "seed_kg",
        "distance_km",
        "arrival_hour",
        "actual_yield_kg",
        "dispatch_attention",
    ]

    for column in numeric_columns:
        converted = pd.to_numeric(df[column], errors="coerce")

        invalid_count = converted.isna().sum() - df[column].isna().sum()

        if invalid_count > 0:
            raise ValueError(
                f"Column '{column}' contains {invalid_count} "
                "non-numeric value(s)."
            )


def validate_dispatch_attention(df):
    """Check that dispatch_attention contains only 0 and 1."""

    values = set(df["dispatch_attention"].dropna().unique())

    if not values.issubset({0, 1}):
        raise ValueError(
            "dispatch_attention must contain only 0 and 1."
        )


def build_data_report(df, file_path, group_code):
    """Create the Stage 1 data report."""

    missing_values = df.isna().sum().to_dict()

    duplicate_record_ids = int(
        df["record_id"].duplicated().sum()
    )

    duplicate_rows = int(df.duplicated().sum())

    # Descriptive statistics for numeric columns
    statistics = (
        df[FEATURE_COLUMNS + [REGRESSION_TARGET]]
        .describe()
        .round(4)
        .to_dict()
    )

    report = {
        "group_code": group_code,
        "row_count": int(len(df)),
        "feature_count": len(FEATURE_COLUMNS),
        "columns": REQUIRED_COLUMNS,
        "missing_values": missing_values,
        "duplicate_record_ids": duplicate_record_ids,
        "duplicate_rows": duplicate_rows,
        "sha256": calculate_sha256(file_path),
        "descriptive_statistics": statistics,
    }

    return report


def run_data_pipeline(data_path, output_dir, group_code):
    """
    Run Stage 1 of the assignment.

    Returns:
        df: validated pandas DataFrame
        X: NumPy feature matrix
        y_regression: regression target
        y_classification: classification target
    """

    data_path = Path(data_path)
    output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------
    # 1. Load CSV
    # ---------------------------------------------------------

    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {data_path}"
        )

    df = pd.read_csv(data_path)

    print(f"Loaded dataset: {data_path}")
    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")

    # ---------------------------------------------------------
    # 2. Validate columns
    # ---------------------------------------------------------

    validate_columns(df)

    # ---------------------------------------------------------
    # 3. Validate data types
    # ---------------------------------------------------------

    validate_data_types(df)

    # ---------------------------------------------------------
    # 4. Validate classification target
    # ---------------------------------------------------------

    validate_dispatch_attention(df)

    # ---------------------------------------------------------
    # 5. Check missing values
    # ---------------------------------------------------------

    total_missing = int(df.isna().sum().sum())

    print(f"Missing values: {total_missing}")

    # ---------------------------------------------------------
    # 6. Check duplicate IDs
    # ---------------------------------------------------------

    duplicate_ids = int(
        df["record_id"].duplicated().sum()
    )

    print(f"Duplicate record IDs: {duplicate_ids}")

    # ---------------------------------------------------------
    # 7. Create NumPy feature matrix
    # ---------------------------------------------------------

    X = df[FEATURE_COLUMNS].to_numpy(dtype=float)

    # Regression target
    y_regression = df[REGRESSION_TARGET].to_numpy(dtype=float)

    # Classification target
    y_classification = df[CLASSIFICATION_TARGET].to_numpy(dtype=int)

    print(f"Feature matrix shape: {X.shape}")

    # ---------------------------------------------------------
    # 8. Generate data report
    # ---------------------------------------------------------

    report = build_data_report(
        df=df,
        file_path=data_path,
        group_code=group_code,
    )

    report_path = output_dir / "data_report.json"

    with open(report_path, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=4)

    print(f"Data report saved to: {report_path}")
    print(f"Group code: {group_code}")
    print(f"SHA-256: {calculate_sha256(data_path)}")

    # ---------------------------------------------------------
    # 9. Return data for later stages
    # ---------------------------------------------------------

    return (
        df,
        X,
        y_regression,
        y_classification,
    )