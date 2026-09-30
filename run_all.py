import argparse

from src.data_pipeline import run_data_pipeline
from src.regression import run_regression
from src.classification import run_classification


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data",
        required=True,
        help="Path to the CSV dataset"
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Directory for generated artifacts"
    )

    parser.add_argument(
        "--group",
        required=True,
        help="Group code"
    )

    args = parser.parse_args()

    # =========================================================
    # STAGE 1 — DATA PIPELINE
    # =========================================================

    (
        df,
        X,
        y_regression,
        y_classification,
    ) = run_data_pipeline(
        data_path=args.data,
        output_dir=args.output,
        group_code=args.group,
    )

    # =========================================================
    # STAGE 2 — REGRESSION
    # =========================================================

    run_regression(
        X=X,
        y=y_regression,
        output_dir=args.output,
        random_seed=3513,
    )

    # =========================================================
    # STAGE 3 — CLASSIFICATION
    # =========================================================

    run_classification(
        X=X,
        y=y_classification,
        output_dir=args.output,
        random_seed=3513,
    )

    print(
        "\nStage 1, Stage 2, and Stage 3 "
        "completed successfully."
    )


if __name__ == "__main__":
    main()