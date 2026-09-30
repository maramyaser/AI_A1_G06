import argparse

from src.data_pipeline import run_data_pipeline


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
        help="Group code, for example AI_A1_G06"
    )

    args = parser.parse_args()

    run_data_pipeline(
        data_path=args.data,
        output_dir=args.output,
        group_code=args.group,
    )

    print("\nStage 1 completed successfully.")


if __name__ == "__main__":
    main()