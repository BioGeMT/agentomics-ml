import argparse
from pathlib import Path

from datasets.csv_converter import convert_inference_csv


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Convert a given input CSV into a contract split folder (input/ + optional labels.csv), ready for inference."
    )
    ap.add_argument("--input", type=Path, required=True, help="Path to the inference CSV")
    ap.add_argument("--output-split", type=Path, required=True, help="Destination split folder to create")
    ap.add_argument("--label-col", default=None, help="Label column in the CSV; when set, labels.csv is written so metrics can be computed")
    ap.add_argument("--id-col", default=None, help="Existing id column in the CSV; ids are generated when omitted")
    args = ap.parse_args()

    convert_inference_csv(
        csv_path=args.input,
        output_split_dir=args.output_split,
        label_column=args.label_col,
        id_column=args.id_col,
    )

if __name__ == "__main__":
    main()
