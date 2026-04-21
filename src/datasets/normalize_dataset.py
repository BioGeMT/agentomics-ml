import argparse
import csv
import sys
from pathlib import Path


def normalize_input_dataset(input_path: Path, output_path: Path):
    """
    Normalize a csv dataset before entering the train/inference pipeline. Ensures it contains the id column.
    If 'id' column is already provided, keeps original csv.
    Uses only stdlib csv to avoid requiring pandas in the calling environment.
    """
    with open(input_path, newline='') as fin:
        reader = csv.reader(fin)
        header = next(reader, None)
        if header is None:
            raise ValueError(f"Input CSV is empty: {input_path}")
        if 'id' in header:
            return False  # original had id, no normalization needed
        rows = list(reader)

    with open(output_path, 'w', newline='') as fout:
        writer = csv.writer(fout)
        writer.writerow(['id'] + header)
        for idx, row in enumerate(rows):
            writer.writerow([idx] + row)

    print("[Warning] Input CSV has no 'id' column. Sequential IDs (0..N-1) added in a temporary file, used for running inference on. If you need specific IDs, include an 'id' column in the input csv", file=sys.stderr)
    return True

def main():
    parser = argparse.ArgumentParser(description="Normalize a CSV dataset for use in the pipeline.")
    parser.add_argument("--input", required=True, help="Path to the input CSV file")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = input_path.parent / f"normalized_{input_path.name}"

    normalized = normalize_input_dataset(input_path, output_path)
    if normalized:
        print(output_path.name)

if __name__ == "__main__":
    main()
