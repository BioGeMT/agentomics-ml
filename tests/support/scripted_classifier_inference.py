import argparse
import csv
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--artifacts-dir", type=Path, required=True)
    args = parser.parse_args()
    model = json.loads((args.artifacts_dir / "model.json").read_text())
    with (args.input / "data.csv").open() as source, args.output.open("w") as output:
        writer = csv.DictWriter(output, fieldnames=["id", "prediction", "probability_0", "probability_1"])
        writer.writeheader()
        for row in csv.DictReader(source):
            prediction = model[row["sequence"][0]]
            writer.writerow({
                "id": row["id"], "prediction": prediction,
                "probability_0": int(prediction == 0), "probability_1": int(prediction == 1),
            })
    print("Predictions generated using trained artifacts")


if __name__ == "__main__":
    main()
