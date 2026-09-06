import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-data", type=Path, required=True)
    parser.add_argument("--validation-data", type=Path, required=True)
    parser.add_argument("--artifacts-dir", type=Path, required=True)
    args = parser.parse_args()
    with (args.train_data / "labels.csv").open() as stream:
        labels = {row["id"]: int(row["numeric_label"]) for row in csv.DictReader(stream)}
    counts = defaultdict(Counter)
    with (args.train_data / "input" / "data.csv").open() as stream:
        for row in csv.DictReader(stream):
            counts[row["sequence"][0]][labels[row["id"]]] += 1
    model = {base: counter.most_common(1)[0][0] for base, counter in counts.items()}
    args.artifacts_dir.mkdir(parents=True, exist_ok=True)
    (args.artifacts_dir / "model.json").write_text(json.dumps(model))
    print("Training completed")


if __name__ == "__main__":
    main()
