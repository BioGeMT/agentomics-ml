from pathlib import Path

import numpy as np
import pandas as pd

TARGET_COLUMN = "{target}"


def prepare(raw: Path, public: Path, private: Path) -> None:
    # Note: private directory unused - we don't generate answers.csv for grading
    train_df = pd.read_csv(raw / "train.csv").copy()
    test_df = pd.read_csv(raw / "test.csv").copy()
    if TARGET_COLUMN not in train_df.columns or TARGET_COLUMN not in test_df.columns:
        raise ValueError(f"Missing target column '{{TARGET_COLUMN}}'")
    if "id" not in train_df.columns:
        train_df.insert(0, "id", range(len(train_df)))
    train_df.to_csv(public / "train.csv", index=False)
    features_df = test_df.drop(columns=[TARGET_COLUMN]).copy()
    if "id" not in features_df.columns:
        features_df.insert(0, "id", range(len(features_df)))
    features_df.to_csv(public / "test_features.csv", index=False)
    # Create sample submission with random floating-point placeholders (15 decimals)
    sample = pd.DataFrame({{"id": features_df["id"]}})
    rng = np.random.default_rng(0)
    sample[TARGET_COLUMN] = np.round(rng.random(len(features_df)), 15)
    sample.to_csv(public / "sample_submission.csv", index=False)
    # Create dummy answers.csv (required by biomlbench but not used - we do our own eval)
    answers = pd.DataFrame({{"id": features_df["id"], TARGET_COLUMN: test_df[TARGET_COLUMN].values}})
    answers.to_csv(private / "answers.csv", index=False)
