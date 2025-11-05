from pathlib import Path

import numpy as np
import pandas as pd

TARGET_COLUMN = "{target}"


def prepare(raw: Path, public: Path, private: Path) -> None:
    train_df = pd.read_csv(raw / "train.csv").copy()
    test_df = pd.read_csv(raw / "test.csv").copy()

    train_df["id"] = range(len(train_df))
    train_df = train_df[["id"] + [c for c in train_df.columns if c != "id"]]
    train_df.to_csv(public / "train.csv", index=False)

    features_df = test_df.drop(columns=[TARGET_COLUMN]).copy()
    features_df["id"] = range(len(features_df))
    features_df = features_df[["id"] + [c for c in features_df.columns if c != "id"]]
    features_df.to_csv(public / "test_features.csv", index=False)

    rng = np.random.default_rng(0)
    sample = pd.DataFrame({{"id": features_df["id"]}})
    sample[TARGET_COLUMN] = np.round(rng.random(len(features_df)), 15)
    sample.to_csv(public / "sample_submission.csv", index=False)

    answers = pd.DataFrame({{"id": features_df["id"], TARGET_COLUMN: test_df[TARGET_COLUMN].values}})
    answers.to_csv(private / "answers.csv", index=False)
