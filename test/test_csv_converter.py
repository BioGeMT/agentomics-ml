import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from agentomics.datasets.csv_converter import convert_csv_dataset
from agentomics.datasets.dataset_preparation import prepare_dataset


class CsvConverterTest(unittest.TestCase):
    def test_writes_test_split_into_dataset_folder(self):
        with TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "datasets" / "toy"

            convert_csv_dataset(
                output_dir=output_dir,
                label_column="target",
                splits={
                    "train": pd.DataFrame({"id": ["train-0"], "feature": [1], "target": ["a"]}),
                    "test": pd.DataFrame({"id": ["test-0"], "feature": [2], "target": ["a"]}),
                },
                task_type="classification",
            )

            self.assertTrue((output_dir / "train" / "input" / "data.csv").is_file())
            self.assertTrue((output_dir / "test" / "input" / "data.csv").is_file())
            self.assertTrue((output_dir / "test" / "labels.csv").is_file())
            self.assertTrue((output_dir / "metadata.json").is_file())

    def test_omits_test_folder_when_no_test_split(self):
        with TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "datasets" / "toy"

            convert_csv_dataset(
                output_dir=output_dir,
                label_column="target",
                splits={
                    "train": pd.DataFrame({"id": ["train-0"], "feature": [1], "target": ["a"]}),
                },
            )

            self.assertTrue((output_dir / "train" / "input" / "data.csv").is_file())
            self.assertFalse((output_dir / "test").exists())

    def test_train_split_is_required(self):
        with TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError) as raised:
                convert_csv_dataset(
                    output_dir=Path(tmp) / "datasets" / "toy",
                    label_column="target",
                    splits={
                        "test": pd.DataFrame({"id": ["test-0"], "feature": [2], "target": ["a"]}),
                    },
                )

            self.assertIn("train", str(raised.exception))

    def test_preparation_can_replay_resolved_csv_metadata(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_dir = root / "datasets" / "toy"
            source_dir.mkdir(parents=True)
            (source_dir / "metadata.json").write_text(
                json.dumps({"id_column": "sample_id"}),
                encoding="utf-8",
            )
            pd.DataFrame(
                {
                    "sample_id": ["row-1", "row-2"],
                    "feature": [1, 2],
                    "outcome": ["no", "yes"],
                }
            ).to_csv(source_dir / "train.csv", index=False)

            first_metadata = prepare_dataset(
                source_dir=source_dir,
                destination_dir=root / "first_prepared",
                task_type="classification",
                label_to_scalar={"no": 0, "yes": 1},
                label_column="outcome",
                interactive=False,
            )
            second_metadata = prepare_dataset(
                source_dir=source_dir,
                destination_dir=root / "second_prepared",
                task_type=first_metadata["task_type"],
                label_to_scalar=first_metadata["label_to_scalar"],
                label_column=first_metadata["label_column"],
                interactive=False,
            )

            self.assertEqual(second_metadata["label_column"], "outcome")
            self.assertEqual(second_metadata["id_column"], "sample_id")
            self.assertEqual(
                second_metadata["label_to_scalar"],
                {"no": 0, "yes": 1},
            )
            replayed_labels = pd.read_csv(
                root / "second_prepared" / "train" / "labels.csv",
                dtype={"id": str},
            )
            self.assertEqual(replayed_labels["id"].tolist(), ["row-1", "row-2"])


if __name__ == "__main__":
    unittest.main()
