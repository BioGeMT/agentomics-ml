import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from datasets.csv_converter import convert_csv_dataset


class CsvConverterTest(unittest.TestCase):
    def test_writes_hidden_test_split_to_test_output_dir(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            output_dir = root / "datasets" / "toy"
            test_output_dir = root / "test_datasets" / "toy"

            convert_csv_dataset(
                output_dir=output_dir,
                test_output_dir=test_output_dir,
                label_column="target",
                splits={
                    "train": pd.DataFrame({"id": ["train-0"], "feature": [1], "target": ["a"]}),
                    "test": pd.DataFrame({"id": ["test-0"], "feature": [2], "target": ["a"]}),
                },
                task_type="classification",
            )

            self.assertTrue((output_dir / "train" / "input" / "data.csv").is_file())
            self.assertFalse((output_dir / "test").exists())
            self.assertTrue((test_output_dir / "test" / "input" / "data.csv").is_file())
            self.assertTrue((output_dir / "metadata.json").is_file())

    def test_requires_test_output_dir_for_hidden_test_split(self):
        with TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError) as raised:
                convert_csv_dataset(
                    output_dir=Path(tmp) / "datasets" / "toy",
                    label_column="target",
                    splits={
                        "train": pd.DataFrame({"id": ["train-0"], "feature": [1], "target": ["a"]}),
                        "test": pd.DataFrame({"id": ["test-0"], "feature": [2], "target": ["a"]}),
                    },
                )

            self.assertIn("test_output_dir", str(raised.exception))


if __name__ == "__main__":
    unittest.main()
