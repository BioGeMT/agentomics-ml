import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from datasets.dataset_preparation import (
    check_dataset_prepared,
    discover_test_split_names,
    prepare_dataset,
    prepare_test_dataset,
)

DATASET_NAME = "publishing_dataset"


def write_raw_split(dataset_dir: Path, split_name: str) -> Path:
    split_dir = dataset_dir / split_name
    input_dir = split_dir / "input"
    input_dir.mkdir(parents=True)
    (input_dir / "examples.txt").write_text(f"{split_name}-data\n", encoding="utf-8")
    (split_dir / "labels.csv").write_text(
        f"id,label\n{split_name}-0,negative\n{split_name}-1,positive\n",
        encoding="utf-8",
    )
    return split_dir


def write_raw_dataset(root: Path) -> tuple[Path, Path]:
    dataset_dir = root / "datasets" / DATASET_NAME
    test_dataset_dir = root / "test_datasets" / DATASET_NAME
    for split_name in ["train", "validation"]:
        write_raw_split(dataset_dir, split_name)
    write_raw_split(test_dataset_dir, "test")
    (dataset_dir / "metadata.json").write_text(
        json.dumps({"task_type": "classification"}), encoding="utf-8"
    )
    (dataset_dir / "dataset_description.md").write_text("description", encoding="utf-8")
    return dataset_dir, test_dataset_dir


class DatasetPreparationPublishingTests(unittest.TestCase):
    def setUp(self):
        self._temp_dir = TemporaryDirectory()
        self.addCleanup(self._temp_dir.cleanup)
        self.root = Path(self._temp_dir.name)
        self.dataset_dir, self.test_dataset_dir = write_raw_dataset(self.root)
        self.prepared_dir = self.root / "prepared" / DATASET_NAME
        self.prepared_test_dir = self.root / "prepared_test" / DATASET_NAME

    def prepare(self):
        return prepare_dataset(source_dir=self.dataset_dir, destination_dir=self.prepared_dir)

    def prepare_test(self, train_metadata):
        prepare_test_dataset(
            source_dir=self.test_dataset_dir,
            destination_dir=self.prepared_test_dir,
            task_type=train_metadata["task_type"],
            input_structure=train_metadata["input_structure"],
            label_to_scalar=train_metadata.get("label_to_scalar"),
        )

    def test_public_dataset_is_prepared_to_destination(self):
        self.prepare()

        labels_text = (self.prepared_dir / "train" / "labels.csv").read_text(encoding="utf-8")
        metadata = json.loads((self.prepared_dir / "metadata.json").read_text(encoding="utf-8"))

        self.assertIn("id,numeric_label", labels_text)
        self.assertTrue(metadata["prepared"])
        self.assertEqual({"negative": 0, "positive": 1}, metadata["label_to_scalar"])
        self.assertTrue(check_dataset_prepared(self.prepared_dir))

    def test_hidden_test_dataset_prepared_to_destination(self):
        train_metadata = self.prepare()
        self.prepare_test(train_metadata)

        test_labels_text = (self.prepared_test_dir / "test" / "labels.csv").read_text(encoding="utf-8")
        test_metadata = json.loads((self.prepared_test_dir / "metadata.json").read_text(encoding="utf-8"))

        self.assertIn("id,numeric_label", test_labels_text)
        self.assertTrue(test_metadata["prepared"])
        self.assertEqual({"negative": 0, "positive": 1}, test_metadata["label_to_scalar"])
        self.assertFalse((self.test_dataset_dir / "metadata.json").exists())

    def test_named_hidden_test_dataset_prepared_to_destination(self):
        write_raw_split(self.test_dataset_dir, "leftout")
        train_metadata = self.prepare()
        prepare_test_dataset(
            source_dir=self.test_dataset_dir,
            destination_dir=self.prepared_test_dir,
            task_type=train_metadata["task_type"],
            input_structure=train_metadata["input_structure"],
            label_to_scalar=train_metadata.get("label_to_scalar"),
            split_name="leftout",
        )

        leftout_labels_text = (self.prepared_test_dir / "leftout" / "labels.csv").read_text(encoding="utf-8")
        test_metadata = json.loads((self.prepared_test_dir / "metadata.json").read_text(encoding="utf-8"))

        self.assertIn("id,numeric_label", leftout_labels_text)
        self.assertTrue((self.prepared_test_dir / "leftout" / "input").is_symlink())
        self.assertEqual({"leftout_rows": 2}, test_metadata["splits"])

    def test_discovers_test_split_first_then_extra_hidden_splits(self):
        write_raw_split(self.test_dataset_dir, "leftout")

        self.assertEqual(["test", "leftout"], discover_test_split_names(self.test_dataset_dir))

    def test_public_test_split_is_rejected(self):
        write_raw_split(self.dataset_dir, "test")

        with self.assertRaises(ValueError) as raised:
            self.prepare()

        self.assertIn("must not contain test/", str(raised.exception))
        self.assertIn("test_datasets", str(raised.exception))
        self.assertFalse(check_dataset_prepared(self.prepared_dir))

    def test_symlinked_public_dataset_is_rejected_with_actionable_error(self):
        external_file = self.root / "external_reference.txt"
        external_file.write_text("external", encoding="utf-8")
        (self.dataset_dir / "train" / "input" / "linked.txt").symlink_to(external_file)

        with self.assertRaises(ValueError) as raised:
            self.prepare()

        self.assertIn("symlink", str(raised.exception))
        self.assertIn("linked.txt", str(raised.exception))
        self.assertFalse(check_dataset_prepared(self.prepared_dir))


if __name__ == "__main__":
    unittest.main()
