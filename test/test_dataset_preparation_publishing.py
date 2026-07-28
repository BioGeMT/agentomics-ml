import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from agentomics.datasets.dataset_preparation import (
    check_dataset,
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

    def test_colocated_test_split_is_allowed_and_ignored_by_training_prep(self):
        write_raw_split(self.dataset_dir, "test")
        write_raw_split(self.dataset_dir, "test_leftout")

        self.prepare()

        # test-prefixed folders are permitted inside the dataset folder but are
        # not part of the prepared training output.
        self.assertTrue(check_dataset_prepared(self.prepared_dir))
        self.assertFalse((self.prepared_dir / "test").exists())
        self.assertFalse((self.prepared_dir / "test_leftout").exists())

    def test_discovers_all_test_prefixed_split_directories_in_stable_order(self):
        for split_name in ["test_z", "testing", "test", "test_a"]:
            write_raw_split(self.dataset_dir, split_name)

        self.assertEqual(
            ["test", "test_a", "test_z", "testing"],
            discover_test_split_names(self.dataset_dir),
        )

    def test_named_test_split_is_prepared_using_training_metadata(self):
        write_raw_split(self.dataset_dir, "test_leftout")
        train_metadata = self.prepare()

        test_metadata = prepare_test_dataset(
            source_dir=self.dataset_dir,
            destination_dir=self.prepared_test_dir,
            task_type=train_metadata["task_type"],
            input_structure=train_metadata["input_structure"],
            label_to_scalar=train_metadata.get("label_to_scalar"),
            split_name="test_leftout",
        )

        labels_path = self.prepared_test_dir / "test_leftout" / "labels.csv"
        self.assertTrue(labels_path.is_file())
        self.assertEqual(2, test_metadata["splits"]["test_leftout_rows"])

    def test_dataset_check_validates_and_reports_every_test_split(self):
        write_raw_split(self.dataset_dir, "test")
        write_raw_split(self.dataset_dir, "test_leftout")

        summary = check_dataset(self.dataset_dir)

        self.assertEqual(
            {"test": 2, "test_leftout": 2},
            summary["test_splits"],
        )

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
