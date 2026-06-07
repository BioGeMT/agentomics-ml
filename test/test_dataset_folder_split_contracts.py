import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from datasets.dataset_preparation import (
    check_dataset_prepared,
    prepare_dataset,
    setup_nonsensitive_dataset_files_for_agent,
)
from datasets.datasets_interactive import _get_single_prepared_dataset_info
from datasets.data_contract import NUMERIC_LABEL_COLUMN_NAME, validate_and_read_labels
DATASET_NAME = "folder_contract_dataset"


def write_split(dataset_dir: Path, split_name: str, rows: int = 2, label_column: str = "label") -> Path:
    split_dir = dataset_dir / split_name
    input_dir = split_dir / "input"
    input_dir.mkdir(parents=True)
    (input_dir / "examples.txt").write_text(
        "\n".join(f"{split_name}-example-{index}" for index in range(rows)),
        encoding="utf-8",
    )
    labels = [f"id,{label_column}"]
    labels.extend(f"{split_name}-{index},{index % 2}" for index in range(rows))
    (split_dir / "labels.csv").write_text("\n".join(labels) + "\n", encoding="utf-8")
    return split_dir


def write_dataset_metadata(dataset_dir: Path) -> None:
    metadata = {
        "task_type": "classification",
        "numeric_label_col": "numeric_label",
        "splits": {
            "train_rows": 2,
            "validation_rows": 2,
            "test_rows": 2,
        },
    }
    (dataset_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    (dataset_dir / "dataset_description.md").write_text("temporary dataset", encoding="utf-8")


def create_prepared_folder_dataset(prepared_datasets_dir: Path) -> Path:
    dataset_dir = prepared_datasets_dir / DATASET_NAME
    dataset_dir.mkdir(parents=True)
    write_dataset_metadata(dataset_dir)
    for split_name in ["train", "validation", "test"]:
        write_split(dataset_dir, split_name, label_column="numeric_label")
    return dataset_dir


def create_config(tmpdir: Path) -> SimpleNamespace:
    agent_id = "agent-under-test"
    return SimpleNamespace(
        agent_id=agent_id,
        runs_dir=tmpdir / "runs",
        split_allowed_iterations=1,
        split_time_deadline=None,
        can_iteration_split_now_cached=lambda iteration: True,
    )


class DatasetFolderSplitContractTest(unittest.TestCase):
    def test_check_dataset_prepared_accepts_folder_only_split_contract(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            prepared_datasets_dir = root / "prepared"
            dataset_dir = create_prepared_folder_dataset(prepared_datasets_dir)

            self.assertTrue(check_dataset_prepared(dataset_dir, prepared_datasets_dir))

    def test_get_single_prepared_dataset_info_counts_folder_labels(self):
        with TemporaryDirectory() as tmp:
            dataset_dir = create_prepared_folder_dataset(Path(tmp) / "prepared")

            info = _get_single_prepared_dataset_info(dataset_dir)

            self.assertEqual("Prepared", info["status"])
            self.assertEqual(2, info["train_rows"])
            self.assertEqual(2, info["validation_rows"])
            self.assertEqual(2, info["test_rows"])

    def test_prepare_dataset_rejects_non_numeric_regression_labels(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw_dataset_dir = root / "raw" / DATASET_NAME
            raw_dataset_dir.mkdir(parents=True)
            write_split(raw_dataset_dir, "train")
            (raw_dataset_dir / "train" / "labels.csv").write_text(
                "id,label\ntrain-0,not-a-number\n",
                encoding="utf-8",
            )

            with self.assertRaises(ValueError):
                prepare_dataset(
                    dataset_dir=raw_dataset_dir,
                    output_dir=root / "prepared",
                    test_sets_output_dir=root / "prepared_tests",
                    task_type="regression",
                )

    def test_labels_csv_rejects_extra_columns(self):
        with TemporaryDirectory() as tmp:
            labels_path = Path(tmp) / "labels.csv"
            labels_path.write_text(
                "id,numeric_label,extra\nsample-1,0,unused\n",
                encoding="utf-8",
            )

            with self.assertRaises(ValueError):
                validate_and_read_labels(labels_path, NUMERIC_LABEL_COLUMN_NAME, require_numeric_values=True)

    def test_prepare_dataset_derives_class_labels_from_folder_labels(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw_dataset_dir = root / "raw" / DATASET_NAME
            raw_dataset_dir.mkdir(parents=True)
            write_split(raw_dataset_dir, "train")

            prepare_dataset(
                dataset_dir=raw_dataset_dir,
                output_dir=root / "prepared",
                test_sets_output_dir=root / "prepared_tests",
                task_type="classification",
            )

            metadata = json.loads((root / "prepared" / DATASET_NAME / "metadata.json").read_text())
            self.assertEqual({"0": 0, "1": 1}, metadata["label_to_scalar"])

    def test_prepare_dataset_preserves_numeric_looking_ids_as_strings(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw_dataset_dir = root / "raw" / DATASET_NAME
            raw_dataset_dir.mkdir(parents=True)
            write_split(raw_dataset_dir, "train")
            (raw_dataset_dir / "train" / "labels.csv").write_text(
                "id,label\n001,0\n002,1\n",
                encoding="utf-8",
            )

            prepare_dataset(
                dataset_dir=raw_dataset_dir,
                output_dir=root / "prepared",
                test_sets_output_dir=root / "prepared_tests",
                task_type="classification",
            )

            prepared_labels = (
                root / "prepared" / DATASET_NAME / "train" / "labels.csv"
            ).read_text(encoding="utf-8").splitlines()
            self.assertEqual(["id,numeric_label", "001,0", "002,1"], prepared_labels)

    def test_prepare_dataset_rejects_unknown_test_labels(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw_dataset_dir = root / "raw" / DATASET_NAME
            raw_dataset_dir.mkdir(parents=True)
            write_split(raw_dataset_dir, "train")
            write_split(raw_dataset_dir, "test")
            (raw_dataset_dir / "test" / "labels.csv").write_text(
                "id,label\ntest-0,7\n",
                encoding="utf-8",
            )

            with self.assertRaises(ValueError) as ctx:
                prepare_dataset(
                    dataset_dir=raw_dataset_dir,
                    output_dir=root / "prepared",
                    test_sets_output_dir=root / "prepared_tests",
                    task_type="classification",
                )
            self.assertIn("not present in train", str(ctx.exception))

    def test_prepare_dataset_removes_stale_prepared_splits(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw_dataset_dir = root / "raw" / DATASET_NAME
            raw_dataset_dir.mkdir(parents=True)
            write_split(raw_dataset_dir, "train")

            stale_prepared_dir = root / "prepared" / DATASET_NAME
            stale_prepared_dir.mkdir(parents=True)
            write_split(stale_prepared_dir, "validation")

            stale_test_dir = root / "prepared_tests" / DATASET_NAME
            stale_test_dir.mkdir(parents=True)
            write_split(stale_test_dir, "test")

            prepare_dataset(
                dataset_dir=raw_dataset_dir,
                output_dir=root / "prepared",
                test_sets_output_dir=root / "prepared_tests",
                task_type="classification",
            )

            self.assertTrue((root / "prepared" / DATASET_NAME / "train").is_dir())
            self.assertFalse((root / "prepared" / DATASET_NAME / "validation").exists())
            self.assertFalse((root / "prepared_tests" / DATASET_NAME).exists())

    def test_setup_nonsensitive_dataset_files_copies_public_split_folders_only(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            prepared_datasets_dir = root / "prepared"
            agent_datasets_dir = root / "agent_datasets"
            create_prepared_folder_dataset(prepared_datasets_dir)

            setup_nonsensitive_dataset_files_for_agent(
                prepared_datasets_dir=prepared_datasets_dir,
                agent_datasets_dir=agent_datasets_dir,
                dataset_name=DATASET_NAME,
            )

            agent_dataset_dir = agent_datasets_dir / DATASET_NAME
            self.assertTrue((agent_dataset_dir / "train" / "input").is_dir())
            self.assertTrue((agent_dataset_dir / "train" / "labels.csv").is_file())
            self.assertTrue((agent_dataset_dir / "validation" / "input").is_dir())
            self.assertTrue((agent_dataset_dir / "validation" / "labels.csv").is_file())
            self.assertFalse((agent_dataset_dir / "test").exists())
            self.assertFalse((agent_dataset_dir / "train.csv").exists())
            self.assertFalse((agent_dataset_dir / "validation.csv").exists())

if __name__ == "__main__":
    unittest.main()
