import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from datasets.data_contract import NUMERIC_LABEL_COLUMN_NAME, validate_and_read_labels
from datasets.dataset_preparation import (
    check_dataset_prepared,
    prepare_dataset,
)
from datasets.datasets_interactive import _get_single_dataset_info
from utils.config import Config

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
        },
        "prepared": True,
    }
    (dataset_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    (dataset_dir / "dataset_description.md").write_text("temporary dataset", encoding="utf-8")


def create_prepared_folder_dataset(datasets_dir: Path) -> Path:
    dataset_dir = datasets_dir / DATASET_NAME
    dataset_dir.mkdir(parents=True)
    write_dataset_metadata(dataset_dir)
    for split_name in ["train", "validation"]:
        write_split(dataset_dir, split_name, label_column="numeric_label")
    return dataset_dir


class DatasetFolderSplitContractTest(unittest.TestCase):
    def test_check_dataset_prepared_accepts_folder_only_split_contract(self):
        with TemporaryDirectory() as tmp:
            dataset_dir = create_prepared_folder_dataset(Path(tmp) / "datasets")

            self.assertTrue(check_dataset_prepared(dataset_dir))

    def test_get_single_dataset_info_counts_folder_labels(self):
        with TemporaryDirectory() as tmp:
            dataset_dir = create_prepared_folder_dataset(Path(tmp) / "datasets")
            metadata = json.loads((dataset_dir / "metadata.json").read_text(encoding="utf-8"))
            metadata["splits"] = {"train_rows": 999, "validation_rows": 999}
            (dataset_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")

            info = _get_single_dataset_info(dataset_dir)

            self.assertEqual("Prepared", info["status"])
            self.assertEqual(2, info["train_rows"])
            self.assertEqual(2, info["validation_rows"])

    def test_prepare_dataset_rejects_non_numeric_regression_labels(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw_dataset_dir = root / "datasets" / DATASET_NAME
            dest = root / "prepared" / DATASET_NAME
            raw_dataset_dir.mkdir(parents=True)
            write_split(raw_dataset_dir, "train")
            (raw_dataset_dir / "train" / "labels.csv").write_text(
                "id,label\ntrain-0,not-a-number\n",
                encoding="utf-8",
            )

            with self.assertRaises(ValueError):
                prepare_dataset(
                    source_dir=raw_dataset_dir,
                    destination_dir=dest,
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
            raw_dataset_dir = root / "datasets" / DATASET_NAME
            dest = root / "prepared" / DATASET_NAME
            raw_dataset_dir.mkdir(parents=True)
            write_split(raw_dataset_dir, "train")

            prepare_dataset(
                source_dir=raw_dataset_dir,
                destination_dir=dest,
                task_type="classification",
            )

            metadata = json.loads((dest / "metadata.json").read_text())
            self.assertEqual({"0": 0, "1": 1}, metadata["label_to_scalar"])

    def test_prepare_dataset_preserves_numeric_looking_ids_as_strings(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw_dataset_dir = root / "datasets" / DATASET_NAME
            dest = root / "prepared" / DATASET_NAME
            raw_dataset_dir.mkdir(parents=True)
            write_split(raw_dataset_dir, "train")
            (raw_dataset_dir / "train" / "labels.csv").write_text(
                "id,label\n001,0\n002,1\n",
                encoding="utf-8",
            )

            prepare_dataset(
                source_dir=raw_dataset_dir,
                destination_dir=dest,
                task_type="classification",
            )

            prepared_labels = (dest / "train" / "labels.csv").read_text(encoding="utf-8").splitlines()
            self.assertEqual(["id,numeric_label", "001,0", "002,1"], prepared_labels)

    def test_prepare_dataset_leaves_absent_validation_split_absent(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw_dataset_dir = root / "datasets" / DATASET_NAME
            dest = root / "prepared" / DATASET_NAME
            raw_dataset_dir.mkdir(parents=True)
            write_split(raw_dataset_dir, "train")

            prepare_dataset(
                source_dir=raw_dataset_dir,
                destination_dir=dest,
                task_type="classification",
            )

            self.assertTrue((dest / "train").is_dir())
            self.assertFalse((dest / "validation").exists())

    def test_dataset_dir_does_not_point_at_shared(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = Config(
                agent_id="agent-under-test",
                model_name="test-model",
                iteration_plan_model_name="test-model",
                dataset=DATASET_NAME,
                tags=[],
                val_metric="ACC",
                workspace_dir=str(root / "workspace"),
                datasets_dir=str(root / "datasets"),
                user_prompt="test",
                task_type="classification",
                input_structure=["data.csv"],
            )

            self.assertFalse(str(config.dataset_dir).endswith("run/shared/datasets/" + DATASET_NAME))


if __name__ == "__main__":
    unittest.main()
