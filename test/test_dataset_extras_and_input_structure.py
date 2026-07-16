import json
import shutil
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from agentomics.datasets.dataset_preparation import prepare_dataset, prepare_test_dataset
from agentomics.datasets.data_contract import _validate_input_structure, record_input_dir_structure

DATASET_NAME = "extras_test_dataset"


def write_split(dataset_dir: Path, split_name: str, rows: int = 2, input_files: dict | None = None) -> Path:
    split_dir = dataset_dir / split_name
    input_dir = split_dir / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
    if input_files is None:
        (input_dir / "data.csv").write_text(
            "id,feature\n" + "\n".join(f"{split_name}-{i},val{i}" for i in range(rows)),
            encoding="utf-8",
        )
    else:
        for filename, content in input_files.items():
            filepath = input_dir / filename
            filepath.parent.mkdir(parents=True, exist_ok=True)
            filepath.write_text(content, encoding="utf-8")
    labels = ["id,label"]
    labels.extend(f"{split_name}-{i},{i % 2}" for i in range(rows))
    (split_dir / "labels.csv").write_text("\n".join(labels) + "\n", encoding="utf-8")
    return split_dir


class RecordInputStructureTest(unittest.TestCase):
    def test_captures_files_and_dirs(self):
        with TemporaryDirectory() as tmp:
            input_dir = Path(tmp) / "input"
            input_dir.mkdir()
            (input_dir / "data.csv").write_text("a,b\n1,2\n")
            subdir = input_dir / "images"
            subdir.mkdir()
            (subdir / "img1.png").write_text("fake")

            structure = record_input_dir_structure(input_dir)

            self.assertIn("data.csv", structure)
            self.assertIn("images/", structure)
            self.assertNotIn("images/img1.png", structure)

    def test_ignores_nested_files_inside_matching_top_level_dirs(self):
        with TemporaryDirectory() as tmp:
            train_input = Path(tmp) / "train_input"
            train_images = train_input / "images"
            train_images.mkdir(parents=True)
            (train_images / "train_1.png").write_text("fake")

            validation_input = Path(tmp) / "validation_input"
            validation_images = validation_input / "images"
            validation_images.mkdir(parents=True)
            (validation_images / "validation_1.png").write_text("fake")

            expected = record_input_dir_structure(train_input)
            _validate_input_structure(validation_input, expected)

    def test_accepts_symlinked_files_inside_matching_top_level_directory(self):
        with TemporaryDirectory() as tmp:
            source_audio = Path(tmp) / "source" / "audio"
            source_audio.mkdir(parents=True)
            (source_audio / "train.wav").write_text("train audio")
            (source_audio / "validation.wav").write_text("validation audio")

            train_input = Path(tmp) / "train_input"
            train_audio = train_input / "audio"
            train_audio.mkdir(parents=True)
            (train_audio / "train.wav").symlink_to(source_audio / "train.wav")

            validation_input = Path(tmp) / "validation_input"
            validation_audio = validation_input / "audio"
            validation_audio.mkdir(parents=True)
            (validation_audio / "validation.wav").symlink_to(source_audio / "validation.wav")

            expected = record_input_dir_structure(train_input)
            _validate_input_structure(validation_input, expected)

    def test_distinguishes_file_from_empty_dir_with_same_name(self):
        with TemporaryDirectory() as tmp:
            file_root = Path(tmp) / "file_variant"
            file_root.mkdir()
            (file_root / "data").write_text("hello")

            dir_root = Path(tmp) / "dir_variant"
            dir_root.mkdir()
            (dir_root / "data").mkdir()

            expected = record_input_dir_structure(file_root)
            with self.assertRaises(ValueError):
                _validate_input_structure(dir_root, expected)

class ValidateInputStructureTest(unittest.TestCase):
    def test_rejects_missing_file(self):
        with TemporaryDirectory() as tmp:
            input_dir = Path(tmp) / "input"
            input_dir.mkdir()
            (input_dir / "data.csv").write_text("content")
            (input_dir / "extra.csv").write_text("content")
            expected = record_input_dir_structure(input_dir)

            (input_dir / "extra.csv").unlink()
            with self.assertRaises(ValueError) as ctx:
                _validate_input_structure(input_dir, expected)
            self.assertIn("Missing", str(ctx.exception))

    def test_rejects_missing_top_level_directory(self):
        with TemporaryDirectory() as tmp:
            input_dir = Path(tmp) / "input"
            images_dir = input_dir / "images"
            images_dir.mkdir(parents=True)
            (images_dir / "sample.png").write_text("content")
            expected = record_input_dir_structure(input_dir)

            shutil.rmtree(images_dir)
            with self.assertRaises(ValueError) as ctx:
                _validate_input_structure(input_dir, expected)
            self.assertIn("Missing", str(ctx.exception))

    def test_rejects_extra_file(self):
        with TemporaryDirectory() as tmp:
            input_dir = Path(tmp) / "input"
            input_dir.mkdir()
            (input_dir / "data.csv").write_text("content")
            expected = record_input_dir_structure(input_dir)

            (input_dir / "unexpected.csv").write_text("content")
            with self.assertRaises(ValueError) as ctx:
                _validate_input_structure(input_dir, expected)
            self.assertIn("Extra", str(ctx.exception))


class PrepareDatasetSplitEntryTest(unittest.TestCase):
    def test_works_without_extra_split_entries(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = root / "raw" / DATASET_NAME
            dest = root / "prepared" / DATASET_NAME
            raw.mkdir(parents=True)
            write_split(raw, "train")

            prepare_dataset(
                source_dir=raw,
                destination_dir=dest,
                task_type="classification",
            )

            self.assertTrue((dest / "train" / "input").exists())

    def test_rejects_split_level_extras(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = root / "raw" / DATASET_NAME
            dest = root / "prepared" / DATASET_NAME
            raw.mkdir(parents=True)
            write_split(raw, "train")
            split_extras = raw / "train" / "extras"
            split_extras.mkdir()
            (split_extras / "augmented.csv").write_text("augmented data")

            with self.assertRaises(ValueError) as ctx:
                prepare_dataset(
                    source_dir=raw,
                    destination_dir=dest,
                    task_type="classification",
                )
            self.assertIn("unsupported", str(ctx.exception))

    def test_rejects_train_supplementary(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = root / "raw" / DATASET_NAME
            dest = root / "prepared" / DATASET_NAME
            raw.mkdir(parents=True)
            write_split(raw, "train")
            train_supp = raw / "train" / "supplementary"
            train_supp.mkdir()
            (train_supp / "augmented.csv").write_text("augmented data")

            with self.assertRaises(ValueError) as ctx:
                prepare_dataset(
                    source_dir=raw,
                    destination_dir=dest,
                    task_type="classification",
                )
            self.assertIn("train", str(ctx.exception))

    def test_records_input_structure_in_metadata(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = root / "raw" / DATASET_NAME
            dest = root / "prepared" / DATASET_NAME
            raw.mkdir(parents=True)
            write_split(raw, "train")

            prepare_dataset(
                source_dir=raw,
                destination_dir=dest,
                task_type="classification",
            )

            metadata = json.loads((dest / "metadata.json").read_text())
            self.assertIn("input_structure", metadata)
            self.assertIn("data.csv", metadata["input_structure"])

    def test_rejects_mismatched_validation_input_structure(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = root / "raw" / DATASET_NAME
            dest = root / "prepared" / DATASET_NAME
            raw.mkdir(parents=True)
            write_split(raw, "train", input_files={"data.csv": "id,a\n1,x\n2,y\n"})
            write_split(raw, "validation", input_files={"different.csv": "id,b\n1,x\n2,y\n"})

            with self.assertRaises(ValueError) as ctx:
                prepare_dataset(
                    source_dir=raw,
                    destination_dir=dest,
                    task_type="classification",
                )
            self.assertIn("validation/input", str(ctx.exception))

    def test_accepts_different_nested_files_with_matching_top_level_directory(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = root / "raw" / DATASET_NAME
            dest = root / "prepared" / DATASET_NAME
            raw_test = root / "raw_tests" / DATASET_NAME
            raw.mkdir(parents=True)
            raw_test.mkdir(parents=True)
            write_split(raw, "train", input_files={"images/train-0.png": "fake train image"})
            write_split(raw, "validation", input_files={"images/validation-0.png": "fake validation image"})
            write_split(raw_test, "test", input_files={"images/test-0.png": "fake test image"})

            train_metadata = prepare_dataset(
                source_dir=raw,
                destination_dir=dest,
                task_type="classification",
            )
            dest_test = root / "prepared_test" / DATASET_NAME
            prepare_test_dataset(
                source_dir=raw_test,
                destination_dir=dest_test,
                task_type=train_metadata["task_type"],
                input_structure=train_metadata["input_structure"],
                label_to_scalar=train_metadata.get("label_to_scalar"),
            )

            metadata = json.loads((dest / "metadata.json").read_text())
            self.assertEqual(["images/"], metadata["input_structure"])

class PrepareDatasetSupplementaryTest(unittest.TestCase):
    def test_preserves_supplementary_folder(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = root / "raw" / DATASET_NAME
            dest = root / "prepared" / DATASET_NAME
            raw.mkdir(parents=True)
            write_split(raw, "train")
            supp = raw / "supplementary"
            supp.mkdir()
            (supp / "paper.pdf").write_text("fake pdf")

            prepare_dataset(
                source_dir=raw,
                destination_dir=dest,
                task_type="classification",
            )

            self.assertTrue((dest / "supplementary").exists())
            self.assertTrue((dest / "supplementary" / "paper.pdf").is_file())

    def test_works_without_supplementary(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = root / "raw" / DATASET_NAME
            dest = root / "prepared" / DATASET_NAME
            raw.mkdir(parents=True)
            write_split(raw, "train")

            prepare_dataset(
                source_dir=raw,
                destination_dir=dest,
                task_type="classification",
            )

            self.assertFalse((dest / "supplementary").exists())

if __name__ == "__main__":
    unittest.main()
