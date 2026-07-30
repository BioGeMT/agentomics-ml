from __future__ import annotations

import argparse
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from agentomics.cli import run as run_cli
from agentomics.datasets.data_contract import (
    DATASET_METADATA_FILE_NAME,
    PREPARED_DATASETS_DIR_NAME,
)
from agentomics.runtime.filesystem import remove_path
from agentomics.runtime.read_write_utils import (
    load_dataset_metadata,
    save_dataset_metadata,
)
from agentomics.utils.config import Config


class TestRemoveTransientDatasetState(unittest.TestCase):
    def test_removes_transient_directories_but_preserves_splits(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            workspace = Path(temporary_directory)
            persistent_split = workspace / "run" / "shared" / "splits" / "split_0"
            persistent_split.mkdir(parents=True)
            (persistent_split / "labels.csv").write_text(
                "id,numeric_label\nrow-1,0\n",
                encoding="utf-8",
            )
            transient_file = (
                workspace
                / PREPARED_DATASETS_DIR_NAME
                / "_csv_converted_source"
                / "toy"
                / "data.csv"
            )
            transient_file.parent.mkdir(parents=True)
            transient_file.write_text("data", encoding="utf-8")

            remove_path(workspace / PREPARED_DATASETS_DIR_NAME)

            self.assertFalse(
                (workspace / PREPARED_DATASETS_DIR_NAME).exists()
            )
            self.assertTrue((persistent_split / "labels.csv").is_file())


class TestDatasetMetadataPersistence(unittest.TestCase):
    def test_metadata_is_stored_in_run_shared_independently_of_prepared_dataset(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            workspace = Path(temporary_directory)
            config = Mock(
                shared_dir=(
                    workspace / Config.RUN_DIRNAME / Config.SHARED_DIRNAME
                )
            )
            metadata = {
                "task_type": "classification",
                "label_to_scalar": {"negative": 0, "positive": 1},
                "label_column": "outcome",
                "id_column": "sample_id",
                "input_structure": ["data.csv"],
            }

            save_dataset_metadata(config, metadata)
            self.assertTrue(
                (config.shared_dir / DATASET_METADATA_FILE_NAME).is_file()
            )
            prepared_datasets_dir = workspace / PREPARED_DATASETS_DIR_NAME
            prepared_datasets_dir.mkdir()
            remove_path(prepared_datasets_dir)

            self.assertEqual(load_dataset_metadata(config), metadata)

class TestTransientDatasetCleanupLifecycle(unittest.TestCase):
    def test_cleanup_runs_after_test_evaluation_and_reporting(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            datasets_dir = root / "datasets"
            dataset_dir = datasets_dir / "toy"
            dataset_dir.mkdir(parents=True)
            workspace = root / "workspace"
            arguments = argparse.Namespace(
                datasets_dir=datasets_dir,
                list_datasets=False,
                cpu_only=True,
                image="test-image",
                workspace_dir=workspace,
                fork_from_run=None,
                dataset="toy",
                test=False,
                list_models=False,
                list_metrics=False,
            )
            parser = Mock()
            parser.parse_args.return_value = arguments
            events = []

            def run_agent(*_args):
                path = (
                    workspace
                    / PREPARED_DATASETS_DIR_NAME
                    / "_csv_converted_source"
                    / "toy"
                )
                path.mkdir(parents=True)
                events.append("run")
                return 0

            def assert_transient_state_exists(event):
                self.assertTrue(
                    (workspace / PREPARED_DATASETS_DIR_NAME).is_dir()
                )
                events.append(event)

            with (
                patch.object(run_cli, "build_parser", return_value=parser),
                patch.object(run_cli, "create_agent_id", return_value="new_run"),
                patch.object(run_cli, "_run_agent_in_docker", side_effect=run_agent),
                patch.object(
                    run_cli,
                    "_run_test_evaluation_in_docker",
                    side_effect=lambda **_kwargs: assert_transient_state_exists(
                        "test evaluation"
                    ),
                ),
                patch.object(
                    run_cli,
                    "_run_reporting_in_docker",
                    side_effect=lambda *_args: assert_transient_state_exists(
                        "reporting"
                    ),
                ),
            ):
                exit_code = run_cli.main()

            self.assertEqual(exit_code, 0)
            self.assertEqual(events, ["run", "test evaluation", "reporting"])
            self.assertFalse(
                (workspace / PREPARED_DATASETS_DIR_NAME).exists()
            )


if __name__ == "__main__":
    unittest.main()
