import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import agentomics.prepare_datasets as prepare_datasets


class PrepareDatasetsCliTest(unittest.TestCase):
    def test_single_dataset_mode_passes_test_sets_output_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            dataset_dir = tmp_path / "demo"
            dataset_dir.mkdir()

            paths = SimpleNamespace(
                datasets_dir=tmp_path / "datasets",
                prepared_datasets_dir=tmp_path / "prepared",
                prepared_test_sets_dir=tmp_path / "prepared_tests",
            )

            with (
                patch.object(sys, "argv", ["agentomics.prepare_datasets", "--dataset-dir", str(dataset_dir)]),
                patch.object(prepare_datasets, "resolve_agentomics_paths", return_value=paths),
                patch.object(prepare_datasets, "prepare_dataset") as prepare_dataset_mock,
            ):
                prepare_datasets.main()

            self.assertTrue(paths.prepared_datasets_dir.is_dir())
            self.assertTrue(paths.prepared_test_sets_dir.is_dir())
            prepare_dataset_mock.assert_called_once_with(
                dataset_dir=dataset_dir,
                target_col=None,
                positive_class=None,
                negative_class=None,
                task_type=None,
                output_dir=paths.prepared_datasets_dir,
                test_sets_output_dir=paths.prepared_test_sets_dir,
            )


if __name__ == "__main__":
    unittest.main()
