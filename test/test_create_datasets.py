import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import agentomics.utils.create_datasets as create_datasets


class CreateDatasetsCliTest(unittest.TestCase):
    def test_main_uses_resolved_dataset_and_cache_paths(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            datasets_dir = tmp_path / "datasets"
            cache_dir = tmp_path / "cache"
            paths = SimpleNamespace(
                datasets_dir=datasets_dir,
                base_dir=tmp_path / "base",
            )

            with (
                patch.object(
                    sys,
                    "argv",
                    [
                        "agentomics-download-datasets",
                        "--datasets-dir",
                        str(datasets_dir),
                        "--cache-dir",
                        str(cache_dir),
                    ],
                ),
                patch.object(create_datasets, "resolve_agentomics_paths", return_value=paths),
                patch.object(create_datasets, "generate_dataset_files") as generate_dataset_files_mock,
            ):
                create_datasets.main()

            generate_dataset_files_mock.assert_called_once_with(datasets_dir, cache_dir.resolve())

    def test_download_genomic_benchmark_dataset_retries_after_stale_cache_cleanup_error(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_root = Path(tmpdir) / ".genomic_benchmarks"
            dataset_path = cache_root / "human_enhancers_cohn"
            stale_split = dataset_path / "train" / "negative"
            stale_split.mkdir(parents=True, exist_ok=True)
            (stale_split / "partial.txt").write_text("stale", encoding="utf-8")

            calls = []

            def fake_download_dataset(dataset_name, dest_path, cache_path):
                calls.append((dataset_name, Path(dest_path), Path(cache_path)))
                if len(calls) == 1:
                    raise OSError(39, "Directory not empty")

                fresh_dataset_path = Path(dest_path) / dataset_name
                fresh_label = fresh_dataset_path / "train" / "positive"
                fresh_label.mkdir(parents=True, exist_ok=True)
                (fresh_label / "seq.txt").write_text("ACGT", encoding="utf-8")
                fresh_test_label = fresh_dataset_path / "test" / "negative"
                fresh_test_label.mkdir(parents=True, exist_ok=True)
                (fresh_test_label / "seq.txt").write_text("TGCA", encoding="utf-8")
                return fresh_dataset_path

            resolved_path = create_datasets._download_genomic_benchmark_dataset(
                fake_download_dataset,
                cache_root,
                "human_enhancers_cohn",
            )

            self.assertEqual("human_enhancers_cohn", resolved_path.name)
            self.assertEqual(2, len(calls))
            self.assertEqual(cache_root, calls[0][1])
            self.assertNotEqual(cache_root, calls[1][1])

    def test_download_genomic_benchmark_dataset_reuses_complete_cache(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_root = Path(tmpdir) / ".genomic_benchmarks"
            dataset_path = cache_root / "human_enhancers_cohn"
            train_label = dataset_path / "train" / "positive"
            test_label = dataset_path / "test" / "negative"
            train_label.mkdir(parents=True, exist_ok=True)
            test_label.mkdir(parents=True, exist_ok=True)
            (train_label / "seq.txt").write_text("ACGT", encoding="utf-8")
            (test_label / "seq.txt").write_text("TGCA", encoding="utf-8")

            download_dataset_mock = unittest.mock.Mock()

            resolved_path = create_datasets._download_genomic_benchmark_dataset(
                download_dataset_mock,
                cache_root,
                "human_enhancers_cohn",
            )

            self.assertEqual(dataset_path, resolved_path)
            download_dataset_mock.assert_not_called()


if __name__ == "__main__":
    unittest.main()
