import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from run_logging.test_evaluation import run_test_evaluation


def _make_config(tmp: Path, dataset_name: str = "test_dataset") -> SimpleNamespace:
    snapshot_dir = tmp / "snapshot"
    snapshot_dir.mkdir(parents=True)
    shared_dir = tmp / "run" / "shared"
    shared_dir.mkdir(parents=True)
    return SimpleNamespace(
        dataset=dataset_name,
        best_iteration_snapshot_dir=snapshot_dir,
        shared_dir=shared_dir,
        agent_id="agent-under-test",
        task_type="classification",
        label_to_scalar={"negative": 0, "positive": 1},
        input_structure=["examples.txt"],
        val_metric="ACC",
    )


def _write_test_dataset(test_datasets_dir: Path, dataset_name: str, rows: int = 2) -> None:
    dataset_dir = test_datasets_dir / dataset_name
    dataset_dir.mkdir(parents=True, exist_ok=True)
    split_dir = dataset_dir / "test"
    (split_dir / "input").mkdir(parents=True)
    (split_dir / "input" / "examples.txt").write_text("example\n", encoding="utf-8")
    labels_list = ["negative", "positive"]
    labels = ["id,label"] + [f"test-{i},{labels_list[i % 2]}" for i in range(rows)]
    (split_dir / "labels.csv").write_text("\n".join(labels) + "\n", encoding="utf-8")


class TestRunTestEvaluationFolderContract(unittest.TestCase):
    def _run_with_mocks(self, workspace_dir, test_datasets_dir, config, inference_result=None):
        if inference_result is None:
            inference_result = SimpleNamespace(returncode=0, stderr=b"", stdout=b"")
        with (
            patch("run_logging.test_evaluation.load_config_from_run_dir", return_value=config),
            patch("run_logging.test_evaluation.resume_wandb_run", return_value=None),
            patch("run_logging.test_evaluation.is_wandb_active", return_value=False),
            patch("run_logging.test_evaluation.create_environment_from_descriptor"),
            patch("run_logging.test_evaluation.run_inference_script", return_value=inference_result) as mock_infer,
            patch("run_logging.test_evaluation.compute_metrics", return_value={"ACC": 1.0}),
            patch("run_logging.test_evaluation.log_test_inference_duration", return_value=None),
            patch("run_logging.test_evaluation.print_best_iteration_metrics", return_value=None),
        ):
            run_test_evaluation(workspace_dir, test_datasets_dir)
            return mock_infer

    def test_skips_when_test_split_absent(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            mock_infer = self._run_with_mocks(root, root / "test_datasets", _make_config(root))
            mock_infer.assert_not_called()

    @patch("builtins.print")
    def test_skips_when_test_split_absent_without_error(self, mock_print):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._run_with_mocks(root, root / "test_datasets", _make_config(root))
            printed = " ".join(str(a) for call in mock_print.call_args_list for a in call.args)
            self.assertNotIn("FINAL TEST EVAL FAIL", printed)

    def test_skips_inference_when_test_split_exists_but_has_no_input_folder(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = _make_config(root)
            dataset_dir = root / "test_datasets" / config.dataset
            dataset_dir.mkdir(parents=True, exist_ok=True)
            test_split_dir = dataset_dir / "test"
            test_split_dir.mkdir(parents=True)
            (test_split_dir / "labels.csv").write_text("id,label\nr-0,negative\n", encoding="utf-8")

            mock_infer = self._run_with_mocks(root, root / "test_datasets", config)
            mock_infer.assert_not_called()

    def test_calls_inference_with_input_folder_not_csv(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = _make_config(root)
            test_datasets_dir = root / "test_datasets"
            _write_test_dataset(test_datasets_dir, config.dataset)

            mock_infer = self._run_with_mocks(root, test_datasets_dir, config)

            mock_infer.assert_called_once()
            input_path = mock_infer.call_args.kwargs["input_path"]
            self.assertTrue(str(input_path).endswith("/input"), f"Expected folder input path, got: {input_path}")
            self.assertNotIn(".csv", str(input_path))
            self.assertNotIn(".no_label", str(input_path))

    def test_compute_metrics_receives_labels_csv_path(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = _make_config(root)
            test_datasets_dir = root / "test_datasets"
            _write_test_dataset(test_datasets_dir, config.dataset)

            metric_calls = []
            with (
                patch("run_logging.test_evaluation.load_config_from_run_dir", return_value=config),
                patch("run_logging.test_evaluation.resume_wandb_run", return_value=None),
                patch("run_logging.test_evaluation.is_wandb_active", return_value=False),
                patch("run_logging.test_evaluation.create_environment_from_descriptor"),
                patch("run_logging.test_evaluation.run_inference_script",
                      return_value=SimpleNamespace(returncode=0, stderr=b"", stdout=b"")),
                patch("run_logging.test_evaluation.compute_metrics",
                      side_effect=lambda **kw: metric_calls.append(kw) or {"ACC": 1.0}),
                patch("run_logging.test_evaluation.log_test_inference_duration", return_value=None),
                patch("run_logging.test_evaluation.print_best_iteration_metrics", return_value=None),
            ):
                run_test_evaluation(root, test_datasets_dir)

            self.assertEqual(1, len(metric_calls))
            labels_path = metric_calls[0]["labels_path"]
            self.assertTrue(str(labels_path).endswith("labels.csv"), f"Expected labels.csv, got: {labels_path}")
            self.assertNotIn("input", str(labels_path))


if __name__ == "__main__":
    unittest.main()
