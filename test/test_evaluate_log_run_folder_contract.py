import sys
import types
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))


def _stub(name, **attrs):
    mod = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    sys.modules.setdefault(name, mod)
    return mod


wandb_errors = _stub("wandb.errors", AuthenticationError=Exception, UsageError=Exception)
wandb_mod = _stub("wandb", log=lambda *a, **k: None, run=None, login=lambda *a, **k: True)
wandb_mod.errors = wandb_errors
_stub("utils.metrics", get_task_to_metrics_names=lambda: {})
_stub("run_logging.logging_helpers", is_wandb_active=lambda: False, log_test_inference_duration=lambda *a, **k: None)
_stub("run_logging.wandb_setup", resume_wandb_run=lambda *a, **k: None)
_stub("utils.printing_utils", print_best_iteration_metrics=lambda *a, **k: None)
_stub("runtime.conda_utils", get_best_iteration_snapshot_environment_path=lambda *a: None)
_stub("runtime.evaluate_result", get_metrics=lambda **k: {})
_stub("runtime.inference_runner", compute_metrics=lambda **k: {}, run_inference_script=lambda **k: None)
_stub("runtime.filesystem", remove_path=lambda *a: None)
_stub("runtime.read_write_utils", load_config_from_run_dir=lambda *a, **k: None, load_dataset_metadata=lambda *a: {})
_stub("utils.exceptions", AgentScriptFailed=Exception)

from run_logging.test_evaluation import run_test_evaluation  # noqa: E402


METADATA = {"numeric_label_col": "numeric_label", "task_type": "classification"}


def _make_config(tmp: Path, dataset_name: str = "test_dataset") -> SimpleNamespace:
    snapshot_dir = tmp / "snapshot"
    snapshot_dir.mkdir(parents=True)
    return SimpleNamespace(
        dataset=dataset_name,
        best_iteration_snapshot_dir=snapshot_dir,
        agent_id="agent-under-test",
        task_type="classification",
        val_metric="ACC",
    )


def write_split(parent: Path, split_name: str, rows: int = 2) -> None:
    split_dir = parent / split_name
    (split_dir / "input").mkdir(parents=True)
    (split_dir / "input" / "examples.txt").write_text("example\n", encoding="utf-8")
    labels = ["id,numeric_label"] + [f"{split_name}-{i},{i % 2}" for i in range(rows)]
    (split_dir / "labels.csv").write_text("\n".join(labels) + "\n", encoding="utf-8")


class TestRunTestEvaluationFolderContract(unittest.TestCase):
    def _run_with_mocks(self, workspace_dir, prepared_test_sets_dir, config, inference_result=None):
        if inference_result is None:
            inference_result = SimpleNamespace(returncode=0, stderr=b"", stdout=b"")
        with (
            patch("run_logging.test_evaluation.load_config_from_run_dir", return_value=config),
            patch("run_logging.test_evaluation.resume_wandb_run", return_value=None),
            patch("run_logging.test_evaluation.load_dataset_metadata", return_value=METADATA),
            patch("run_logging.test_evaluation.is_wandb_active", return_value=False),
            patch("run_logging.test_evaluation.run_inference_script", return_value=inference_result) as mock_infer,
            patch("run_logging.test_evaluation.compute_metrics", return_value={"ACC": 1.0}),
            patch("run_logging.test_evaluation.log_test_inference_duration", return_value=None),
            patch("run_logging.test_evaluation.print_best_iteration_metrics", return_value=None),
        ):
            run_test_evaluation(workspace_dir, prepared_test_sets_dir)
            return mock_infer

    def test_skips_when_test_split_absent(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            mock_infer = self._run_with_mocks(root, root / "prepared_test_sets", _make_config(root))
            mock_infer.assert_not_called()

    def test_skips_inference_when_test_split_exists_but_has_no_input_folder(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = _make_config(root)
            test_split_dir = root / "prepared_test_sets" / config.dataset / "test"
            test_split_dir.mkdir(parents=True)
            # labels.csv present but no input/ folder — malformed split
            (test_split_dir / "labels.csv").write_text("id,numeric_label\nr-0,0\n", encoding="utf-8")

            # run_test_evaluation catches the FileNotFoundError internally (to not crash the run),
            # so we verify inference is not called rather than expecting an exception.
            mock_infer = self._run_with_mocks(root, root / "prepared_test_sets", config)
            mock_infer.assert_not_called()

    def test_calls_inference_with_input_folder_not_csv(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = _make_config(root)
            prepared_test_sets_dir = root / "prepared_test_sets"
            write_split(prepared_test_sets_dir / config.dataset, "test")

            mock_infer = self._run_with_mocks(root, prepared_test_sets_dir, config)

            mock_infer.assert_called_once()
            input_path = mock_infer.call_args.kwargs["input_path"]
            self.assertTrue(str(input_path).endswith("/input"), f"Expected folder input path, got: {input_path}")
            self.assertNotIn(".csv", str(input_path))
            self.assertNotIn(".no_label", str(input_path))

    def test_compute_metrics_receives_labels_csv_path(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = _make_config(root)
            prepared_test_sets_dir = root / "prepared_test_sets"
            write_split(prepared_test_sets_dir / config.dataset, "test")

            metric_calls = []
            with (
                patch("run_logging.test_evaluation.load_config_from_run_dir", return_value=config),
                patch("run_logging.test_evaluation.resume_wandb_run", return_value=None),
                patch("run_logging.test_evaluation.load_dataset_metadata", return_value=METADATA),
                patch("run_logging.test_evaluation.is_wandb_active", return_value=False),
                patch("run_logging.test_evaluation.run_inference_script",
                      return_value=SimpleNamespace(returncode=0, stderr=b"", stdout=b"")),
                patch("run_logging.test_evaluation.compute_metrics",
                      side_effect=lambda **kw: metric_calls.append(kw) or {"ACC": 1.0}),
                patch("run_logging.test_evaluation.log_test_inference_duration", return_value=None),
                patch("run_logging.test_evaluation.print_best_iteration_metrics", return_value=None),
            ):
                run_test_evaluation(root, prepared_test_sets_dir)

            self.assertEqual(1, len(metric_calls))
            labels_path = metric_calls[0]["labels_path"]
            self.assertTrue(str(labels_path).endswith("labels.csv"), f"Expected labels.csv, got: {labels_path}")
            self.assertNotIn("input", str(labels_path))


if __name__ == "__main__":
    unittest.main()
