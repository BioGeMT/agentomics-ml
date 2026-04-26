import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import agentomics.run_logging.evaluate_stealth_test as evaluate_stealth_test


class EvaluateStealthTestPathResolutionTest(unittest.TestCase):
    def test_uses_prepared_paths_from_config_when_repo_paths_are_unavailable(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            prepared_datasets_dir = tmp_path / "prepared_datasets"
            prepared_test_sets_dir = tmp_path / "prepared_test_sets"
            experiment_folder = tmp_path / "experiment"
            test_output_dir = tmp_path / "test_outputs"

            (prepared_datasets_dir / "demo").mkdir(parents=True)
            (prepared_test_sets_dir / "demo").mkdir(parents=True)
            (experiment_folder / "extras").mkdir(parents=True)
            test_output_dir.mkdir()

            (prepared_datasets_dir / "demo" / "metadata.json").write_text(
                json.dumps({"task_type": "classification", "numeric_label_col": "numeric_label"}),
                encoding="utf-8",
            )
            (prepared_test_sets_dir / "demo" / "test.csv").write_text(
                "id,numeric_label\n1,1\n",
                encoding="utf-8",
            )
            (test_output_dir / "iteration_0_test_predictions.csv").write_text(
                "id,prediction\n1,0.9\n",
                encoding="utf-8",
            )
            (experiment_folder / "extras" / "config.json").write_text(
                json.dumps(
                    {
                        "agent_id": "demo-agent",
                        "model_name": "demo-model",
                        "feedback_model_name": "demo-model",
                        "dataset": "demo",
                        "tags": [],
                        "val_metric": "ACC",
                        "workspace_dir": str(tmp_path / "workspace"),
                        "prepared_dataset_dir": str(prepared_datasets_dir / "demo"),
                        "prepared_test_set_dir": str(prepared_test_sets_dir / "demo"),
                        "agent_dataset_dir": str(tmp_path / "agent_datasets" / "demo"),
                        "user_prompt": "demo",
                        "iterations": 1,
                    }
                ),
                encoding="utf-8",
            )

            mock_run = Mock()
            with (
                patch.object(evaluate_stealth_test, "resume_wandb_run", return_value=mock_run),
                patch.object(evaluate_stealth_test, "get_metrics", return_value={"ACC": 1.0}) as get_metrics,
                patch.object(evaluate_stealth_test.wandb, "log"),
                patch.object(evaluate_stealth_test.wandb, "finish"),
            ):
                evaluate_stealth_test.evaluate_stealth_test(
                    dataset="demo",
                    test_output_dir=test_output_dir,
                    experiment_folder=experiment_folder,
                )

            get_metrics.assert_called_once()
            self.assertEqual(
                Path(get_metrics.call_args.kwargs["test_file"]).resolve(),
                (prepared_test_sets_dir / "demo" / "test.csv").resolve(),
            )
            mock_run.define_metric.assert_called_once_with(step_metric='iteration_index', name='ACC')


if __name__ == "__main__":
    unittest.main()
