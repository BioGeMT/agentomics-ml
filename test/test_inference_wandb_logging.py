import json
import sys
import types
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from agentomics.runtime.inference_workflow import _log_metrics_to_wandb


class InferenceWandbLoggingTest(unittest.TestCase):
    def test_metrics_are_logged_under_the_test_split_prefix(self):
        with TemporaryDirectory() as tmp:
            agent_dir = Path(tmp)
            metrics_path = agent_dir / "predictions.metrics.json"
            metrics_path.write_text(
                json.dumps({"ACC": 0.75, "F1": 0.8}),
                encoding="utf-8",
            )
            wandb_run = Mock()
            resume_wandb_run = Mock(return_value=wandb_run)
            fake_wandb_setup = types.ModuleType(
                "agentomics.run_logging.wandb_setup"
            )
            fake_wandb_setup.resume_wandb_run = resume_wandb_run

            with (
                patch.dict(
                    sys.modules,
                    {"agentomics.run_logging.wandb_setup": fake_wandb_setup},
                ),
                patch(
                    "agentomics.runtime.inference_workflow.are_wandb_vars_available",
                    return_value=True,
                ),
                patch(
                    "agentomics.runtime.inference_workflow.load_config_from_run_dir_and_reroot",
                    return_value=Mock(),
                ),
            ):
                logged = _log_metrics_to_wandb(
                    agent_dir=agent_dir,
                    metrics_path=metrics_path,
                    prefix="test_leftout",
                )

        self.assertTrue(logged)
        wandb_run.log.assert_called_once_with(
            {
                "test_leftout/ACC": 0.75,
                "test_leftout/F1": 0.8,
            }
        )
        wandb_run.finish.assert_called_once_with()

    def test_logging_failure_does_not_interrupt_inference(self):
        with TemporaryDirectory() as tmp:
            agent_dir = Path(tmp)
            metrics_path = agent_dir / "predictions.metrics.json"
            metrics_path.write_text(json.dumps({"ACC": 0.75}), encoding="utf-8")
            fake_wandb_setup = types.ModuleType(
                "agentomics.run_logging.wandb_setup"
            )
            fake_wandb_setup.resume_wandb_run = Mock(
                side_effect=RuntimeError("network unavailable")
            )

            with (
                patch.dict(
                    sys.modules,
                    {"agentomics.run_logging.wandb_setup": fake_wandb_setup},
                ),
                patch(
                    "agentomics.runtime.inference_workflow.are_wandb_vars_available",
                    return_value=True,
                ),
                patch(
                    "agentomics.runtime.inference_workflow.load_config_from_run_dir_and_reroot",
                    return_value=Mock(),
                ),
            ):
                logged = _log_metrics_to_wandb(
                    agent_dir=agent_dir,
                    metrics_path=metrics_path,
                    prefix="test",
                )

        self.assertFalse(logged)


if __name__ == "__main__":
    unittest.main()
