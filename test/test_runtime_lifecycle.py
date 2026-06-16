import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from agents.steps.data_split import DataSplitOutput
from runtime.read_write_utils import (
    archive_current_iteration,
    get_last_successful_iteration,
    get_next_iteration_index,
    initialize_current_iteration_metadata,
    initialize_current_iteration_state,
    initialize_current_iteration_workspace,
    initialize_run_directories,
    load_iteration_state,
    save_config,
    update_current_iteration_state,
)
from runtime.step_outputs import (
    _deserialize_step_output_record,
    load_step_output,
    load_step_outputs,
    save_step_output,
)
from utils.config import Config


class TestStepOutputRoundtrip(unittest.TestCase):
    """Step outputs must survive a save→archive→load roundtrip with correct types."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        (self.root / "workspace").mkdir()
        self.config = self._make_config("roundtrip_agent")

    def tearDown(self):
        self.temp_dir.cleanup()

    def _make_config(self, agent_id: str) -> Config:
        config = Config(
            agent_id=agent_id,
            model_name="test-model",
            iteration_plan_model_name="test-iteration-plan-model",
            dataset="toy",
            tags=[],
            val_metric="ACC",
            workspace_dir=str(self.root / "workspace"),
            datasets_dir=str(self.root / "datasets"),
            user_prompt="test",
            task_type="classification",
            input_structure=["data.csv"],
        )
        initialize_run_directories(config)
        save_config(config)
        initialize_current_iteration_workspace(config)
        initialize_current_iteration_metadata(config, iteration=0)
        initialize_current_iteration_state(config, started_at=100.0)
        return config

    def _save_and_archive_step(self, step_id: str, output) -> None:
        """Mimics RuntimeStep.on_step_success: save output then rename current_step → step_id."""
        self.config.current_step_dir.mkdir(exist_ok=True)
        save_step_output(self.config, step_id, output)
        archived = self.config.archived_step_dir(step_id)
        self.config.current_step_dir.rename(archived)

    def test_pydantic_output_is_deserialized_to_original_type(self):
        split_output = DataSplitOutput(
            train_path="/tmp/train",
            val_path="/tmp/validation",
            mini_train_path="/tmp/mini_train",
            splitting_strategy="stratified 80/20",
            split_changed=True,
        )
        self._save_and_archive_step("data_split", split_output)

        loaded = load_step_output(self.config, "data_split", self.config.current_iteration_dir)

        self.assertIsInstance(loaded, DataSplitOutput)
        self.assertEqual(loaded.splitting_strategy, "stratified 80/20")
        self.assertTrue(loaded.split_changed)

    def test_plain_dict_output_survives_roundtrip(self):
        self._save_and_archive_step("data_exploration", {"summary": "10 features, 1000 rows"})

        loaded = load_step_output(self.config, "data_exploration", self.config.current_iteration_dir)

        self.assertIsInstance(loaded, dict)
        self.assertEqual(loaded["summary"], "10 features, 1000 rows")

    def test_save_creates_current_step_dir_for_skipped_step_flow(self):
        split_output = DataSplitOutput(
            train_path="/tmp/train",
            val_path="/tmp/validation",
            mini_train_path="/tmp/mini_train",
            splitting_strategy="reused previous split",
            split_changed=False,
        )

        save_step_output(self.config, "data_split", split_output)

        self.assertTrue(self.config.current_step_dir.exists())
        archived = self.config.archived_step_dir("data_split")
        self.config.current_step_dir.rename(archived)

        loaded = load_step_output(self.config, "data_split", self.config.current_iteration_dir)

        self.assertIsInstance(loaded, DataSplitOutput)
        self.assertEqual(loaded.splitting_strategy, "reused previous split")
        self.assertFalse(loaded.split_changed)

    def test_load_step_outputs_returns_all_archived_outputs_in_sequence_order(self):
        self._save_and_archive_step("data_exploration", {"origin": "exploration"})
        self._save_and_archive_step("data_split", DataSplitOutput(
            train_path="/a", val_path="/b", mini_train_path="/c", splitting_strategy="s", split_changed=False,
        ))

        outputs = load_step_outputs(self.config)

        self.assertEqual(len(outputs), 2)
        self.assertEqual(outputs[0]["origin"], "exploration")
        self.assertIsInstance(outputs[1], DataSplitOutput)

    def test_save_rejects_duplicate_step_id_in_same_iteration(self):
        self.config.current_step_dir.mkdir(exist_ok=True)
        save_step_output(self.config, "data_exploration", {"first": True})

        with self.assertRaises(FileExistsError):
            save_step_output(self.config, "data_exploration", {"second": True})

    def test_deserialize_unknown_step_id_returns_plain_payload(self):
        record = {
            "step_id": "nonexistent_custom_step",
            "model_type": "dict",
            "payload": {"custom": "data"},
        }
        output = _deserialize_step_output_record(record)
        self.assertEqual(output, {"custom": "data"})


class TestIterationArchival(unittest.TestCase):
    """Archive moves current_iteration → iteration_N and updates iteration tracking."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.workspace_dir = self.root / "workspace"
        self.workspace_dir.mkdir()
        self.datasets_dir = self.root / "datasets"
        self.datasets_dir.mkdir()

    def tearDown(self):
        self.temp_dir.cleanup()

    def _make_config(self, agent_id: str) -> Config:
        config = Config(
            agent_id=agent_id,
            model_name="test-model",
            iteration_plan_model_name="test-iteration-plan-model",
            dataset="toy",
            tags=[],
            val_metric="ACC",
            workspace_dir=str(self.workspace_dir),
            datasets_dir=str(self.datasets_dir),
            user_prompt="test",
            task_type="classification",
            input_structure=["data.csv"],
        )
        initialize_run_directories(config)
        save_config(config)
        return config

    def _run_full_iteration(self, config: Config, iteration: int, status: str = "success") -> None:
        initialize_current_iteration_workspace(config)
        initialize_current_iteration_metadata(config, iteration=iteration)
        initialize_current_iteration_state(config, started_at=100.0)
        update_current_iteration_state(config, ended_at=110.0, status=status)
        archive_current_iteration(config, iteration)

    def test_archive_moves_current_to_numbered_dir(self):
        config = self._make_config("archive_agent")
        self._run_full_iteration(config, iteration=0)

        self.assertTrue(config.iteration_dir(0).exists())
        self.assertFalse(config.current_iteration_dir.exists())

    def test_next_iteration_index_advances_after_archive(self):
        config = self._make_config("index_agent")

        self.assertEqual(get_next_iteration_index(config), 0)

        self._run_full_iteration(config, iteration=0)
        self.assertEqual(get_next_iteration_index(config), 1)

        self._run_full_iteration(config, iteration=1)
        self.assertEqual(get_next_iteration_index(config), 2)

    def test_last_successful_iteration_skips_failed(self):
        config = self._make_config("success_tracking_agent")

        self._run_full_iteration(config, iteration=0, status="success")
        self._run_full_iteration(config, iteration=1, status="failed")

        self.assertEqual(get_last_successful_iteration(config), 0)

    def test_last_successful_iteration_returns_none_when_all_failed(self):
        config = self._make_config("all_failed_agent")

        self._run_full_iteration(config, iteration=0, status="failed")

        self.assertIsNone(get_last_successful_iteration(config))

    def test_init_workspace_succeeds_if_already_exists(self):
        config = self._make_config("reject_agent")
        initialize_current_iteration_workspace(config)
        initialize_current_iteration_workspace(config)
        self.assertTrue(config.current_iteration_dir.is_dir())

    def test_iteration_state_preserves_fields_across_partial_updates(self):
        config = self._make_config("state_agent")
        initialize_current_iteration_workspace(config)
        initialize_current_iteration_metadata(config, iteration=0)
        initialize_current_iteration_state(config, started_at=100.0)

        update_current_iteration_state(config, status="running")
        update_current_iteration_state(config, ended_at=112.5)

        state = load_iteration_state(config.current_iteration_dir)
        self.assertEqual(state["started_at"], 100.0)
        self.assertEqual(state["ended_at"], 112.5)
        self.assertEqual(state["status"], "running")


if __name__ == "__main__":
    unittest.main()
