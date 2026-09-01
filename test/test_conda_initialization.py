import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from agentomics.runtime import conda_utils, run_lifecycle


class TestSharedEnvironmentInitialization(unittest.TestCase):
    def setUp(self):
        self._temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self._temp_dir.name)
        self.environment_path = self.root / "environment"
        self.config = SimpleNamespace(agent_id="test-agent")

    def tearDown(self):
        self._temp_dir.cleanup()

    def test_reuses_an_initialized_environment(self):
        activation_script = self.environment_path / "bin" / "activate"
        activation_script.parent.mkdir(parents=True)
        activation_script.touch()

        with mock.patch.object(
            conda_utils,
            "get_shared_environment_path",
            return_value=self.environment_path,
        ), mock.patch.object(conda_utils.subprocess, "run") as subprocess_run:
            result = conda_utils.initialize_shared_environment(self.config)

        self.assertEqual(result, self.environment_path)
        subprocess_run.assert_not_called()


class TestRunInitialization(unittest.IsolatedAsyncioTestCase):
    async def test_initializes_environment_before_tool_creation(self):
        config = SimpleNamespace(
            agent_id="test-agent",
            iterations=0,
            tool_ids=["run_python"],
        )
        call_order = []

        with mock.patch.object(
            run_lifecycle,
            "initialize_shared_environment",
            side_effect=lambda _: call_order.append("environment"),
        ) as initialize_environment, mock.patch.object(
            run_lifecycle,
            "create_tools",
            side_effect=lambda *_: call_order.append("tools") or [],
        ) as create_tools, mock.patch.object(
            run_lifecycle,
            "get_next_iteration_index",
            return_value=0,
        ):
            await run_lifecycle.run_agentomics.__wrapped__(
                config,
                mock.sentinel.default_model,
                mock.sentinel.iteration_plan_model,
                mock.sentinel.provider,
            )

        self.assertEqual(call_order, ["environment", "tools"])
        initialize_environment.assert_called_once_with(config)
        create_tools.assert_called_once_with(config, ["run_python"])


if __name__ == "__main__":
    unittest.main()
