import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
RUN_SH = REPO_ROOT / "run.sh"


class RunLauncherContainerContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.script = RUN_SH.read_text(encoding="utf-8")

    def _block(self, start: str, end: str) -> str:
        start_index = self.script.index(start)
        end_index = self.script.index(end, start_index)
        return self.script[start_index:end_index]

    def test_run_config_uses_datasets_dir(self):
        self.assertIn(
            "AGENTOMICS_ARGS+=(--workspace-dir /workspace --datasets-dir /repository/datasets)",
            self.script,
        )
        self.assertNotIn("--prepared-datasets-dir", self.script)

    def test_agent_container_has_no_hidden_test_mount(self):
        agent_block = self._block(
            "--name agentomics_cont_${AGENT_ID}",
            '${RUN_EXEC[@]+"${RUN_EXEC[@]}"}',
        )

        self.assertIn('-v "$(pwd)/datasets":/repository/datasets:ro', agent_block)
        self.assertIn("-v temp_agentomics_volume_${AGENT_ID}:/workspace", agent_block)
        self.assertNotIn("/repository/test_datasets", agent_block)

    def test_final_eval_uses_hidden_tests_read_only(self):
        test_eval_block = self._block(
            "--name agentomics_test_eval_cont_${AGENT_ID}",
            "else",
        )

        self.assertNotIn('/repository/datasets', test_eval_block)
        self.assertIn('-v "$(pwd)/test_datasets":/repository/test_datasets:ro', test_eval_block)
        self.assertIn("--test-datasets-dir /repository/test_datasets", test_eval_block)

    def test_help_text_has_no_prepared_datasets_user_concept(self):
        help_block = self._block("show_help()", "EOF")
        self.assertNotIn("prepared dataset", help_block.lower())

    def test_list_datasets_label(self):
        self.assertIn("List available datasets and exit", self.script)


if __name__ == "__main__":
    unittest.main()
