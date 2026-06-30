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

    def test_run_config_passes_datasets_and_workspace_dirs(self):
        self.assertIn('AGENTOMICS_ARGS+=(--workspace-dir "$WORKSPACE_DIR")', self.script)
        self.assertIn('AGENTOMICS_ARGS+=(--datasets-dir "$AGENTOMICS_DIR/datasets")', self.script)
        self.assertNotIn("--prepared-datasets-dir", self.script)

    def test_no_hidden_test_handling(self):
        self.assertNotIn("test_datasets", self.script)
        self.assertNotIn("--test-datasets-dir", self.script)

    def test_help_text_has_no_prepared_datasets_user_concept(self):
        help_block = self._block("show_help()", "EOF")
        self.assertNotIn("prepared dataset", help_block.lower())

    def test_list_datasets_label(self):
        self.assertIn("List available datasets and exit", self.script)


if __name__ == "__main__":
    unittest.main()
