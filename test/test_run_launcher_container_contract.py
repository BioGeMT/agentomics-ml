import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from agentomics.cli.run import (  # noqa: E402
    CONTAINER_DATASETS_DIRECTORY,
    PUBLIC_DATASET_ENTRY_NAMES,
    _build_dataset_mounts,
)

RUN_LAUNCHER = REPO_ROOT / "src" / "agentomics" / "cli" / "run.py"
RUN_ARGUMENTS = REPO_ROOT / "src" / "agentomics" / "cli" / "run_arguments.py"


class RunLauncherSourceContractTest(unittest.TestCase):
    """The launcher must hand the container its directories and drive the
    package workflow modules; the retired local-execution and separate
    hidden-test-tree concepts must stay gone."""

    @classmethod
    def setUpClass(cls):
        cls.launcher = RUN_LAUNCHER.read_text(encoding="utf-8")
        cls.arguments = RUN_ARGUMENTS.read_text(encoding="utf-8")

    def test_run_config_passes_datasets_and_workspace_dirs(self):
        self.assertIn(
            '["--workspace-dir", str(CONTAINER_WORKSPACE_DIRECTORY)]',
            self.launcher,
        )
        self.assertIn(
            '["--datasets-dir", str(CONTAINER_DATASETS_DIRECTORY)]',
            self.launcher,
        )
        self.assertNotIn("--prepared-datasets-dir", self.launcher)

    def test_launcher_has_no_local_execution_mode(self):
        self.assertNotIn("--execution-mode", self.launcher)
        self.assertNotIn("dst=/outputs", self.launcher)

    def test_launcher_invokes_container_workflow(self):
        self.assertIn("agentomics.runtime.run_workflow", self.launcher)

    def test_test_evaluation_reuses_inference_on_co_located_splits(self):
        # The old separate hidden-test tree and flag are gone; evaluation runs
        # the best model on every test-prefixed split via inference.
        self.assertNotIn("test_datasets", self.launcher)
        self.assertNotIn("--test-datasets-dir", self.launcher)
        self.assertIn("path.name.startswith(TEST_SPLIT)", self.launcher)
        self.assertIn("run_inference_in_docker", self.launcher)
        self.assertIn('f"eval_predictions_{test_input.name}.csv"', self.launcher)
        self.assertIn("agentomics.runtime.report_workflow", self.launcher)

    def test_help_text_has_no_prepared_datasets_user_concept(self):
        self.assertNotIn("prepared dataset", self.arguments.lower())

    def test_list_datasets_label(self):
        self.assertIn("List available datasets and exit", self.arguments)


class DatasetMountContractTest(unittest.TestCase):
    """The agent container only receives the public dataset entries; the
    co-located ``test`` split must never be mounted into the agent's view."""

    def _mounts_for(self, entry_names: list[str]) -> list[str]:
        with TemporaryDirectory() as tmp:
            dataset_dir = Path(tmp) / "example_dataset"
            dataset_dir.mkdir()
            for name in entry_names:
                path = dataset_dir / name
                if name.endswith(".csv") or name.endswith(".json") or name.endswith(".md"):
                    path.write_text("x", encoding="utf-8")
                else:
                    path.mkdir()
            return _build_dataset_mounts(dataset_dir)

    def test_public_entries_are_mounted_readonly(self):
        mounts = self._mounts_for(["train", "validation", "metadata.json"])
        destinations = " ".join(mounts)
        self.assertIn(f"dst={CONTAINER_DATASETS_DIRECTORY}/example_dataset/train", destinations)
        self.assertIn(
            f"dst={CONTAINER_DATASETS_DIRECTORY}/example_dataset/validation", destinations
        )
        for mount in mounts:
            if "type=bind" in mount:
                self.assertIn("readonly", mount)

    def test_co_located_test_split_is_not_mounted_to_agent(self):
        self.assertNotIn("test", PUBLIC_DATASET_ENTRY_NAMES)
        self.assertNotIn("test.csv", PUBLIC_DATASET_ENTRY_NAMES)

        mounts = self._mounts_for(["train", "validation", "test", "test.csv"])
        for mount in mounts:
            self.assertNotIn("/example_dataset/test", mount)

    def test_symlinked_dataset_entry_is_rejected(self):
        with TemporaryDirectory() as tmp:
            dataset_dir = Path(tmp) / "example_dataset"
            (dataset_dir / "train").mkdir(parents=True)
            external = Path(tmp) / "external"
            external.mkdir()
            (dataset_dir / "validation").symlink_to(external)

            with self.assertRaises(ValueError) as raised:
                _build_dataset_mounts(dataset_dir)
            self.assertIn("symlink", str(raised.exception).lower())


if __name__ == "__main__":
    unittest.main()
