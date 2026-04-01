import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from agentomics.utils.path_defaults import find_repo_root, resolve_agentomics_paths


class PathDefaultsTest(unittest.TestCase):
    def test_find_repo_root_from_nested_repo_path(self):
        nested_path = REPO_ROOT / "src" / "agentomics" / "utils"
        self.assertEqual(find_repo_root(nested_path), REPO_ROOT)

    def test_resolve_paths_uses_repo_root_when_detected(self):
        paths = resolve_agentomics_paths(cwd=REPO_ROOT / "src")
        self.assertEqual(paths.base_dir, REPO_ROOT)
        self.assertEqual(paths.workspace_dir, REPO_ROOT / "workspace")
        self.assertEqual(paths.prepared_datasets_dir, REPO_ROOT / "prepared_datasets")
        self.assertEqual(paths.prepared_test_sets_dir, REPO_ROOT / "prepared_test_sets")

    def test_resolve_paths_falls_back_to_current_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            paths = resolve_agentomics_paths(cwd=tmp_path)
            self.assertEqual(paths.base_dir, tmp_path)
            self.assertEqual(paths.workspace_dir, tmp_path / "workspace")
            self.assertEqual(paths.datasets_dir, tmp_path / "datasets")
            self.assertEqual(paths.prepared_datasets_dir, tmp_path / "prepared_datasets")
            self.assertEqual(paths.prepared_test_sets_dir, tmp_path / "prepared_test_sets")

    def test_workspace_override_updates_agent_dataset_default(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace_dir = Path(tmpdir) / "custom-workspace"
            paths = resolve_agentomics_paths(cwd=REPO_ROOT, workspace_dir=workspace_dir)
            self.assertEqual(paths.workspace_dir, workspace_dir.resolve())
            self.assertEqual(paths.agent_datasets_dir, workspace_dir.resolve() / "datasets")

    def test_legacy_prepared_tests_env_var_is_still_supported(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            env_path = str(Path(tmpdir) / "legacy-tests")
            with patch.dict(os.environ, {"PREPARED_TESTS_DIR": env_path}, clear=False):
                paths = resolve_agentomics_paths(cwd=REPO_ROOT)
            self.assertEqual(paths.prepared_test_sets_dir, Path(env_path).resolve())


if __name__ == "__main__":
    unittest.main()
