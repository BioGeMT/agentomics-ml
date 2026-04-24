from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import call, patch

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from runtime.setup_fork import fork_run
from runtime.read_write_utils import replace_string_in_tree_files
from runtime.git_checkpoints import (
    build_iteration_end_commit_message,
    build_step_commit_message,
    _find_checkpoint_commit,
)
from runtime.read_write_utils import initialize_run_directories, save_config
from utils.config import Config


def _git(repo_dir: Path, args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=repo_dir,
        check=True,
        capture_output=True,
        text=True,
    )


def _setup_git_repo(repo_dir: Path) -> None:
    repo_dir.mkdir(parents=True, exist_ok=True)
    _git(repo_dir, ["init"])
    _git(repo_dir, ["config", "user.name", "Test"])
    _git(repo_dir, ["config", "user.email", "test@test.com"])


def _commit(repo_dir: Path, message: str, filename: str = "state.txt", content: str = "") -> str:
    file_path = repo_dir / filename
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(content or message, encoding="utf-8")
    _git(repo_dir, ["add", "-A"])
    _git(repo_dir, ["commit", "-m", message])
    result = _git(repo_dir, ["rev-parse", "HEAD"])
    return result.stdout.strip()


class TestFindForkCheckpointCommit(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.workspace_dir = Path(self.temp_dir.name) / "workspace"
        self.run_id = "test_run_abc"
        _setup_git_repo(self.workspace_dir)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_finds_step_commit_with_exact_iteration(self):
        _commit(self.workspace_dir, f"agentomics/{self.run_id}/start", "f0.txt")
        _commit(self.workspace_dir, build_step_commit_message(self.run_id, 0, "data_split"), "f1.txt")
        target_hash = _commit(self.workspace_dir, build_step_commit_message(self.run_id, 0, "model_training"), "f2.txt")
        _commit(self.workspace_dir, build_step_commit_message(self.run_id, 1, "model_training"), "f3.txt")

        result = _find_checkpoint_commit(self.workspace_dir, self.run_id, "model_training", iteration=0)
        self.assertEqual(result, target_hash)

    def test_finds_latest_step_commit_for_given_step(self):
        _commit(self.workspace_dir, f"agentomics/{self.run_id}/start", "f0.txt")
        _commit(self.workspace_dir, build_step_commit_message(self.run_id, 0, "model_training"), "f1.txt")
        latest_hash = _commit(self.workspace_dir, build_step_commit_message(self.run_id, 1, "model_training"), "f2.txt")

        result = _find_checkpoint_commit(self.workspace_dir, self.run_id, "model_training", iteration=None)
        self.assertEqual(result, latest_hash)

    def test_finds_latest_step_commit_when_no_step_or_iteration(self):
        _commit(self.workspace_dir, f"agentomics/{self.run_id}/start", "f0.txt")
        _commit(self.workspace_dir, build_step_commit_message(self.run_id, 0, "data_split"), "f1.txt")
        _commit(self.workspace_dir, build_step_commit_message(self.run_id, 0, "model_training"), "f2.txt")
        latest_hash = _commit(self.workspace_dir, build_iteration_end_commit_message(self.run_id, 0), "f3.txt")

        result = _find_checkpoint_commit(self.workspace_dir, self.run_id, step_id=None, iteration=None)
        self.assertEqual(result, latest_hash)

    def test_finds_latest_checkpoint_for_iteration_when_step_omitted(self):
        _commit(self.workspace_dir, build_step_commit_message(self.run_id, 0, "model_training"), "f0.txt")
        latest_hash = _commit(self.workspace_dir, build_iteration_end_commit_message(self.run_id, 0), "f1.txt")
        _commit(self.workspace_dir, build_step_commit_message(self.run_id, 1, "model_training"), "f2.txt")

        result = _find_checkpoint_commit(self.workspace_dir, self.run_id, step_id=None, iteration=0)
        self.assertEqual(result, latest_hash)

    def test_raises_when_step_not_found(self):
        _commit(self.workspace_dir, f"agentomics/{self.run_id}/start", "f0.txt")

        with self.assertRaises(ValueError):
            _find_checkpoint_commit(self.workspace_dir, self.run_id, "nonexistent_step", iteration=None)

    def test_raises_when_iteration_not_found(self):
        _commit(self.workspace_dir, build_step_commit_message(self.run_id, 0, "model_training"), "f0.txt")

        with self.assertRaises(ValueError):
            _find_checkpoint_commit(self.workspace_dir, self.run_id, "model_training", iteration=5)


class TestReplaceStringInTreeFiles(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.run_dir = Path(self.temp_dir.name) / Config.RUN_DIRNAME
        self.run_dir.mkdir()

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_replaces_path_in_text_files(self):
        script = self.run_dir / "train.py"
        script.write_text(f'data = "/old/workspace/{Config.RUN_DIRNAME}/shared/data.csv"', encoding="utf-8")

        replace_string_in_tree_files(self.run_dir, "/old/workspace", "/new/workspace")

        self.assertIn("/new/workspace", script.read_text(encoding="utf-8"))
        self.assertNotIn("/old/workspace", script.read_text(encoding="utf-8"))

    def test_skips_files_in_conda_dirs(self):
        conda_dir = self.run_dir / ".conda" / "envs"
        conda_dir.mkdir(parents=True)
        conda_file = conda_dir / "activate.sh"
        original = f"prefix: /old/workspace/{Config.RUN_DIRNAME}/shared/.conda"
        conda_file.write_text(original, encoding="utf-8")

        replace_string_in_tree_files(self.run_dir, "/old/workspace", "/new/workspace", skip_dirs={".conda"})

        self.assertEqual(conda_file.read_text(encoding="utf-8"), original)


class TestForkRun(unittest.TestCase):
    """Integration tests for fork_run: builds a minimal source run, forks it, and verifies the result."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.source_workspace = self.root / "source_workspace"
        self.target_workspace = self.root / "target_workspace"

    def tearDown(self):
        self.temp_dir.cleanup()

    def _build_source_run(self, run_id: str) -> None:
        source_config = Config(
            agent_id=run_id,
            model_name="test-model",
            iteration_plan_model_name="test-model",
            dataset="toy",
            tags=[],
            val_metric="ACC",
            workspace_dir=str(self.source_workspace),
            prepared_datasets_dir=str(self.root / "prepared_datasets"),
            user_prompt="test",
            task_type="classification",
        )
        initialize_run_directories(source_config)
        save_config(source_config)

        _setup_git_repo(self.source_workspace)
        run_dir = source_config.run_dir
        (self.source_workspace / ".gitignore").write_text(
            f"run/shared/.conda/\n{Config.BEST_ITERATION_SNAPSHOT_DIRNAME}/.conda/\n__pycache__/\n.cache/\n*.pyc\n",
            encoding="utf-8",
        )

        script = run_dir / "shared" / "train.py"
        script.write_text(f'data = "{self.source_workspace}/{Config.RUN_DIRNAME}/shared/data.csv"', encoding="utf-8")
        (run_dir / "shared" / "environment.yml").write_text("name: fork-test\n", encoding="utf-8")
        (run_dir / "shared" / ".conda" / "envs" / f"{run_id}_env").mkdir(parents=True)

        _commit(self.source_workspace, f"agentomics/{run_id}/start", "init.txt")

        current_iter = run_dir / "current_iteration" / "runtime_info"
        current_iter.mkdir(parents=True)
        (current_iter / "iteration_metadata.json").write_text('{"iteration": 0}', encoding="utf-8")
        (current_iter / "iteration_state.json").write_text('{"status": "running"}', encoding="utf-8")
        (run_dir / "current_iteration" / "data_split").mkdir()
        (run_dir / "current_iteration" / "data_split" / "output.json").write_text('{}', encoding="utf-8")
        _commit(self.source_workspace, build_step_commit_message(run_id, 0, "data_split"), "data_split_marker.txt")

        (run_dir / "current_iteration" / "model_training").mkdir()
        (run_dir / "current_iteration" / "model_training" / "output.json").write_text('{}', encoding="utf-8")
        _commit(self.source_workspace, build_step_commit_message(run_id, 0, "model_training"), "model_training_marker.txt")

        archived_iteration_dir = run_dir / "iteration_0"
        shutil.move(run_dir / "current_iteration", archived_iteration_dir)
        best_snapshot_runtime_info = self.source_workspace / Config.BEST_ITERATION_SNAPSHOT_DIRNAME / Config.RUNTIME_INFO_DIRNAME
        best_snapshot_runtime_info.mkdir(parents=True, exist_ok=True)
        (best_snapshot_runtime_info / Config.ITERATION_METADATA_FILENAME).write_text('{"iteration": 0}', encoding="utf-8")
        (best_snapshot_runtime_info / "environment.yml").write_text("name: fork-test\n", encoding="utf-8")
        (self.source_workspace / Config.BEST_ITERATION_SNAPSHOT_DIRNAME / "snapshot_marker.txt").write_text("best", encoding="utf-8")
        (self.source_workspace / Config.BEST_ITERATION_SNAPSHOT_DIRNAME / "environment.yml").write_text("name: fork-test\n", encoding="utf-8")
        (self.source_workspace / Config.BEST_ITERATION_SNAPSHOT_DIRNAME / ".conda" / "envs" / f"{run_id}_env").mkdir(parents=True)
        (self.source_workspace / "reports" / "markdown").mkdir(parents=True, exist_ok=True)
        (self.source_workspace / "reports" / "markdown" / "run_report_iter_0.md").write_text("report", encoding="utf-8")
        (self.source_workspace / "extras" / "run_logs").mkdir(parents=True, exist_ok=True)
        (self.source_workspace / "extras" / "run_logs" / "latest.log").write_text("log", encoding="utf-8")
        _commit(self.source_workspace, build_iteration_end_commit_message(run_id, 0), "iteration_end_marker.txt")

    def _fork(self, source_run_id: str, target_run_id: str, **kwargs):
        self._build_source_run(source_run_id)
        with patch("runtime.setup_fork.ensure_environment_from_descriptor") as ensure_environment:
            fork_run(
                source_workspace_dir=self.source_workspace,
                target_agent_id=target_run_id,
                target_workspace_dir=self.target_workspace,
                **kwargs,
            )
        return ensure_environment

    def test_checks_out_correct_state(self):
        self._fork("src_run", "tgt_run", fork_from_step="data_split", fork_from_iteration=0)
        target_run_dir = self.target_workspace / Config.RUN_DIRNAME
        self.assertTrue((self.target_workspace / "data_split_marker.txt").exists())
        self.assertFalse((self.target_workspace / "model_training_marker.txt").exists())
        self.assertFalse((self.target_workspace / "iteration_end_marker.txt").exists())

    def test_defaults_to_latest_checkpoint_when_none_given(self):
        self._fork("src_run", "tgt_run", fork_from_step=None, fork_from_iteration=None)
        target_run_dir = self.target_workspace / Config.RUN_DIRNAME
        self.assertTrue((self.target_workspace / "iteration_end_marker.txt").exists())
        self.assertTrue((target_run_dir / "iteration_0" / "model_training" / "output.json").exists())
        self.assertFalse((target_run_dir / "current_iteration").exists())
        self.assertTrue((self.target_workspace / Config.BEST_ITERATION_SNAPSHOT_DIRNAME / "snapshot_marker.txt").exists())
        self.assertTrue((self.target_workspace / "reports" / "markdown" / "run_report_iter_0.md").exists())
        self.assertTrue((self.target_workspace / "extras" / "run_logs" / "latest.log").exists())

    def test_rewrites_workspace_paths(self):
        self._fork("src_run", "tgt_run", fork_from_step="data_split", fork_from_iteration=0)
        content = (self.target_workspace / Config.RUN_DIRNAME / "shared" / "train.py").read_text(encoding="utf-8")
        self.assertIn(str(self.target_workspace), content)
        self.assertNotIn(str(self.source_workspace), content)

    def test_rewinds_workspace_artifacts_with_checkpoint(self):
        ensure_environment = self._fork("src_run", "tgt_run", fork_from_step="data_split", fork_from_iteration=0)

        self.assertFalse((self.target_workspace / Config.BEST_ITERATION_SNAPSHOT_DIRNAME).exists())
        self.assertFalse((self.target_workspace / "reports").exists())
        self.assertFalse((self.target_workspace / "extras").exists())
        ensure_environment.assert_called_once_with(
            self.target_workspace / Config.RUN_DIRNAME / "shared" / "environment.yml",
            self.target_workspace / Config.RUN_DIRNAME / "shared" / ".conda" / "envs" / "tgt_run_env",
        )

    def test_syncs_shared_and_snapshot_environments_from_descriptors(self):
        target_run_id = "tgt_run"
        ensure_environment = self._fork("src_run", target_run_id, fork_from_step=None, fork_from_iteration=None)
        target_run_dir = self.target_workspace / Config.RUN_DIRNAME
        target_snapshot_dir = self.target_workspace / Config.BEST_ITERATION_SNAPSHOT_DIRNAME
        self.assertFalse((target_run_dir / "shared" / ".conda").exists())
        self.assertFalse((target_snapshot_dir / ".conda").exists())
        ensure_environment.assert_has_calls([
            call(
                target_run_dir / "shared" / "environment.yml",
                target_run_dir / "shared" / ".conda" / "envs" / f"{target_run_id}_env",
            ),
            call(
                target_snapshot_dir / "environment.yml",
                target_snapshot_dir / ".conda" / "envs" / f"{target_run_id}_env",
            ),
        ])
        self.assertEqual(ensure_environment.call_count, 2)

    def test_raises_when_target_workspace_not_empty(self):
        self._build_source_run("src_run")
        self.target_workspace.mkdir(parents=True)
        (self.target_workspace / "stale.txt").write_text("stale", encoding="utf-8")

        with self.assertRaises(FileExistsError):
            with patch("runtime.setup_fork.ensure_environment_from_descriptor"):
                fork_run(
                    source_workspace_dir=self.source_workspace,
                    target_agent_id="tgt_run",
                    target_workspace_dir=self.target_workspace,
                    fork_from_step="data_split",
                    fork_from_iteration=0,
                )

    def test_creates_target_branch(self):
        target_run_id = "tgt_run"
        self._fork("src_run", target_run_id, fork_from_step="data_split", fork_from_iteration=0)
        branch = _git(self.target_workspace, ["branch", "--show-current"]).stdout.strip()
        self.assertEqual(branch, f"agentomics-run-{target_run_id}")

    def test_handles_dangling_symlink_in_source(self):
        # Regression: wandb creates debug-core.log as a symlink to a path inside the container
        # (/root/.cache/wandb/...) which is dangling on the host. copytree must not raise.
        self._build_source_run("src_run")
        dangling = self.source_workspace / "extras" / "run_logs" / "debug-core.log"
        dangling.parent.mkdir(parents=True, exist_ok=True)
        dangling.symlink_to("/nonexistent/container/path/debug.log")

        with patch("runtime.setup_fork.ensure_environment_from_descriptor"):
            fork_run(
                source_workspace_dir=self.source_workspace,
                target_agent_id="tgt_run",
                target_workspace_dir=self.target_workspace,
                fork_from_step=None,
                fork_from_iteration=None,
            )

    def test_handles_tracked_file_modified_after_last_commit(self):
        # Regression: wandb appends to its .wandb binary log after the last git commit.
        # The copied workspace therefore has modified tracked files, which made
        # `git checkout -b` fail. The fix is `git reset --hard HEAD` before checkout.
        self._build_source_run("src_run")
        tracked_file = self.source_workspace / "model_training_marker.txt"
        tracked_file.write_text("modified after commit", encoding="utf-8")

        with patch("runtime.setup_fork.ensure_environment_from_descriptor"):
            fork_run(
                source_workspace_dir=self.source_workspace,
                target_agent_id="tgt_run",
                target_workspace_dir=self.target_workspace,
                fork_from_step="data_split",
                fork_from_iteration=0,
            )


class TestBashHelpers(unittest.TestCase):
    def _run_bash(self, snippet: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["bash", "-c", f"set -euo pipefail; source scripts/bash_helpers.sh; {snippet}"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )

    def test_build_setup_fork_args_includes_optional_args_when_provided(self):
        result = self._run_bash(
            "build_setup_fork_args /src /tgt agent123 data_split 2; "
            'printf "%s\\n" "${SETUP_FORK_ARGS[@]}"'
        )
        self.assertEqual(result.returncode, 0, msg=result.stderr)
        args = result.stdout.splitlines()
        self.assertIn("--fork-from-step", args)
        self.assertIn("data_split", args)
        self.assertIn("--fork-from-iteration", args)
        self.assertIn("2", args)

    def test_build_setup_fork_args_omits_optional_args_when_empty(self):
        result = self._run_bash(
            "build_setup_fork_args /src /tgt agent123; "
            'printf "%s\\n" "${SETUP_FORK_ARGS[@]}"'
        )
        self.assertEqual(result.returncode, 0, msg=result.stderr)
        args = result.stdout.splitlines()
        self.assertNotIn("--fork-from-step", args)
        self.assertNotIn("--fork-from-iteration", args)


if __name__ == "__main__":
    unittest.main()
