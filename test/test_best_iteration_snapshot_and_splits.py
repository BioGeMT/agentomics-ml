import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from agents.steps.data_split import DataSplitOutput, DataSplitStep
from agents.steps.validation_evaluation import ValidationEvaluationOutput, ValidationEvaluationStep
from runtime.best_iteration_snapshot import update_best_iteration_snapshot
from runtime.generate_final_reports import gather_iteration_inputs
from runtime.read_write_utils import (
    initialize_current_iteration_metadata,
    initialize_current_iteration_state,
    initialize_current_iteration_workspace,
    initialize_run_directories,
    load_iteration_state,
    save_config,
    update_current_iteration_state,
)
from runtime.filesystem import rewrite_symlinks_to_absolute, validate_symlinks_targets_in
from utils.config import Config


def _write_split_folder(split_path: Path, row_id: str = "row-1") -> None:
    (split_path / "input").mkdir(parents=True, exist_ok=True)
    (split_path / "input" / "examples.txt").write_text(f"{row_id}\n", encoding="utf-8")
    (split_path / "labels.csv").write_text(f"id,numeric_label\n{row_id},0\n", encoding="utf-8")


def _write_step_output(iteration_dir: Path, step_id: str, output) -> None:
    """Write a step output file to an iteration directory, mimicking the archived layout."""
    payload = output.model_dump() if hasattr(output, "model_dump") else output
    step_dir = iteration_dir / step_id
    step_dir.mkdir(parents=True, exist_ok=True)
    (step_dir / Config.STEP_OUTPUT_FILENAME).write_text(
        json.dumps({
            "step_id": step_id,
            "model_type": type(output).__name__,
            "payload": payload,
        }, indent=2),
        encoding="utf-8",
    )


class TestBestIterationSnapshot(unittest.TestCase):
    """update_best_iteration_snapshot must publish, clear, or skip based on is_new_best and split_changed."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        (self.root / "workspace").mkdir()
        self.config = self._make_config("best_iteration_snapshot_agent")

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
        return config

    def _create_iteration_with_validation(
        self,
        iteration: int,
        is_new_best: bool,
        split_changed: bool,
        split_version: int = 0,
    ) -> None:
        iteration_dir = self.config.iteration_dir(iteration)
        iteration_dir.mkdir(parents=True, exist_ok=True)
        (iteration_dir / Config.RUNTIME_INFO_DIRNAME).mkdir()
        (iteration_dir / Config.RUNTIME_INFO_DIRNAME / Config.ITERATION_METADATA_FILENAME).write_text(
            json.dumps({"iteration": iteration}), encoding="utf-8",
        )
        _write_step_output(iteration_dir, "data_split", DataSplitOutput(
            train_path=str(self.config.splits_dir / f"split_{split_version}" / "train"),
            val_path=str(self.config.splits_dir / f"split_{split_version}" / "validation"),
            mini_train_path=str(self.config.splits_dir / f"split_{split_version}" / "mini_train"),
            splitting_strategy="test",
            split_changed=split_changed,
        ))
        _write_step_output(iteration_dir, "validation_evaluation", ValidationEvaluationOutput(
            metrics={"validation/ACC": 0.9},
            is_new_best=is_new_best,
            status="success",
        ))

    def _seed_best_iteration_snapshot(self) -> None:
        best_iteration_snapshot_dir = self.config.best_iteration_snapshot_dir
        best_iteration_snapshot_dir.mkdir(parents=True, exist_ok=True)
        (best_iteration_snapshot_dir / Config.RUNTIME_INFO_DIRNAME).mkdir()
        (best_iteration_snapshot_dir / Config.RUNTIME_INFO_DIRNAME / Config.ITERATION_METADATA_FILENAME).write_text(
            json.dumps({"iteration": 0}), encoding="utf-8",
        )
        (best_iteration_snapshot_dir / "old_model.bin").write_text("old", encoding="utf-8")

    def test_new_best_publishes(self):
        self._create_iteration_with_validation(iteration=1, is_new_best=True, split_changed=False)
        iteration_dir = self.config.iteration_dir(1)
        (iteration_dir / "inference.py").write_text("print('ok')", encoding="utf-8")
        conda_env = self.config.shared_dir / ".conda" / "envs" / f"{self.config.agent_id}_env"
        conda_env.mkdir(parents=True, exist_ok=True)

        with patch("runtime.best_iteration_snapshot.export_environment_descriptor_to_path"):
            update_best_iteration_snapshot(self.config, iteration=1)

        self.assertTrue((self.config.best_iteration_snapshot_dir / "inference.py").exists())

    def test_split_changed_without_new_best_clears_best_iteration_snapshot(self):
        self._seed_best_iteration_snapshot()
        self._create_iteration_with_validation(
            iteration=1, is_new_best=False, split_changed=True, split_version=1,
        )

        update_best_iteration_snapshot(self.config, iteration=1)

        self.assertFalse(self.config.best_iteration_snapshot_dir.exists())

    def test_no_change_leaves_best_iteration_snapshot_untouched(self):
        self._seed_best_iteration_snapshot()
        self._create_iteration_with_validation(
            iteration=1, is_new_best=False, split_changed=False,
        )

        update_best_iteration_snapshot(self.config, iteration=1)

        self.assertTrue((self.config.best_iteration_snapshot_dir / "old_model.bin").exists())


class TestIsNewBest(unittest.TestCase):
    """ValidationEvaluationStep._is_new_best correctly compares metrics across split versions."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        (self.root / "workspace").mkdir()
        self.config = self._make_config("is_new_best_agent")

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
        initialize_current_iteration_metadata(config, iteration=1)
        initialize_current_iteration_state(config, started_at=100.0)
        return config

    def _setup_best_iteration(self, best_metrics: dict, split_version: int = 0) -> None:
        """Create a best-iteration snapshot and an archived winning iteration with the given metrics and split version."""
        best_dir = self.config.iteration_dir(0)
        best_dir.mkdir(parents=True, exist_ok=True)
        _write_step_output(best_dir, "data_split", DataSplitOutput(
            train_path=str(self.config.splits_dir / f"split_{split_version}" / "train"),
            val_path=str(self.config.splits_dir / f"split_{split_version}" / "validation"),
            mini_train_path=str(self.config.splits_dir / f"split_{split_version}" / "mini_train"),
            splitting_strategy="test", split_changed=False,
        ))
        _write_step_output(best_dir, "validation_evaluation", ValidationEvaluationOutput(
            metrics=best_metrics, is_new_best=True, status="success",
        ))
        best_iteration_snapshot_dir = self.config.best_iteration_snapshot_dir
        best_iteration_snapshot_dir.mkdir(parents=True, exist_ok=True)
        (best_iteration_snapshot_dir / Config.RUNTIME_INFO_DIRNAME).mkdir(exist_ok=True)
        (best_iteration_snapshot_dir / Config.RUNTIME_INFO_DIRNAME / Config.ITERATION_METADATA_FILENAME).write_text(
            json.dumps({"iteration": 0}), encoding="utf-8",
        )

    def _set_current_split_version(self, split_version: int) -> None:
        _write_step_output(self.config.current_iteration_dir, "data_split", DataSplitOutput(
            train_path=str(self.config.splits_dir / f"split_{split_version}" / "train"),
            val_path=str(self.config.splits_dir / f"split_{split_version}" / "validation"),
            mini_train_path=str(self.config.splits_dir / f"split_{split_version}" / "mini_train"),
            splitting_strategy="test", split_changed=split_version > 0,
            split_version=split_version,
        ))

    def test_better_score_on_same_split_is_new_best(self):
        self._setup_best_iteration({"validation/ACC": 0.8}, split_version=0)
        self._set_current_split_version(0)
        step = ValidationEvaluationStep(self.config, Mock(), Mock(), Mock(), [])

        self.assertTrue(step._is_new_best({"validation/ACC": 0.9}))

    def test_worse_score_on_same_split_is_not_new_best(self):
        self._setup_best_iteration({"validation/ACC": 0.8}, split_version=0)
        self._set_current_split_version(0)
        step = ValidationEvaluationStep(self.config, Mock(), Mock(), Mock(), [])

        self.assertFalse(step._is_new_best({"validation/ACC": 0.7}))

    def test_different_split_version_always_counts_as_new_best(self):
        self._setup_best_iteration({"validation/ACC": 0.99}, split_version=0)
        self._set_current_split_version(1)
        step = ValidationEvaluationStep(self.config, Mock(), Mock(), Mock(), [])

        # Even a much worse score is "new best" when splits differ (metrics aren't comparable)
        self.assertTrue(step._is_new_best({"validation/ACC": 0.5}))

    def test_no_existing_best_iteration_is_always_new_best(self):
        self._set_current_split_version(0)
        step = ValidationEvaluationStep(self.config, Mock(), Mock(), Mock(), [])

        self.assertTrue(step._is_new_best({"validation/ACC": 0.1}))

    def test_missing_validation_metric_is_never_new_best(self):
        self._setup_best_iteration({"validation/ACC": 0.8}, split_version=0)
        self._set_current_split_version(0)
        step = ValidationEvaluationStep(self.config, Mock(), Mock(), Mock(), [])

        # Only train metric present, no validation/ACC
        self.assertFalse(step._is_new_best({"train/ACC": 1.0}))


class TestDataSplitSkipReuse(unittest.TestCase):
    """DataSplitStep.should_run and build_skipped_output handle split budget and reuse."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        (self.root / "workspace").mkdir()
        self.config = self._make_config("split_agent")

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
            split_allowed_iterations=1,
        )
        initialize_run_directories(config)
        save_config(config)
        config.dataset_dir.mkdir(parents=True, exist_ok=True)
        _write_split_folder(config.dataset_dir / "train", row_id="train-row")
        return config

    def _create_split_dir(self, version: int) -> Path:
        split_dir = self.config.splits_dir / f"split_{version}"
        split_dir.mkdir(parents=True, exist_ok=True)
        _write_split_folder(split_dir / "train", row_id="train-row")
        _write_split_folder(split_dir / "validation", row_id="validation-row")
        return split_dir

    def _archive_iteration_with_split(self, iteration: int, split_version: int, status: str = "success") -> None:
        iteration_dir = self.config.iteration_dir(iteration)
        iteration_dir.mkdir(parents=True, exist_ok=True)
        (iteration_dir / Config.RUNTIME_INFO_DIRNAME).mkdir(parents=True)
        (iteration_dir / Config.RUNTIME_INFO_DIRNAME / "iteration_state.json").write_text(
            json.dumps({"status": status, "started_at": 100.0, "ended_at": 110.0}), encoding="utf-8",
        )
        _write_step_output(iteration_dir, "data_split", DataSplitOutput(
            train_path=str(self.config.splits_dir / f"split_{split_version}" / "train"),
            val_path=str(self.config.splits_dir / f"split_{split_version}" / "validation"),
            mini_train_path=str(self.config.splits_dir / f"split_{split_version}" / "mini_train"),
            splitting_strategy=f"strategy for split {split_version}",
            split_changed=False,
            split_version=split_version,
        ))

    def _make_step_for_iteration(self, iteration: int) -> DataSplitStep:
        initialize_current_iteration_workspace(self.config)
        initialize_current_iteration_metadata(self.config, iteration=iteration)
        initialize_current_iteration_state(self.config, started_at=100.0)
        return DataSplitStep(self.config, Mock(), Mock(), Mock(), [])

    def test_split_allowed_on_first_iteration(self):
        step = self._make_step_for_iteration(iteration=0)
        step.on_iteration_start(iteration=0)

        self.assertFalse(step.should_be_simulated())

    def test_split_blocked_after_budget_exhausted(self):
        self._create_split_dir(0)
        step = self._make_step_for_iteration(iteration=1)
        step.on_iteration_start(iteration=1)

        self.assertTrue(step.should_be_simulated())

    def test_mini_train_only_when_explicit_validation_exists(self):
        _write_split_folder(self.config.dataset_dir / "validation", row_id="validation-row")
        step = self._make_step_for_iteration(iteration=0)
        step.on_iteration_start(iteration=0)

        self.assertFalse(step.should_be_simulated())
        iteration_state = load_iteration_state(self.config.current_iteration_dir)
        self.assertTrue(iteration_state["only_mini_split_allowed_at_start"])
        self.assertFalse(iteration_state["full_split_allowed_at_start"])

    def test_split_simulated_when_explicit_validation_and_split_dir_exist(self):
        _write_split_folder(self.config.dataset_dir / "validation", row_id="validation-row")
        self._create_split_dir(0)
        step = self._make_step_for_iteration(iteration=1)
        step.on_iteration_start(iteration=1)

        self.assertTrue(step.should_be_simulated())
        iteration_state = load_iteration_state(self.config.current_iteration_dir)
        self.assertFalse(iteration_state["only_mini_split_allowed_at_start"])
        self.assertFalse(iteration_state["full_split_allowed_at_start"])

    def test_simulated_output_reuses_latest_split(self):
        self._create_split_dir(0)
        self._archive_iteration_with_split(iteration=0, split_version=0)
        initialize_current_iteration_workspace(self.config)
        initialize_current_iteration_metadata(self.config, iteration=1)
        initialize_current_iteration_state(self.config, started_at=100.0)
        step = DataSplitStep(self.config, Mock(), Mock(), Mock(), [])

        output = step.build_simulated_output()

        self.assertFalse(output.split_changed)
        self.assertIn("split_0", output.train_path)
        self.assertEqual(output.splitting_strategy, "strategy for split 0")

    def test_simulated_output_resolves_symlinked_train_val(self):
        split_dir = self.config.splits_dir / "split_0"
        split_dir.mkdir(parents=True)
        _write_split_folder(self.config.dataset_dir / "validation", row_id="val-row")
        os.symlink(self.config.dataset_dir / "train", split_dir / "train")
        os.symlink(self.config.dataset_dir / "validation", split_dir / "validation")
        _write_split_folder(split_dir / "mini_train", row_id="train-row")
        self._archive_iteration_with_split(iteration=0, split_version=0)
        step = self._make_step_for_iteration(iteration=1)

        output = step.build_simulated_output()

        self.assertTrue(Path(output.train_path).is_symlink())
        self.assertTrue((Path(output.train_path) / "labels.csv").exists())
        self.assertTrue((Path(output.val_path) / "labels.csv").exists())

    def test_simulated_output_fallback_creates_symlinks_not_copies(self):
        _write_split_folder(self.config.dataset_dir / "validation", row_id="val-row")
        _write_split_folder(self.config.dataset_dir / "mini_train", row_id="train-row")
        step = self._make_step_for_iteration(iteration=0)

        output = step.build_simulated_output()

        split_dir = Path(output.train_path).parent
        self.assertTrue((split_dir / "train").is_symlink())
        self.assertTrue((split_dir / "validation").is_symlink())
        self.assertTrue((split_dir / "mini_train").is_symlink())
        self.assertEqual((split_dir / "train").resolve(), (self.config.dataset_dir / "train").resolve())

    def test_latest_split_strategy_returns_last_successful_iteration_strategy(self):
        self._archive_iteration_with_split(iteration=0, split_version=0)
        self._archive_iteration_with_split(iteration=1, split_version=1)
        step = self._make_step_for_iteration(iteration=2)

        self.assertEqual("strategy for split 1", step._get_latest_split_strategy())


class TestFinalReportSplitLabels(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.config = Config(
            agent_id="report_agent",
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
        initialize_run_directories(self.config)
        save_config(self.config)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_iteration_inputs_use_iteration_split_version_labels(self):
        _write_split_folder(self.config.splits_dir / "split_0" / "train", row_id="old-train")
        _write_split_folder(self.config.splits_dir / "split_0" / "validation", row_id="old-validation")
        _write_split_folder(self.config.splits_dir / "split_1" / "train", row_id="new-train")
        _write_split_folder(self.config.splits_dir / "split_1" / "validation", row_id="new-validation")

        iteration_dir = self.config.iteration_dir(0)
        iteration_dir.mkdir(parents=True, exist_ok=True)
        _write_step_output(iteration_dir, "data_split", DataSplitOutput(
            train_path=str(self.config.splits_dir / "split_1" / "train"),
            val_path=str(self.config.splits_dir / "split_1" / "validation"),
            mini_train_path=str(self.config.splits_dir / "split_1" / "mini_train"),
            splitting_strategy="new split",
            split_changed=True,
            split_version=1,
        ))

        inputs = gather_iteration_inputs(
            config=self.config,
            iteration=0,
        )

        labels_by_split = {split.split_name: split.labeled_csv for split in inputs.splits}
        self.assertEqual(self.config.splits_dir / "split_1" / "train" / "labels.csv", labels_by_split["train"])
        self.assertEqual(
            self.config.splits_dir / "split_1" / "validation" / "labels.csv",
            labels_by_split["validation"],
        )

    def test_iteration_inputs_resolve_symlinked_splits(self):
        self.config.dataset_dir.mkdir(parents=True, exist_ok=True)
        _write_split_folder(self.config.dataset_dir / "train", row_id="ds-train")
        _write_split_folder(self.config.dataset_dir / "validation", row_id="ds-val")

        split_dir = self.config.splits_dir / "split_0"
        split_dir.mkdir(parents=True)
        os.symlink(self.config.dataset_dir / "train", split_dir / "train")
        os.symlink(self.config.dataset_dir / "validation", split_dir / "validation")
        _write_split_folder(split_dir / "mini_train", row_id="ds-train")

        iteration_dir = self.config.iteration_dir(0)
        iteration_dir.mkdir(parents=True, exist_ok=True)
        _write_step_output(iteration_dir, "data_split", DataSplitOutput(
            train_path=str(split_dir / "train"),
            val_path=str(split_dir / "validation"),
            mini_train_path=str(split_dir / "mini_train"),
            splitting_strategy="",
            split_changed=False,
            split_version=0,
        ))

        inputs = gather_iteration_inputs(
            config=self.config,
            iteration=0,
        )

        labels_by_split = {split.split_name: split.labeled_csv for split in inputs.splits}
        self.assertIsNotNone(labels_by_split["train"])
        self.assertIsNotNone(labels_by_split["validation"])
        self.assertTrue(labels_by_split["train"].exists())
        self.assertTrue(labels_by_split["validation"].exists())

    def test_iteration_inputs_include_test_artifacts_from_snapshot(self):
        # The best iteration snapshot points at iteration 0.
        snapshot_dir = self.config.best_iteration_snapshot_dir
        (snapshot_dir / Config.RUNTIME_INFO_DIRNAME).mkdir(parents=True, exist_ok=True)
        (snapshot_dir / Config.RUNTIME_INFO_DIRNAME / Config.ITERATION_METADATA_FILENAME).write_text(
            json.dumps({"iteration": 0}), encoding="utf-8"
        )
        # Artifacts written post-run by scripts/inference.sh into the snapshot.
        (snapshot_dir / "eval_predictions_test.csv").write_text(
            "id,prediction,probability_1\nt-0,1,0.9\n", encoding="utf-8"
        )
        (snapshot_dir / "eval_predictions_test.numeric_labels.csv").write_text(
            "id,numeric_label\nt-0,1\n", encoding="utf-8"
        )
        (snapshot_dir / "eval_predictions_test.metrics.json").write_text(
            json.dumps({"ACC": 1.0}), encoding="utf-8"
        )

        inputs = gather_iteration_inputs(config=self.config, iteration=0)

        test_split = next((s for s in inputs.splits if s.split_name == "test"), None)
        self.assertIsNotNone(test_split)
        self.assertEqual(snapshot_dir / "eval_predictions_test.numeric_labels.csv", test_split.labeled_csv)
        self.assertEqual(snapshot_dir / "eval_predictions_test.csv", test_split.preds_csv)
        self.assertEqual({"ACC": 1.0}, test_split.metrics)


class TestSplitSymlinkHelpers(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_rewrite_symlinks_to_absolute_rewrites_relative(self):
        target_file = self.root / "data" / "file.txt"
        target_file.parent.mkdir()
        target_file.write_text("content", encoding="utf-8")
        link_dir = self.root / "links"
        link_dir.mkdir()
        os.symlink("../data/file.txt", link_dir / "link.txt")

        self.assertFalse(os.readlink(link_dir / "link.txt").startswith("/"))

        rewrite_symlinks_to_absolute(link_dir)

        self.assertTrue(os.readlink(link_dir / "link.txt").startswith("/"))
        self.assertEqual((link_dir / "link.txt").resolve(), target_file.resolve())

    def test_validate_symlinks_targets_in_rejects_outside(self):
        allowed = self.root / "dataset"
        allowed.mkdir()
        (allowed / "ok.txt").write_text("ok", encoding="utf-8")
        outside = self.root / "secret"
        outside.mkdir()
        (outside / "bad.txt").write_text("bad", encoding="utf-8")
        split_dir = self.root / "split"
        split_dir.mkdir()
        os.symlink(allowed / "ok.txt", split_dir / "good_link")
        os.symlink(outside / "bad.txt", split_dir / "bad_link")

        with self.assertRaises(ValueError) as ctx:
            validate_symlinks_targets_in(split_dir, allowed)
        self.assertIn("bad_link", str(ctx.exception))

    def test_validate_symlinks_targets_in_accepts_all_inside(self):
        allowed = self.root / "dataset"
        allowed.mkdir()
        (allowed / "a.txt").write_text("a", encoding="utf-8")
        (allowed / "b.txt").write_text("b", encoding="utf-8")
        split_dir = self.root / "split"
        split_dir.mkdir()
        os.symlink(allowed / "a.txt", split_dir / "link_a")
        os.symlink(allowed / "b.txt", split_dir / "link_b")

        validate_symlinks_targets_in(split_dir, allowed)

    def test_validate_symlinks_targets_in_accepts_logical_prepared_target(self):
        raw_data = self.root / "raw" / "input"
        raw_data.mkdir(parents=True)
        (raw_data / "file.txt").write_text("content", encoding="utf-8")
        prepared = self.root / "prepared" / "dataset"
        prepared.mkdir(parents=True)
        os.symlink(raw_data, prepared / "input")
        split_dir = self.root / "split"
        split_dir.mkdir()
        os.symlink(prepared / "input" / "file.txt", split_dir / "link.txt")

        self.assertFalse((split_dir / "link.txt").resolve().is_relative_to(prepared.resolve()))

        validate_symlinks_targets_in(split_dir, prepared)

    def _make_step_with_dataset(self, dataset_dir: Path) -> DataSplitStep:
        config = Mock()
        config.dataset_dir = dataset_dir
        return DataSplitStep(config, Mock(), Mock(), Mock(), [])

    def test_raise_on_unnecessary_copies_detects_copied_file(self):
        dataset_dir = self.root / "dataset"
        (dataset_dir / "train" / "input").mkdir(parents=True)
        (dataset_dir / "train" / "input" / "file.txt").write_text("hello", encoding="utf-8")

        split = self.root / "split"
        (split / "input").mkdir(parents=True)
        (split / "input" / "file.txt").write_text("hello", encoding="utf-8")

        step = self._make_step_with_dataset(dataset_dir)
        from pydantic_ai import ModelRetry
        with self.assertRaises(ModelRetry):
            step._raise_on_unnecessary_copies(split)

    def test_raise_on_unnecessary_copies_allows_different_size(self):
        dataset_dir = self.root / "dataset"
        (dataset_dir / "train" / "input").mkdir(parents=True)
        (dataset_dir / "train" / "input" / "data.csv").write_text("a,b,c\n1,2,3\n4,5,6", encoding="utf-8")

        split = self.root / "split"
        (split / "input").mkdir(parents=True)
        (split / "input" / "data.csv").write_text("a,b,c\n1,2,3", encoding="utf-8")

        step = self._make_step_with_dataset(dataset_dir)
        step._raise_on_unnecessary_copies(split)

    def test_raise_on_unnecessary_copies_ignores_symlinks(self):
        dataset_dir = self.root / "dataset"
        (dataset_dir / "train" / "input").mkdir(parents=True)
        (dataset_dir / "train" / "input" / "file.txt").write_text("hello", encoding="utf-8")

        split = self.root / "split"
        (split / "input").mkdir(parents=True)
        os.symlink(dataset_dir / "train" / "input" / "file.txt", split / "input" / "file.txt")

        step = self._make_step_with_dataset(dataset_dir)
        step._raise_on_unnecessary_copies(split)


if __name__ == "__main__":
    unittest.main()
