from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from agentomics.datasets.data_contract import (
    read_split_samples,
    record_input_dir_structure,
    validate_generated_cells,
    validate_splits,
)
from agentomics.datasets.dataset_preparation import prepare_dataset
from agentomics.utils.metrics import get_default_val_metric, get_higher_is_better_map, resolve_val_metric
from agentomics.utils.task_types import TaskTypes, is_generation_task

ad = pytest.importorskip("anndata")
sparse = pytest.importorskip("scipy.sparse")

TASK = TaskTypes.TEMPORAL_GENERATION
N_GENES = 60


def _cells(stages: list[str], seed: int = 0, shift: float = 0.0, genes: int = N_GENES):
    """Two noisy cell states; shift moves half the genes up, so later stages differ from earlier ones."""
    rng = np.random.default_rng(seed)
    n = len(stages)
    base = rng.gamma(1.0, 1.0, size=(2, genes)) * (rng.random((2, genes)) > 0.3)
    state = rng.integers(0, 2, n)
    values = base[state] + rng.normal(0, 0.3, (n, genes))
    values[:, : genes // 2] += shift
    values = np.clip(values, 0, None) * (rng.random((n, genes)) > 0.4)
    adata = ad.AnnData(sparse.csr_matrix(values.astype(np.float32)))
    adata.obs = pd.DataFrame({"stage": stages}, index=[f"{s}:{i}" for i, s in enumerate(stages)])
    adata.var_names = [f"G{i}" for i in range(genes)]
    return adata


def _write_split(split: Path, observed: list[str], targets: dict[str, int], label_cells: int = 200) -> Path:
    (split / "input").mkdir(parents=True)
    _cells([stage for stage in observed for _ in range(label_cells)], seed=1).write_h5ad(split / "input" / "controls.h5ad")
    pd.DataFrame(
        [(stage, n) for stage, n in targets.items()], columns=["stage", "n_cells"]
    ).to_csv(split / "input" / "perturbations.csv", index=False)
    _cells([stage for stage in targets for _ in range(label_cells)], seed=2, shift=1.0).write_h5ad(split / "labels.h5ad")
    return split


def _validate(split: Path) -> None:
    validate_splits({"train": split}, record_input_dir_structure(split / "input"), TASK)


def test_task_type_is_a_generation_task():
    assert is_generation_task(TASK)
    assert is_generation_task(TaskTypes.CONDITIONAL_GENERATION)
    assert not is_generation_task(TaskTypes.REGRESSION)


def test_valid_split_passes_and_ids_are_stages(tmp_path):
    split = _write_split(tmp_path / "train", ["E8.5"], {"E9.5": 50, "E10.5": 50})
    _validate(split)
    assert sorted(read_split_samples(split, TASK)["id"]) == ["E10.5", "E9.5"]


def test_target_not_later_than_observed_is_rejected(tmp_path):
    split = _write_split(tmp_path / "train", ["E8.5", "E9.5"], {"E9.5": 50})
    with pytest.raises(ValueError, match="later than every observed stage"):
        _validate(split)


def test_malformed_stage_is_rejected(tmp_path):
    split = _write_split(tmp_path / "train", ["E8.5"], {"day10": 50})
    with pytest.raises(ValueError, match="must look like"):
        _validate(split)


def test_labels_must_match_target_stages(tmp_path):
    split = _write_split(tmp_path / "train", ["E8.5"], {"E9.5": 50})
    _cells(["E10.5"] * 10).write_h5ad(split / "labels.h5ad")
    with pytest.raises(ValueError, match="stages must match"):
        _validate(split)


def _write_prediction(path: Path, stage: str, n: int, scale: float = 1.0, genes: int = N_GENES):
    cells = _cells([stage] * n, seed=3, shift=1.0, genes=genes)
    cells.X = cells.X * scale
    cells.write_h5ad(path)
    return path


def test_generated_cells_validation(tmp_path):
    split = _write_split(tmp_path / "train", ["E8.5"], {"E9.5": 40})
    validate_generated_cells(_write_prediction(tmp_path / "ok.h5ad", "E9.5", 40), split / "input", TASK)
    with pytest.raises(ValueError, match="exactly its n_cells"):
        validate_generated_cells(_write_prediction(tmp_path / "few.h5ad", "E9.5", 39), split / "input", TASK)
    with pytest.raises(ValueError, match="looks like raw counts"):
        validate_generated_cells(_write_prediction(tmp_path / "raw.h5ad", "E9.5", 40, scale=1000), split / "input", TASK)
    with pytest.raises(ValueError, match="genes"):
        validate_generated_cells(_write_prediction(tmp_path / "genes.h5ad", "E9.5", 40, genes=10), split / "input", TASK)


def test_prepare_dataset_links_generation_labels(tmp_path):
    source = tmp_path / "source"
    _write_split(source / "train", ["E8.5"], {"E9.5": 50, "E10.5": 50})
    (source / "metadata.json").write_text('{"task_type": "temporal_generation"}')
    (source / "dataset_description.md").write_text("test")
    metadata = prepare_dataset(source, tmp_path / "prepared")
    assert metadata["task_type"] == TASK
    assert (tmp_path / "prepared" / "train" / "labels.h5ad").is_file()


def test_metrics_registry():
    assert get_default_val_metric(TASK) == "VE_SCORE"
    assert resolve_val_metric(TASK, "VE_MMD") == "VE_MMD"
    higher_is_better = get_higher_is_better_map()
    assert higher_is_better["VE_SCORE"] and not higher_is_better["VE_MMD"] and not higher_is_better["VE_VARIOGRAM"]


def test_score_anchors_copy_last_at_50_and_replicate_near_100(tmp_path, monkeypatch):
    from agentomics.utils import ve_t1_scoring

    monkeypatch.setattr(ve_t1_scoring, "ANCHOR_CACHE_DIR", tmp_path / "cache")
    reference = ve_t1_scoring._dense(_cells(["E8.5"] * 400, seed=1).X)
    truth_cells = ve_t1_scoring._dense(_cells(["E9.5"] * 800, seed=2, shift=1.0).X)
    stage = ve_t1_scoring.StageData(reference=reference, truth=truth_cells[:400])  # unsubsampled, small test data
    stage_anchors = ve_t1_scoring.anchors(stage)

    copy_last = ve_t1_scoring.score_stage(stage, reference, stage_anchors)
    replicate = ve_t1_scoring.score_stage(stage, truth_cells[400:], stage_anchors)
    average_cell = ve_t1_scoring.score_stage(
        stage, np.repeat(truth_cells[400:].mean(axis=0, keepdims=True), 400, axis=0), stage_anchors
    )
    assert copy_last["SCORE"] == pytest.approx(50.0)
    assert replicate["SCORE"] > 90
    assert average_cell["MMD_SKILL"] < 0.5
    assert copy_last["SCORE"] < replicate["SCORE"]


def test_skill_mapping():
    from agentomics.utils.ve_t1_scoring import skill

    assert skill("DE_SCORE", 0.0, 0.0, 0.8) == 0.5
    assert skill("DE_SCORE", 0.8, 0.0, 0.8) == 1.0
    assert skill("DE_SCORE", 2.0, 0.0, 0.8) == 1.0
    assert 0 < skill("DE_SCORE", -0.4, 0.0, 0.8) < 0.5
    # lower is better: the ceiling is below the floor
    assert skill("MMD", 0.04, 0.08, 0.0) == pytest.approx(0.75)
    assert 0 < skill("MMD", 0.2, 0.08, 0.0) < 0.5


def test_split_scoring_through_get_metrics(tmp_path, monkeypatch):
    from agentomics.runtime.evaluate_result import get_metrics
    from agentomics.utils import ve_t1_scoring

    monkeypatch.setattr(ve_t1_scoring, "ANCHOR_CACHE_DIR", tmp_path / "cache")
    split = _write_split(tmp_path / "validation", ["E8.5"], {"E9.5": 150})
    prediction = _write_prediction(tmp_path / "pred.h5ad", "E9.5", 150)
    metrics = get_metrics(prediction, split / "labels.h5ad", TASK)
    assert set(metrics) == {"VE_SCORE", "VE_DE_SCORE", "VE_DE_DIRECTION", "VE_MMD", "VE_VARIOGRAM"}
    assert 0 < metrics["VE_SCORE"] <= 100


def test_adversarial_controls_do_not_beat_copy_last():
    """The organisers' T1 controls: rescaled reference and a tiled mean cell must not score above the floor."""
    from agentomics.utils import ve_t1_scoring

    reference = ve_t1_scoring._dense(_cells(["E8.5"] * 400, seed=1).X)
    truth_cells = ve_t1_scoring._dense(_cells(["E9.5"] * 400, seed=2, shift=1.0).X)
    stage = ve_t1_scoring.StageData(reference=reference, truth=truth_cells)
    stage_anchors = ve_t1_scoring.anchors(stage)

    shrink = ve_t1_scoring.score_stage(stage, reference * 0.99, stage_anchors)
    assert shrink["DE_SCORE"] == pytest.approx(0.0) and shrink["DE_DIRECTION"] == pytest.approx(0.0)
    scale = ve_t1_scoring.score_stage(stage, reference * 2.0, stage_anchors)
    one_cell = ve_t1_scoring.score_stage(stage, np.repeat(reference.mean(axis=0, keepdims=True), 400, axis=0), stage_anchors)
    assert scale["SCORE"] < 50 and one_cell["SCORE"] < 50
