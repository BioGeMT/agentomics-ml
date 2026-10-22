import math
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from agentomics.utils import cell_eval2_scoring
from agentomics.utils.metrics import get_conditional_generation_metrics_functions

ad = pytest.importorskip("anndata")
pytestmark = pytest.mark.skipif(
    not (Path(sys.executable).parent / "cell-eval2").is_file(),
    reason="cell-eval2 is not installed",
)

CONTEXTS = ["A", "B"]
TARGETS = ["G0", "G1", "G2", "G3"]
N_GENES = 50
N_CELLS = 60
N_CONTROLS = 120


def _make_cells(rng, context, target, n_cells):
    means = np.full(N_GENES, 20.0)
    if target != cell_eval2_scoring.CONTROL_LABEL:
        index = TARGETS.index(target)
        means[index] = 1.0
        shifted = 4 + 10 * index + np.arange(15)
        means[shifted % N_GENES] *= 3.0
    counts = rng.poisson(means, size=(n_cells, N_GENES)).astype(np.float32)
    adata = ad.AnnData(counts)
    adata.obs["context"] = context
    adata.obs["target_gene"] = target
    return adata


def _write_dataset(directory: Path, seed: int = 0) -> tuple[Path, Path]:
    rng = np.random.default_rng(seed)
    real_parts, pred_parts = [], []
    for context in CONTEXTS:
        real_parts.append(_make_cells(rng, context, cell_eval2_scoring.CONTROL_LABEL, N_CONTROLS))
        for target in TARGETS:
            real_parts.append(_make_cells(rng, context, target, N_CELLS))
            pred_parts.append(_make_cells(rng, context, target, N_CELLS))
    real = ad.concat(real_parts, index_unique="-")
    pred = ad.concat(pred_parts, index_unique="-")
    for adata in (real, pred):
        adata.var_names = [f"G{i}" for i in range(N_GENES)]
    real_path, pred_path = directory / "real.h5ad", directory / "pred.h5ad"
    real.write_h5ad(real_path)
    pred.write_h5ad(pred_path)
    return real_path, pred_path


# Scoring is slow (cell-eval2 runs differential expression), so the module shares one dataset
# and one real-bundle cache across tests.
@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    directory = tmp_path_factory.mktemp("conditional_generation")
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(cell_eval2_scoring, "BUNDLE_CACHE_DIR", directory / "bundles")
        yield _write_dataset(directory)


@pytest.fixture(autouse=True)
def empty_score_cache(monkeypatch):
    monkeypatch.setattr(cell_eval2_scoring, "_score_cache", {})


def _score_all(real_path, pred_path):
    return {
        name: metric(real_path, pred_path)
        for name, metric in get_conditional_generation_metrics_functions().items()
    }


def _write_variant(pred_path: Path, name: str, edit) -> Path:
    pred = ad.read_h5ad(pred_path)
    edit(pred)
    variant_path = pred_path.parent / name
    pred.write_h5ad(variant_path)
    return variant_path


def _shuffle_targets(pred):
    pred.obs["target_gene"] = np.random.default_rng(1).permutation(pred.obs["target_gene"].to_numpy())


def test_matching_prediction_scores_higher_than_shuffled(dataset):
    real_path, pred_path = dataset
    shuffled_path = _write_variant(pred_path, "shuffled.h5ad", _shuffle_targets)

    scores = _score_all(real_path, pred_path)
    shuffled_scores = _score_all(real_path, shuffled_path)

    assert all(math.isfinite(value) for value in scores.values())
    assert scores["AVG_SCORE"] > shuffled_scores["AVG_SCORE"]


def test_metrics_share_one_scoring_and_reuse_the_real_bundle(dataset, monkeypatch, tmp_path):
    real_path, pred_path = dataset
    monkeypatch.setattr(cell_eval2_scoring, "BUNDLE_CACHE_DIR", tmp_path / "bundles")
    other_pred_path = _write_variant(pred_path, "other_pred.h5ad", _shuffle_targets)

    with patch.object(
        cell_eval2_scoring, "_run_cell_eval2", wraps=cell_eval2_scoring._run_cell_eval2
    ) as run_cell_eval2:
        _score_all(real_path, pred_path)
        first_subcommands = [call.args[0] for call in run_cell_eval2.call_args_list]
        run_cell_eval2.reset_mock()
        _score_all(real_path, other_pred_path)
        second_subcommands = [call.args[0] for call in run_cell_eval2.call_args_list]

    assert first_subcommands.count("run") == len(CONTEXTS)
    assert first_subcommands.count("prep-real-bundle") == len(CONTEXTS)
    assert second_subcommands.count("run") == len(CONTEXTS)
    assert "prep-real-bundle" not in second_subcommands


def test_prediction_with_control_cells_is_rejected(dataset):
    real_path, pred_path = dataset

    def add_control_cell(pred):
        pred.obs["target_gene"] = pred.obs["target_gene"].astype(str)
        pred.obs.iloc[0, pred.obs.columns.get_loc("target_gene")] = cell_eval2_scoring.CONTROL_LABEL

    with_controls_path = _write_variant(pred_path, "with_controls.h5ad", add_control_cell)

    with pytest.raises(ValueError, match="control cells"):
        _score_all(real_path, with_controls_path)


def test_prediction_with_different_genes_is_rejected(dataset):
    real_path, pred_path = dataset

    def reverse_genes(pred):
        pred.var_names = list(reversed(pred.var_names))

    reversed_genes_path = _write_variant(pred_path, "reversed_genes.h5ad", reverse_genes)

    with pytest.raises(ValueError, match="var_names"):
        _score_all(real_path, reversed_genes_path)
