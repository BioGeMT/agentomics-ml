import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from agentomics.datasets.conditional_generation_contract import CONTROL_LABEL
from agentomics.runtime.conditional_generation_evaluation import write_reference
from agentomics.runtime.evaluate_result import get_metrics
from agentomics.utils.metrics import get_conditional_generation_metrics_names
from agentomics.utils.task_types import TaskTypes

ad = pytest.importorskip("anndata")

TARGETS = ["G0", "G1", "G2", "G3"]
N_GENES = 50
N_CELLS = 60


def _cells(rng, context, target, n_cells):
    means = np.full(N_GENES, 20.0)
    if target != CONTROL_LABEL:
        index = TARGETS.index(target)
        means[index] = 1.0
        means[(4 + 10 * index + np.arange(15)) % N_GENES] *= 3.0
    adata = ad.AnnData(rng.poisson(means, size=(n_cells, N_GENES)).astype(np.float32))
    adata.obs["context"] = context
    adata.obs["target_gene"] = target
    adata.var_names = [f"G{i}" for i in range(N_GENES)]
    return adata


def _write_split(split: Path, measured_contexts=("A", "B"), control_contexts=("A", "B"), seed=0) -> Path:
    """A split in the conditional generation layout, with a strong signal per target."""
    rng = np.random.default_rng(seed)
    (split / "input").mkdir(parents=True)
    ad.concat([_cells(rng, context, CONTROL_LABEL, 120) for context in control_contexts], index_unique="-").write_h5ad(
        split / "input" / "controls.h5ad"
    )
    conditions = [(context, target) for context in measured_contexts for target in TARGETS]
    pd.DataFrame(
        [(context, target, N_CELLS) for context, target in conditions],
        columns=["context", "target_gene", "n_cells"],
    ).to_csv(split / "input" / "perturbations.csv", index=False)
    pd.DataFrame(
        {"id": [f"{context}|{target}" for context, target in conditions], "numeric_label": N_CELLS}
    ).to_csv(split / "labels.csv", index=False)
    ad.concat([_cells(rng, context, target, N_CELLS) for context, target in conditions], index_unique="-").write_h5ad(
        split / "labels.h5ad"
    )
    return split


def test_reference_holds_measured_cells_and_only_their_contexts_controls(tmp_path):
    split = _write_split(tmp_path / "validation", measured_contexts=("A",), control_contexts=("A", "B"))

    write_reference(split, tmp_path / "reference.h5ad")

    reference = ad.read_h5ad(tmp_path / "reference.h5ad")
    assert list(reference.obs.columns) == ["context", "target_gene"]
    assert set(reference.obs["context"]) == {"A"}
    counts = reference.obs["target_gene"].value_counts()
    assert counts[CONTROL_LABEL] == 120
    assert all(counts[target] == N_CELLS for target in TARGETS)


@pytest.mark.skipif(not (Path(sys.executable).parent / "cell-eval2").is_file(), reason="cell-eval2 is not installed")
def test_get_metrics_scores_a_prediction_against_its_split(tmp_path, monkeypatch):
    from agentomics.utils import cell_eval2_scoring

    monkeypatch.setattr(cell_eval2_scoring, "BUNDLE_CACHE_DIR", tmp_path / "bundles")
    split = _write_split(tmp_path / "validation")
    prediction = _write_split(tmp_path / "generated", seed=1) / "labels.h5ad"

    metrics = get_metrics(
        results_file=prediction,
        test_file=split / "labels.csv",
        task_type=TaskTypes.CONDITIONAL_GENERATION,
    )

    assert list(metrics) == get_conditional_generation_metrics_names()
    assert all(math.isfinite(value) for value in metrics.values())
    assert metrics["AVG_SCORE"] > 0
