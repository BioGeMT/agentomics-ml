"""NaN handling on small subsets of a real dataset whose sources measured different gene panels.

GSE127199 combines four sources (two DE_TERA and two HE_Dropseq replicates), each with its own
measured-gene mask. A cell's NaNs must be exactly the genes its source did not measure, and scoring
a context must keep exactly the genes measured by every source in that context. The DE_TERA masks
are identical; the HE_Dropseq masks differ, and differ more from the DE_TERA ones.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from agentomics.datasets.conditional_generation_contract import CONTROL_LABEL, validate_generated_cells
from agentomics.datasets.data_contract import record_input_dir_structure, validate_splits
from agentomics.runtime.conditional_generation_evaluation import write_reference
from agentomics.utils import cell_eval2_scoring as scoring
from agentomics.utils.task_types import TaskTypes

ad = pytest.importorskip("anndata")

SOURCE = Path("/SCRATCH/dtzim01/vcc_run/data/heldout/GSE127199__H1.h5ad")
pytestmark = pytest.mark.skipif(not SOURCE.is_file(), reason=f"{SOURCE} is not available")

# context -> (sources, targets). Contexts never share cells, even when they share a source.
CONTEXTS = {
    "DE": (["DE_TERA_rep1", "DE_TERA_rep2"], ["FOXH1", "SMAD2"]),
    "HE": (["HE_Dropseq_rep1", "HE_Dropseq_rep2"], ["FOXA2"]),
    "MIXED": (["DE_TERA_rep1", "HE_Dropseq_rep2"], ["FOXA2"]),
}
HETEROGENEOUS_CONTEXTS = ["HE", "MIXED"]
N_CONTROLS_PER_SOURCE = 40
N_CELLS_PER_SOURCE = 25


def _subset() -> tuple[ad.AnnData, pd.DataFrame]:
    source = ad.read_h5ad(SOURCE, backed="r")
    try:
        masks = pd.DataFrame(
            np.asarray(source.varm["measured_by_source"]),
            index=source.var_names,
            columns=list(source.uns["measured_mask_ids"]),
        )
        obs = source.obs
        rows, contexts, used = [], [], {}
        for context, (sources, targets) in CONTEXTS.items():
            for mask_id in sources:
                in_source = obs["measured_mask_id"].astype(str) == mask_id
                for target, n_cells in [(CONTROL_LABEL, N_CONTROLS_PER_SOURCE)] + [(t, N_CELLS_PER_SOURCE) for t in targets]:
                    offset = used.get((mask_id, target), 0)
                    used[(mask_id, target)] = offset + n_cells
                    candidates = np.flatnonzero((in_source & (obs["target_gene"].astype(str) == target)).to_numpy())
                    chosen = candidates[offset:offset + n_cells]
                    assert len(chosen) == n_cells, f"{SOURCE.name} has too few {target} cells from {mask_id}"
                    rows.extend(chosen)
                    contexts.extend([context] * n_cells)
        order = np.argsort(rows)
        cells = source[np.asarray(rows)[order]].to_memory()
    finally:
        source.file.close()
    cells.obs = pd.DataFrame(
        {
            "context": np.asarray(contexts)[order],
            "target_gene": cells.obs["target_gene"].astype(str).to_numpy(),
            "measured_mask_id": cells.obs["measured_mask_id"].astype(str).to_numpy(),
        },
        index=cells.obs_names.astype(str),
    )
    cells.varm.clear()
    cells.uns.clear()
    return cells, masks


@pytest.fixture(scope="module")
def real_cells():
    return _subset()


def _measured_in_every_source(masks: pd.DataFrame, context: str) -> np.ndarray:
    return masks[CONTEXTS[context][0]].all(axis=1).to_numpy()


@pytest.fixture(scope="module")
def split(real_cells, tmp_path_factory) -> Path:
    cells, _ = real_cells
    split = tmp_path_factory.mktemp("real") / "validation"
    (split / "input").mkdir(parents=True)
    is_control = (cells.obs["target_gene"] == CONTROL_LABEL).to_numpy()
    with ad.settings.override(allow_write_nullable_strings=False):
        cells[is_control].write_h5ad(split / "input" / "controls.h5ad")
        cells[~is_control].write_h5ad(split / "labels.h5ad")
    perturbed = cells.obs[~is_control]
    perturbed.groupby(["context", "target_gene"]).size().rename("n_cells").reset_index().to_csv(
        split / "input" / "perturbations.csv", index=False
    )
    return split


def _generate(split: Path, sample) -> ad.AnnData:
    perturbations = pd.read_csv(split / "input" / "perturbations.csv")
    controls = ad.read_h5ad(split / "input" / "controls.h5ad")
    obs = perturbations.loc[perturbations.index.repeat(perturbations["n_cells"]), ["context", "target_gene"]]
    obs.index = [f"generated-{i}" for i in range(len(obs))]
    generated = ad.AnnData(sample(controls, obs), obs=obs.reset_index(drop=True).set_axis(obs.index))
    generated.var_names = controls.var_names
    return generated


def _poisson_from_control_means(controls, obs):
    """A valid model: per-context control means, with unavailable genes predicted from observed ones."""
    rng = np.random.default_rng(0)
    rows = []
    for context in obs["context"]:
        in_context = controls[(controls.obs["context"] == context).to_numpy()].X.toarray()
        means = np.nanmean(in_context, axis=0)
        means = np.where(np.isnan(means), np.nanmean(means), means)
        rows.append(rng.poisson(means))
    return sparse.csr_matrix(np.asarray(rows, dtype=np.float32))


def _copy_controls(controls, obs):
    """An invalid model: repeats control cells verbatim, including their NaNs."""
    rows = []
    for context in obs["context"]:
        rows.append(np.flatnonzero((controls.obs["context"] == context).to_numpy())[len(rows) % N_CONTROLS_PER_SOURCE])
    return sparse.csr_matrix(controls.X[rows])


def test_cell_nans_are_exactly_the_genes_its_source_did_not_measure(real_cells):
    cells, masks = real_cells
    matrix = sparse.csr_matrix(cells.X)
    assert np.isnan(matrix.data).any()
    for row, mask_id in enumerate(cells.obs["measured_mask_id"]):
        start, end = matrix.indptr[row], matrix.indptr[row + 1]
        unavailable = matrix.indices[start:end][np.isnan(matrix.data[start:end])]
        np.testing.assert_array_equal(np.sort(unavailable), np.flatnonzero(~masks[mask_id].to_numpy()))


@pytest.mark.parametrize("context", list(CONTEXTS))
def test_subset_covers_identical_and_differing_source_panels(real_cells, context):
    cells, masks = real_cells
    sources = CONTEXTS[context][0]
    assert (masks[sources[0]] != masks[sources[1]]).any() == (context in HETEROGENEOUS_CONTEXTS)
    assert cells.obs_names.is_unique


def test_real_split_with_missing_values_passes_the_contract(split):
    validate_splits({"validation": split}, record_input_dir_structure(split / "input"), TaskTypes.CONDITIONAL_GENERATION)


@pytest.mark.parametrize("context", list(CONTEXTS))
def test_scoring_genes_are_those_measured_by_every_source_of_the_context(real_cells, context):
    cells, masks = real_cells
    in_context = cells[(cells.obs["context"] == context).to_numpy()]
    keep = scoring._scoring_genes(in_context, context)
    np.testing.assert_array_equal(keep, _measured_in_every_source(masks, context))
    assert 0 < keep.sum() < len(keep)


def test_valid_prediction_on_the_full_panel_passes(split):
    generated = _generate(split, _poisson_from_control_means)
    generated.write_h5ad(split.parent / "generated.h5ad")
    validate_generated_cells(split.parent / "generated.h5ad", split / "input")


def test_prediction_copying_control_nans_is_rejected(split):
    generated = _generate(split, _copy_controls)
    assert np.isnan(generated.X.data).any()
    generated.write_h5ad(split.parent / "copied.h5ad")
    with pytest.raises(ValueError, match="finite, non-negative whole numbers"):
        validate_generated_cells(split.parent / "copied.h5ad", split / "input")
    with pytest.raises(ValueError, match="finite, non-negative whole counts"):
        scoring._validate_pair(ad.read_h5ad(split / "labels.h5ad"), generated)


def test_scorer_receives_only_measured_genes_for_each_context(split, real_cells, tmp_path, monkeypatch):
    _, masks = real_cells
    reference_path = tmp_path / "reference.h5ad"
    write_reference(split, reference_path)
    predictions_path = tmp_path / "predictions.h5ad"
    _generate(split, _poisson_from_control_means).write_h5ad(predictions_path)
    received = {}

    def record(reference, prediction):
        received[reference.obs["context"].iloc[0]] = (reference.copy(), prediction.copy())
        return {"avg_score": 0.0}

    monkeypatch.setattr(scoring, "_score_context", record)
    scoring._score(reference_path, predictions_path)

    labels = ad.read_h5ad(split / "labels.h5ad")
    for context, (sources, targets) in CONTEXTS.items():
        reference, prediction = received[context]
        expected_genes = masks.index[_measured_in_every_source(masks, context)].tolist()
        assert reference.var_names.tolist() == prediction.var_names.tolist() == expected_genes
        assert not np.isnan(reference.X.data).any()
        assert not np.isnan(prediction.X.data).any()
        n_controls = len(sources) * N_CONTROLS_PER_SOURCE
        n_perturbed = len(sources) * N_CELLS_PER_SOURCE * len(targets)
        assert reference.obs["target_gene"].value_counts().to_dict() == {
            CONTROL_LABEL: n_controls, **{target: len(sources) * N_CELLS_PER_SOURCE for target in targets}
        }
        assert (prediction.obs["target_gene"] == CONTROL_LABEL).sum() == n_controls
        assert len(prediction) == n_controls + n_perturbed
        # Observed values reach the scorer unchanged.
        measured = labels[(labels.obs["context"] == context).to_numpy(), expected_genes]
        perturbed = reference[(reference.obs["target_gene"] != CONTROL_LABEL).to_numpy()]
        np.testing.assert_array_equal(
            perturbed.X.toarray(), measured[perturbed.obs_names.str.removesuffix("-perturbed")].X.toarray()
        )
