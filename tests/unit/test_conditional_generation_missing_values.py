"""Missing reference measurements (NaN) must never become observed zeros or hide invalid predictions."""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from agentomics.datasets.conditional_generation_contract import (
    CONTROL_LABEL,
    MAX_COUNTS_PER_CELL,
    _read_cells,
    validate_generated_cells,
)
from agentomics.datasets.data_contract import record_input_dir_structure, validate_splits
from agentomics.datasets.dataset_preparation import prepare_dataset
from agentomics.utils import cell_eval2_scoring as scoring
from agentomics.utils.task_types import TaskTypes

ad = pytest.importorskip("anndata")

STORAGES = ["dense", "csr", "csc"]
BAD_OBSERVED_VALUES = [np.inf, -np.inf, -1, 0.5, 2.25]
BAD_PREDICTED_VALUES = [np.nan, np.inf, -np.inf, -1, 0.5]


def cells(values, contexts, targets, storage="dense", dtype=np.float32):
    matrix = np.asarray(values, dtype=dtype)
    if storage == "csr":
        matrix = sparse.csr_matrix(matrix)
    elif storage == "csc":
        matrix = sparse.csc_matrix(matrix)
    return ad.AnnData(
        matrix,
        obs=pd.DataFrame({"context": contexts, "target_gene": targets}, index=[str(i) for i in range(len(contexts))]),
        var=pd.DataFrame(index=[f"G{i}" for i in range(matrix.shape[1])]),
    )


def write(adata, path: Path) -> Path:
    adata.write_h5ad(path)
    return path


def capture_score_context(monkeypatch, result=lambda context: {"avg_score": 0.0}):
    """Replaces cell-eval2 with a recorder of the matrices it would receive."""
    calls = {}

    def record(reference, prediction):
        context = reference.obs["context"].iloc[0]
        calls[context] = (reference.copy(), prediction.copy())
        return result(context)

    monkeypatch.setattr(scoring, "_score_context", record)
    return calls


def dense(adata) -> np.ndarray:
    return adata.X.toarray() if sparse.issparse(adata.X) else np.asarray(adata.X)


# --- Reference validation in the dataset contract ---------------------------------------------


@pytest.mark.parametrize("storage", STORAGES)
def test_reference_nans_pass_backed_validation(tmp_path, storage):
    real = cells([[0, np.nan, 2], [3, 1, 0]], ["A", "A"], [CONTROL_LABEL, "G0"], storage)
    obs, genes = _read_cells(write(real, tmp_path / "reference.h5ad"), validate_counts=True)
    assert genes == ["G0", "G1", "G2"]
    assert obs["target_gene"].tolist() == [CONTROL_LABEL, "G0"]


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int32, np.int64])
def test_reference_validation_accepts_any_numeric_count_dtype(tmp_path, dtype):
    real = cells([[0, 1], [3, 7]], ["A", "A"], [CONTROL_LABEL, "G0"], "csr", dtype=dtype)
    _read_cells(write(real, tmp_path / "reference.h5ad"), validate_counts=True)


def test_reference_validation_accepts_explicit_zeros_and_negative_zero(tmp_path):
    matrix = sparse.csr_matrix(
        (np.array([0.0, -0.0, np.nan, 4.0], dtype=np.float32), np.array([0, 1, 2, 0]), np.array([0, 3, 4])),
        shape=(2, 3),
    )
    real = cells(np.zeros((2, 3)), ["A", "A"], [CONTROL_LABEL, "G0"])
    real.X = matrix
    _read_cells(write(real, tmp_path / "reference.h5ad"), validate_counts=True)


def test_all_missing_reference_is_a_valid_file(tmp_path):
    # The file contract only describes values; scoring decides whether enough genes remain.
    real = cells(np.full((2, 2), np.nan), ["A", "A"], [CONTROL_LABEL, "G0"], "csr")
    _read_cells(write(real, tmp_path / "reference.h5ad"), validate_counts=True)


@pytest.mark.parametrize("storage", STORAGES)
@pytest.mark.parametrize("bad", BAD_OBSERVED_VALUES)
def test_bad_reference_value_is_rejected_even_in_a_column_with_nan(tmp_path, storage, bad):
    real = cells([[0, np.nan], [3, bad]], ["A", "A"], [CONTROL_LABEL, "G0"], storage)
    with pytest.raises(ValueError, match="invalid reference values"):
        _read_cells(write(real, tmp_path / "reference.h5ad"), validate_counts=True)


def test_reference_values_are_only_inspected_on_request(tmp_path):
    real = cells([[0, -1], [3, 0.5]], ["A", "A"], [CONTROL_LABEL, "G0"], "csr")
    _read_cells(write(real, tmp_path / "reference.h5ad"))


@pytest.mark.parametrize("storage", ["dense", "csr"])
def test_invalid_reference_value_after_the_first_block_names_its_rows(tmp_path, storage):
    matrix = np.ones((2100, 3), dtype=np.float32)
    matrix[:, 2] = np.nan
    matrix[2050, 1] = -1
    real = cells(matrix, ["A"] * 2100, ["G0"] * 2100, storage)
    with pytest.raises(ValueError, match=r"rows \[2048, 2100\)"):
        _read_cells(write(real, tmp_path / "reference.h5ad"), validate_counts=True)


GENES = ["G0", "G1", "G2", "G3"]
CONDITIONS = [("A", "G1"), ("A", "G2"), ("B", "G1")]
N_CELLS = 3


def _write_split(split: Path, controls_values=None, labels_values=None) -> Path:
    (split / "input").mkdir(parents=True)
    control_rows = [(context, CONTROL_LABEL) for context in ("A", "B") for _ in range(5)]
    label_rows = [condition for condition in CONDITIONS for _ in range(N_CELLS)]
    for rows, values, path in (
        (control_rows, controls_values, split / "input" / "controls.h5ad"),
        (label_rows, labels_values, split / "labels.h5ad"),
    ):
        matrix = np.ones((len(rows), len(GENES)), dtype=np.float32) if values is None else values
        contexts, targets = zip(*rows)
        adata = cells(matrix, list(contexts), list(targets), "csr")
        adata.var_names = GENES
        write(adata, path)
    pd.DataFrame(
        [(context, target, N_CELLS) for context, target in CONDITIONS],
        columns=["context", "target_gene", "n_cells"],
    ).to_csv(split / "input" / "perturbations.csv", index=False)
    return split


def _with_value(shape, row, column, value):
    matrix = np.ones(shape, dtype=np.float32)
    matrix[:, 3] = np.nan  # G3 is unavailable everywhere; G1 is unavailable in one cell.
    matrix[0, 1] = np.nan
    matrix[row, column] = value
    return matrix


def _validate(split: Path) -> None:
    validate_splits({"train": split}, record_input_dir_structure(split / "input"), TaskTypes.CONDITIONAL_GENERATION)


def test_split_with_missing_controls_and_responses_passes(tmp_path):
    split = _write_split(
        tmp_path / "train",
        controls_values=_with_value((10, 4), 0, 0, 0),
        labels_values=_with_value((9, 4), 0, 0, 2),
    )
    _validate(split)


@pytest.mark.parametrize("bad", BAD_OBSERVED_VALUES)
@pytest.mark.parametrize("broken_file", ["controls.h5ad", "labels.h5ad"])
def test_split_with_invalid_observed_value_is_rejected(tmp_path, bad, broken_file):
    controls = _with_value((10, 4), 9, 2, bad if broken_file == "controls.h5ad" else 1)
    labels = _with_value((9, 4), 8, 2, bad if broken_file == "labels.h5ad" else 1)
    split = _write_split(tmp_path / "train", controls_values=controls, labels_values=labels)
    with pytest.raises(ValueError, match=rf"{broken_file}.*invalid reference values"):
        _validate(split)


def test_preparation_accepts_missing_values_and_keeps_them(tmp_path):
    source = tmp_path / "source"
    controls, labels = _with_value((10, 4), 0, 0, 0), _with_value((9, 4), 0, 0, 2)
    for split_name in ("train", "validation"):
        _write_split(source / split_name, controls_values=controls, labels_values=labels)
    (source / "metadata.json").write_text('{"task_type": "conditional_generation"}', encoding="utf-8")

    prepare_dataset(source, tmp_path / "prepared")

    for split_name in ("train", "validation"):
        prepared = tmp_path / "prepared" / split_name
        np.testing.assert_array_equal(dense(ad.read_h5ad(prepared / "labels.h5ad")), labels)
        np.testing.assert_array_equal(dense(ad.read_h5ad(prepared / "input" / "controls.h5ad")), controls)


# --- Prediction validation ------------------------------------------------------------------------


def _write_generated(path: Path, values) -> Path:
    rows = [condition for condition in CONDITIONS for _ in range(N_CELLS)]
    contexts, targets = zip(*rows)
    generated = cells(np.full((len(rows), len(GENES)), values, dtype=np.float32), list(contexts), list(targets), "csr")
    generated.var_names = GENES
    return write(generated, path)


def test_complete_prediction_passes_when_input_controls_have_missing_values(tmp_path):
    split = _write_split(tmp_path / "mini_train", controls_values=_with_value((10, 4), 0, 0, 0))
    validate_generated_cells(_write_generated(tmp_path / "generated.h5ad", 2.0), split / "input")


@pytest.mark.parametrize("bad", BAD_PREDICTED_VALUES)
def test_generated_cells_cannot_copy_missing_values_from_the_controls(tmp_path, bad):
    split = _write_split(tmp_path / "mini_train", controls_values=_with_value((10, 4), 0, 0, 0))
    with pytest.raises(ValueError, match="finite, non-negative whole numbers"):
        validate_generated_cells(_write_generated(tmp_path / "generated.h5ad", bad), split / "input")


@pytest.mark.parametrize("storage", STORAGES)
@pytest.mark.parametrize("bad", BAD_PREDICTED_VALUES)
def test_invalid_prediction_cannot_hide_in_an_excluded_gene(storage, bad):
    real = cells([[1, np.nan], [2, np.nan]], ["A", "A"], [CONTROL_LABEL, "G0"])
    pred = cells([[1, bad]], ["A"], ["G0"], storage)
    with pytest.raises(ValueError, match="finite, non-negative whole counts"):
        scoring._validate_pair(real, pred)


@pytest.mark.parametrize("storage", STORAGES)
def test_prediction_total_counts_include_excluded_genes(storage):
    real = cells([[1, np.nan], [2, np.nan]], ["A", "A"], [CONTROL_LABEL, "G0"])
    pred = cells([[1, MAX_COUNTS_PER_CELL]], ["A"], ["G0"], storage)
    with pytest.raises(ValueError, match="must not total more than"):
        scoring._validate_pair(real, pred)


@pytest.mark.parametrize("storage", ["dense", "csr"])
def test_invalid_prediction_after_the_first_block_names_its_rows(storage):
    real = cells([[1, 1], [2, 2]], ["A", "A"], [CONTROL_LABEL, "G0"])
    matrix = np.ones((1500, 2), dtype=np.float32)
    matrix[1400, 0] = np.nan
    pred = cells(matrix, ["A"] * 1500, ["G0"] * 1500, storage)
    with pytest.raises(ValueError, match=r"pred rows \[1024, 1500\)"):
        scoring._validate_pair(real, pred)


@pytest.mark.parametrize("storage", STORAGES)
def test_valid_prediction_passes_when_the_reference_has_missing_values(storage):
    real = cells([[1, np.nan], [2, 0]], ["A", "A"], [CONTROL_LABEL, "G0"], storage)
    pred = cells([[0, 7], [3, 0]], ["A", "A"], ["G0", "G0"], storage)
    scoring._validate_pair(real, pred)


def test_invalid_prediction_is_rejected_before_any_scoring(tmp_path, monkeypatch):
    real = cells([[1, np.nan], [2, np.nan]], ["A", "A"], [CONTROL_LABEL, "G0"])
    pred = cells([[1, np.nan]], ["A"], ["G0"], "csr")
    calls = capture_score_context(monkeypatch)
    with pytest.raises(ValueError, match="pred rows"):
        scoring._score(write(real, tmp_path / "real.h5ad"), write(pred, tmp_path / "pred.h5ad"))
    assert calls == {}


# --- Choosing the scoring genes -------------------------------------------------------------------


@pytest.mark.parametrize("storage", STORAGES)
def test_genes_with_nan_are_excluded_and_zeros_remain_observed(storage):
    real = cells([[0, np.nan, 2, 0], [3, 1, 0, 0]], ["A", "A"], [CONTROL_LABEL, "G0"], storage)
    keep = scoring._scoring_genes(real, "A")
    assert keep.dtype == bool
    np.testing.assert_array_equal(keep, [True, False, True, True])


def test_explicitly_stored_sparse_zeros_remain_observed():
    real = cells(np.zeros((2, 3)), ["A", "A"], [CONTROL_LABEL, "G0"])
    real.X = sparse.csr_matrix(
        (np.array([0.0, 0.0, np.nan], dtype=np.float32), np.array([0, 1, 2]), np.array([0, 2, 3])),
        shape=(2, 3),
    )
    np.testing.assert_array_equal(scoring._scoring_genes(real, "A"), [True, True, False])


@pytest.mark.parametrize("storage", STORAGES)
@pytest.mark.parametrize("missing_row", [0, 1], ids=["control", "perturbed"])
def test_a_single_missing_cell_excludes_the_gene(storage, missing_row):
    matrix = np.ones((2, 3), dtype=np.float32)
    matrix[missing_row, 1] = np.nan
    real = cells(matrix, ["A", "A"], [CONTROL_LABEL, "G0"], storage)
    np.testing.assert_array_equal(scoring._scoring_genes(real, "A"), [True, False, True])


@pytest.mark.parametrize("storage", STORAGES)
def test_complete_reference_keeps_every_gene(storage):
    real = cells([[0, 4, 2], [3, 1, 0]], ["A", "A"], [CONTROL_LABEL, "G0"], storage)
    assert scoring._scoring_genes(real, "A").all()


@pytest.mark.parametrize("dtype", [np.int32, np.int64])
def test_integer_reference_keeps_every_gene(dtype):
    real = cells([[0, 4], [3, 1]], ["A", "A"], [CONTROL_LABEL, "G0"], "csr", dtype=dtype)
    assert scoring._scoring_genes(real, "A").all()


@pytest.mark.parametrize("storage", STORAGES)
def test_missingness_after_the_first_block_is_detected(storage):
    matrix = np.ones((3000, 4), dtype=np.float32)
    matrix[1025, 1] = np.nan
    matrix[-1, 3] = np.nan
    real = cells(matrix, ["A"] * 3000, ["G0"] * 3000, storage)
    np.testing.assert_array_equal(scoring._scoring_genes(real, "A"), [True, False, True, False])


@pytest.mark.parametrize(
    "matrix",
    [[[np.nan, np.nan], [np.nan, np.nan]], [[np.nan, 1], [1, np.nan]]],
    ids=["all_missing", "every_gene_missing_somewhere"],
)
@pytest.mark.parametrize("storage", STORAGES)
def test_empty_common_gene_panel_is_rejected(matrix, storage):
    real = cells(matrix, ["A", "A"], [CONTROL_LABEL, "G0"], storage)
    with pytest.raises(ValueError, match="Context 'A' has no genes observed"):
        scoring._scoring_genes(real, "A")


@pytest.mark.parametrize("storage", STORAGES)
@pytest.mark.parametrize("bad", BAD_OBSERVED_VALUES)
def test_invalid_observed_reference_value_is_rejected_even_in_an_excluded_gene(storage, bad):
    real = cells([[0, np.nan], [3, bad]], ["A", "A"], [CONTROL_LABEL, "G0"], storage)
    with pytest.raises(ValueError, match="Reference context 'A' rows"):
        scoring._scoring_genes(real, "A")


# --- Scoring each context on its own panel --------------------------------------------------------


@pytest.mark.parametrize("real_storage", STORAGES)
@pytest.mark.parametrize("pred_storage", STORAGES)
def test_each_context_filters_reference_predictions_and_controls_before_scoring(
    tmp_path, monkeypatch, real_storage, pred_storage,
):
    real = cells(
        [[10, 0, np.nan], [4, 3, 9], [5, np.nan, 0], [2, 7, 6]],
        ["A", "A", "B", "B"], [CONTROL_LABEL, "G0"] * 2, real_storage,
    )
    pred = cells([[4, 3, 100], [2, 200, 6]], ["A", "B"], ["G0", "G0"], pred_storage)
    calls = capture_score_context(monkeypatch, lambda context: {"avg_score": 0.2 if context == "A" else 0.8})

    scores = scoring._score(write(real, tmp_path / "real.h5ad"), write(pred, tmp_path / "pred.h5ad"))

    assert scores == pytest.approx({"avg_score": 0.5})
    assert sorted(calls) == ["A", "B"]
    for context, genes, observed_control, generated in (
        ("A", ["G0", "G1"], [10, 0], [4, 3]),
        ("B", ["G0", "G2"], [5, 0], [2, 6]),
    ):
        reference, prediction = calls[context]
        assert reference.var_names.tolist() == prediction.var_names.tolist() == genes
        assert not np.isnan(dense(reference)).any()
        assert not np.isnan(dense(prediction)).any()
        assert set(reference.obs["context"]) == set(prediction.obs["context"]) == {context}
        assert prediction.obs["target_gene"].tolist() == ["G0", CONTROL_LABEL]
        np.testing.assert_array_equal(dense(prediction)[0], generated)
        # The appended controls are the filtered real controls of the same context.
        np.testing.assert_array_equal(dense(prediction)[1], observed_control)
        np.testing.assert_array_equal(dense(reference)[0], observed_control)


def test_missingness_in_one_context_does_not_exclude_the_gene_elsewhere(tmp_path, monkeypatch):
    real = cells(
        [[1, np.nan, 1], [1, 1, 1], [2, 2, 2], [3, 3, 3]],
        ["A", "A", "B", "B"], [CONTROL_LABEL, "G0"] * 2, "csr",
    )
    pred = cells([[1, 1, 1], [1, 1, 1]], ["A", "B"], ["G0", "G0"], "csr")
    calls = capture_score_context(monkeypatch)

    scoring._score(write(real, tmp_path / "real.h5ad"), write(pred, tmp_path / "pred.h5ad"))

    assert calls["A"][0].var_names.tolist() == ["G0", "G2"]
    assert calls["B"][0].var_names.tolist() == ["G0", "G1", "G2"]


def test_complete_reference_scores_the_full_panel(tmp_path, monkeypatch):
    real = cells([[1, 0, 1], [1, 2, 1]], ["A", "A"], [CONTROL_LABEL, "G0"], "csr")
    pred = cells([[1, 5, 1]], ["A"], ["G0"], "csr")
    calls = capture_score_context(monkeypatch)

    scoring._score(write(real, tmp_path / "real.h5ad"), write(pred, tmp_path / "pred.h5ad"))

    reference, prediction = calls["A"]
    assert reference.var_names.tolist() == prediction.var_names.tolist() == ["G0", "G1", "G2"]
    np.testing.assert_array_equal(dense(reference), [[1, 0, 1], [1, 2, 1]])


def test_kept_genes_keep_their_order_and_their_cells_keep_their_values(tmp_path, monkeypatch):
    rng = np.random.default_rng(0)
    n_genes = 40
    real_matrix = rng.poisson(5, size=(30, n_genes)).astype(np.float32)
    missing_genes = [3, 4, 17, 39]
    for gene in missing_genes:
        real_matrix[rng.integers(30), gene] = np.nan
    targets = [CONTROL_LABEL] * 10 + ["G1"] * 10 + ["G2"] * 10
    real = cells(real_matrix, ["A"] * 30, targets, "csr")
    pred_matrix = rng.poisson(5, size=(20, n_genes)).astype(np.float32)
    pred = cells(pred_matrix, ["A"] * 20, ["G1"] * 10 + ["G2"] * 10, "csr")
    calls = capture_score_context(monkeypatch)

    scoring._score(write(real, tmp_path / "real.h5ad"), write(pred, tmp_path / "pred.h5ad"))

    kept = [gene for gene in range(n_genes) if gene not in missing_genes]
    reference, prediction = calls["A"]
    assert reference.var_names.tolist() == [f"G{gene}" for gene in kept]
    np.testing.assert_array_equal(dense(reference), real_matrix[:, kept])
    np.testing.assert_array_equal(dense(prediction)[:20], pred_matrix[:, kept])
    np.testing.assert_array_equal(dense(prediction)[20:], real_matrix[:10, kept])
    assert reference.obs["target_gene"].tolist() == targets


def test_scores_average_every_metric_over_contexts(tmp_path, monkeypatch):
    real = cells(
        [[1, np.nan], [1, 1], [1, 1], [1, 1], [np.nan, 1], [1, 1]],
        ["A", "A", "B", "B", "C", "C"], [CONTROL_LABEL, "G0"] * 3, "csr",
    )
    pred = cells([[1, 1]] * 3, ["A", "B", "C"], ["G0"] * 3, "csr")
    values = {"A": 0.0, "B": 0.3, "C": 0.9}
    capture_score_context(monkeypatch, lambda context: {"avg_score": values[context], "pds": 2 * values[context]})

    scores = scoring._score(write(real, tmp_path / "real.h5ad"), write(pred, tmp_path / "pred.h5ad"))

    assert scores == pytest.approx({"avg_score": 0.4, "pds": 0.8})


def test_scoring_does_not_modify_the_input_files(tmp_path, monkeypatch):
    real_matrix = np.array([[10, 0, np.nan], [4, 3, 9]], dtype=np.float32)
    pred_matrix = np.array([[4, 3, 100]], dtype=np.float32)
    real_path = write(cells(real_matrix, ["A", "A"], [CONTROL_LABEL, "G0"], "csr"), tmp_path / "real.h5ad")
    pred_path = write(cells(pred_matrix, ["A"], ["G0"], "csr"), tmp_path / "pred.h5ad")
    capture_score_context(monkeypatch)

    scoring._score(real_path, pred_path)

    np.testing.assert_array_equal(dense(ad.read_h5ad(real_path)), real_matrix)
    np.testing.assert_array_equal(dense(ad.read_h5ad(pred_path)), pred_matrix)


def test_context_without_observed_genes_fails_before_scoring(tmp_path, monkeypatch):
    real = cells(
        [[1, 1], [1, 1], [np.nan, 1], [1, np.nan]],
        ["A", "A", "B", "B"], [CONTROL_LABEL, "G0"] * 2, "csr",
    )
    pred = cells([[1, 1], [1, 1]], ["A", "B"], ["G0", "G0"], "csr")
    calls = capture_score_context(monkeypatch)
    with pytest.raises(ValueError, match="Context 'B' has no genes observed"):
        scoring._score(write(real, tmp_path / "real.h5ad"), write(pred, tmp_path / "pred.h5ad"))
    assert list(calls) == ["A"]


def test_scorer_failure_names_its_context(tmp_path, monkeypatch):
    real = cells([[1, np.nan], [1, 1]], ["A", "A"], [CONTROL_LABEL, "G0"], "csr")
    pred = cells([[1, 1]], ["A"], ["G0"], "csr")

    def fail(reference, prediction):
        raise RuntimeError("cell-eval2 run failed")

    monkeypatch.setattr(scoring, "_score_context", fail)
    with pytest.raises(RuntimeError, match="Scoring context 'A' failed: cell-eval2 run failed"):
        scoring._score(write(real, tmp_path / "real.h5ad"), write(pred, tmp_path / "pred.h5ad"))


def test_unscorable_context_is_left_out_of_the_average(tmp_path, monkeypatch, capsys):
    real = cells([[1, 1], [1, 1]] * 3, ["A", "A", "B", "B", "C", "C"], [CONTROL_LABEL, "G0"] * 3, "csr")
    pred = cells([[1, 1]] * 3, ["A", "B", "C"], ["G0"] * 3, "csr")
    values = {"A": 0.2, "C": 0.6}

    def score_context(reference, prediction):
        context = reference.obs["context"].iloc[0]
        if context == "B":
            raise scoring.UnscorableContext("metric 'pds_cosine' has no usable replicate scale")
        return {"avg_score": values[context]}

    monkeypatch.setattr(scoring, "_score_context", score_context)
    scores = scoring._score(write(real, tmp_path / "real.h5ad"), write(pred, tmp_path / "pred.h5ad"))

    assert scores == pytest.approx({"avg_score": 0.4})
    output = capsys.readouterr().out
    assert "[context B]: skipped, left out of the average: metric 'pds_cosine'" in output
    assert "averaged over 2 contexts; skipped ['B']" in output


def test_single_cell_perturbations_are_left_out_on_both_sides(capsys):
    real = cells(
        [[1, 1], [2, 2], [2, 2], [3, 3], [4, 4], [4, 4]],
        ["A"] * 6, [CONTROL_LABEL, "G0", "G0", "G1", "G2", "G2"], "csr",
    )
    pred = cells([[2, 2], [2, 2], [5, 5], [4, 4], [4, 4], [1, 1]], ["A"] * 6, ["G0", "G0", "G1", "G2", "G2", CONTROL_LABEL], "csr")

    real, pred = scoring._without_unsplittable_perturbations(real, pred)

    assert real.obs["target_gene"].tolist() == [CONTROL_LABEL, "G0", "G0", "G2", "G2"]
    assert pred.obs["target_gene"].tolist() == ["G0", "G0", "G2", "G2", CONTROL_LABEL]
    assert 5 not in dense(pred)
    assert "[context A]: left out 1 perturbations with fewer than 2 measured cells: ['G1']" in capsys.readouterr().out


def test_complete_perturbations_are_kept_unchanged():
    real = cells([[1, 1], [2, 2], [2, 2], [3, 3], [3, 3]], ["A"] * 5, [CONTROL_LABEL, "G0", "G0", "G1", "G1"], "csr")
    pred = cells([[2, 2], [2, 2]], ["A"] * 2, ["G0", "G1"], "csr")

    assert scoring._without_unsplittable_perturbations(real, pred) == (real, pred)


def test_context_with_too_few_splittable_perturbations_is_unscorable():
    real = cells([[1, 1], [2, 2], [2, 2], [3, 3]], ["A"] * 4, [CONTROL_LABEL, "G0", "G0", "G1"], "csr")
    pred = cells([[2, 2], [2, 2], [3, 3]], ["A"] * 3, ["G0", "G0", "G1"], "csr")

    with pytest.raises(scoring.UnscorableContext, match="fewer than 2 perturbations with at least 2 measured cells"):
        scoring._without_unsplittable_perturbations(real, pred)


def test_context_with_a_single_perturbation_is_unscorable_even_when_nothing_is_dropped():
    # GSE127199 hepatic endoderm: one target (FOXA2) with many cells, so no perturbation is dropped.
    real = cells([[1, 1], [2, 2], [2, 2]], ["A"] * 3, [CONTROL_LABEL, "G0", "G0"], "csr")
    pred = cells([[2, 2], [2, 2]], ["A"] * 2, ["G0", "G0"], "csr")

    with pytest.raises(scoring.UnscorableContext, match="fewer than 2 perturbations with at least 2 measured cells"):
        scoring._without_unsplittable_perturbations(real, pred)


def test_single_perturbation_context_is_left_out_of_the_average(tmp_path, monkeypatch):
    rows = [("A", CONTROL_LABEL), ("A", "G0"), ("A", "G0"), ("A", "G1"), ("A", "G1")]
    rows += [("B", CONTROL_LABEL), ("B", "G0"), ("B", "G0")]
    contexts, targets = zip(*rows)
    real = cells(np.ones((len(rows), 2)), list(contexts), list(targets), "csr")
    pred = cells(np.ones((6, 2)), ["A"] * 4 + ["B"] * 2, ["G0", "G0", "G1", "G1", "G0", "G0"], "csr")
    monkeypatch.setattr(scoring, "_get_real_bundle", lambda real_context, path: tmp_path)
    monkeypatch.setattr(scoring, "_run_cell_eval2", lambda *arguments: None)
    monkeypatch.setattr(scoring, "_cell_eval2_flags", lambda: scoring.CELL_EVAL2_FLAGS)
    monkeypatch.setattr(pd, "read_csv", lambda path: pd.DataFrame({"metric": ["avg_score"], "from_replicate": [0.5]}))

    assert scoring._score(write(real, tmp_path / "real.h5ad"), write(pred, tmp_path / "pred.h5ad")) == {"avg_score": 0.5}


def test_scoring_fails_when_no_context_is_scorable(tmp_path, monkeypatch):
    real = cells([[1, 1], [1, 1]], ["A", "A"], [CONTROL_LABEL, "G0"], "csr")
    pred = cells([[1, 1]], ["A"], ["G0"], "csr")

    def unscorable(reference, prediction):
        raise scoring.UnscorableContext("no usable replicate scale")

    monkeypatch.setattr(scoring, "_score_context", unscorable)
    with pytest.raises(RuntimeError, match="No context could be scored: {'A': 'no usable replicate scale'}"):
        scoring._score(write(real, tmp_path / "real.h5ad"), write(pred, tmp_path / "pred.h5ad"))


@pytest.mark.parametrize("failure", ["invalid_prediction", "empty_panel", "scorer"])
def test_reference_file_is_closed_after_a_failure(tmp_path, monkeypatch, failure):
    real_matrix = [[np.nan, np.nan], [np.nan, np.nan]] if failure == "empty_panel" else [[1, np.nan], [1, 1]]
    pred_value = np.inf if failure == "invalid_prediction" else 1
    real_path = write(cells(real_matrix, ["A", "A"], [CONTROL_LABEL, "G0"], "csr"), tmp_path / "real.h5ad")
    pred_path = write(cells([[1, pred_value]], ["A"], ["G0"], "csr"), tmp_path / "pred.h5ad")

    def fail(reference, prediction):
        raise RuntimeError("cell-eval2 run failed")

    opened = []
    read_h5ad = ad.read_h5ad

    def spy(path, *args, **kwargs):
        adata = read_h5ad(path, *args, **kwargs)
        if kwargs.get("backed"):
            opened.append(adata)
        return adata

    monkeypatch.setattr(scoring, "_score_context", fail)
    monkeypatch.setattr(ad, "read_h5ad", spy)
    with pytest.raises((ValueError, RuntimeError)):
        scoring._score(real_path, pred_path)
    assert len(opened) == 1 and not opened[0].file.is_open


def test_score_cache_tracks_rewritten_files(tmp_path, monkeypatch):
    monkeypatch.setattr(scoring, "_score_cache", {})
    real = cells([[1, np.nan], [1, 1]], ["A", "A"], [CONTROL_LABEL, "G0"], "csr")
    real_path = write(real, tmp_path / "real.h5ad")
    pred_path = write(cells([[1, 1]], ["A"], ["G0"], "csr"), tmp_path / "pred.h5ad")
    calls = []
    monkeypatch.setattr(scoring, "_score_context", lambda reference, prediction: calls.append(1) or {"avg_score": 1.0})

    scoring.score_vcc2026(real_path, pred_path)
    scoring.score_vcc2026(real_path, pred_path)
    assert len(calls) == 1
    write(cells([[1, 2, 3]], ["A"], ["G0"], "csr")[:, :2], pred_path)
    scoring.score_vcc2026(real_path, pred_path)
    assert len(calls) == 2


# --- Real bundle cache ----------------------------------------------------------------------------


@pytest.fixture
def fake_cell_eval2(tmp_path, monkeypatch):
    """Stands in for the cell-eval2 subcommands that build a real bundle."""
    monkeypatch.setattr(scoring, "BUNDLE_CACHE_DIR", tmp_path / "bundles")
    monkeypatch.setattr(scoring, "_cell_eval2_flags", lambda: scoring.CELL_EVAL2_FLAGS)
    subcommands = []

    def run(subcommand, *arguments):
        subcommands.append(subcommand)
        output = Path(arguments[list(arguments).index("-o") + 1])
        output.mkdir(parents=True)
        if subcommand == "baseline":
            prediction = Path(arguments[list(arguments).index("--save-pred") + 1])
            prediction.write_text("baseline")

    monkeypatch.setattr(scoring, "_run_cell_eval2", run)
    return subcommands


def test_bundle_records_the_filtered_gene_panel(tmp_path, fake_cell_eval2):
    real = cells([[1, np.nan, 1], [1, 1, 1]], ["A", "A"], [CONTROL_LABEL, "G0"], "csr")
    filtered = real[:, scoring._scoring_genes(real, "A")].copy()

    bundle = scoring._get_real_bundle(filtered, tmp_path / "unused.h5ad")

    assert pd.read_csv(bundle / "scoring_genes.csv")["gene"].tolist() == ["G0", "G2"]
    assert fake_cell_eval2 == ["baseline", "prep-real-bundle"]
    assert scoring._get_real_bundle(filtered, tmp_path / "unused.h5ad") == bundle
    assert fake_cell_eval2 == ["baseline", "prep-real-bundle"]
    assert [path.name for path in (tmp_path / "bundles").iterdir()] == [bundle.name]


REFUSAL = (
    "prep-real-bundle: metric 'de_wilcoxon_sig_jaccard' has no usable replicate scale: the anchor (0.59855) "
    "leaves no usable headroom over the baseline (0.704545). Every vcc2026 member is decisive"
)


BASELINE_REFUSAL = (
    "ValueError: expr_mse_unbiased_capped_norm: the sum of expr_distance_unbiased over 10 perturbation(s) is "
    "-0.0031749501165267862, which is not positive, so the ratio of sums is undefined or sign-flipped. That means "
    "this reference panel carries no measurable aggregate effect at this cell depth -- a property of the REFERENCE, "
    "not of the submission. Fix or replace the reference panel."
)


def _fail_prep_real_bundle(monkeypatch, message, failing="prep-real-bundle"):
    run = scoring._run_cell_eval2

    def fail(subcommand, *arguments):
        run(subcommand, *arguments)
        if subcommand == failing:
            raise RuntimeError(f"cell-eval2 {failing} failed with exit code 1:\nsome warning\n{message}\n")

    monkeypatch.setattr(scoring, "_run_cell_eval2", fail)


@pytest.mark.parametrize(
    "failing, message, subcommands",
    [
        ("prep-real-bundle", REFUSAL, ["baseline", "prep-real-bundle"]),
        ("baseline", BASELINE_REFUSAL, ["baseline"]),
    ],
)
def test_refused_bundle_is_unscorable_and_the_refusal_is_cached(
    tmp_path, fake_cell_eval2, monkeypatch, failing, message, subcommands
):
    real = cells([[1, 1], [1, 1]], ["A", "A"], [CONTROL_LABEL, "G0"], "csr")
    _fail_prep_real_bundle(monkeypatch, message, failing)

    for _ in range(2):
        with pytest.raises(scoring.UnscorableContext) as error:
            scoring._get_real_bundle(real, tmp_path / "unused.h5ad")
        assert str(error.value) == message
    assert fake_cell_eval2 == subcommands
    assert len(list((tmp_path / "bundles").iterdir())) == 1


def test_other_bundle_failures_still_fail_and_are_not_cached(tmp_path, fake_cell_eval2, monkeypatch):
    real = cells([[1, 1], [1, 1]], ["A", "A"], [CONTROL_LABEL, "G0"], "csr")
    _fail_prep_real_bundle(monkeypatch, "CUDA_ERROR_OUT_OF_MEMORY")

    for _ in range(2):
        with pytest.raises(RuntimeError, match="CUDA_ERROR_OUT_OF_MEMORY") as error:
            scoring._get_real_bundle(real, tmp_path / "unused.h5ad")
        assert not isinstance(error.value, scoring.UnscorableContext)
    assert fake_cell_eval2 == ["baseline", "prep-real-bundle"] * 2
    assert list((tmp_path / "bundles").iterdir()) == []


def _real_with_contexts():
    """Context A has 2 splittable perturbations; context B has only one (G1; G2 has a single cell)."""
    rows = [("A", CONTROL_LABEL)] * 2 + [("A", "G1")] * 2 + [("A", "G2")] * 2
    rows += [("B", CONTROL_LABEL)] * 2 + [("B", "G1")] * 2 + [("B", "G2")]
    contexts, targets = zip(*rows)
    matrix = np.ones((len(rows), 3), dtype=np.float32)
    matrix[0, 2] = np.nan
    return cells(matrix, list(contexts), list(targets), "csr")


def test_unscorable_contexts_reports_refused_and_unsplittable_contexts(tmp_path, fake_cell_eval2, monkeypatch):
    real_path = write(_real_with_contexts(), tmp_path / "real.h5ad")
    _fail_prep_real_bundle(monkeypatch, REFUSAL)

    assert scoring.unscorable_contexts(real_path) == {
        "A": REFUSAL,
        "B": "fewer than 2 perturbations with at least 2 measured cells",
    }


def test_unscorable_contexts_is_empty_when_every_bundle_builds(tmp_path, fake_cell_eval2):
    real = _real_with_contexts()
    real = real[(real.obs["context"] == "A").to_numpy()].copy()

    assert scoring.unscorable_contexts(write(real, tmp_path / "real.h5ad")) == {}
    assert fake_cell_eval2 == ["baseline", "prep-real-bundle"]


def test_checked_bundle_is_the_one_scoring_reuses(tmp_path, fake_cell_eval2, monkeypatch):
    real = _real_with_contexts()
    real_path = write(real[(real.obs["context"] == "A").to_numpy()].copy(), tmp_path / "real.h5ad")
    pred_path = write(cells(np.ones((4, 3)), ["A"] * 4, ["G1", "G1", "G2", "G2"], "csr"), tmp_path / "pred.h5ad")
    assert scoring.unscorable_contexts(real_path) == {}

    run = scoring._run_cell_eval2

    def run_or_score(subcommand, *arguments):
        if subcommand != "score":
            return run(subcommand, *arguments)
        fake_cell_eval2.append(subcommand)
        output = Path(arguments[list(arguments).index("-o") + 1])
        pd.DataFrame({"metric": ["avg_score"], "from_replicate": [0.5]}).to_csv(output, index=False)

    monkeypatch.setattr(scoring, "_run_cell_eval2", run_or_score)
    assert scoring._score(real_path, pred_path) == {"avg_score": 0.5}
    assert fake_cell_eval2 == ["baseline", "prep-real-bundle", "run", "score"]


def test_filtered_gene_panel_changes_bundle_identity(monkeypatch):
    real = cells([[1, 2, 3]], ["A"], ["G0"], "csr")
    monkeypatch.setattr(scoring, "_cell_eval2_flags", lambda: scoring.CELL_EVAL2_FLAGS)
    assert scoring._content_digest(real[:, [0, 1]]) != scoring._content_digest(real[:, [0, 2]])
    assert scoring._content_digest(real[:, [0, 1]]) == scoring._content_digest(real[:, [0, 1]].copy())


def test_bundle_identity_distinguishes_missing_from_zero(monkeypatch):
    monkeypatch.setattr(scoring, "_cell_eval2_flags", lambda: scoring.CELL_EVAL2_FLAGS)
    missing = cells([[1, np.nan]], ["A"], ["G0"], "csr")
    zero = cells([[1, 0]], ["A"], ["G0"], "csr")
    assert scoring._content_digest(missing) == scoring._content_digest(missing.copy())
    assert scoring._content_digest(missing) != scoring._content_digest(zero)


# --- Scoring reference built from a split ---------------------------------------------------------


def test_reference_keeps_missing_values_of_labels_and_controls(tmp_path):
    from agentomics.runtime.conditional_generation_evaluation import write_reference

    controls, labels = _with_value((10, 4), 0, 0, 0), _with_value((9, 4), 0, 0, 2)
    split = _write_split(tmp_path / "validation", controls_values=controls, labels_values=labels)

    write_reference(split, tmp_path / "reference.h5ad")

    reference = ad.read_h5ad(tmp_path / "reference.h5ad")
    np.testing.assert_array_equal(dense(reference), np.vstack([labels, controls]))
    for context in ("A", "B"):
        in_context = reference[(reference.obs["context"] == context).to_numpy()]
        expected = [True, True, True, False] if context == "B" else [True, False, True, False]
        np.testing.assert_array_equal(scoring._scoring_genes(in_context, context), expected)


# --- Agent instructions ---------------------------------------------------------------------------


def test_task_description_explains_missing_values_only_for_generation():
    from agentomics.agents.prompt_builder import _task_description

    description = _task_description(TaskTypes.CONDITIONAL_GENERATION)
    assert "NaN for unavailable gene measurements" in description
    assert "do not interpret them as observed zeros" in description
    assert "NaN" not in _task_description(TaskTypes.REGRESSION)


def test_metric_description_states_what_the_scorer_leaves_out():
    from agentomics.utils.metrics import CONDITIONAL_GENERATION_METRICS

    description = CONDITIONAL_GENERATION_METRICS["AVG_SCORE"][1]
    assert "genes observed (not NaN) in all its real cells" in description
    assert "fewer than 2 real cells" in description
    assert "'no usable replicate scale'" in description and "left out of the average" in description


def test_inference_instructions_forbid_missing_values_in_predictions():
    from agentomics.agents.steps.model_inference import ModelInferenceStep

    description = ModelInferenceStep._generated_cells_description()
    assert "must not contain NaN or infinity" in description
    assert "do not copy missing values into the output" in description


def test_split_instructions_preserve_missing_values():
    from agentomics.agents.steps.data_split import DataSplitStep

    def instructions(task_type):
        return DataSplitStep._split_files_instructions(SimpleNamespace(config=SimpleNamespace(task_type=task_type)))

    assert "NaNs for unavailable measurements" in instructions(TaskTypes.CONDITIONAL_GENERATION)
    assert "Do not fill missing measurements with zeros" in instructions(TaskTypes.CONDITIONAL_GENERATION)
    assert instructions(TaskTypes.CLASSIFICATION) == ""


# --- mini_train check -----------------------------------------------------------------------------


def test_mini_train_check_rejects_real_data_cell_eval2_refuses(tmp_path, monkeypatch):
    from pydantic_ai import ModelRetry

    from agentomics.agents.steps.data_split import DataSplitStep

    checked = []

    def unscorable(reference_path):
        reference = ad.read_h5ad(reference_path)
        checked.append(sorted(set(reference.obs["target_gene"])))
        return {"B": REFUSAL}

    monkeypatch.setattr(scoring, "unscorable_contexts", unscorable)
    split = _write_split(tmp_path / "mini_train")

    with pytest.raises(ModelRetry, match=r"Choose other conditions\. Refused contexts: B: prep-real-bundle"):
        DataSplitStep._validate_mini_train_scorable(split)
    assert checked == [["G1", "G2", CONTROL_LABEL]]


def test_mini_train_check_accepts_scorable_real_data(tmp_path, monkeypatch):
    from agentomics.agents.steps.data_split import DataSplitStep

    monkeypatch.setattr(scoring, "unscorable_contexts", lambda reference_path: {})
    DataSplitStep._validate_mini_train_scorable(_write_split(tmp_path / "mini_train"))
