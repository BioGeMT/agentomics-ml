from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from agentomics.datasets.conditional_generation_contract import CONTROL_LABEL, validate_generated_cells
from agentomics.datasets.data_contract import record_input_dir_structure, validate_splits
from agentomics.datasets.dataset_preparation import prepare_dataset, prepare_test_dataset
from agentomics.utils.task_types import TaskTypes

ad = pytest.importorskip("anndata")

GENES = ["G0", "G1", "G2", "G3"]
CONDITIONS = [("A", "G1"), ("A", "G2"), ("B", "G1")]
N_CELLS = 3


def _cells(obs_rows: list[tuple[str, str]], genes=GENES):
    adata = ad.AnnData(np.ones((len(obs_rows), len(genes)), dtype=np.float32))
    adata.obs = pd.DataFrame(obs_rows, columns=["context", "target_gene"], index=[str(i) for i in range(len(obs_rows))])
    adata.var_names = genes
    return adata


def _write_split(split: Path, label_column: str = "numeric_label") -> Path:
    (split / "input").mkdir(parents=True)
    _cells([(context, CONTROL_LABEL) for context in ("A", "B") for _ in range(5)]).write_h5ad(
        split / "input" / "controls.h5ad"
    )
    pd.DataFrame(
        [(context, target, N_CELLS) for context, target in CONDITIONS],
        columns=["context", "target_gene", "n_cells"],
    ).to_csv(split / "input" / "perturbations.csv", index=False)
    pd.DataFrame(
        {"id": [f"{context}|{target}" for context, target in CONDITIONS], label_column: N_CELLS}
    ).to_csv(split / "labels.csv", index=False)
    _cells([condition for condition in CONDITIONS for _ in range(N_CELLS)]).write_h5ad(split / "labels.h5ad")
    return split


def _validate(split: Path, task_type=TaskTypes.CONDITIONAL_GENERATION) -> None:
    validate_splits({"train": split}, record_input_dir_structure(split / "input"), task_type)


def test_valid_split_passes(tmp_path):
    _validate(_write_split(tmp_path / "train"))


def test_labels_h5ad_is_only_allowed_for_conditional_generation(tmp_path):
    split = _write_split(tmp_path / "train")
    with pytest.raises(ValueError, match="unsupported top-level entries"):
        _validate(split, task_type=TaskTypes.REGRESSION)


def _drop_condition_from_truth(split):
    _cells([condition for condition in CONDITIONS[:-1] for _ in range(N_CELLS)]).write_h5ad(split / "labels.h5ad")


def _add_controls_to_truth(split):
    _cells([*CONDITIONS, ("A", CONTROL_LABEL)]).write_h5ad(split / "labels.h5ad")


def _drop_context_from_controls(split):
    _cells([("A", CONTROL_LABEL)] * 5).write_h5ad(split / "input" / "controls.h5ad")


def _reorder_truth_genes(split):
    _cells(CONDITIONS, genes=list(reversed(GENES))).write_h5ad(split / "labels.h5ad")


def _change_label(split):
    labels = pd.read_csv(split / "labels.csv")
    labels.loc[0, "numeric_label"] = N_CELLS + 1
    labels.to_csv(split / "labels.csv", index=False)


def _duplicate_perturbation(split):
    perturbations = pd.read_csv(split / "input" / "perturbations.csv")
    pd.concat([perturbations, perturbations.head(1)]).to_csv(split / "input" / "perturbations.csv", index=False)


def _remove_truth(split):
    (split / "labels.h5ad").unlink()


@pytest.mark.parametrize(
    ("break_split", "message"),
    [
        (_drop_condition_from_truth, "labels.h5ad conditions must match"),
        (_add_controls_to_truth, "must not contain control cells"),
        (_drop_context_from_controls, "no control cells for contexts"),
        (_reorder_truth_genes, "same genes"),
        (_change_label, "labels must equal"),
        (_duplicate_perturbation, "more than once"),
        (_remove_truth, "labels.h5ad does not exist"),
    ],
)
def test_inconsistent_split_is_rejected(tmp_path, break_split, message):
    split = _write_split(tmp_path / "train")
    break_split(split)
    with pytest.raises(ValueError, match=message):
        _validate(split)


def _write_source_dataset(dataset: Path, splits=("train", "validation")) -> Path:
    for split_name in splits:
        _write_split(dataset / split_name, label_column="label")
    (dataset / "metadata.json").write_text('{"task_type": "conditional_generation"}', encoding="utf-8")
    return dataset


def test_preparation_links_ground_truth_cells_next_to_labels(tmp_path):
    source = _write_source_dataset(tmp_path / "source")

    metadata = prepare_dataset(source, tmp_path / "prepared")

    assert metadata["task_type"] == TaskTypes.CONDITIONAL_GENERATION
    for split_name in ("train", "validation"):
        prepared_split = tmp_path / "prepared" / split_name
        assert (prepared_split / "labels.h5ad").resolve() == (source / split_name / "labels.h5ad").resolve()
        labels = pd.read_csv(prepared_split / "labels.csv")
        assert list(labels.columns) == ["id", "numeric_label"]
        assert (labels["numeric_label"] == N_CELLS).all()


def test_preparation_rejects_a_split_without_ground_truth_cells(tmp_path):
    source = _write_source_dataset(tmp_path / "source")
    (source / "validation" / "labels.h5ad").unlink()

    with pytest.raises(ValueError, match="validation split is missing labels.h5ad"):
        prepare_dataset(source, tmp_path / "prepared")


def test_test_split_preparation_links_ground_truth_cells(tmp_path):
    source = _write_source_dataset(tmp_path / "source", splits=("train", "test"))
    metadata = prepare_dataset(source, tmp_path / "prepared")

    prepare_test_dataset(
        source,
        tmp_path / "prepared_test",
        task_type=metadata["task_type"],
        input_structure=metadata["input_structure"],
    )

    assert (tmp_path / "prepared_test" / "test" / "labels.h5ad").is_symlink()


def _write_generated(path: Path, rows=None, genes=GENES, values=1.0) -> Path:
    rows = rows if rows is not None else [condition for condition in CONDITIONS for _ in range(N_CELLS)]
    generated = _cells(rows, genes=genes)
    generated.X = np.full(generated.shape, values, dtype=np.float32)
    generated.write_h5ad(path)
    return path


def test_generated_cells_matching_the_input_pass(tmp_path):
    split = _write_split(tmp_path / "mini_train")
    validate_generated_cells(_write_generated(tmp_path / "generated.h5ad"), split / "input")


@pytest.mark.parametrize(
    ("generated_kwargs", "message"),
    [
        ({"rows": [condition for condition in CONDITIONS[:-1] for _ in range(N_CELLS)]}, "conditions must match"),
        ({"rows": [condition for condition in CONDITIONS for _ in range(N_CELLS - 1)]}, "wrong counts"),
        ({"rows": [*[c for c in CONDITIONS for _ in range(N_CELLS)], ("A", CONTROL_LABEL)]}, "control"),
        ({"genes": list(reversed(GENES))}, "same order"),
        ({"values": 0.5}, "whole numbers"),
        ({"values": -1.0}, "whole numbers"),
        ({"values": 300_000.0}, "must not total more than"),
    ],
)
def test_invalid_generated_cells_are_rejected(tmp_path, generated_kwargs, message):
    split = _write_split(tmp_path / "mini_train")
    generated = _write_generated(tmp_path / "generated.h5ad", **generated_kwargs)

    with pytest.raises(ValueError, match=message):
        validate_generated_cells(generated, split / "input")


def test_generated_output_that_is_not_h5ad_is_rejected(tmp_path):
    split = _write_split(tmp_path / "mini_train")
    not_h5ad = tmp_path / "generated.h5ad"
    not_h5ad.write_text("id,prediction\n", encoding="utf-8")

    with pytest.raises(ValueError, match="could not be read as an h5ad"):
        validate_generated_cells(not_h5ad, split / "input")
