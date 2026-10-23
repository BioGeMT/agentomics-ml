"""Split contract for the conditional generation task.

A split folder holds:
    input/controls.h5ad      control cells (raw counts), obs: context, target_gene == "non-targeting"
    input/perturbations.csv  conditions to generate: context, target_gene, n_cells
    labels.h5ad              measured perturbed cells (raw counts), obs: context, target_gene

input/ is all inference receives; labels.h5ad is the ground truth.
"""
from pathlib import Path

import pandas as pd

from agentomics.datasets.data_contract import (
    GENERATION_LABELS_FILE_NAME,
    ID_COLUMN_NAME,
    INPUT_DIR_NAME,
)

CONTROLS_FILE_NAME = "controls.h5ad"
PERTURBATIONS_FILE_NAME = "perturbations.csv"
CONTEXT_COLUMN_NAME = "context"
TARGET_GENE_COLUMN_NAME = "target_gene"
N_CELLS_COLUMN_NAME = "n_cells"
CONTROL_LABEL = "non-targeting"
CONDITION_ID_SEPARATOR = "|"
MAX_COUNTS_PER_CELL = 1_000_000

PERTURBATION_COLUMNS = [CONTEXT_COLUMN_NAME, TARGET_GENE_COLUMN_NAME, N_CELLS_COLUMN_NAME]
CELL_OBS_COLUMNS = [CONTEXT_COLUMN_NAME, TARGET_GENE_COLUMN_NAME]


def condition_ids(contexts, target_genes) -> list[str]:
    return [f"{context}{CONDITION_ID_SEPARATOR}{target}" for context, target in zip(contexts, target_genes)]


def validate_conditional_generation_split(split_path: Path) -> None:
    """Validates the generation files of one split against each other."""
    split_path = Path(split_path)
    input_path = split_path / INPUT_DIR_NAME
    perturbations = read_perturbations(input_path / PERTURBATIONS_FILE_NAME)
    controls_obs, controls_genes = _read_cells(input_path / CONTROLS_FILE_NAME)
    truth_obs, truth_genes = _read_cells(split_path / GENERATION_LABELS_FILE_NAME)
    expected_ids = set(perturbations[ID_COLUMN_NAME])

    # controls.h5ad: only control cells, for every context that has conditions
    non_controls = controls_obs[TARGET_GENE_COLUMN_NAME] != CONTROL_LABEL
    if non_controls.any():
        raise ValueError(
            f"{CONTROLS_FILE_NAME} must contain only '{CONTROL_LABEL}' cells, found "
            f"{_head(controls_obs.loc[non_controls, TARGET_GENE_COLUMN_NAME].unique())}"
        )
    contexts_without_controls = set(perturbations[CONTEXT_COLUMN_NAME]) - set(controls_obs[CONTEXT_COLUMN_NAME])
    if contexts_without_controls:
        raise ValueError(f"{CONTROLS_FILE_NAME} has no control cells for contexts {_head(contexts_without_controls)}")

    # labels.h5ad: measured cells for exactly the listed conditions, on the controls' genes
    if (truth_obs[TARGET_GENE_COLUMN_NAME] == CONTROL_LABEL).any():
        raise ValueError(
            f"{GENERATION_LABELS_FILE_NAME} must not contain control cells; they belong in {CONTROLS_FILE_NAME}"
        )
    truth_ids = set(condition_ids(truth_obs[CONTEXT_COLUMN_NAME], truth_obs[TARGET_GENE_COLUMN_NAME]))
    _require_same_ids(truth_ids, expected_ids, GENERATION_LABELS_FILE_NAME)
    if truth_genes != controls_genes:
        raise ValueError(
            f"{GENERATION_LABELS_FILE_NAME} and {CONTROLS_FILE_NAME} must have the same genes (var_names), in the same order"
        )


def validate_generated_cells(predictions_path: Path, input_path: Path) -> None:
    """Validates inference output (an h5ad of generated cells) against the input/ it was generated from."""
    import anndata as ad
    import numpy as np
    from scipy import sparse

    perturbations = read_perturbations(Path(input_path) / PERTURBATIONS_FILE_NAME)
    _, control_genes = _read_cells(Path(input_path) / CONTROLS_FILE_NAME)
    try:
        generated = ad.read_h5ad(predictions_path)
    except Exception as error:
        raise ValueError(f"{predictions_path} could not be read as an h5ad file: {error}") from error

    missing_columns = set(CELL_OBS_COLUMNS) - set(generated.obs.columns)
    if missing_columns:
        raise ValueError(f"generated cells are missing obs columns: {sorted(missing_columns)}")
    obs = generated.obs[CELL_OBS_COLUMNS].astype(str)
    if (obs[TARGET_GENE_COLUMN_NAME] == CONTROL_LABEL).any():
        raise ValueError(f"generated cells must not include control ('{CONTROL_LABEL}') cells")
    if list(generated.var_names) != control_genes:
        raise ValueError(f"generated cells must have the genes (var_names) of {CONTROLS_FILE_NAME}, in the same order")

    generated_counts = pd.Series(
        condition_ids(obs[CONTEXT_COLUMN_NAME], obs[TARGET_GENE_COLUMN_NAME])
    ).value_counts()
    expected_counts = perturbations.set_index(ID_COLUMN_NAME)[N_CELLS_COLUMN_NAME]
    _require_same_ids(set(generated_counts.index), set(expected_counts.index), "generated cells")
    wrong_counts = expected_counts.index[generated_counts.reindex(expected_counts.index) != expected_counts]
    if len(wrong_counts):
        raise ValueError(
            f"each condition needs exactly its {N_CELLS_COLUMN_NAME} generated cells; wrong counts for {_head(wrong_counts)}"
        )

    values = generated.X.data if sparse.issparse(generated.X) else np.asarray(generated.X)
    if not np.isfinite(values).all() or (values < 0).any() or (values != np.round(values)).any():
        raise ValueError("generated values must be raw counts: finite, non-negative whole numbers")
    if (np.asarray(generated.X.sum(axis=1)).ravel() > MAX_COUNTS_PER_CELL).any():
        raise ValueError(f"a generated cell's counts must not total more than {MAX_COUNTS_PER_CELL}")


def read_perturbations(perturbations_path: Path) -> pd.DataFrame:
    """Reads and validates perturbations.csv, adding the condition id column."""
    perturbations_path = Path(perturbations_path)
    if not perturbations_path.is_file():
        raise ValueError(f"{PERTURBATIONS_FILE_NAME} does not exist: {perturbations_path}")
    perturbations = pd.read_csv(perturbations_path, dtype=str, keep_default_na=False)

    if set(perturbations.columns) != set(PERTURBATION_COLUMNS):
        raise ValueError(f"{perturbations_path} must contain exactly columns {PERTURBATION_COLUMNS}")
    if perturbations.empty:
        raise ValueError(f"{perturbations_path} contains no conditions")
    for column in CELL_OBS_COLUMNS:
        if (perturbations[column].str.strip() == "").any():
            raise ValueError(f"{perturbations_path} has empty {column} values")
    if (perturbations[TARGET_GENE_COLUMN_NAME] == CONTROL_LABEL).any():
        raise ValueError(f"{perturbations_path} must not list '{CONTROL_LABEL}' as a condition")

    n_cells = pd.to_numeric(perturbations[N_CELLS_COLUMN_NAME], errors="coerce")
    if not ((n_cells > 0) & (n_cells == n_cells.round())).all():
        raise ValueError(f"{perturbations_path} {N_CELLS_COLUMN_NAME} must be positive integers")
    perturbations[N_CELLS_COLUMN_NAME] = n_cells.astype(int)

    perturbations[ID_COLUMN_NAME] = condition_ids(
        perturbations[CONTEXT_COLUMN_NAME], perturbations[TARGET_GENE_COLUMN_NAME]
    )
    duplicates = perturbations[ID_COLUMN_NAME][perturbations[ID_COLUMN_NAME].duplicated()]
    if not duplicates.empty:
        raise ValueError(f"{perturbations_path} lists conditions more than once: {_head(duplicates.unique())}")
    return perturbations


def _read_cells(h5ad_path: Path) -> tuple[pd.DataFrame, list[str]]:
    """Reads the cell annotations and gene names of an h5ad without loading its matrix."""
    import anndata as ad

    if not h5ad_path.is_file():
        raise ValueError(f"{h5ad_path.name} does not exist: {h5ad_path}")
    adata = ad.read_h5ad(h5ad_path, backed="r")
    try:
        missing = set(CELL_OBS_COLUMNS) - set(adata.obs.columns)
        if missing:
            raise ValueError(f"{h5ad_path} is missing obs columns: {sorted(missing)}")
        return adata.obs[CELL_OBS_COLUMNS].astype(str), list(adata.var_names)
    finally:
        adata.file.close()


def _require_same_ids(actual: set[str], expected: set[str], file_name: str) -> None:
    missing = expected - actual
    extra = actual - expected
    if missing or extra:
        raise ValueError(
            f"{file_name} conditions must match {PERTURBATIONS_FILE_NAME}. "
            f"Missing: {_head(missing)}; extra: {_head(extra)} (showing up to 10)"
        )


def _head(values, n: int = 10) -> list[str]:
    return sorted(map(str, values))[:n]
