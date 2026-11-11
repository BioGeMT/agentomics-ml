"""Split contract for the temporal generation task (Virtual Embryo Challenge, Task 1).

Same layout as conditional generation, with developmental stages as the conditions:
    input/controls.h5ad      observed cells of the earlier stages (log1p-normalised), obs: stage
    input/perturbations.csv  target stages to generate cells for: stage, n_cells
    labels.h5ad              measured cells of the target stages (log1p-normalised), obs: stage

input/ is all inference receives; labels.h5ad is the ground truth.
Every target stage must be later than every observed stage of its split.
Values must be finite, non-negative and log-scale (at most MAX_LOG_EXPRESSION).
"""
import re
from pathlib import Path

import pandas as pd

from agentomics.datasets.conditional_generation_contract import (
    CONTROLS_FILE_NAME,
    N_CELLS_COLUMN_NAME,
    PERTURBATIONS_FILE_NAME,
)
from agentomics.datasets.data_contract import (
    GENERATION_LABELS_FILE_NAME,
    ID_COLUMN_NAME,
    INPUT_DIR_NAME,
)

STAGE_COLUMN_NAME = "stage"
STAGE_PATTERN = re.compile(r"^E(\d+(?:\.\d+)?)$")
# log1p of a normalised count above this is implausible; raw counts usually exceed it.
MAX_LOG_EXPRESSION = 20.0

CONDITION_COLUMNS = [STAGE_COLUMN_NAME, N_CELLS_COLUMN_NAME]
CELL_OBS_COLUMNS = [STAGE_COLUMN_NAME]


def stage_value(stage: str) -> float:
    """'E9.5' -> 9.5 (embryonic day)."""
    match = STAGE_PATTERN.match(str(stage))
    if match is None:
        raise ValueError(f"stage '{stage}' must look like 'E9.5' (E followed by the embryonic day)")
    return float(match.group(1))


def validate_temporal_generation_split(split_path: Path) -> None:
    """Validates the generation files of one split against each other."""
    split_path = Path(split_path)
    input_path = split_path / INPUT_DIR_NAME
    conditions = read_conditions(input_path / PERTURBATIONS_FILE_NAME)
    observed_obs, observed_genes = read_cells(input_path / CONTROLS_FILE_NAME)
    truth_obs, truth_genes = read_cells(split_path / GENERATION_LABELS_FILE_NAME)

    observed_stages = set(observed_obs[STAGE_COLUMN_NAME])
    target_stages = set(conditions[ID_COLUMN_NAME])
    latest_observed = max(observed_stages, key=stage_value)
    not_later = sorted(s for s in target_stages if stage_value(s) <= stage_value(latest_observed))
    if not_later:
        raise ValueError(
            f"target stages must be later than every observed stage in {CONTROLS_FILE_NAME} "
            f"(latest observed: {latest_observed}); not later: {not_later}"
        )
    _require_same_stages(set(truth_obs[STAGE_COLUMN_NAME]), target_stages, GENERATION_LABELS_FILE_NAME)
    if truth_genes != observed_genes:
        raise ValueError(
            f"{GENERATION_LABELS_FILE_NAME} and {CONTROLS_FILE_NAME} must have the same genes (var_names), in the same order"
        )


def validate_generated_cells(predictions_path: Path, input_path: Path) -> None:
    """Validates inference output (an h5ad of generated cells) against the input/ it was generated from."""
    import anndata as ad
    import numpy as np
    from scipy import sparse

    conditions = read_conditions(Path(input_path) / PERTURBATIONS_FILE_NAME)
    _, observed_genes = read_cells(Path(input_path) / CONTROLS_FILE_NAME)
    try:
        generated = ad.read_h5ad(predictions_path)
    except Exception as error:
        raise ValueError(f"{predictions_path} could not be read as an h5ad file: {error}") from error

    if STAGE_COLUMN_NAME not in generated.obs.columns:
        raise ValueError(f"generated cells are missing obs column '{STAGE_COLUMN_NAME}'")
    if list(generated.var_names) != observed_genes:
        raise ValueError(f"generated cells must have the genes (var_names) of {CONTROLS_FILE_NAME}, in the same order")

    generated_counts = generated.obs[STAGE_COLUMN_NAME].astype(str).value_counts()
    expected_counts = conditions.set_index(ID_COLUMN_NAME)[N_CELLS_COLUMN_NAME]
    _require_same_stages(set(generated_counts.index), set(expected_counts.index), "generated cells")
    wrong_counts = expected_counts.index[generated_counts.reindex(expected_counts.index) != expected_counts]
    if len(wrong_counts):
        raise ValueError(
            f"each target stage needs exactly its {N_CELLS_COLUMN_NAME} generated cells; wrong counts for {sorted(wrong_counts)}"
        )

    values = generated.X.data if sparse.issparse(generated.X) else np.asarray(generated.X)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("generated values must be finite and non-negative")
    if values.size and values.max() > MAX_LOG_EXPRESSION:
        raise ValueError(
            f"generated values must be log1p-normalised expression like the inputs (max {MAX_LOG_EXPRESSION}); "
            f"found {float(values.max()):.1f}, which looks like raw counts"
        )


def read_conditions(conditions_path: Path) -> pd.DataFrame:
    """Reads and validates the target stages file, adding the condition id column (the stage)."""
    conditions_path = Path(conditions_path)
    if not conditions_path.is_file():
        raise ValueError(f"{PERTURBATIONS_FILE_NAME} does not exist: {conditions_path}")
    conditions = pd.read_csv(conditions_path, dtype=str, keep_default_na=False)

    if set(conditions.columns) != set(CONDITION_COLUMNS):
        raise ValueError(f"{conditions_path} must contain exactly columns {CONDITION_COLUMNS}")
    if conditions.empty:
        raise ValueError(f"{conditions_path} contains no target stages")
    for stage in conditions[STAGE_COLUMN_NAME]:
        stage_value(stage)
    n_cells = pd.to_numeric(conditions[N_CELLS_COLUMN_NAME], errors="coerce")
    if not ((n_cells > 0) & (n_cells == n_cells.round())).all():
        raise ValueError(f"{conditions_path} {N_CELLS_COLUMN_NAME} must be positive integers")
    conditions[N_CELLS_COLUMN_NAME] = n_cells.astype(int)

    conditions[ID_COLUMN_NAME] = conditions[STAGE_COLUMN_NAME]
    duplicates = conditions[ID_COLUMN_NAME][conditions[ID_COLUMN_NAME].duplicated()]
    if not duplicates.empty:
        raise ValueError(f"{conditions_path} lists stages more than once: {sorted(duplicates.unique())}")
    return conditions


def read_cells(h5ad_path: Path, *, validate_values: bool = False) -> tuple[pd.DataFrame, list[str]]:
    """Reads stage annotations and genes, optionally validating the expression values in blocks."""
    import anndata as ad

    h5ad_path = Path(h5ad_path)
    if not h5ad_path.is_file():
        raise ValueError(f"{h5ad_path.name} does not exist: {h5ad_path}")
    adata = ad.read_h5ad(h5ad_path, backed="r")
    try:
        if STAGE_COLUMN_NAME not in adata.obs.columns:
            raise ValueError(f"{h5ad_path} is missing obs column '{STAGE_COLUMN_NAME}'")
        obs = adata.obs[CELL_OBS_COLUMNS].astype(str)
        for stage in obs[STAGE_COLUMN_NAME].unique():
            stage_value(stage)
        if validate_values:
            _validate_log_expression(adata, h5ad_path)
        return obs, list(adata.var_names)
    finally:
        adata.file.close()


def _validate_log_expression(adata, h5ad_path: Path) -> None:
    import numpy as np
    from scipy import sparse

    for matrix, start, end in adata.chunked_X(4096):
        values = matrix.data if sparse.issparse(matrix) else np.asarray(matrix)
        if not np.isfinite(values).all() or (values < 0).any() or (values.size and values.max() > MAX_LOG_EXPRESSION):
            raise ValueError(
                f"{h5ad_path} rows [{start}, {end}) contain invalid values: expected finite, non-negative "
                f"log1p-normalised expression (at most {MAX_LOG_EXPRESSION})"
            )


def _require_same_stages(actual: set[str], expected: set[str], file_name: str) -> None:
    missing = expected - actual
    extra = actual - expected
    if missing or extra:
        raise ValueError(
            f"{file_name} stages must match {PERTURBATIONS_FILE_NAME}. Missing: {sorted(missing)}; extra: {sorted(extra)}"
        )
