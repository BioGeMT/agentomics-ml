"""Scores conditional generation predictions against the measured cells of a split."""
from __future__ import annotations

import tempfile
from pathlib import Path

from agentomics.datasets.conditional_generation_contract import (
    CELL_OBS_COLUMNS,
    CONTEXT_COLUMN_NAME,
    CONTROLS_FILE_NAME,
)
from agentomics.datasets.data_contract import GENERATION_LABELS_FILE_NAME, INPUT_DIR_NAME
from agentomics.utils.metrics import get_conditional_generation_metrics_functions


def get_conditional_generation_metrics(predictions_path: Path, labels_path: Path) -> dict[str, float]:
    """Scores predictions.h5ad against the split that labels_path (its labels.csv) belongs to."""
    split_path = Path(labels_path).parent
    with tempfile.TemporaryDirectory() as temporary_directory:
        reference_path = Path(temporary_directory) / "reference.h5ad"
        write_reference(split_path, reference_path)
        return {
            name: metric(reference_path, predictions_path)
            for name, metric in get_conditional_generation_metrics_functions().items()
        }


def write_reference(split_path: Path, reference_path: Path) -> None:
    """Writes the scoring reference: the measured perturbed cells plus the control cells of their contexts."""
    import anndata as ad

    perturbed = ad.read_h5ad(split_path / GENERATION_LABELS_FILE_NAME)
    controls = ad.read_h5ad(split_path / INPUT_DIR_NAME / CONTROLS_FILE_NAME)
    # Controls of contexts without measured cells would make the scorer expect predictions for them.
    contexts = perturbed.obs[CONTEXT_COLUMN_NAME].unique()
    controls = controls[controls.obs[CONTEXT_COLUMN_NAME].isin(contexts).to_numpy()]

    reference = ad.concat([perturbed, controls], keys=["perturbed", "control"], index_unique="-")
    reference.obs = reference.obs[CELL_OBS_COLUMNS]
    # Plain strings, so the h5ad is writable and readable under pandas 2 (see cell_eval2_scoring).
    with ad.settings.override(allow_write_nullable_strings=False):
        reference.write_h5ad(reference_path)
